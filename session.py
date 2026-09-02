"""A recording session, batch mode, and naming a finished transcript."""

import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime

import numpy as np

from audio import Chunker, Vad
from backends import GeminiLive
from config import (FRAME_BYTES, FRAME_MS, LABELS, LIMIT_ASD, LIMIT_ASH, LIMIT_RPD,
                    LIMIT_RPM, SAMPLE_RATE, compact, confirm, die, hms, sanitize_title)
from devices import Capture, resolve_source
from quota import print_quota
from ui import KeyReader, Screen, State


def apply_title(path, gemini, mode):
    """Rename a finished transcript to '<original stem> <topic>'.

    mode: True — always, False — never, None — ask, which is what happens when
    the filename was generated rather than chosen. A non-interactive run keeps
    the default and titles anyway, so unattended batches still get named.

    Returns the path to use from here on — the original one if anything went
    wrong, because a transcript sitting under a dull name beats one lost to a
    failed rename.
    """
    if mode is False or gemini is None or not path.exists():
        return path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return path
    body = "\n".join(l for l in text.splitlines() if l and not l.startswith("#"))
    if len(body) < 200:                       # too little said to name it
        return path
    if mode is None and not confirm("Name this transcript by its subject?"):
        return path
    if len(body) > 40000:                     # head and tail are enough for a title
        body = body[:20000] + "\n[...]\n" + body[-20000:]
    try:
        raw, model = gemini.title(body)
    except Exception as e:
        print(f"title: skipped ({e})", file=sys.stderr)
        return path
    slug = sanitize_title(raw)
    if not slug:
        return path
    target = path.with_name(f"{path.stem} {slug}{path.suffix}")
    i = 2
    while target.exists() and target != path:
        target = path.with_name(f"{path.stem} {slug} ({i}){path.suffix}")
        i += 1
    try:
        path.rename(target)
    except OSError as e:
        print(f"title: could not rename ({e})", file=sys.stderr)
        return path
    print(f"Title: {slug}   [{model}]")
    return target


class Session:
    """One recording session: capture -> VAD/chunking -> backend -> file and screen.

    This used to be a single 330-line function whose thirteen closures all shared
    the same handful of objects. They are methods now, which is what that shared
    state was asking for all along, and each one can be exercised on its own.
    """

    def __init__(self, args, backend, usage, out_path, gemini=None):
        self.args = args
        self.backend = backend
        self.usage = usage
        self.gemini = gemini
        self.out_path = out_path

        self.stop = threading.Event()
        self.state = State()
        self.work = queue.Queue()
        self.sources = ["mic", "speaker"] if args.source == "both" else [args.source]
        self.caps = []

        # Resolve the devices before creating Screen: resolve_source may terminate
        # the program via die(), and by then Screen would have already hidden the
        # cursor, narrowed the scrolling region and enabled the kitty protocol.
        self.devices = {}
        if "mic" in self.sources:
            self.devices["mic"] = resolve_source("mic", args.mic_device)
        if "speaker" in self.sources:
            self.devices["speaker"] = resolve_source("speaker", args.speaker_device)

        self.screen = Screen(not args.no_tui)
        self.vads = {s: Vad(args.vad_threshold) for s in self.sources}
        self.prompts = {s: "" for s in self.sources}
        # one writer per source: Wave_write is not thread-safe, and mixing mic and
        # speaker into a single mono stream is pointless — two speakers become mush
        self.dump = {"w": {}, "by_frame": bool(args.keep_audio)}

        self.out = open(out_path, "a", encoding="utf-8", buffering=1)
        self.out.write(f"# Transcript — {datetime.now():%Y-%m-%d %H:%M}\n")
        if args.backend == "gemini-live":
            self.out.write(f"# Source: {args.source} | engine: {args.live_model}\n\n")
        else:
            self.out.write(f"# Source: {args.source} | language: {args.lang} "
                           f"| model: {args.model}\n\n")

        self.chunkers = {s: Chunker(s, args.min_chunk, args.max_chunk, args.silence,
                                    self._emit_chunk) for s in self.sources}
        self.live = self._make_live()

    # --- output ----------------------------------------------------------

    def _dump_path(self, src):
        if self.args.source == "both":
            return self.out_path.with_suffix(f".{LABELS[src]}.wav")
        return self.out_path.with_suffix(".wav")

    def _wout(self, text):
        """Background threads may append after the file is already closed (for
        example if a request hung and join timed out) — silently skip it."""
        if not self.out.closed:
            self.out.write(text)

    def _start_audio_dump(self, reason=None):
        if self.dump["w"]:
            return
        for src in self.sources:
            w = wave.open(str(self._dump_path(src)), "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            self.dump["w"][src] = w
        if reason:
            names = ", ".join(self._dump_path(x).name for x in self.sources)
            self.screen.line(f"\x1b[33m!! {reason} — writing raw audio to {names}, "
                             f"transcribe it later: ./rt.py --file <file>\x1b[0m")

    # --- audio in --------------------------------------------------------

    def _emit_chunk(self, src, audio):
        self.state.pending += 1
        self.work.put((src, audio))

    def _on_frame(self, src, frame):
        if self.state.paused:
            return
        vad = self.vads[src]
        speaking = vad.push(frame)
        self.state.rms[src] = vad.rms
        self.state.speaking[src] = speaking
        if self.live is not None:
            self.live.speaking = speaking
            self.live.feed(frame)
        else:
            self.chunkers[src].feed(frame, speaking)
        if self.dump["by_frame"]:
            w = self.dump["w"].get(src)
            if w is not None:
                w.writeframes((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())

    # --- Gemini Live backend ---------------------------------------------

    def _make_live(self):
        if self.args.backend != "gemini-live":
            return None
        if self.gemini is None:
            die("--backend gemini-live needs a Gemini key")
        if self.args.source == "both":
            die("--backend gemini-live works with one source: -s mic or -s speaker")
        return GeminiLive(self.gemini.key, self._on_final, self._on_interim, self._on_note,
                          model=self.args.live_model, silence_ms=self.args.live_silence,
                          flush_sec=self.args.live_flush)

    def _on_final(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.state.transcript.append((ts, self.sources[0], text))
        self.state.lines += 1
        for i, ln in enumerate(self.screen.wrap(text, indent="") or [""]):
            self.screen.line((f"\x1b[90m[{ts}]\x1b[0m " if i == 0 else "         ") + ln)
        self._wout(f"[{ts}] {text}\n")

    def _on_interim(self, text):
        self.state.interim = text

    def _on_note(self, msg):
        self.screen.line(f"\x1b[33m{msg}\x1b[0m")

    # --- Groq backend ----------------------------------------------------

    def _gate(self, chunk_sec):
        """'ok' — safe to send; 'quota' — the daily limit is used up.

        Hourly limits are simply waited out. But if we are already stopping, wait
        no longer than a minute so Ctrl+C does not hang — the rest goes to wav.
        """
        waited = 0.0
        while True:
            st = self.usage.stats()
            if st["req_day"] >= LIMIT_RPD or st["sec_day"] + chunk_sec > LIMIT_ASD:
                return "quota"
            if st["rpm"] >= LIMIT_RPM - 1 or st["sec_hour"] + chunk_sec > LIMIT_ASH:
                if self.stop.is_set() and waited >= 60:
                    return "quota"
                time.sleep(3)
                waited += 3
                continue
            return "ok"

    def _spill_chunk(self, src, audio):
        """Quota is gone — keep the audio so it can be transcribed later."""
        if self.dump["by_frame"]:
            return
        w = self.dump["w"].get(src)
        if w is not None:
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())

    def _worker(self):
        while True:
            item = self.work.get()
            if item is None:
                break
            src, audio = item
            dur = len(audio) / SAMPLE_RATE
            try:
                if not self.args.no_quota_guard and self._gate(dur) == "quota":
                    if not self.state.quota_out:
                        self.state.quota_out = True
                        self._start_audio_dump("today's limit is used up")
                    self._spill_chunk(src, audio)
                    continue
                text = self.backend.transcribe(audio, self.prompts[src])
                self.usage.add(dur)
                self.state.sent += 1
                if text:
                    self.prompts[src] = (self.prompts[src] + " " + text)[-600:]
                    ts = datetime.now().strftime("%H:%M:%S")
                    self.state.transcript.append((ts, src, text))
                    label = f"[{LABELS[src]}] " if self.args.source == "both" else ""
                    self.screen.line(f"\x1b[90m[{ts}]\x1b[0m {label}{text}")
                    self._wout(f"[{ts}] {label}{text}\n")
                    self.state.lines += 1
            except Exception as e:
                self.state.failed += 1
                self.screen.line(f"\x1b[31m[!] chunk lost: {e}\x1b[0m")
                self._wout(f"[!! chunk of {dur:.0f}s lost: {e}]\n")
            finally:
                self.state.pending -= 1
                self.work.task_done()

    # --- questions -------------------------------------------------------

    def _ask(self, question, full):
        if self.gemini is None:
            self.screen.line("\x1b[31mGemini is not configured: put the key in "
                             "~/.config/transcribe/gemini_key\x1b[0m")
            return
        lines = list(self.state.transcript)
        if not full:
            lines = lines[-self.args.recent:]
        if not lines:
            self.screen.line("\x1b[31mnothing to ask about yet — "
                             "the transcript is empty\x1b[0m")
            return
        scope = (f"whole transcript, {len(lines)} lines" if full
                 else f"last {len(lines)} lines")
        both = self.args.source == "both"
        ctx = "\n".join(f"[{ts}] " + (f"[{LABELS[sr]}] " if both else "") + tx
                        for ts, sr, tx in lines)
        self.screen.line(f"\x1b[1;35m>>> [{scope}] {question}\x1b[0m")
        self._wout(f"\n>>> QUESTION ({scope}): {question}\n")
        threading.Thread(target=self._answer, args=(ctx, question), daemon=True).start()

    def _answer(self, ctx, question):
        self.state.asking += 1
        try:
            answer, model = self.gemini.ask(ctx, question)
        except Exception as e:
            answer, model = f"(Gemini error: {e})", "—"
        finally:
            self.state.asking -= 1
        for ln in self.screen.wrap(answer):
            self.screen.line(f"\x1b[36m{ln}\x1b[0m")
        self.screen.line(f"\x1b[90m    ── {model}\x1b[0m")
        self._wout(f"<<< [{model}] {answer}\n\n")

    # --- keyboard --------------------------------------------------------

    def _hotkey(self, ch):
        """Returns "" to start typing a question about recent lines, "?" about
        the whole transcript, None if the key was handled or means nothing."""
        if ch in ("q", "Q"):
            self.state.quit = True
            self.stop.set()
        elif ch == " ":
            self.state.paused = not self.state.paused
        elif ch in ("m", "M"):
            ts = datetime.now().strftime("%H:%M:%S")
            self.screen.line(f"\x1b[1;36m─── mark {ts} ───\x1b[0m")
            self._wout(f"\n=== MARK {ts} ===\n\n")
        elif ch in ("/", "?"):
            return ch
        return None

    def _keys(self):
        buf = None                      # not None => typing a question
        full_default = False            # scope picked by the opening key
        reader = KeyReader()
        while not self.stop.is_set():
            if not sys.stdin.isatty():
                time.sleep(1)
                continue
            k = reader.key()
            if k is None:
                continue
            kind = k[0]

            if kind == "interrupt":     # Ctrl+C where the signal misses it
                self.state.quit = True
                self.stop.set()
                continue

            if buf is None:
                if kind == "char":
                    opened = self._hotkey(k[1])
                    if opened is not None:
                        # `?` asks about the whole transcript right away: on Windows
                        # Shift+Enter is the same as Enter, so it needs its own key
                        buf = ""
                        full_default = opened == "?"
                        self.screen.set_input(buf, full_default)
                continue

            if kind == "char":
                buf += k[1]
                self.screen.set_input(buf, full_default)
            elif kind == "backspace":
                buf = buf[:-1]
                self.screen.set_input(buf, full_default)
            elif kind == "clear":
                buf = ""
                self.screen.set_input(buf, full_default)
            elif kind == "esc":
                buf = None
                self.screen.set_input(None)
            elif kind in ("enter", "shift_enter"):
                question = buf.strip()
                full = full_default or kind == "shift_enter"
                buf = None
                self.screen.set_input(None)
                if question:
                    self._ask(question, full=full)

    # --- status line -----------------------------------------------------

    def _status(self):
        st = self.usage.stats()
        elapsed = time.time() - self.state.started
        src = self.sources[0]
        level = min(1.0, self.state.rms[src] / 0.15)
        bar = ("▁▂▃▄▅▆▇█"[min(7, int(level * 8))] * 3
               if not self.state.paused else "···")
        mark = ("● PAUSED" if self.state.paused
                else "● SPEECH" if any(self.state.speaking.values()) else "○ silence")
        if self.live is not None:
            link = "live" if self.live.connected else "no link"
            head = (f" {mark} {hms(elapsed)} │ {bar} │ {link} · "
                    f"sessions {self.live.sessions} │ lines {self.state.lines} │ ")
            tail = self.state.interim.replace("\n", " ")
            room = max(0, self.screen.w - len(head) - 2)
            return head + ("…" + tail[-room + 1:] if len(tail) > room else tail)
        buffered = len(self.chunkers[src].buf) * FRAME_MS / 1000
        return (f" {mark} {hms(elapsed)} │ {bar} │ buf {buffered:4.1f}s │ "
                f"queue {self.state.pending} │ lines {self.state.lines} │ "
                f"today {st['req_day']}/{LIMIT_RPD} req · "
                f"{compact(st['sec_day'])}/{compact(LIMIT_ASD)} audio"
                + (" │ ✦ Gemini…" if self.state.asking else "")
                + ("  QUOTA EXHAUSTED" if self.state.quota_out else ""))

    # --- lifecycle -------------------------------------------------------

    def _start(self):
        if self.args.keep_audio:
            self._start_audio_dump()
        self.screen.raw_input_mode()
        for s in self.sources:
            self.screen.line(f"\x1b[90m{LABELS[s]}: {self.devices[s]}\x1b[0m")
        self.screen.line(f"\x1b[90mwriting to {self.out_path}\x1b[0m")
        hint = ("space — pause, m — mark, / — question about the recent lines, "
                "? — about the whole transcript, q — stop")
        self.screen.line("\x1b[90m" + "─" * 3 + f" {hint} " + "─" * 3 + "\x1b[0m")

        self.caps = [Capture(s, self.devices[s], self._on_frame, self.stop, self.state)
                     for s in self.sources]
        if self.live is not None:
            self.live.start()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.keys_thread = threading.Thread(target=self._keys, daemon=True)
        for c in self.caps:
            c.start()
        self.worker_thread.start()
        self.keys_thread.start()

    def _shutdown(self):
        self.screen.set_status(" stopping, finishing the queue…")
        if self.live is not None:
            self.live.stop.set()
            self.live.join(timeout=60)
        for c in self.caps:         # otherwise feed() from capture races with flush()
            c.join(timeout=3)
        for c in self.chunkers.values():
            c.flush()
        self.work.put(None)
        self.worker_thread.join(timeout=180)

        for w in self.dump["w"].values():
            w.close()
        self.out.write(f"\n# end, {hms(time.time() - self.state.started)}\n")
        self.out.close()
        self.screen.close()

    def _report(self):
        self.out_path = apply_title(self.out_path, self.gemini, self.args.title)
        print(f"\nSaved: {self.out_path}")
        print(f"Lines: {self.state.lines} | requests: {self.state.sent} "
              f"| chunks lost: {self.state.failed}")
        for src in self.dump["w"]:
            print(f"Raw audio: {self._dump_path(src)}")
        if self.state.fatal:
            print(f"\nError: {self.state.fatal}", file=sys.stderr)
        print_quota(self.usage)

    def run(self):
        self._start()
        try:
            while not self.stop.is_set():
                self.screen.set_status(self._status())
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        except Exception as e:          # otherwise we would fly past screen.close()
            self.state.fatal = f"UI failure: {e}"
        finally:
            self.stop.set()
        self._shutdown()
        self._report()


def run(args, backend, usage, out_path, gemini=None):
    Session(args, backend, usage, out_path, gemini).run()


def run_file(args, backend, usage, src_path, out_path, gemini=None):
    if not src_path.exists():
        die(f"no such file: {src_path}")
    # stderr into a temp file rather than a pipe: nobody reads the pipe until stdout
    # is fully parsed, and with a noisy ffmpeg a 64KB buffer would be enough to deadlock
    errf = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(src_path), "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=errf, bufsize=FRAME_BYTES * 64)

    vad = Vad(args.vad_threshold)
    pending = []
    ck = Chunker("file", args.min_chunk, args.max_chunk, args.silence,
                 lambda _s, a: pending.append(a))
    out = open(out_path, "a", encoding="utf-8", buffering=1)
    out.write(f"# Transcript of {src_path.name} — {datetime.now():%Y-%m-%d %H:%M}\n")
    out.write(f"# language: {args.lang} | model: {args.model}\n\n")

    prompt = ""
    pos = 0.0
    sent = failed = lines = 0

    def flush_pending():
        nonlocal prompt, sent, failed, lines
        while pending:
            audio = pending.pop(0)
            dur = len(audio) / SAMPLE_RATE
            if not args.no_quota_guard:
                while True:
                    st = usage.stats()
                    if st["req_day"] >= LIMIT_RPD or st["sec_day"] + dur > LIMIT_ASD:
                        die("daily limit used up — continue tomorrow, the file is saved")
                    if st["rpm"] >= LIMIT_RPM - 1 or st["sec_hour"] + dur > LIMIT_ASH:
                        print(f"\r  waiting for a quota slot… ({hms(pos)})", end="", flush=True)
                        time.sleep(3)
                        continue
                    break
            try:
                text = backend.transcribe(audio, prompt)
                usage.add(dur)
                sent += 1
                if text:
                    prompt = (prompt + " " + text)[-600:]
                    out.write(f"[{hms(pos - dur)}] {text}\n")
                    lines += 1
            except Exception as e:
                failed += 1
                out.write(f"[!! chunk at {hms(pos - dur)} lost: {e}]\n")
            print(f"\r  {hms(pos)} | chunks {sent} | lines {lines}"
                  + (f" | lost {failed}" if failed else "") + "   ", end="", flush=True)

    print(f"Transcribing {src_path.name} -> {out_path}")
    while True:
        data = proc.stdout.read(FRAME_BYTES)
        if not data or len(data) < FRAME_BYTES:
            break
        frame = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
        pos += FRAME_MS / 1000
        ck.feed(frame, vad.push(frame))
        flush_pending()
    ck.flush()
    flush_pending()

    errf.seek(0)
    err = errf.read().decode(errors="replace").strip()
    errf.close()
    proc.wait()
    if proc.returncode != 0 and err:
        print(f"\nffmpeg: {err[:300]}", file=sys.stderr)

    out.write(f"\n# end, {hms(pos)} of audio\n")
    out.close()
    out_path = apply_title(out_path, gemini, args.title)
    print(f"\n\nSaved: {out_path}")
    print(f"Lines: {lines} | requests: {sent} | chunks lost: {failed}")
    print_quota(usage)
