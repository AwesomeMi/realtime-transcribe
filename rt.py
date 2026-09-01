#!/usr/bin/env python3
"""
Realtime transcription via the Groq Whisper API. Capture — PipeWire/PulseAudio.

  ./rt.py                     # system audio (Zoom, browser, speakers)
  ./rt.py -s mic -l uk        # microphone
  ./rt.py -s both             # both sources, lines tagged [mic]/[spk]
  ./rt.py --list-devices
  ./rt.py --quota             # how much of the limits is left

Hotkeys:
  space — pause, m — mark, q — stop and save
  /     — ask Gemini about the last N lines (--recent, 8 by default)
  ?     — ask about the whole transcript
            Shift+Enter — also "whole transcript" (needs the kitty keyboard
                          protocol; Windows has none, so only `?` works there)
            Esc         — cancel input, Ctrl+U — clear the line

Keys: $GROQ_API_KEY or ~/.config/transcribe/key
      $GEMINI_API_KEY or ~/.config/transcribe/gemini_key
"""

import argparse
import asyncio
import base64
import codecs
import collections
import io
import json
import os
import queue
import re
import select
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:
    import ctypes
    import msvcrt
else:
    import termios
    import tty

# --- audio ---------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000       # 320
FRAME_BYTES = FRAME_SAMPLES * 2                      # s16le mono

# --- Groq ----------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODELS = {"turbo": "whisper-large-v3-turbo", "large": "whisper-large-v3"}

# free tier limits
LIMIT_RPM = 20
LIMIT_RPD = 2000
LIMIT_ASH = 7200      # seconds of audio per hour
LIMIT_ASD = 28800     # seconds of audio per day
MIN_BILLED = 10       # anything shorter is still billed as 10s

if IS_WINDOWS:
    _APPDATA = Path(os.environ.get("APPDATA") or Path.home())
    CONFIG_DIR = _APPDATA / "transcribe"
    STATE_DIR = Path(os.environ.get("LOCALAPPDATA") or _APPDATA) / "transcribe"
else:
    STATE_DIR = Path(os.environ.get("XDG_STATE_HOME",
                                    Path.home() / ".local/state")) / "transcribe"
    CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "transcribe"

LABELS = {"mic": "mic", "speaker": "spk"}

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

# Fallback chain: the good ones on top, but with 20 requests per day; below —
# the backups with 500 and 14400. Numbers are free tier daily limits as of 2026-09-02.
GEMINI_CHAIN = [
    ("gemini-3.7-flash", 20),
    ("gemini-3.6-flash", 20),
    ("gemini-3-flash-preview", 20),
    ("gemini-2.5-flash", 20),
    ("gemini-3.5-flash", 20),
    ("gemini-3.5-flash-lite", 500),
    ("gemini-3.1-flash-lite", 500),
    ("gemma-4-31b-it", 14400),
]
GEMINI_SYSTEM = (
    "You are helping a student in the middle of a lecture. You are given a speech "
    "transcript produced by automatic recognition: it contains errors, names and terms "
    "may be garbled, punctuation is imprecise — keep that in mind and do not nitpick "
    "typos. Answer briefly and to the point, in the language of the question. If the "
    "transcript does not contain the answer, say so instead of making things up."
)


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def compact(sec):
    sec = int(sec)
    if sec >= 3600:
        return f"{sec // 3600}h{sec % 3600 // 60:02d}m"
    if sec >= 60:
        return f"{sec // 60}m"
    return f"{sec}s"


# ==========================================================================
# Devices
# ==========================================================================

def pactl(*args):
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def _win_pa():
    """PyAudioWPatch is imported lazily: it is not needed on Linux."""
    try:
        import pyaudiowpatch
        return pyaudiowpatch
    except ImportError:
        die("on Windows PyAudioWPatch is required: pip install PyAudioWPatch")


def _win_device_id(index):
    return f"wasapi:{index}"


def _win_device_index(device):
    try:
        prefix, index = device.split(":", 1)
        if prefix != "wasapi":
            raise ValueError
        return int(index)
    except (AttributeError, TypeError, ValueError):
        raise ValueError(f"invalid WASAPI device id: {device!r}") from None


def _win_sources():
    pa = _win_pa()
    out = []
    with pa.PyAudio() as audio:
        host = audio.get_host_api_info_by_type(pa.paWASAPI)
        for index in range(audio.get_device_count()):
            d = audio.get_device_info_by_index(index)
            if d["hostApi"] != host["index"] or d.get("maxInputChannels", 0) <= 0:
                continue
            is_loopback = bool(d.get("isLoopbackDevice"))
            desc = f"Monitor of {d['name'].removesuffix(' [Loopback]')}" if is_loopback else d["name"]
            out.append((_win_device_id(index), desc, is_loopback))
    return out


def _win_default_mic():
    try:
        pa = _win_pa()
        with pa.PyAudio() as audio:
            return _win_device_id(audio.get_default_wasapi_device(d_in=True)["index"])
    except Exception:
        return None


def _win_default_monitor():
    try:
        pa = _win_pa()
        with pa.PyAudio() as audio:
            return _win_device_id(audio.get_default_wasapi_loopback()["index"])
    except Exception:
        return None


def all_sources():
    """[(name, description, is_monitor)] — every recording source."""
    if IS_WINDOWS:
        return _win_sources()
    out = []
    raw = pactl("-f", "json", "list", "sources")
    if raw:
        try:
            for s in json.loads(raw):
                name = s.get("name", "")
                desc = s.get("description") or name
                out.append((name, desc, name.endswith(".monitor")))
            return out
        except Exception:
            pass
    for line in pactl("list", "short", "sources").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append((parts[1], parts[1], parts[1].endswith(".monitor")))
    return out


def default_mic():
    if IS_WINDOWS:
        return _win_default_mic()
    return pactl("get-default-source") or None


def default_monitor():
    if IS_WINDOWS:
        return _win_default_monitor()
    sink = pactl("get-default-sink")
    return f"{sink}.monitor" if sink else None


def resolve_source(kind, hint):
    """kind: 'mic' | 'speaker'. hint — part of the name or description."""
    srcs = all_sources()
    if hint:
        h = hint.lower()
        for name, desc, is_mon in srcs:
            if h in name.lower() or h in desc.lower():
                return name
        die(f"device '{hint}' not found, check --list-devices")
    name = default_monitor() if kind == "speaker" else default_mic()
    if not name:
        die(f"could not determine the default device for '{kind}'")
    known = {s[0] for s in srcs}
    if name not in known:
        # the default sink's monitor may not be up — take the first matching one
        want_mon = kind == "speaker"
        for n, _d, is_mon in srcs:
            if is_mon == want_mon:
                return n
        die(f"no available source for '{kind}'")
    return name


def list_devices():
    srcs = all_sources()
    dm, dmon = default_mic(), default_monitor()
    print("\n=== Microphones (--source mic) ===")
    for name, desc, is_mon in srcs:
        if not is_mon:
            print(f"  {'*' if name == dm else ' '} {desc}\n      {name}")
    print("\n=== Output monitors — system audio (--source speaker) ===")
    for name, desc, is_mon in srcs:
        if is_mon:
            print(f"  {'*' if name == dmon else ' '} {desc}\n      {name}")
    print("\n* — default.  Pick another: --mic-device NAME_PART / --speaker-device NAME_PART\n")


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# ==========================================================================
# Quota accounting (sliding hour / day windows, survives a restart)
# ==========================================================================

class Usage:
    def __init__(self, path):
        self.path = path
        self.events = []          # [[epoch, billed_seconds], ...]
        self.lock = threading.Lock()
        try:
            self.events = json.loads(path.read_text())["events"]
        except Exception:
            self.events = []
        self._prune()

    def _prune(self):
        cutoff = time.time() - 25 * 3600
        self.events = [e for e in self.events if e[0] >= cutoff]

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"events": self.events}))
            tmp.replace(self.path)
        except Exception:
            pass

    def add(self, seconds):
        with self.lock:
            self.events.append([time.time(), max(seconds, MIN_BILLED)])
            self._prune()
            self._save()

    def stats(self):
        now = time.time()
        with self.lock:
            ev = list(self.events)
        rpm = sum(1 for t, _ in ev if t >= now - 60)
        req_h = sum(1 for t, _ in ev if t >= now - 3600)
        day = now - 24 * 3600      # the stored window is longer, cut it explicitly
        req_d = sum(1 for t, _ in ev if t >= day)
        sec_h = sum(s for t, s in ev if t >= now - 3600)
        sec_d = sum(s for t, s in ev if t >= day)
        return {"rpm": rpm, "req_hour": req_h, "req_day": req_d,
                "sec_hour": sec_h, "sec_day": sec_d}


def print_quota(usage):
    st = usage.stats()
    print(f"""
Usage over the sliding windows (local accounting, free tier):

  requests per minute  {st['rpm']:>6} / {LIMIT_RPM}
  requests per day     {st['req_day']:>6} / {LIMIT_RPD}
  audio per hour       {compact(st['sec_hour']):>6} / {compact(LIMIT_ASH)}
  audio per day        {compact(st['sec_day']):>6} / {compact(LIMIT_ASD)}

  left for today       ~{compact(max(0, LIMIT_ASD - st['sec_day']))} of audio,
                       ~{LIMIT_RPD - st['req_day']} requests
""")


# ==========================================================================
# Energy-based VAD with an adaptive noise threshold
# ==========================================================================

class Vad:
    """Threshold = noise floor (p10 over the last minute) x3, clamped both ways.

    Tuned on a real lecture recording: on dense speech it lets 95% of the frames
    through, on pauses it cuts ~75% of the input. Biased towards "better to send
    something extra than to lose quiet speech".
    """

    ENTER_FRAMES = 3      # 60ms of speech — turn on
    EXIT_FRAMES = 25      # 500ms of silence — turn off
    FLOOR_PCT = 10
    MULT = 3.0
    ABS_MIN = 0.0015      # below this — digital silence
    ABS_MAX = 0.02        # never go higher, even in a noisy lecture hall

    def __init__(self, fixed=None):
        self.hist = collections.deque(maxlen=3000)   # 60s of RMS
        self.speaking = False
        self.disagree = 0
        self.fixed = fixed
        self.thr = self.ABS_MIN if fixed is None else fixed
        self.n = 0
        self.rms = 0.0

    def push(self, frame):
        self.rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        self.hist.append(self.rms)
        self.n += 1
        if self.fixed is None and self.n % 25 == 0 and len(self.hist) >= 50:
            floor = float(np.percentile(
                np.fromiter(self.hist, np.float32, len(self.hist)), self.FLOOR_PCT))
            self.thr = min(max(floor * self.MULT, self.ABS_MIN), self.ABS_MAX)
        loud = self.rms > self.thr
        if loud != self.speaking:
            self.disagree += 1
            if self.disagree >= (self.ENTER_FRAMES if loud else self.EXIT_FRAMES):
                self.speaking = loud
                self.disagree = 0
        else:
            self.disagree = 0
        return self.speaking


# ==========================================================================
# Splitting into chunks on pauses
# ==========================================================================

class Chunker:
    PAD_FRAMES = 10       # 200ms of tail after speech

    def __init__(self, src, min_sec, max_sec, silence_sec, emit):
        self.src = src
        self.min_frames = int(min_sec * 1000 / FRAME_MS)
        self.max_frames = int(max_sec * 1000 / FRAME_MS)
        self.silence_need = int(silence_sec * 1000 / FRAME_MS)
        self.min_voiced = int(0.6 * 1000 / FRAME_MS)
        self.emit = emit
        self.buf = []
        self.voiced = 0
        self.silence = 0
        self.cut_at = None

    def feed(self, frame, speaking):
        self.buf.append(frame)
        if speaking:
            self.voiced += 1
            self.silence = 0
            self.cut_at = None
        else:
            self.silence += 1
            if self.cut_at is None:
                self.cut_at = len(self.buf)

        n = len(self.buf)
        if n >= self.max_frames:
            self._cut(n)
        elif n >= self.min_frames and self.silence >= self.silence_need:
            end = min(n, (n if self.cut_at is None else self.cut_at) + self.PAD_FRAMES)
            self._cut(end)

    def _cut(self, end):
        head, tail = self.buf[:end], self.buf[end:]
        voiced_enough = self.voiced >= self.min_voiced
        self.buf = tail
        self.voiced = 0
        self.silence = len(tail)
        self.cut_at = 0 if tail else None
        if head and voiced_enough:
            self.emit(self.src, np.concatenate(head))

    def flush(self):
        if len(self.buf) >= self.min_voiced and self.voiced >= self.min_voiced // 2:
            self.emit(self.src, np.concatenate(self.buf))
        self.buf = []


# ==========================================================================
# Capture
# ==========================================================================

class WindowsCapture(threading.Thread):
    """Capture microphones and output loopbacks through WASAPI via PyAudioWPatch."""

    def __init__(self, src, device, sink, stop, state):
        super().__init__(daemon=True)
        self.src, self.device, self.sink, self.stop, self.state = src, device, sink, stop, state

    def run(self):
        pa = _win_pa()
        try:
            audio = pa.PyAudio()
            index = _win_device_index(self.device)
            info = audio.get_device_info_by_index(index)
            rate = int(info["defaultSampleRate"])
            channels = min(2, int(info["maxInputChannels"]))
            native_frames = round(rate * FRAME_MS / 1000)
            stream = audio.open(format=pa.paFloat32, channels=channels, rate=rate,
                                input=True, input_device_index=index,
                                frames_per_buffer=native_frames)
        except Exception as e:
            if "audio" in locals():
                audio.terminate()
            self.state.fatal = f"device {self.device!r} failed to open: {type(e).__name__}: {e!r}"
            self.stop.set()
            return
        try:
            while not self.stop.is_set():
                raw = stream.read(native_frames, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels)
                frame = data.mean(axis=1)
                if len(frame) != FRAME_SAMPLES:
                    if len(frame) % FRAME_SAMPLES == 0:
                        frame = frame.reshape(FRAME_SAMPLES, -1).mean(axis=1)
                    else:
                        x = np.linspace(0, len(frame) - 1, FRAME_SAMPLES)
                        frame = np.interp(x, np.arange(len(frame)), frame)
                self.sink(self.src, np.ascontiguousarray(frame, dtype=np.float32))
        except Exception as e:
            if not self.stop.is_set():
                self.state.fatal = f"stream {self.src} died: {type(e).__name__}: {e!r}"
                self.stop.set()
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


class PosixCapture(threading.Thread):
    def __init__(self, src, device, sink, stop, state):
        super().__init__(daemon=True)
        self.src, self.device, self.sink, self.stop, self.state = src, device, sink, stop, state
        self.proc = None

    def _read_frame(self):
        """Reads exactly one frame.

        read() may return less than requested — that is a normal short read,
        not a broken stream. Only an empty result (EOF) means it is broken.
        """
        buf = b""
        while len(buf) < FRAME_BYTES:
            chunk = self.proc.stdout.read(FRAME_BYTES - len(buf))
            if not chunk or self.stop.is_set():
                return None
            buf += chunk
        return buf

    def run(self):
        cmd = ["parec", "--format=s16le", f"--rate={SAMPLE_RATE}", "--channels=1",
               "--latency-msec=100", "-d", self.device]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, bufsize=0)
        except FileNotFoundError:
            self.state.fatal = "parec not found (needs the libpulse / pipewire-pulse package)"
            self.stop.set()
            return
        while not self.stop.is_set():
            data = self._read_frame()
            if data is None:
                if self.stop.is_set():
                    break
                self.state.fatal = f"stream {self.src} died"
                self.stop.set()
                break
            frame = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
            self.sink(self.src, frame)
        if self.proc:
            self.proc.terminate()


# ==========================================================================
# Recognition backends
# ==========================================================================

def to_wav(audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def retry_after(value, fallback):
    """Retry-After can be a number, a number with a unit ('7.66s') or an HTTP date."""
    if value:
        m = re.match(r"\s*([\d.]+)\s*s?\s*$", value)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return fallback


class GroqBackend:
    def __init__(self, key, model, lang):
        self.key = key
        self.model = MODELS[model]
        self.lang = lang
        self.session = requests.Session()

    def transcribe(self, audio, prompt):
        data = {"model": self.model, "response_format": "json", "temperature": "0"}
        if self.lang and self.lang != "auto":
            data["language"] = self.lang
        if prompt:
            data["prompt"] = prompt[-400:]
        files = {"file": ("chunk.wav", to_wav(audio), "audio/wav")}
        delay = 2.0
        for attempt in range(4):
            r = self.session.post(GROQ_URL, headers={"Authorization": f"Bearer {self.key}"},
                                  data=data, files=files, timeout=120)
            if r.status_code == 200:
                return r.json().get("text", "").strip()
            if r.status_code == 429:
                wait = retry_after(r.headers.get("retry-after"), delay)
                time.sleep(min(wait, 30))
                delay *= 2
                continue
            if r.status_code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError("failed after 4 attempts")


class Gemini:
    """Interactions API — the current Gemini interface (v1beta/interactions)."""

    def __init__(self, key, chain, thinking="low"):
        self.key = key
        self.chain = [m for m, _ in chain]
        self.limits = dict(chain)
        self.thinking = thinking
        self.session = requests.Session()
        # per-model: thinking — whether the model accepts thinking_level (gemma has
        # its own level vocabulary); until — skip it until this time; dead — the daily
        # quota is used up, do not touch it until the end of the session.
        self.state = {m: {"thinking": True, "until": 0.0, "dead": False}
                      for m in self.chain}

    @staticmethod
    def _extract(d):
        """The text lives in steps[] under type=model_output.

        The docs describe an output_text field, but the API does not return it — we
        read it only in case it ever shows up, the main path goes through steps.
        """
        if d.get("output_text"):
            return d["output_text"].strip()
        outs = []
        for step in d.get("steps") or []:
            if step.get("type") != "model_output":
                continue                        # skip type=thought blocks
            txt = "\n".join(c["text"] for c in step.get("content") or []
                             if isinstance(c, dict) and c.get("text"))
            if txt:
                outs.append(txt)
        return outs[-1].strip() if outs else None

    def _call(self, model, prompt, retry_without_thinking=True):
        st = self.state[model]
        body = {
            "model": model,
            "system_instruction": GEMINI_SYSTEM,
            "input": prompt,
            "generation_config": {"temperature": 0.3},
        }
        if st["thinking"]:
            body["generation_config"]["thinking_level"] = self.thinking
        r = self.session.post(
            GEMINI_URL, json=body, timeout=180,
            headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})

        if r.status_code == 400 and "thinking" in r.text.lower() and retry_without_thinking:
            st["thinking"] = False          # the model has its own levels — go without it
            return self._call(model, prompt, retry_without_thinking=False)
        if r.status_code == 429:
            daily = any(w in r.text.lower() for w in ("per day", "perday", "daily"))
            if daily:
                st["dead"] = True
            else:
                st["until"] = time.time() + 60
            raise RuntimeError("daily quota" if daily else "per-minute limit")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
        text = self._extract(r.json())
        if not text:
            raise RuntimeError(f"empty response: {json.dumps(r.json())[:200]}")
        return text

    def ask(self, context, question):
        """Returns (answer, model). Walks the chain until someone answers."""
        prompt = f"Lecture transcript:\n\n{context}\n\n---\n\nQuestion: {question}"
        problems = []
        now = time.time()
        for model in self.chain:
            st = self.state[model]
            if st["dead"] or now < st["until"]:
                continue
            try:
                return self._call(model, prompt), model
            except Exception as e:
                problems.append(f"{model}: {e}")
        raise RuntimeError("all models unavailable — " + "; ".join(problems[-3:]))


Capture = WindowsCapture if IS_WINDOWS else PosixCapture


class GeminiLive(threading.Thread):
    """Continuous transcription via the Live API (WebSocket).

    Lives in its own thread with an asyncio loop. Audio is pushed in with feed() from
    the capture thread, piles up in a buffer and leaves it in 100ms chunks.

    Measured model behaviour: interim transcripts are cumulative and keep growing
    until a final arrives, after which the counter resets. The model emits finals of
    its own rarely — on continuous speech it can stay silent for minutes — so if
    there is no final for longer than flush_sec, the session is closed by force
    (audioStreamEnd makes it hand over a final) and reopened. While the session is
    being recreated, audio piles up in the same buffer, so nothing is lost.
    """

    URL = ("wss://generativelanguage.googleapis.com/ws/"
           "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=")
    CHUNK_MS = 100
    CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000

    def __init__(self, key, on_final, on_interim, on_note,
                 model="gemini-3.5-transcribe-live", silence_ms=800, flush_sec=90):
        super().__init__(daemon=True)
        self.key = key
        self.model = model
        self.silence_ms = silence_ms
        self.flush_sec = flush_sec
        self.on_final = on_final
        self.on_interim = on_interim
        self.on_note = on_note
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sent_sec = 0.0
        self.sessions = 0
        self.connected = False
        self.speaking = False     # set from outside by VAD: rotating during a pause is safe

    # --- from the capture thread side ---
    def feed(self, frame):
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        with self.lock:
            self.buf.extend(pcm)

    def _take(self):
        with self.lock:
            if len(self.buf) < self.CHUNK_BYTES:
                return None
            out = bytes(self.buf[:self.CHUNK_BYTES])
            del self.buf[:self.CHUNK_BYTES]
            return out

    # --- thread ---
    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:
            self.on_note(f"Live API stopped: {e}")

    async def _main(self):
        backoff = 2.0
        while not self.stop.is_set():
            try:
                await self._session()
                backoff = 2.0
            except Exception as e:
                self.connected = False
                if self.stop.is_set():
                    break
                self.on_note(f"Live API dropped ({e}) — reconnecting in {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _session(self):
        from websockets.asyncio.client import connect

        setup = {
            "model": f"models/{self.model}",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": {"languageCodes": []},
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "silenceDurationMs": self.silence_ms,
                    "prefixPaddingMs": 200,
                }
            },
        }
        async with connect(self.URL + self.key, max_size=None) as ws:
            await ws.send(json.dumps({"setup": setup}))
            first = await asyncio.wait_for(ws.recv(), timeout=30)
            raw = first if isinstance(first, str) else first.decode(errors="replace")
            if "setupComplete" not in raw:
                raise RuntimeError(f"setup rejected: {raw[:200]}")
            self.connected = True
            self.sessions += 1

            last_final = [time.time()]
            closing = [False]

            async def receiver():
                async for msg in ws:
                    d = json.loads(msg if isinstance(msg, str) else msg.decode())
                    sc = d.get("serverContent") or {}
                    it = sc.get("interimInputTranscription")
                    if it and it.get("text"):
                        self.on_interim(it["text"])
                    fin = sc.get("inputTranscription")
                    if fin and fin.get("text"):
                        last_final[0] = time.time()
                        self.on_interim("")
                        self.on_final(fin["text"].strip())
                    if d.get("goAway"):
                        closing[0] = True
                        return

            rx = asyncio.create_task(receiver())
            try:
                while not self.stop.is_set() and not closing[0] and not rx.done():
                    chunk = self._take()
                    if chunk is None:
                        await asyncio.sleep(0.02)
                    else:
                        await ws.send(json.dumps({"realtimeInput": {"audio": {
                            "data": base64.b64encode(chunk).decode(),
                            "mimeType": f"audio/pcm;rate={SAMPLE_RATE}"}}}))
                        self.sent_sec += self.CHUNK_MS / 1000

                    # No final for a long time — force one, otherwise the text piles
                    # up until the end of the class. Cut only during a pause: a cut in
                    # mid-word makes the model drop the unfinished tail and the phrase
                    # is lost. If speech never stops, wait up to twice as long and cut
                    # anyway.
                    overdue = time.time() - last_final[0]
                    if overdue > self.flush_sec and (not self.speaking
                                                     or overdue > self.flush_sec * 2):
                        await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                        mark = last_final[0]
                        deadline = time.time() + 8
                        while time.time() < deadline and not rx.done():
                            # wait until the finals stop coming: there may be
                            # several, and leaving on the first one loses the tail
                            if last_final[0] != mark and time.time() - last_final[0] > 1.5:
                                break
                            await asyncio.sleep(0.1)
                        return                              # reconnect

                # normal shutdown — collect the tail
                if self.stop.is_set():
                    while True:
                        chunk = self._take()
                        if chunk is None:
                            break
                        await ws.send(json.dumps({"realtimeInput": {"audio": {
                            "data": base64.b64encode(chunk).decode(),
                            "mimeType": f"audio/pcm;rate={SAMPLE_RATE}"}}}))
                    await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                    for _ in range(60):
                        if rx.done():
                            break
                        await asyncio.sleep(0.1)
            finally:
                self.connected = False
                rx.cancel()


class LocalBackend:
    """Offline fallback on faster-whisper. Untested here — needs a pip install."""

    def __init__(self, model, lang):
        from faster_whisper import WhisperModel
        try:
            self.m = WhisperModel(model, device="cuda", compute_type="float16")
        except Exception:
            self.m = WhisperModel(model, device="cpu", compute_type="int8")
        self.lang = lang

    def transcribe(self, audio, prompt):
        segs, _ = self.m.transcribe(audio, language=None if self.lang == "auto" else self.lang,
                                    beam_size=5, vad_filter=True, initial_prompt=prompt or None)
        return " ".join(s.text.strip() for s in segs).strip()


# ==========================================================================
# Screen: scrolling region + pinned status line
# ==========================================================================

def enable_vt():
    """Old conhost does not understand ANSI until VT output processing is on."""
    if not IS_WINDOWS:
        return
    try:
        k = ctypes.windll.kernel32
        handle = k.GetStdHandle(-11)                  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(handle, ctypes.byref(mode)):
            k.SetConsoleMode(handle, mode.value | 0x0004)   # VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


class Screen:
    def __init__(self, tui):
        self.tui = tui and sys.stdout.isatty()
        self.lock = threading.RLock()
        self.status = ""
        self.input = None          # not None => question input line at the bottom
        self.input_full = False    # which scope was picked when it opened
        self.old_term = None
        self.w, self.h = shutil.get_terminal_size((100, 24))
        if not self.tui:
            return
        enable_vt()
        sys.stdout.write("\x1b[?25l")                 # hide the cursor
        sys.stdout.write(f"\x1b[1;{self.h - 1}r")     # scrolling region
        sys.stdout.write(f"\x1b[{self.h - 1};1H")
        sys.stdout.write("\x1b[>1u")                  # kitty keyboard: tell Shift+Enter apart
        sys.stdout.flush()

    def _apply_resize(self):
        """The size is polled instead of catching SIGWINCH.

        There is no SIGWINCH on Windows, and on Unix the handler runs in the
        main thread between bytecodes and, since the lock is reentrant, it
        happily wedged its own escape sequences into the middle of someone
        else's write. The poll is called from _draw() — that is, at least once
        every 0.25s — and costs a single ioctl call.
        """
        size = shutil.get_terminal_size((100, 24))
        if (size.columns, size.lines) == (self.w, self.h):
            return
        old_h = self.h
        self.w, self.h = size
        # when the window grows, the old status line ends up inside the scrolling
        # region and stays there as garbage — erase it at its old position
        sys.stdout.write(f"\x1b[{old_h};1H\x1b[2K")
        sys.stdout.write(f"\x1b[r\x1b[1;{self.h - 1}r")
        sys.stdout.write(f"\x1b[{self.h - 1};1H")

    def raw_input_mode(self):
        if not (self.tui and sys.stdin.isatty()):
            return
        if IS_WINDOWS:
            return                  # msvcrt reads without echo, no mode change needed
        self.old_term = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    def line(self, text):
        with self.lock:
            if not self.tui:
                print(text, flush=True)
                return
            self._apply_resize()
            sys.stdout.write(f"\x1b[{self.h - 1};1H" + text + "\n")
            self._draw()

    def set_status(self, text):
        with self.lock:
            self.status = text
            if self.tui:
                self._draw()

    def set_input(self, text, full=False):
        """text=None — close the input line."""
        with self.lock:
            self.input = text
            self.input_full = full
            if self.tui:
                self._draw()

    def _draw(self):
        self._apply_resize()
        if self.input is not None:
            line = ("?> " if self.input_full else "> ") + self.input
            if len(line) > self.w - 1:
                line = line[-(self.w - 1):]
            sys.stdout.write(f"\x1b[{self.h};1H\x1b[2K\x1b[1;35m{line}\x1b[0m\x1b[?25h")
            sys.stdout.flush()      # leave the cursor at the end of the input
            return
        s = self.status[: self.w]
        sys.stdout.write(f"\x1b[?25l\x1b[{self.h};1H\x1b[2K\x1b[7m{s.ljust(self.w)}\x1b[0m")
        sys.stdout.write(f"\x1b[{self.h - 1};1H")
        sys.stdout.flush()

    def wrap(self, text, indent="  "):
        width = max(40, self.w - len(indent) - 1)
        out = []
        for para in text.splitlines():
            out.extend(textwrap.wrap(para, width=width) or [""])
        return [indent + ln for ln in out]

    def close(self):
        with self.lock:
            if self.old_term is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_term)
            if self.tui:
                sys.stdout.write("\x1b[<u")     # restore the normal keyboard mode
                sys.stdout.write(f"\x1b[r\x1b[{self.h};1H\x1b[2K\x1b[?25h")
                sys.stdout.flush()


# ==========================================================================
# Key reading
# ==========================================================================

class WindowsKeyReader:
    """Reads via msvcrt: select on Windows accepts sockets only, not stdin.

    Windows Terminal cannot do the kitty keyboard protocol, so Shift+Enter here is
    indistinguishable from a plain Enter — "the whole transcript" is selected with
    the `?` key instead of `/` when opening the question line.
    """

    def key(self, timeout=0.3):
        deadline = time.time() + timeout
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):     # function key prefix
                    msvcrt.getwch()             # eat the scan code and ignore it
                    return None
                if ch in ("\r", "\n"):
                    return ("enter",)
                if ch == "\x08":
                    return ("backspace",)
                if ch == "\x15":
                    return ("clear",)
                if ch == "\x1b":
                    return ("esc",)
                if ch == "\x03":
                    return ("interrupt",)
                return ("char", ch)
            if time.time() >= deadline:
                return None
            time.sleep(0.01)


class PosixKeyReader:
    """Reads straight from the file descriptor.

    select() only sees the fd, while sys.stdin.read(1) drags the whole available
    chunk into the TextIOWrapper buffer — after the first character select stays
    quiet and the rest of a pasted line is lost. Hence our own buffer and an
    incremental decoder.
    """

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.buf = ""

    def _getch(self, timeout):
        if not self.buf:
            r, _, _ = select.select([self.fd], [], [], timeout)
            if not r:
                return None
            try:
                data = os.read(self.fd, 4096)
            except OSError:
                return None
            if not data:
                return None
            self.buf += self.dec.decode(data)
            if not self.buf:
                return None
        ch, self.buf = self.buf[0], self.buf[1:]
        return ch

    def key(self, timeout=0.3):
        """('char', c) | ('enter',) | ('shift_enter',) | ('esc',) | ('backspace',) | ('clear',)"""
        ch = self._getch(timeout)
        if ch is None:
            return None
        if ch == "\x1b":
            seq = ""
            while len(seq) < 16:
                c = self._getch(0.05)
                if c is None:
                    break
                seq += c
                if c.isalpha() or c == "~":
                    break
            if not seq:
                return ("esc",)
            if seq in ("\r", "\n"):
                return ("shift_enter",)     # Alt+Enter — fallback without CSI-u
            if seq.startswith("[") and seq.endswith("u"):
                parts = seq[1:-1].split(";")
                if parts[0] == "13":        # Enter in the kitty keyboard protocol
                    mod = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                    return ("shift_enter",) if (mod - 1) & 1 else ("enter",)
            return None
        if ch in ("\r", "\n"):
            return ("enter",)
        if ch in ("\x7f", "\b"):
            return ("backspace",)
        if ch == "\x15":                   # Ctrl+U — clear the line
            return ("clear",)
        if ch == "\x03":
            return ("interrupt",)
        return ("char", ch)


KeyReader = WindowsKeyReader if IS_WINDOWS else PosixKeyReader


# ==========================================================================
# Session state
# ==========================================================================

class State:
    def __init__(self):
        self.paused = False
        self.quit = False
        self.fatal = None
        self.quota_out = False
        self.sent = 0
        self.failed = 0
        self.lines = 0
        self.rms = {"mic": 0.0, "speaker": 0.0}
        self.speaking = {"mic": False, "speaker": False}
        self.pending = 0
        self.asking = 0
        self.transcript = []      # [(ts, src, text)] — context for Gemini
        self.interim = ""         # the current unfinished phrase in live mode
        self.started = time.time()


# ==========================================================================
# Main run
# ==========================================================================

def run(args, backend, usage, out_path, gemini=None):
    stop = threading.Event()
    state = State()
    work = queue.Queue()
    sources = ["mic", "speaker"] if args.source == "both" else [args.source]

    # Resolve the devices before creating Screen: resolve_source may terminate the
    # program via die(), and by then Screen would have already hidden the cursor,
    # narrowed the scrolling region and enabled the kitty keyboard protocol.
    devices = {}
    if "mic" in sources:
        devices["mic"] = resolve_source("mic", args.mic_device)
    if "speaker" in sources:
        devices["speaker"] = resolve_source("speaker", args.speaker_device)

    screen = Screen(not args.no_tui)
    vads = {s: Vad(args.vad_threshold) for s in sources}
    prompts = {s: "" for s in sources}
    # one writer per source: Wave_write is not thread-safe, and mixing mic and
    # speaker into a single mono stream is pointless — two speakers become mush
    dump = {"w": {}, "by_frame": bool(args.keep_audio)}

    def dump_path(src):
        if args.source == "both":
            return out_path.with_suffix(f".{LABELS[src]}.wav")
        return out_path.with_suffix(".wav")

    out = open(out_path, "a", encoding="utf-8", buffering=1)

    def wout(text):
        """Background threads may append after the file is already closed (for
        example if a request hung and join timed out) — silently skip it."""
        if not out.closed:
            out.write(text)

    out.write(f"# Transcript — {datetime.now():%Y-%m-%d %H:%M}\n")
    if args.backend == "gemini-live":
        out.write(f"# Source: {args.source} | engine: {args.live_model}\n\n")
    else:
        out.write(f"# Source: {args.source} | language: {args.lang} | model: {args.model}\n\n")

    def emit_chunk(src, audio):
        state.pending += 1
        work.put((src, audio))

    chunkers = {s: Chunker(s, args.min_chunk, args.max_chunk, args.silence, emit_chunk)
                for s in sources}

    live = None
    if args.backend == "gemini-live":
        if gemini is None:
            die("--backend gemini-live needs a Gemini key")
        if args.source == "both":
            die("--backend gemini-live works with one source: -s mic or -s speaker")

        def on_final(text):
            ts = datetime.now().strftime("%H:%M:%S")
            state.transcript.append((ts, sources[0], text))
            state.lines += 1
            for i, ln in enumerate(screen.wrap(text, indent="") or [""]):
                screen.line((f"\x1b[90m[{ts}]\x1b[0m " if i == 0 else "         ") + ln)
            wout(f"[{ts}] {text}\n")

        def on_interim(text):
            state.interim = text

        def on_note(msg):
            screen.line(f"\x1b[33m{msg}\x1b[0m")

        live = GeminiLive(gemini.key, on_final, on_interim, on_note,
                          model=args.live_model, silence_ms=args.live_silence,
                          flush_sec=args.live_flush)

    def on_frame(src, frame):
        if state.paused:
            return
        v = vads[src]
        speaking = v.push(frame)
        state.rms[src] = v.rms
        state.speaking[src] = speaking
        if live is not None:
            live.speaking = speaking
            live.feed(frame)
        else:
            chunkers[src].feed(frame, speaking)
        if dump["by_frame"]:
            w = dump["w"].get(src)
            if w is not None:
                w.writeframes((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())

    def start_audio_dump(reason=None):
        if dump["w"]:
            return
        for src in sources:
            w = wave.open(str(dump_path(src)), "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            dump["w"][src] = w
        if reason:
            names = ", ".join(dump_path(x).name for x in sources)
            screen.line(f"\x1b[33m!! {reason} — writing raw audio to {names}, "
                        f"transcribe it later: ./rt.py --file <file>\x1b[0m")

    def gate(chunk_sec):
        """'ok' — safe to send; 'quota' — the daily limit is used up.

        Hourly limits are simply waited out. But if we are already stopping, wait
        no longer than a minute so Ctrl+C does not hang — the rest goes to wav.
        """
        waited = 0.0
        while True:
            st = usage.stats()
            if st["req_day"] >= LIMIT_RPD or st["sec_day"] + chunk_sec > LIMIT_ASD:
                return "quota"
            if st["rpm"] >= LIMIT_RPM - 1 or st["sec_hour"] + chunk_sec > LIMIT_ASH:
                if stop.is_set() and waited >= 60:
                    return "quota"
                time.sleep(3)
                waited += 3
                continue
            return "ok"

    def worker():
        while True:
            item = work.get()
            if item is None:
                break
            src, audio = item
            dur = len(audio) / SAMPLE_RATE
            try:
                if not args.no_quota_guard and gate(dur) == "quota":
                    if not state.quota_out:
                        state.quota_out = True
                        start_audio_dump("today's limit is used up")
                    if not dump["by_frame"]:
                        w = dump["w"].get(src)
                        if w is not None:
                            w.writeframes(
                                (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
                    continue
                text = backend.transcribe(audio, prompts[src])
                usage.add(dur)
                state.sent += 1
                if text:
                    prompts[src] = (prompts[src] + " " + text)[-600:]
                    ts = datetime.now().strftime("%H:%M:%S")
                    state.transcript.append((ts, src, text))
                    label = f"[{LABELS[src]}] " if args.source == "both" else ""
                    screen.line(f"\x1b[90m[{ts}]\x1b[0m {label}{text}")
                    wout(f"[{ts}] {label}{text}\n")
                    state.lines += 1
            except Exception as e:
                state.failed += 1
                screen.line(f"\x1b[31m[!] chunk lost: {e}\x1b[0m")
                wout(f"[!! chunk of {dur:.0f}s lost: {e}]\n")
            finally:
                state.pending -= 1
                work.task_done()

    def ask(question, full):
        if gemini is None:
            screen.line("\x1b[31mGemini is not configured: put the key in "
                        "~/.config/transcribe/gemini_key\x1b[0m")
            return
        lines = list(state.transcript)
        if not full:
            lines = lines[-args.recent:]
        if not lines:
            screen.line("\x1b[31mnothing to ask about yet — the transcript is empty\x1b[0m")
            return
        scope = (f"whole transcript, {len(lines)} lines" if full
                 else f"last {len(lines)} lines")
        ctx = "\n".join(
            f"[{ts}] " + (f"[{LABELS[sr]}] " if args.source == "both" else "") + tx
            for ts, sr, tx in lines)
        screen.line(f"\x1b[1;35m>>> [{scope}] {question}\x1b[0m")
        wout(f"\n>>> QUESTION ({scope}): {question}\n")

        def do_ask():
            state.asking += 1
            try:
                answer, model = gemini.ask(ctx, question)
            except Exception as e:
                answer, model = f"(Gemini error: {e})", "—"
            finally:
                state.asking -= 1
            for ln in screen.wrap(answer):
                screen.line(f"\x1b[36m{ln}\x1b[0m")
            screen.line(f"\x1b[90m    ── {model}\x1b[0m")
            wout(f"<<< [{model}] {answer}\n\n")

        threading.Thread(target=do_ask, daemon=True).start()

    def keys():
        buf = None                      # not None => typing a question
        full_default = False            # scope picked by the opening key
        reader = KeyReader()
        while not stop.is_set():
            if not sys.stdin.isatty():
                time.sleep(1)
                continue
            k = reader.key()
            if k is None:
                continue
            kind = k[0]

            if kind == "interrupt":     # Ctrl+C where the signal misses it
                state.quit = True
                stop.set()
                continue

            if buf is None:
                if kind != "char":
                    continue
                ch = k[1]
                if ch in ("q", "Q"):
                    state.quit = True
                    stop.set()
                elif ch == " ":
                    state.paused = not state.paused
                elif ch in ("m", "M"):
                    ts = datetime.now().strftime("%H:%M:%S")
                    screen.line(f"\x1b[1;36m─── mark {ts} ───\x1b[0m")
                    out.write(f"\n=== MARK {ts} ===\n\n")
                elif ch in ("/", "?"):
                    # `?` opens a question about the whole transcript right away:
                    # on Windows Shift+Enter is the same as Enter, so we need another way
                    buf = ""
                    full_default = (ch == "?")
                    screen.set_input(buf, full_default)
                continue

            if kind == "char":
                buf += k[1]
                screen.set_input(buf, full_default)
            elif kind == "backspace":
                buf = buf[:-1]
                screen.set_input(buf, full_default)
            elif kind == "clear":
                buf = ""
                screen.set_input(buf, full_default)
            elif kind == "esc":
                buf = None
                screen.set_input(None)
            elif kind in ("enter", "shift_enter"):
                question = buf.strip()
                full = full_default or kind == "shift_enter"
                buf = None
                screen.set_input(None)
                if question:
                    ask(question, full=full)

    # start
    if args.keep_audio:
        start_audio_dump()

    screen.raw_input_mode()
    for s in sources:
        screen.line(f"\x1b[90m{LABELS[s]}: {devices[s]}\x1b[0m")
    screen.line(f"\x1b[90mwriting to {out_path}\x1b[0m")
    hint = ("space — pause, m — mark, / — question about the recent lines, "
            "? — about the whole transcript, q — stop")
    screen.line("\x1b[90m" + "─" * 3 + f" {hint} " + "─" * 3 + "\x1b[0m")

    caps = [Capture(s, devices[s], on_frame, stop, state) for s in sources]
    if live is not None:
        live.start()
    wrk = threading.Thread(target=worker, daemon=True)
    kb = threading.Thread(target=keys, daemon=True)
    for c in caps:
        c.start()
    wrk.start()
    kb.start()

    try:
        while not stop.is_set():
            st = usage.stats()
            el = time.time() - state.started
            bar_src = sources[0]
            level = min(1.0, state.rms[bar_src] / 0.15)
            bar = "▁▂▃▄▅▆▇█"[min(7, int(level * 8))] * 3 if not state.paused else "···"
            mark = "● PAUSED" if state.paused else ("● SPEECH" if any(state.speaking.values()) else "○ silence")
            buf = len(chunkers[bar_src].buf) * FRAME_MS / 1000
            if live is not None:
                link = "live" if live.connected else "no link"
                head = (f" {mark} {hms(el)} │ {bar} │ {link} · sessions {live.sessions} │ "
                        f"lines {state.lines} │ ")
                tail = state.interim.replace("\n", " ")
                room = max(0, screen.w - len(head) - 2)
                screen.set_status(head + ("…" + tail[-room + 1:] if len(tail) > room else tail))
            else:
                screen.set_status(
                    f" {mark} {hms(el)} │ {bar} │ buf {buf:4.1f}s │ queue {state.pending} │ "
                    f"lines {state.lines} │ today {st['req_day']}/{LIMIT_RPD} req · "
                    f"{compact(st['sec_day'])}/{compact(LIMIT_ASD)} audio"
                    + (f" │ \u2726 Gemini…" if state.asking else "")
                    + ("  QUOTA EXHAUSTED" if state.quota_out else "")
                )
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    except Exception as e:                  # otherwise we would fly past screen.close()
        state.fatal = f"UI failure: {e}"
    finally:
        stop.set()

    screen.set_status(" stopping, finishing the queue…")
    if live is not None:
        live.stop.set()
        live.join(timeout=60)
    for c in caps:              # otherwise feed() from capture races with flush()
        c.join(timeout=3)
    for c in chunkers.values():
        c.flush()
    work.put(None)
    wrk.join(timeout=180)

    for w in dump["w"].values():
        w.close()
    out.write(f"\n# end, {hms(time.time() - state.started)}\n")
    out.close()
    screen.close()

    print(f"\nSaved: {out_path}")
    print(f"Lines: {state.lines} | requests: {state.sent} | chunks lost: {state.failed}")
    for src in dump["w"]:
        print(f"Raw audio: {dump_path(src)}")
    if state.fatal:
        print(f"\nError: {state.fatal}", file=sys.stderr)
    print_quota(usage)


# ==========================================================================
# Batch mode: transcribe an existing file through the same pipeline
# ==========================================================================

def run_file(args, backend, usage, src_path, out_path):
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
    print(f"\n\nSaved: {out_path}")
    print(f"Lines: {lines} | requests: {sent} | chunks lost: {failed}")
    print_quota(usage)


# ==========================================================================

def read_key(cli_key, env_name, filename):
    if cli_key:
        return cli_key.strip()
    if os.environ.get(env_name):
        return os.environ[env_name].strip()
    kf = CONFIG_DIR / filename
    if kf.exists():
        k = kf.read_text().strip()
        if k:
            return k
    return None


def find_key(cli_key):
    k = read_key(cli_key, "GROQ_API_KEY", "key")
    if not k:
        die("no key: put it in $GROQ_API_KEY, in ~/.config/transcribe/key or pass --api-key")
    return k


def build_parser():
    p = argparse.ArgumentParser(
        description="Realtime lecture transcription via Groq Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("-s", "--source", default="speaker", choices=["mic", "speaker", "both"],
                   help="audio source (default speaker — system audio)")
    p.add_argument("-l", "--lang", default="auto", help="language code or auto (default auto)")
    p.add_argument("-m", "--model", default="turbo", choices=list(MODELS),
                   help="turbo — faster and cheaper, large — more accurate")
    p.add_argument("-o", "--out", metavar="FILE", help="transcript file")
    p.add_argument("--min-chunk", type=float, default=15, metavar="SEC",
                   help="minimum chunk length (default 15)")
    p.add_argument("--max-chunk", type=float, default=30, metavar="SEC",
                   help="hard chunk limit (default 30)")
    p.add_argument("--silence", type=float, default=0.5, metavar="SEC",
                   help="pause length to cut on (default 0.5)")
    p.add_argument("--vad-threshold", type=float, default=None, metavar="RMS",
                   help="fixed VAD threshold instead of the adaptive one (e.g. 0.004)")
    p.add_argument("--mic-device", metavar="NAME", help="part of the microphone name")
    p.add_argument("--speaker-device", metavar="NAME", help="part of the output monitor name")
    p.add_argument("--backend", default="groq", choices=["groq", "local", "gemini-live"],
                   help="gemini-live — a continuous stream to the Gemini Live API (no "
                        "request-count limits); local — offline via faster-whisper")
    p.add_argument("--live-model", default="gemini-3.5-transcribe-live",
                   help="model for --backend gemini-live")
    p.add_argument("--live-silence", type=int, default=800, metavar="MS",
                   help="pause after which the Live API ends a phrase (default 800)")
    p.add_argument("--live-flush", type=int, default=90, metavar="SEC",
                   help="recreate the session if there is no final for longer (default 90)")
    p.add_argument("--local-model", default="large-v3", help="model for --backend local")
    p.add_argument("--api-key", help="Groq key (otherwise $GROQ_API_KEY)")
    p.add_argument("--keep-audio", action="store_true", default=None,
                   help="also write a raw wav (otherwise only when the quota runs out)")
    p.add_argument("--no-quota-guard", action="store_true",
                   help="do not throttle for free-tier limits (for a paid plan)")
    p.add_argument("--no-tui", action="store_true", help="plain line-by-line output, no status bar")
    p.add_argument("--list-devices", action="store_true", help="list the devices and exit")
    p.add_argument("--quota", action="store_true", help="show quota usage and exit")
    p.add_argument("--recent", type=int, default=8, metavar="N",
                   help="how many recent lines Enter picks up for a question (default 8)")
    p.add_argument("--gemini-model", metavar="MODEL",
                   help="pin a single model instead of the fallback chain "
                        "(e.g. gemini-3.5-flash-lite — the fastest one)")
    p.add_argument("--thinking", default="low", choices=["low", "medium", "high"],
                   help="Gemini reasoning depth (default low — for speed)")
    p.add_argument("--gemini-key", help="Gemini key (otherwise $GEMINI_API_KEY)")
    p.add_argument("--file", metavar="FILE",
                   help="do not record from a device, transcribe an existing file "
                        "(any format ffmpeg can read)")
    return p


def main():
    args = build_parser().parse_args()

    if args.list_devices:
        list_devices()
        return

    usage = Usage(STATE_DIR / "usage.json")
    if args.quota:
        print_quota(usage)
        return

    if args.min_chunk < MIN_BILLED:
        print(f"warning: chunks shorter than {MIN_BILLED}s are still billed as {MIN_BILLED}s",
              file=sys.stderr)
    if args.source == "both":
        print("warning: mode both sends two streams — the limits run out twice as fast",
              file=sys.stderr)

    if args.backend == "local":
        backend = LocalBackend(args.local_model, args.lang)
    elif args.backend == "gemini-live":
        if args.file:
            die("--backend gemini-live is for live capture only, use groq for files")
        backend = None
    else:
        backend = GroqBackend(find_key(args.api_key), args.model, args.lang)

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        out_path = Path.home() / "Documents/transcripts" / f"{datetime.now():%Y-%m-%d_%H%M}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.file:
        run_file(args, backend, usage, Path(args.file).expanduser(), out_path)
        return

    gkey = read_key(args.gemini_key, "GEMINI_API_KEY", "gemini_key")
    chain = [(args.gemini_model, 0)] if args.gemini_model else GEMINI_CHAIN
    gemini = Gemini(gkey, chain, args.thinking) if gkey else None
    if gemini is None:
        print("note: no Gemini key — questions about the transcript ('/') are unavailable",
              file=sys.stderr)
    run(args, backend, usage, out_path, gemini)


if __name__ == "__main__":
    main()
