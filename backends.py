"""Speech-to-text backends and the Gemini clients."""

import asyncio
import base64
import json
import threading
import time

import numpy as np
import requests

from audio import to_wav
from config import (GEMINI_SYSTEM, GEMINI_TITLE_CHAIN, GEMINI_TITLE_SYSTEM,
                    GEMINI_URL, GROQ_URL, MODELS, SAMPLE_RATE, retry_after)


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

    def _call(self, model, prompt, system=GEMINI_SYSTEM, retry_without_thinking=True):
        st = self.state[model]
        body = {
            "model": model,
            "system_instruction": system,
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
            return self._call(model, prompt, system, retry_without_thinking=False)
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

    def _walk(self, chain, prompt, system):
        """Returns (text, model). Walks the chain until someone answers."""
        problems = []
        now = time.time()
        for model in chain:
            st = self.state.setdefault(model, {"thinking": True, "until": 0.0, "dead": False})
            if st["dead"] or now < st["until"]:
                continue
            try:
                return self._call(model, prompt, system), model
            except Exception as e:
                problems.append(f"{model}: {e}")
        raise RuntimeError("every model is unavailable — " + "; ".join(problems[-3:]))

    def title(self, transcript):
        """Short subject line for a finished transcript."""
        return self._walk([m for m, _ in GEMINI_TITLE_CHAIN],
                          f"Transcript:\n\n{transcript}\n\n---\n\nTitle:",
                          GEMINI_TITLE_SYSTEM)

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
