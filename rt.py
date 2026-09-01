#!/usr/bin/env python3
"""
Realtime-транскрипция через Groq Whisper API. Захват — PipeWire/PulseAudio.

  ./rt.py                     # системный звук (Zoom, браузер, колонки)
  ./rt.py -s mic -l uk        # микрофон
  ./rt.py -s both             # оба источника, строки помечаются [mic]/[spk]
  ./rt.py --list-devices
  ./rt.py --quota             # сколько лимитов осталось

Горячие клавиши:
  space — пауза, m — метка, q — стоп и сохранить
  /     — спросить Gemini по расшифровке:
            Enter       — контекст из последних N реплик (--recent, по умолчанию 8)
            Shift+Enter — контекст из всей расшифровки
            Esc         — отменить ввод

Ключи: $GROQ_API_KEY или ~/.config/transcribe/key
       $GEMINI_API_KEY или ~/.config/transcribe/gemini_key
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
import signal
import subprocess
import sys
import tempfile
import termios
import textwrap
import threading
import time
import tty
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# --- аудио ---------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000       # 320
FRAME_BYTES = FRAME_SAMPLES * 2                      # s16le mono

# --- Groq ----------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODELS = {"turbo": "whisper-large-v3-turbo", "large": "whisper-large-v3"}

# лимиты free tier
LIMIT_RPM = 20
LIMIT_RPD = 2000
LIMIT_ASH = 7200      # секунд аудио в час
LIMIT_ASD = 28800     # секунд аудио в сутки
MIN_BILLED = 10       # короче — всё равно тарифицируется как 10с

STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "transcribe"
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "transcribe"

LABELS = {"mic": "mic", "speaker": "spk"}

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

# Цепочка фолбэка: сверху качественные, но с 20 запросами в сутки; ниже —
# запасные с 500 и 14400. Числа — суточные лимиты free tier на 2026-09-02.
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
    "Ты помогаешь студенту прямо во время лекции. Тебе дают расшифровку речи из "
    "автоматического распознавания: в ней бывают ошибки, имена и термины могут быть "
    "искажены, пунктуация неточная — учитывай это и не цепляйся к опечаткам. "
    "Отвечай кратко и по делу, на языке вопроса. Если в расшифровке нет ответа, "
    "так и скажи, не выдумывай."
)


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def compact(sec):
    sec = int(sec)
    if sec >= 3600:
        return f"{sec // 3600}ч{sec % 3600 // 60:02d}м"
    if sec >= 60:
        return f"{sec // 60}м"
    return f"{sec}с"


# ==========================================================================
# Устройства
# ==========================================================================

def pactl(*args):
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def all_sources():
    """[(name, description, is_monitor)] — все источники записи."""
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
    return pactl("get-default-source") or None


def default_monitor():
    sink = pactl("get-default-sink")
    return f"{sink}.monitor" if sink else None


def resolve_source(kind, hint):
    """kind: 'mic' | 'speaker'. hint — часть имени или описания."""
    srcs = all_sources()
    if hint:
        h = hint.lower()
        for name, desc, is_mon in srcs:
            if h in name.lower() or h in desc.lower():
                return name
        die(f"устройство '{hint}' не найдено, глянь --list-devices")
    name = default_monitor() if kind == "speaker" else default_mic()
    if not name:
        die(f"не удалось определить устройство по умолчанию для '{kind}'")
    known = {s[0] for s in srcs}
    if name not in known:
        # монитор дефолтного сина может быть не поднят — берём первый подходящий
        want_mon = kind == "speaker"
        for n, _d, is_mon in srcs:
            if is_mon == want_mon:
                return n
        die(f"нет доступного источника для '{kind}'")
    return name


def list_devices():
    srcs = all_sources()
    dm, dmon = default_mic(), default_monitor()
    print("\n=== Микрофоны (--source mic) ===")
    for name, desc, is_mon in srcs:
        if not is_mon:
            print(f"  {'*' if name == dm else ' '} {desc}\n      {name}")
    print("\n=== Мониторы вывода — системный звук (--source speaker) ===")
    for name, desc, is_mon in srcs:
        if is_mon:
            print(f"  {'*' if name == dmon else ' '} {desc}\n      {name}")
    print("\n* — по умолчанию.  Выбрать другое: --mic-device ЧАСТЬ_ИМЕНИ / --speaker-device ЧАСТЬ_ИМЕНИ\n")


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# ==========================================================================
# Учёт лимитов (скользящие окна час / сутки, переживает перезапуск)
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
        day = now - 24 * 3600      # хранимое окно длиннее, режем явно
        req_d = sum(1 for t, _ in ev if t >= day)
        sec_h = sum(s for t, s in ev if t >= now - 3600)
        sec_d = sum(s for t, s in ev if t >= day)
        return {"rpm": rpm, "req_hour": req_h, "req_day": req_d,
                "sec_hour": sec_h, "sec_day": sec_d}


def print_quota(usage):
    st = usage.stats()
    print(f"""
Расход за скользящие окна (локальный учёт, free tier):

  запросов за минуту   {st['rpm']:>6} / {LIMIT_RPM}
  запросов за сутки    {st['req_day']:>6} / {LIMIT_RPD}
  аудио за час         {compact(st['sec_hour']):>6} / {compact(LIMIT_ASH)}
  аудио за сутки       {compact(st['sec_day']):>6} / {compact(LIMIT_ASD)}

  осталось на сегодня  ~{compact(max(0, LIMIT_ASD - st['sec_day']))} аудио,
                       ~{LIMIT_RPD - st['req_day']} запросов
""")


# ==========================================================================
# VAD по энергии с адаптивным порогом шума
# ==========================================================================

class Vad:
    """Порог = шумовой пол (p10 за последнюю минуту) x3, зажатый снизу и сверху.

    Подобрано на реальной записи лекции: на плотной речи пропускает 95% кадров,
    на паузах отсекает ~75% входа. Смещено в сторону «лучше лишнее отправить,
    чем потерять тихую речь».
    """

    ENTER_FRAMES = 3      # 60мс речи — включаемся
    EXIT_FRAMES = 25      # 500мс тишины — выключаемся
    FLOOR_PCT = 10
    MULT = 3.0
    ABS_MIN = 0.0015      # ниже — цифровая тишина
    ABS_MAX = 0.02        # выше не поднимаемся даже в шумной аудитории

    def __init__(self, fixed=None):
        self.hist = collections.deque(maxlen=3000)   # 60с RMS
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
# Нарезка на куски по паузам
# ==========================================================================

class Chunker:
    PAD_FRAMES = 10       # 200мс хвоста после речи

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
# Захват
# ==========================================================================

class Capture(threading.Thread):
    def __init__(self, src, device, sink, stop, state):
        super().__init__(daemon=True)
        self.src, self.device, self.sink, self.stop, self.state = src, device, sink, stop, state
        self.proc = None

    def _read_frame(self):
        """Дочитывает ровно кадр.

        read() вправе вернуть меньше запрошенного — это нормальное короткое
        чтение, а не обрыв. Обрыв — только пустой ответ (EOF).
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
            self.state.fatal = "parec не найден (нужен пакет libpulse / pipewire-pulse)"
            self.stop.set()
            return
        while not self.stop.is_set():
            data = self._read_frame()
            if data is None:
                if self.stop.is_set():
                    break
                self.state.fatal = f"поток {self.src} оборвался"
                self.stop.set()
                break
            frame = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
            self.sink(self.src, frame)
        if self.proc:
            self.proc.terminate()


# ==========================================================================
# Бэкенды распознавания
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
    """Retry-After бывает числом, числом с единицей ('7.66s') и HTTP-датой."""
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
        raise RuntimeError("не удалось после 4 попыток")


class Gemini:
    """Interactions API — текущий интерфейс Gemini (v1beta/interactions)."""

    def __init__(self, key, chain, thinking="low"):
        self.key = key
        self.chain = [m for m, _ in chain]
        self.limits = dict(chain)
        self.thinking = thinking
        self.session = requests.Session()
        # per-model: thinking — принимает ли модель thinking_level (у gemma свой
        # словарь уровней); until — до какого времени пропускать; dead — квота
        # на сутки выбрана, до конца сессии не трогаем.
        self.state = {m: {"thinking": True, "until": 0.0, "dead": False}
                      for m in self.chain}

    @staticmethod
    def _extract(d):
        """Текст лежит в steps[] под type=model_output.

        В доках описано поле output_text, но API его не возвращает — берём его
        только если однажды появится, а основной путь идёт по steps.
        """
        if d.get("output_text"):
            return d["output_text"].strip()
        outs = []
        for step in d.get("steps") or []:
            if step.get("type") != "model_output":
                continue                        # пропускаем блоки type=thought
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
            st["thinking"] = False          # у модели свой словарь уровней — идём без него
            return self._call(model, prompt, retry_without_thinking=False)
        if r.status_code == 429:
            daily = any(w in r.text.lower() for w in ("per day", "perday", "daily"))
            if daily:
                st["dead"] = True
            else:
                st["until"] = time.time() + 60
            raise RuntimeError("квота на сутки" if daily else "лимит в минуту")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
        text = self._extract(r.json())
        if not text:
            raise RuntimeError(f"пустой ответ: {json.dumps(r.json())[:200]}")
        return text

    def ask(self, context, question):
        """Возвращает (ответ, модель). Идёт по цепочке, пока кто-то не ответит."""
        prompt = f"Расшифровка лекции:\n\n{context}\n\n---\n\nВопрос: {question}"
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
        raise RuntimeError("все модели недоступны — " + "; ".join(problems[-3:]))


class GeminiLive(threading.Thread):
    """Непрерывная транскрипция через Live API (WebSocket).

    Живёт в своём потоке с asyncio-циклом. Аудио кладётся через feed() из потока
    захвата, копится в буфере, оттуда уходит кусками по 100мс.

    Замеренное поведение модели: промежуточные расшифровки кумулятивны и растут,
    пока не придёт финал, после чего счётчик сбрасывается. Свои финалы модель
    отбивает редко — при непрерывной речи может молчать минутами, — поэтому если
    финала нет дольше flush_sec, сессия закрывается принудительно (audioStreamEnd
    заставляет отдать финал) и открывается заново. Аудио на время пересоздания
    копится в том же буфере, так что ничего не теряется.
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
        self.speaking = False     # ставится снаружи по VAD: в паузе ротировать безопасно

    # --- со стороны потока захвата ---
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

    # --- поток ---
    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:
            self.on_note(f"Live API остановлен: {e}")

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
                self.on_note(f"обрыв Live API ({e}) — переподключаюсь через {backoff:.0f}с")
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
                raise RuntimeError(f"setup отклонён: {raw[:200]}")
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

                    # Финала давно нет — форсируем, иначе текст копится до конца пары.
                    # Рвём только в паузе: обрыв посреди слова заставляет модель
                    # выбросить недоговорённый хвост, и фраза теряется. Если речь
                    # не прекращается, ждём до двойного срока и рвём всё равно.
                    overdue = time.time() - last_final[0]
                    if overdue > self.flush_sec and (not self.speaking
                                                     or overdue > self.flush_sec * 2):
                        await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                        mark = last_final[0]
                        deadline = time.time() + 8
                        while time.time() < deadline and not rx.done():
                            # ждём, пока финалы не перестанут приходить: их может
                            # быть несколько, и уйти по первому значит потерять хвост
                            if last_final[0] != mark and time.time() - last_final[0] > 1.5:
                                break
                            await asyncio.sleep(0.1)
                        return                              # переподключаемся

                # штатная остановка — добираем хвост
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
    """Оффлайн-фолбэк на faster-whisper. Здесь не проверялся — нужен pip install."""

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
# Экран: прокручиваемая область + закреплённая строка статуса
# ==========================================================================

class Screen:
    def __init__(self, tui):
        self.tui = tui and sys.stdout.isatty()
        self.lock = threading.RLock()
        self.status = ""
        self.input = None          # не None => внизу строка ввода вопроса
        self.old_term = None
        self.dirty_size = False    # SIGWINCH только поднимает флаг, см. _apply_resize
        self.w, self.h = shutil.get_terminal_size((100, 24))
        if not self.tui:
            return
        sys.stdout.write("\x1b[?25l")                 # спрятать курсор
        sys.stdout.write(f"\x1b[1;{self.h - 1}r")     # область прокрутки
        sys.stdout.write(f"\x1b[{self.h - 1};1H")
        sys.stdout.write("\x1b[>1u")                  # kitty keyboard: различать Shift+Enter
        sys.stdout.flush()
        try:
            signal.signal(signal.SIGWINCH, self._resize)
        except ValueError:
            pass

    def _resize(self, *_):
        """Обработчик сигнала: только флаг, никакого вывода.

        Сигналы исполняются в основном потоке между байткодами, а lock здесь
        реентрантный — обработчик спокойно захватит его повторно и вклинит свои
        escape-последовательности в середину чужой записи. Поэтому вся работа
        отложена в _apply_resize, который вызывается из обычного кода.
        """
        self.dirty_size = True

    def _apply_resize(self):
        if not self.dirty_size:
            return
        self.dirty_size = False
        old_h = self.h
        self.w, self.h = shutil.get_terminal_size((100, 24))
        # при увеличении окна прежняя строка статуса оказывается внутри области
        # прокрутки и остаётся там мусором — стираем её на старом месте
        sys.stdout.write(f"\x1b[{old_h};1H\x1b[2K")
        sys.stdout.write(f"\x1b[r\x1b[1;{self.h - 1}r")
        sys.stdout.write(f"\x1b[{self.h - 1};1H")

    def raw_input_mode(self):
        if not (self.tui and sys.stdin.isatty()):
            return
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

    def set_input(self, text):
        """text=None — закрыть строку ввода."""
        with self.lock:
            self.input = text
            if self.tui:
                self._draw()

    def _draw(self):
        self._apply_resize()
        if self.input is not None:
            line = "> " + self.input
            if len(line) > self.w - 1:
                line = line[-(self.w - 1):]
            sys.stdout.write(f"\x1b[{self.h};1H\x1b[2K\x1b[1;35m{line}\x1b[0m\x1b[?25h")
            sys.stdout.flush()      # курсор оставляем в конце ввода
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
                sys.stdout.write("\x1b[<u")     # вернуть обычный режим клавиатуры
                sys.stdout.write(f"\x1b[r\x1b[{self.h};1H\x1b[2K\x1b[?25h")
                sys.stdout.flush()


# ==========================================================================
# Чтение клавиш
# ==========================================================================

class KeyReader:
    """Читает с файлового дескриптора напрямую.

    select() видит только fd, а sys.stdin.read(1) утаскивает весь доступный кусок
    в буфер TextIOWrapper — после первого символа select молчит, и остаток
    вставленной строки теряется. Поэтому свой буфер и инкрементальный декодер.
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
                return ("shift_enter",)     # Alt+Enter — фолбэк без CSI-u
            if seq.startswith("[") and seq.endswith("u"):
                parts = seq[1:-1].split(";")
                if parts[0] == "13":        # Enter в kitty keyboard protocol
                    mod = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                    return ("shift_enter",) if (mod - 1) & 1 else ("enter",)
            return None
        if ch in ("\r", "\n"):
            return ("enter",)
        if ch in ("\x7f", "\b"):
            return ("backspace",)
        if ch == "\x15":                   # Ctrl+U — стереть строку
            return ("clear",)
        return ("char", ch)


# ==========================================================================
# Состояние сессии
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
        self.transcript = []      # [(ts, src, text)] — контекст для Gemini
        self.interim = ""         # текущая недоговорённая фраза в live-режиме
        self.started = time.time()


# ==========================================================================
# Основной прогон
# ==========================================================================

def run(args, backend, usage, out_path, gemini=None):
    stop = threading.Event()
    state = State()
    work = queue.Queue()
    sources = ["mic", "speaker"] if args.source == "both" else [args.source]

    # Резолвим устройства до создания Screen: resolve_source может завершить
    # программу через die(), а Screen к тому моменту уже спрятал бы курсор,
    # сузил область прокрутки и включил kitty keyboard protocol.
    devices = {}
    if "mic" in sources:
        devices["mic"] = resolve_source("mic", args.mic_device)
    if "speaker" in sources:
        devices["speaker"] = resolve_source("speaker", args.speaker_device)

    screen = Screen(not args.no_tui)
    vads = {s: Vad(args.vad_threshold) for s in sources}
    prompts = {s: "" for s in sources}
    # по writer'у на источник: Wave_write не потокобезопасен, а смешивать mic и
    # speaker в один моно-поток бессмысленно — выйдет каша из двух говорящих
    dump = {"w": {}, "by_frame": bool(args.keep_audio)}

    def dump_path(src):
        if args.source == "both":
            return out_path.with_suffix(f".{LABELS[src]}.wav")
        return out_path.with_suffix(".wav")

    out = open(out_path, "a", encoding="utf-8", buffering=1)

    def wout(text):
        """Фоновые потоки могут дописать уже после закрытия файла (например,
        если запрос завис и join истёк по таймауту) — молча пропускаем."""
        if not out.closed:
            out.write(text)

    out.write(f"# Транскрипция — {datetime.now():%Y-%m-%d %H:%M}\n")
    if args.backend == "gemini-live":
        out.write(f"# Источник: {args.source} | движок: {args.live_model}\n\n")
    else:
        out.write(f"# Источник: {args.source} | язык: {args.lang} | модель: {args.model}\n\n")

    def emit_chunk(src, audio):
        state.pending += 1
        work.put((src, audio))

    chunkers = {s: Chunker(s, args.min_chunk, args.max_chunk, args.silence, emit_chunk)
                for s in sources}

    live = None
    if args.backend == "gemini-live":
        if gemini is None:
            die("для --backend gemini-live нужен ключ Gemini")
        if args.source == "both":
            die("--backend gemini-live работает с одним источником: -s mic или -s speaker")

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
            screen.line(f"\x1b[33m!! {reason} — пишу сырое аудио в {names}, "
                        f"расшифруешь потом: ./rt.py --file <файл>\x1b[0m")

    def gate(chunk_sec):
        """'ok' — можно слать; 'quota' — суточный лимит выбран.

        Часовые лимиты просто пережидаем. Но если уже останавливаемся, ждём
        не дольше минуты, чтобы Ctrl+C не подвисал — остаток уйдёт в wav.
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
                        start_audio_dump("лимит на сегодня выбран")
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
                screen.line(f"\x1b[31m[!] чанк потерян: {e}\x1b[0m")
                wout(f"[!! чанк {dur:.0f}с потерян: {e}]\n")
            finally:
                state.pending -= 1
                work.task_done()

    def ask(question, full):
        if gemini is None:
            screen.line("\x1b[31mGemini не настроен: положи ключ в "
                        "~/.config/transcribe/gemini_key\x1b[0m")
            return
        lines = list(state.transcript)
        if not full:
            lines = lines[-args.recent:]
        if not lines:
            screen.line("\x1b[31mещё нечего спрашивать — расшифровка пустая\x1b[0m")
            return
        scope = (f"вся расшифровка, {len(lines)} реплик" if full
                 else f"последние {len(lines)} реплик")
        ctx = "\n".join(
            f"[{ts}] " + (f"[{LABELS[sr]}] " if args.source == "both" else "") + tx
            for ts, sr, tx in lines)
        screen.line(f"\x1b[1;35m>>> [{scope}] {question}\x1b[0m")
        wout(f"\n>>> ВОПРОС ({scope}): {question}\n")

        def do_ask():
            state.asking += 1
            try:
                answer, model = gemini.ask(ctx, question)
            except Exception as e:
                answer, model = f"(ошибка Gemini: {e})", "—"
            finally:
                state.asking -= 1
            for ln in screen.wrap(answer):
                screen.line(f"\x1b[36m{ln}\x1b[0m")
            screen.line(f"\x1b[90m    ── {model}\x1b[0m")
            wout(f"<<< [{model}] {answer}\n\n")

        threading.Thread(target=do_ask, daemon=True).start()

    def keys():
        buf = None                      # не None => набираем вопрос
        reader = KeyReader()
        while not stop.is_set():
            if not sys.stdin.isatty():
                time.sleep(1)
                continue
            k = reader.key()
            if k is None:
                continue
            kind = k[0]

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
                    screen.line(f"\x1b[1;36m─── метка {ts} ───\x1b[0m")
                    out.write(f"\n=== МЕТКА {ts} ===\n\n")
                elif ch in ("/", "?"):
                    buf = ""
                    screen.set_input(buf)
                continue

            if kind == "char":
                buf += k[1]
                screen.set_input(buf)
            elif kind == "backspace":
                buf = buf[:-1]
                screen.set_input(buf)
            elif kind == "clear":
                buf = ""
                screen.set_input(buf)
            elif kind == "esc":
                buf = None
                screen.set_input(None)
            elif kind in ("enter", "shift_enter"):
                question = buf.strip()
                buf = None
                screen.set_input(None)
                if question:
                    ask(question, full=(kind == "shift_enter"))

    # запуск
    if args.keep_audio:
        start_audio_dump()

    screen.raw_input_mode()
    for s in sources:
        screen.line(f"\x1b[90m{LABELS[s]}: {devices[s]}\x1b[0m")
    screen.line(f"\x1b[90mпишу в {out_path}\x1b[0m")
    hint = "space — пауза, m — метка, / — спросить Gemini, q — стоп"
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
            mark = "● ПАУЗА" if state.paused else ("● ГОВОРЯТ" if any(state.speaking.values()) else "○ тишина")
            buf = len(chunkers[bar_src].buf) * FRAME_MS / 1000
            if live is not None:
                link = "live" if live.connected else "нет связи"
                head = (f" {mark} {hms(el)} │ {bar} │ {link} · сессий {live.sessions} │ "
                        f"строк {state.lines} │ ")
                tail = state.interim.replace("\n", " ")
                room = max(0, screen.w - len(head) - 2)
                screen.set_status(head + ("…" + tail[-room + 1:] if len(tail) > room else tail))
            else:
                screen.set_status(
                    f" {mark} {hms(el)} │ {bar} │ буфер {buf:4.1f}с │ очередь {state.pending} │ "
                    f"строк {state.lines} │ сегодня {st['req_day']}/{LIMIT_RPD} зап · "
                    f"{compact(st['sec_day'])}/{compact(LIMIT_ASD)} аудио"
                    + (f" │ \u2726 Gemini…" if state.asking else "")
                    + ("  ЛИМИТ ИСЧЕРПАН" if state.quota_out else "")
                )
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    except Exception as e:                  # иначе улетим мимо screen.close()
        state.fatal = f"сбой интерфейса: {e}"
    finally:
        stop.set()

    screen.set_status(" останавливаюсь, дорабатываю очередь…")
    if live is not None:
        live.stop.set()
        live.join(timeout=60)
    for c in caps:              # иначе feed() из захвата гонится с flush()
        c.join(timeout=3)
    for c in chunkers.values():
        c.flush()
    work.put(None)
    wrk.join(timeout=180)

    for w in dump["w"].values():
        w.close()
    out.write(f"\n# конец, {hms(time.time() - state.started)}\n")
    out.close()
    screen.close()

    print(f"\nСохранено: {out_path}")
    print(f"Строк: {state.lines} | запросов: {state.sent} | потеряно чанков: {state.failed}")
    for src in dump["w"]:
        print(f"Сырое аудио: {dump_path(src)}")
    if state.fatal:
        print(f"\nОшибка: {state.fatal}", file=sys.stderr)
    print_quota(usage)


# ==========================================================================
# Пакетный режим: расшифровать готовый файл тем же конвейером
# ==========================================================================

def run_file(args, backend, usage, src_path, out_path):
    if not src_path.exists():
        die(f"нет файла {src_path}")
    # stderr во временный файл, а не в трубу: её никто не вычитывает до конца
    # разбора stdout, и на шумном ffmpeg 64КБ буфера хватило бы для взаимной блокировки
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
    out.write(f"# Транскрипция файла {src_path.name} — {datetime.now():%Y-%m-%d %H:%M}\n")
    out.write(f"# язык: {args.lang} | модель: {args.model}\n\n")

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
                        die("суточный лимит выбран — продолжишь завтра, файл дописан")
                    if st["rpm"] >= LIMIT_RPM - 1 or st["sec_hour"] + dur > LIMIT_ASH:
                        print(f"\r  ждём слот под лимиты… ({hms(pos)})", end="", flush=True)
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
                out.write(f"[!! кусок на {hms(pos - dur)} потерян: {e}]\n")
            print(f"\r  {hms(pos)} | кусков {sent} | строк {lines}"
                  + (f" | потеряно {failed}" if failed else "") + "   ", end="", flush=True)

    print(f"Расшифровываю {src_path.name} -> {out_path}")
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

    out.write(f"\n# конец, {hms(pos)} аудио\n")
    out.close()
    print(f"\n\nСохранено: {out_path}")
    print(f"Строк: {lines} | запросов: {sent} | потеряно кусков: {failed}")
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
        die("нет ключа: положи в $GROQ_API_KEY, в ~/.config/transcribe/key или передай --api-key")
    return k


def build_parser():
    p = argparse.ArgumentParser(
        description="Realtime-транскрипция лекций через Groq Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("-s", "--source", default="speaker", choices=["mic", "speaker", "both"],
                   help="источник звука (по умолчанию speaker — системный звук)")
    p.add_argument("-l", "--lang", default="uk", help="код языка или auto (по умолчанию uk)")
    p.add_argument("-m", "--model", default="turbo", choices=list(MODELS),
                   help="turbo — быстрее и дешевле, large — точнее")
    p.add_argument("-o", "--out", metavar="FILE", help="файл расшифровки")
    p.add_argument("--min-chunk", type=float, default=15, metavar="СЕК",
                   help="минимальная длина куска (по умолчанию 15)")
    p.add_argument("--max-chunk", type=float, default=30, metavar="СЕК",
                   help="жёсткий предел куска (по умолчанию 30)")
    p.add_argument("--silence", type=float, default=0.5, metavar="СЕК",
                   help="пауза, по которой режем (по умолчанию 0.5)")
    p.add_argument("--vad-threshold", type=float, default=None, metavar="RMS",
                   help="фиксированный порог VAD вместо адаптивного (напр. 0.004)")
    p.add_argument("--mic-device", metavar="ИМЯ", help="часть имени микрофона")
    p.add_argument("--speaker-device", metavar="ИМЯ", help="часть имени монитора вывода")
    p.add_argument("--backend", default="groq", choices=["groq", "local", "gemini-live"],
                   help="gemini-live — непрерывный поток в Gemini Live API (лимитов по "
                        "числу запросов нет); local — оффлайн через faster-whisper")
    p.add_argument("--live-model", default="gemini-3.5-transcribe-live",
                   help="модель для --backend gemini-live")
    p.add_argument("--live-silence", type=int, default=800, metavar="МС",
                   help="пауза, по которой Live API отбивает фразу (по умолчанию 800)")
    p.add_argument("--live-flush", type=int, default=90, metavar="СЕК",
                   help="если финала нет дольше этого, пересоздать сессию (по умолчанию 90)")
    p.add_argument("--local-model", default="large-v3", help="модель для --backend local")
    p.add_argument("--api-key", help="ключ Groq (иначе $GROQ_API_KEY)")
    p.add_argument("--keep-audio", action="store_true", default=None,
                   help="писать рядом сырой wav (иначе только при исчерпании лимита)")
    p.add_argument("--no-quota-guard", action="store_true",
                   help="не притормаживать под free-tier лимиты (для платного тарифа)")
    p.add_argument("--no-tui", action="store_true", help="простой построчный вывод, без статус-бара")
    p.add_argument("--list-devices", action="store_true", help="показать устройства и выйти")
    p.add_argument("--quota", action="store_true", help="показать расход лимитов и выйти")
    p.add_argument("--recent", type=int, default=8, metavar="N",
                   help="сколько последних реплик берёт Enter в вопросе (по умолчанию 8)")
    p.add_argument("--gemini-model", metavar="МОДЕЛЬ",
                   help="жёстко закрепить одну модель вместо цепочки фолбэка "
                        "(напр. gemini-3.5-flash-lite — самая быстрая)")
    p.add_argument("--thinking", default="low", choices=["low", "medium", "high"],
                   help="глубина рассуждений Gemini (по умолчанию low — ради скорости)")
    p.add_argument("--gemini-key", help="ключ Gemini (иначе $GEMINI_API_KEY)")
    p.add_argument("--file", metavar="ФАЙЛ",
                   help="не писать с устройства, а расшифровать готовый файл "
                        "(любой формат, который читает ffmpeg)")
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
        print(f"warning: куски короче {MIN_BILLED}с всё равно тарифицируются как {MIN_BILLED}с",
              file=sys.stderr)
    if args.source == "both":
        print("warning: режим both шлёт два потока — лимиты тратятся вдвое быстрее",
              file=sys.stderr)

    if args.backend == "local":
        backend = LocalBackend(args.local_model, args.lang)
    elif args.backend == "gemini-live":
        if args.file:
            die("--backend gemini-live только для живого захвата, для файлов используй groq")
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
        print("note: ключа Gemini нет — вопросы по расшифровке ('/') будут недоступны",
              file=sys.stderr)
    run(args, backend, usage, out_path, gemini)


if __name__ == "__main__":
    main()
