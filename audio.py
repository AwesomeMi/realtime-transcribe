"""Voice activity detection, chunking, and WAV encoding."""

import collections
import io
import wave

import numpy as np

from config import FRAME_MS, SAMPLE_RATE


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


def to_wav(audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()
