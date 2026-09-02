"""Constants, paths and the small helpers everything else leans on."""

import os
import re
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt"


SAMPLE_RATE = 16000


FRAME_MS = 20


FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000       # 320


FRAME_BYTES = FRAME_SAMPLES * 2                      # s16le mono


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
# Naming a file is a mechanical job — start from the models with the widest daily
# allowance so titling never eats the 20-per-day quota reserved for real questions.
GEMINI_TITLE_CHAIN = [
    ("gemini-3.5-flash-lite", 500),
    ("gemini-3.1-flash-lite", 500),
    ("gemma-4-31b-it", 14400),
    ("gemini-3.5-flash", 20),
]


GEMINI_TITLE_SYSTEM = (
    "You name transcript files. Reply with the title only: 2 to 6 words, in the "
    "language of the transcript, naming its actual subject. No quotes, no date, "
    "no trailing period, and never words like 'lecture', 'meeting' or 'transcript'."
)


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


BAD_IN_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_title(title, limit=60):
    """Make a model-written line safe as a filename on Linux and Windows alike."""
    # substitute rather than delete: dropping the separator in "C#/SQL" or a
    # newline would glue neighbouring words into one
    t = BAD_IN_FILENAME.sub(" ", title)
    t = re.sub(r"\s+", " ", t).strip(" .")     # Windows rejects trailing dots/spaces
    if len(t) > limit:
        t = t[:limit].rsplit(" ", 1)[0]
    return t.strip(" .")


def confirm(question, default=True):
    """Yes/no prompt. Keeps the default when nobody is there to answer."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return default
    try:
        answer = input(f"{question} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False                    # bailing out means leave the file alone
    if not answer:
        return default
    return answer[0] in "yд"            # 'д' so "да" works too


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


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
