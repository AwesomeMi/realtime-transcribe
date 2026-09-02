# rt.py — real-time lecture transcription

Live speech-to-text for lectures and meetings, straight into a text file, with a
terminal UI you can ask questions in. Captures either your microphone or the
system audio (Zoom, browser, anything playing through your speakers).

Three transcription backends:

- **Groq Whisper** — audio is cut into chunks on silence and sent over HTTP.
- **Gemini Live API** — a continuous WebSocket stream, no per-request limits.
- **faster-whisper** — optional offline transcription on your GPU or CPU.

While it runs you can hit `/` and ask a question about what has been said so far.
The answer is written into the same transcript, clearly marked.

```
 ● SPEAKING 00:42:15 │ ▅▅▅ │ live · sessions 3 │ lines 187 │ …and that brings us to
```

## Requirements

- Python 3.10+
- Linux: PipeWire or PulseAudio (`parec`, `pactl` — from `pipewire-pulse` or `libpulse`)
- Windows: nothing extra beyond `pip install -r requirements.txt`, which pulls in
  `PyAudioWPatch` for microphone and WASAPI loopback capture
- `ffmpeg` for `--file` mode
- `pip install -r requirements.txt`

## Setup

For Linux and macOS, keys can go in `~/.config/transcribe/`:

```sh
mkdir -p ~/.config/transcribe
printf '%s' 'gsk_...'  > ~/.config/transcribe/key         # Groq, or $GROQ_API_KEY
printf '%s' 'AIza...'  > ~/.config/transcribe/gemini_key  # Gemini, or $GEMINI_API_KEY
chmod 600 ~/.config/transcribe/*
```

On Windows, put the same two files in `%APPDATA%\transcribe`:

```powershell
$keyDir = Join-Path $env:APPDATA "transcribe"
New-Item -ItemType Directory -Force $keyDir | Out-Null
Set-Content (Join-Path $keyDir "key") "gsk_..." -NoNewline
Set-Content (Join-Path $keyDir "gemini_key") "AIza..." -NoNewline
```

Alternatively, set `GROQ_API_KEY` and `GEMINI_API_KEY` environment variables on
any platform, or pass `--api-key` and `--gemini-key` for a single run. Do not
commit real keys to this repository.

The Gemini key is optional for the Groq backend — it only powers the question
feature. It is required for `--backend gemini-live`.

The offline backend is optional and requires `faster-whisper` separately:

```sh
pip install faster-whisper
```

## Usage

```sh
./rt.py                                  # system audio via Groq
./rt.py --backend gemini-live            # system audio via Gemini Live
./rt.py --backend local                  # offline via faster-whisper
./rt.py -s mic -l uk                     # microphone, pinned to Ukrainian
./rt.py -s both                          # mic + speakers, lines tagged [mic]/[spk]
./rt.py --file lecture.mp4               # transcribe an existing recording
./rt.py --list-devices
./rt.py --quota                          # how much of today's free tier is left
```

The spoken language is detected automatically by default. Pass `-l uk`, `-l ru`,
or another language code when you want to pin it explicitly.

Transcripts land in `~/Documents/transcripts/` unless you pass `-o`. They are
written line by line and flushed immediately, so a crash costs you nothing.

### Keys while running

| key | what it does |
|---|---|
| `space` | pause / resume capture |
| `m` | drop a marker at the current timestamp |
| `/` | ask about the **last N lines** (`--recent`, default 8) |
| `?` | ask about the **entire transcript** |
| `q` | stop and save |

Inside the question prompt: `Enter` sends, `Esc` cancels, `Ctrl+U` clears the
line. `Shift+Enter` also forces whole-transcript scope, but that needs a terminal
speaking the kitty keyboard protocol (Ghostty, kitty, foot, WezTerm) — Windows
Terminal cannot distinguish it from plain `Enter`, which is why `?` exists.

## Choosing a backend

Both are free-tier friendly, but they run out in completely different ways.
Numbers below were measured against the live APIs, not copied from docs.

| | Groq Whisper | Gemini Live |
|---|---|---|
| transport | HTTP, chunked on silence | WebSocket, continuous |
| requests/day | 2 000 | unlimited |
| audio/day | 8 h | unlimited |
| audio/hour | 2 h | ~13× real time (20K TPM) |
| a 90-minute lecture costs | ~200 requests, 5 400 s | nothing countable |
| phrase boundaries | cut every 15–30 s | continuous, no seams |

Groq is the safer default: its limits are published and generous, and it handles
`--file` batch work. Gemini Live has no daily ceiling at all, which is what you
want for back-to-back lectures, and in testing it was noticeably more accurate on
Ukrainian — it produced correct grammar where Whisper garbled case endings.

`gemini-3.5-transcribe` (the non-live batch model) is deliberately **not** wired
up: 25 requests/day and 10K TPM cap it at under three hours of audio per day,
which is worse than Groq for the same job.

### Question models

Questions go down a fallback chain, dropping to the next model on a quota error:

```
gemini-3.7-flash → 3.6-flash → 3-flash-preview → 2.5-flash → 3.5-flash   (20/day each)
  → 3.5-flash-lite → 3.1-flash-lite                                      (500/day each)
  → gemma-4-31b-it                                                       (14 400/day)
```

Latency differs a lot: the 3.x flash models take 30–50 s, the lite models answer
in 2–3 s. Pin one with `--gemini-model` if you care more about speed than depth.

## How it works

**Silence-aware chunking (Groq backend).** A fixed-interval cut lands mid-word and
splits phrases. Instead, an energy VAD tracks the noise floor as the 10th
percentile of RMS over the last minute and puts the threshold at 3× that, clamped
to a sane range. A chunk is closed at the first pause after 15 s, or hard-cut at
30 s. Chunks with no speech in them are never sent, which on a quiet stretch cuts
upload volume by about three quarters.

**Context across chunks.** The tail of the previous transcript is passed as the
`prompt` parameter so terminology and names stay consistent across a boundary.

**Quota accounting that survives restarts.** Usage is journalled to
`~/.local/state/transcribe/usage.json` on Linux/macOS and
`%LOCALAPPDATA%\transcribe\usage.json` on Windows, with rolling hour and day
windows. Before each request the client waits for a free slot instead of eating
a 429. If the daily allowance really is gone, it starts writing raw audio to a
`.wav` next to the transcript so you can finish the job later with `--file`.

**Session rotation (Live backend).** The Live model emits interim transcripts
that accumulate and get revised in place, and commits a final one only at an
activity boundary — on a continuously speaking lecturer that can mean nothing for
minutes. So the session is rotated on a timer, but **only during a pause**:
cutting the stream mid-word makes the model discard its unfinished tail. Verified
by comparing a rotated run against an unrotated one — identical character counts.

## Platform support

| | status |
|---|---|
| Linux (PipeWire / PulseAudio) | works; built and regression-tested here |
| Windows | device enumeration and system-audio capture smoke-tested on real hardware |
| macOS | untested; would need a capture backend of its own |

Roughly 13% of the code is platform-specific and it is isolated at three seams —
device enumeration, audio capture, and raw keyboard reads — each with a `Posix*`
and a `Windows*` implementation picked at import time. The other 87% (VAD,
chunking, quota accounting, both API clients, the TUI) is shared, which is
exactly why this is one branch and not a permanent fork: a fix in shared code
would otherwise have to be applied twice.

Windows uses `PyAudioWPatch` for microphone and WASAPI loopback capture,
`msvcrt` for keystrokes, and
`%APPDATA%` / `%LOCALAPPDATA%` for config and state. Terminal resize is polled
rather than signalled, so `SIGWINCH` is gone from both platforms.

**The Windows path has been smoke-tested on real hardware.** Device enumeration
and system-audio loopback capture work; microphone support depends on the active
Windows endpoint and its driver. Expect rough edges; issue reports welcome.

## Repository layout

```
rt.py         entry point and the command line
session.py    Session — one recording; batch mode; naming a finished transcript
devices.py    device enumeration and capture, Posix and Windows side by side
backends.py   Groq Whisper, Gemini questions, Gemini Live
audio.py      voice activity detection, chunking, WAV encoding
quota.py      free-tier accounting that survives a restart
config.py     constants, paths, small shared helpers
ui.py         the pinned status line, keystrokes, session state
```

## License

MIT
