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
import os
import sys
from datetime import datetime
from pathlib import Path

from backends import Gemini, GroqBackend, LocalBackend
from config import (CONFIG_DIR, GEMINI_CHAIN, MIN_BILLED, MODELS,
                    STATE_DIR, die)
from devices import list_devices
from quota import Usage, print_quota
from session import run, run_file


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
    p.add_argument("--title", dest="title", action="store_true", default=None,
                   help="rename the finished transcript to '<date> <topic>' without "
                        "asking; by default you are asked once the session ends, and "
                        "never asked at all when -o already names the file")
    p.add_argument("--no-title", dest="title", action="store_false",
                   help="keep the plain timestamp filename, no question asked")
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

    if args.title is None and args.out is not None:
        args.title = False                  # never touch a name the user picked

    gkey = read_key(args.gemini_key, "GEMINI_API_KEY", "gemini_key")
    chain = [(args.gemini_model, 0)] if args.gemini_model else GEMINI_CHAIN
    gemini = Gemini(gkey, chain, args.thinking) if gkey else None

    if args.file:
        run_file(args, backend, usage, Path(args.file).expanduser(), out_path, gemini)
        return

    if gemini is None:
        print("note: no Gemini key — questions about the transcript ('/') are unavailable",
              file=sys.stderr)
    run(args, backend, usage, out_path, gemini)


if __name__ == "__main__":
    main()
