#!/usr/bin/env python3
"""
Real-time transcription using faster-whisper.
Captures from microphone, speaker (loopback), or both simultaneously.
Press Ctrl+C to stop — saves everything to a text file.

Usage:
  python realtime.py                          # mic only
  python realtime.py --source speaker         # system audio (Zoom, browser, etc.)
  python realtime.py --source both            # mic + speaker mixed
  python realtime.py --list-devices
  python realtime.py --lang uk --chunk 15 --out lecture.txt
"""

import os
import sys
import queue
import threading
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

# CUDA DLL setup for Windows
_sp = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
_extra = [os.path.join(_sp, p, "bin") for p in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc")]
os.environ["PATH"] = os.pathsep.join(p for p in _extra if os.path.isdir(p)) + os.pathsep + os.environ.get("PATH", "")

import soundcard as sc
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_FRAMES = SAMPLE_RATE // 2   # 0.5s per read block


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def list_devices():
    print("\n=== Microphones ===")
    for m in sc.all_microphones(include_loopback=False):
        marker = " <-- default" if m == sc.default_microphone() else ""
        print(f"  {m.name}{marker}")

    print("\n=== Speakers (available for loopback capture) ===")
    for s in sc.all_speakers():
        marker = " <-- default" if s == sc.default_speaker() else ""
        print(f"  {s.name}{marker}")

    print("\nUse --mic-device NAME and --speaker-device NAME to select non-default devices.")
    sys.exit(0)


def resolve_mic(name: str | None):
    if name:
        mics = [m for m in sc.all_microphones(include_loopback=False) if name.lower() in m.name.lower()]
        if not mics:
            print(f"Microphone '{name}' not found. Run --list-devices.")
            sys.exit(1)
        return mics[0]
    return sc.default_microphone()


def resolve_speaker(name: str | None):
    if name:
        speakers = [s for s in sc.all_speakers() if name.lower() in s.name.lower()]
        if not speakers:
            print(f"Speaker '{name}' not found. Run --list-devices.")
            sys.exit(1)
        return speakers[0]
    return sc.default_speaker()


# ---------------------------------------------------------------------------
# Capture threads
# ---------------------------------------------------------------------------

def capture_mic(mic, audio_queue: queue.Queue, stop_event: threading.Event):
    print(f"Mic:     {mic.name}")
    with mic.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_FRAMES) as rec:
        while not stop_event.is_set():
            data = rec.record(numframes=BLOCK_FRAMES)
            audio_queue.put(("mic", data.flatten().astype(np.float32)))


def capture_speaker(speaker, audio_queue: queue.Queue, stop_event: threading.Event):
    loopback = sc.get_microphone(speaker.id, include_loopback=True)
    print(f"Speaker: {speaker.name} (loopback)")
    with loopback.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_FRAMES) as rec:
        while not stop_event.is_set():
            data = rec.record(numframes=BLOCK_FRAMES)
            audio_queue.put(("speaker", data.flatten().astype(np.float32)))


# ---------------------------------------------------------------------------
# Transcription loop
# ---------------------------------------------------------------------------

def transcribe_loop(model, lang, chunk_sec, audio_queue: queue.Queue,
                    stop_event: threading.Event, segments_all: list, lock: threading.Lock,
                    source: str):
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    buffers = {"mic": np.empty(0, dtype=np.float32),
               "speaker": np.empty(0, dtype=np.float32)}

    def process(audio, src_label):
        segments, _ = model.transcribe(
            audio,
            language=lang if lang != "auto" else None,
            beam_size=5,
            vad_filter=True,
        )
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            ts = datetime.now().strftime("%H:%M:%S")
            with lock:
                segments_all.append((ts, src_label, text))
            prefix = f"[{src_label}] " if source == "both" else ""
            print(f"[{ts}] {prefix}{text}", flush=True)

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            src, chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        buffers[src] = np.concatenate([buffers[src], chunk])

        while len(buffers[src]) >= chunk_samples:
            audio = buffers[src][:chunk_samples].copy()
            buffers[src] = buffers[src][chunk_samples:]
            process(audio, src)

    # flush remainders
    for src, buf in buffers.items():
        if len(buf) > SAMPLE_RATE // 4:
            process(buf, src)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(model, args):
    out_path = Path(args.out) if args.out else Path(
        f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    segments_all = []
    lock = threading.Lock()

    capture_threads = []

    if args.source in ("mic", "both"):
        mic = resolve_mic(args.mic_device)
        t = threading.Thread(target=capture_mic, args=(mic, audio_queue, stop_event), daemon=True)
        capture_threads.append(t)

    if args.source in ("speaker", "both"):
        speaker = resolve_speaker(args.speaker_device)
        t = threading.Thread(target=capture_speaker, args=(speaker, audio_queue, stop_event), daemon=True)
        capture_threads.append(t)

    transcribe_thread = threading.Thread(
        target=transcribe_loop,
        args=(model, args.lang, args.chunk, audio_queue, stop_event, segments_all, lock, args.source),
        daemon=True,
    )

    for t in capture_threads:
        t.start()
    transcribe_thread.start()

    print(f"\nRecording [{args.source}]... Press Ctrl+C to stop.")
    print(f"Output will be saved to: {out_path}")
    print("-" * 60)

    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        pass

    print("\n\nStopping — flushing remaining audio...")
    stop_event.set()
    transcribe_thread.join(timeout=60)

    # save
    with lock:
        segs = list(segments_all)

    if not segs:
        print("Nothing was transcribed.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcription — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# Source: {args.source} | Language: {args.lang}\n\n")
        for ts, src, text in segs:
            label = f"[{src}] " if args.source == "both" else ""
            f.write(f"[{ts}] {label}{text}\n")

    print(f"\nSaved {len(segs)} segments → {out_path}")
    print("\n--- Preview ---")
    for ts, src, text in segs[:5]:
        label = f"[{src}] " if args.source == "both" else ""
        print(f"[{ts}] {label}{text}")
    if len(segs) > 5:
        print(f"... and {len(segs) - 5} more lines")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real-time transcription from mic and/or speaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source", "-s", default="mic",
                        choices=["mic", "speaker", "both"],
                        help="Audio source (default: mic)")
    parser.add_argument("--model", "-m", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model (default: large-v3)")
    parser.add_argument("--lang", "-l", default="auto",
                        help="Language code: uk, ru, en... or 'auto' (default: auto)")
    parser.add_argument("--chunk", "-c", type=int, default=10,
                        help="Chunk size in seconds (default: 10)")
    parser.add_argument("--out", "-o", metavar="FILE",
                        help="Output file path (default: realtime_YYYYMMDD_HHMMSS.txt)")
    parser.add_argument("--mic-device", metavar="NAME",
                        help="Partial mic name (default: system default)")
    parser.add_argument("--speaker-device", metavar="NAME",
                        help="Partial speaker name (default: system default)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode (slower)")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()

    device_str = "cpu" if args.cpu else "cuda"
    compute = "int8" if args.cpu else "float16"

    print(f"Loading model '{args.model}' on {device_str}...")
    try:
        model = WhisperModel(args.model, device=device_str, compute_type=compute)
    except Exception:
        print("CUDA unavailable, falling back to CPU...")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")

    run(model, args)


if __name__ == "__main__":
    main()
