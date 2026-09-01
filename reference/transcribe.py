#!/usr/bin/env python3
"""
Audio transcription CLI using faster-whisper.

Usage:
  python transcribe.py audio.mp3
  python transcribe.py *.ogg --lang uk --format srt
  python transcribe.py file.wav --model medium --out-dir results/
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# CUDA DLL setup for Windows
_sp = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
_extra = [os.path.join(_sp, p, "bin") for p in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc")]
os.environ["PATH"] = os.pathsep.join(p for p in _extra if os.path.isdir(p)) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

SUPPORTED_EXT = {".mp3", ".mp4", ".wav", ".ogg", ".flac", ".m4a", ".mkv", ".webm", ".opus", ".aac"}


def format_timestamp(seconds: float, vtt: bool = False) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_txt(segments, path: Path, timestamps: bool):
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            if timestamps:
                f.write(f"[{seg.start:.1f}s] {seg.text.strip()}\n")
            else:
                f.write(seg.text.strip() + "\n")


def write_srt(segments, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")


def write_vtt(segments, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{format_timestamp(seg.start, vtt=True)} --> {format_timestamp(seg.end, vtt=True)}\n")
            f.write(seg.text.strip() + "\n\n")


def transcribe_file(model: WhisperModel, input_path: Path, args) -> None:
    print(f"\n>>> {input_path.name}")

    segments, info = model.transcribe(
        str(input_path),
        language=args.lang if args.lang != "auto" else None,
        beam_size=args.beam_size,
        vad_filter=args.vad,
    )

    # collect segments (generator — iterate once)
    seg_list = list(segments)

    if not args.lang or args.lang == "auto":
        print(f"    Detected language: {info.language} ({info.language_probability:.0%})")

    # determine output path
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    fmt = args.format

    out_path = out_dir / f"{stem}.{fmt}"

    if fmt == "srt":
        write_srt(seg_list, out_path)
    elif fmt == "vtt":
        write_vtt(seg_list, out_path)
    else:  # txt
        write_txt(seg_list, out_path, timestamps=not args.no_timestamps)

    print(f"    Saved → {out_path}  ({len(seg_list)} segments)")


def collect_files(patterns: list[str]) -> list[Path]:
    files = []
    for pattern in patterns:
        expanded = glob.glob(pattern, recursive=True)
        if expanded:
            for p in expanded:
                path = Path(p)
                if path.suffix.lower() in SUPPORTED_EXT:
                    files.append(path)
        else:
            path = Path(pattern)
            if path.exists():
                files.append(path)
            else:
                print(f"Warning: no files matched '{pattern}'", file=sys.stderr)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help="Audio file(s) or glob patterns")
    parser.add_argument("--model", "-m", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--lang", "-l", default="auto",
                        help="Language code (e.g. uk, ru, en) or 'auto' for detection (default: auto)")
    parser.add_argument("--format", "-f", default="txt",
                        choices=["txt", "srt", "vtt"],
                        help="Output format (default: txt)")
    parser.add_argument("--no-timestamps", action="store_true",
                        help="Omit timestamps in txt output (plain text)")
    parser.add_argument("--out-dir", "-o", metavar="DIR",
                        help="Output directory (default: same as input file)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--compute-type", default="float16",
                        choices=["float16", "int8", "float32"],
                        help="Compute type (default: float16 for GPU, use int8 for CPU)")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam search size (default: 5)")
    parser.add_argument("--vad", action="store_true", default=True,
                        help="Enable VAD filter to remove silence (default: on)")
    parser.add_argument("--no-vad", dest="vad", action="store_false",
                        help="Disable VAD filter")

    args = parser.parse_args()

    files = collect_files(args.files)
    if not files:
        print("No audio files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model '{args.model}' on {args.device}...")
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    except Exception:
        print("CUDA unavailable, falling back to CPU with int8...")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"Transcribing {len(files)} file(s)...")
    for f in files:
        try:
            transcribe_file(model, f, args)
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
