import os
import sys

_sp = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
_extra = [os.path.join(_sp, p, "bin") for p in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc")]
_extra = [p for p in _extra if os.path.isdir(p)]
os.environ["PATH"] = os.pathsep.join(_extra) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

for filename in ["peredzahist.ogg", "peredzahist2.ogg"]:
    print(f"Транскрибирую {filename}...")
    segments, info = model.transcribe(filename, language="uk")  # или "uk"
    
    out = filename.replace(".ogg", ".txt")
    with open(out, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(f"[{segment.start:.1f}s] {segment.text}\n")
    
    print(f"Готово → {out}")