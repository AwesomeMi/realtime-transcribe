"""Audio device enumeration and capture, per platform."""

import json
import subprocess
import threading

import numpy as np

from config import (FRAME_BYTES, FRAME_MS, FRAME_SAMPLES, IS_WINDOWS, SAMPLE_RATE, die)


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
            # each step guarded on its own: if stop_stream() raises on an
            # already-dead stream, terminate() must still run or the PyAudio
            # instance leaks, and the exception would escape the thread and
            # bury the real reason in state.fatal
            for cleanup in (stream.stop_stream, stream.close, audio.terminate):
                try:
                    cleanup()
                except Exception:
                    pass


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


Capture = WindowsCapture if IS_WINDOWS else PosixCapture
