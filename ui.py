"""Terminal UI: the pinned status line, keystrokes, and session state."""

import codecs
import os
import select
import shutil
import sys
import textwrap
import threading
import time

from config import IS_WINDOWS

if IS_WINDOWS:
    import ctypes
    import msvcrt
else:
    import termios
    import tty


def enable_vt():
    """Old conhost does not understand ANSI until VT output processing is on."""
    if not IS_WINDOWS:
        return
    try:
        k = ctypes.windll.kernel32
        handle = k.GetStdHandle(-11)                  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(handle, ctypes.byref(mode)):
            k.SetConsoleMode(handle, mode.value | 0x0004)   # VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


class Screen:
    def __init__(self, tui):
        self.tui = tui and sys.stdout.isatty()
        self.lock = threading.RLock()
        self.status = ""
        self.input = None          # not None => question input line at the bottom
        self.input_full = False    # which scope was picked when it opened
        self.old_term = None
        self.w, self.h = shutil.get_terminal_size((100, 24))
        if not self.tui:
            return
        enable_vt()
        sys.stdout.write("\x1b[?25l")                 # hide the cursor
        sys.stdout.write(f"\x1b[1;{self.h - 1}r")     # scrolling region
        sys.stdout.write(f"\x1b[{self.h - 1};1H")
        sys.stdout.write("\x1b[>1u")                  # kitty keyboard: tell Shift+Enter apart
        sys.stdout.flush()

    def _apply_resize(self):
        """The size is polled instead of catching SIGWINCH.

        There is no SIGWINCH on Windows, and on Unix the handler runs in the
        main thread between bytecodes and, since the lock is reentrant, it
        happily wedged its own escape sequences into the middle of someone
        else's write. The poll is called from _draw() — that is, at least once
        every 0.25s — and costs a single ioctl call.
        """
        size = shutil.get_terminal_size((100, 24))
        if (size.columns, size.lines) == (self.w, self.h):
            return
        old_h = self.h
        self.w, self.h = size
        # when the window grows, the old status line ends up inside the scrolling
        # region and stays there as garbage — erase it at its old position
        sys.stdout.write(f"\x1b[{old_h};1H\x1b[2K")
        sys.stdout.write(f"\x1b[r\x1b[1;{self.h - 1}r")
        sys.stdout.write(f"\x1b[{self.h - 1};1H")

    def raw_input_mode(self):
        if not (self.tui and sys.stdin.isatty()):
            return
        if IS_WINDOWS:
            return                  # msvcrt reads without echo, no mode change needed
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

    def set_input(self, text, full=False):
        """text=None — close the input line."""
        with self.lock:
            self.input = text
            self.input_full = full
            if self.tui:
                self._draw()

    def _draw(self):
        self._apply_resize()
        if self.input is not None:
            line = ("?> " if self.input_full else "> ") + self.input
            if len(line) > self.w - 1:
                line = line[-(self.w - 1):]
            sys.stdout.write(f"\x1b[{self.h};1H\x1b[2K\x1b[1;35m{line}\x1b[0m\x1b[?25h")
            sys.stdout.flush()      # leave the cursor at the end of the input
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
                sys.stdout.write("\x1b[<u")     # restore the normal keyboard mode
                sys.stdout.write(f"\x1b[r\x1b[{self.h};1H\x1b[2K\x1b[?25h")
                sys.stdout.flush()


class WindowsKeyReader:
    """Reads via msvcrt: select on Windows accepts sockets only, not stdin.

    Windows Terminal cannot do the kitty keyboard protocol, so Shift+Enter here is
    indistinguishable from a plain Enter — "the whole transcript" is selected with
    the `?` key instead of `/` when opening the question line.
    """

    def key(self, timeout=0.3):
        deadline = time.time() + timeout
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):     # function key prefix
                    msvcrt.getwch()             # eat the scan code and ignore it
                    return None
                if ch in ("\r", "\n"):
                    return ("enter",)
                if ch == "\x08":
                    return ("backspace",)
                if ch == "\x15":
                    return ("clear",)
                if ch == "\x1b":
                    return ("esc",)
                if ch == "\x03":
                    return ("interrupt",)
                return ("char", ch)
            if time.time() >= deadline:
                return None
            time.sleep(0.01)


class PosixKeyReader:
    """Reads straight from the file descriptor.

    select() only sees the fd, while sys.stdin.read(1) drags the whole available
    chunk into the TextIOWrapper buffer — after the first character select stays
    quiet and the rest of a pasted line is lost. Hence our own buffer and an
    incremental decoder.
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
                return ("shift_enter",)     # Alt+Enter — fallback without CSI-u
            if seq.startswith("[") and seq.endswith("u"):
                parts = seq[1:-1].split(";")
                if parts[0] == "13":        # Enter in the kitty keyboard protocol
                    mod = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                    return ("shift_enter",) if (mod - 1) & 1 else ("enter",)
            return None
        if ch in ("\r", "\n"):
            return ("enter",)
        if ch in ("\x7f", "\b"):
            return ("backspace",)
        if ch == "\x15":                   # Ctrl+U — clear the line
            return ("clear",)
        if ch == "\x03":
            return ("interrupt",)
        return ("char", ch)


KeyReader = WindowsKeyReader if IS_WINDOWS else PosixKeyReader


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
        self.transcript = []      # [(ts, src, text)] — context for Gemini
        self.interim = ""         # the current unfinished phrase in live mode
        self.started = time.time()
