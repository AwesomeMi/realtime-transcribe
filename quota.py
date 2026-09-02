"""Free-tier accounting that survives a restart."""

import json
import threading
import time

from config import (LIMIT_ASD, LIMIT_ASH, LIMIT_RPD, LIMIT_RPM, MIN_BILLED, compact)


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
        day = now - 24 * 3600      # the stored window is longer, cut it explicitly
        req_d = sum(1 for t, _ in ev if t >= day)
        sec_h = sum(s for t, s in ev if t >= now - 3600)
        sec_d = sum(s for t, s in ev if t >= day)
        return {"rpm": rpm, "req_hour": req_h, "req_day": req_d,
                "sec_hour": sec_h, "sec_day": sec_d}


def print_quota(usage):
    st = usage.stats()
    print(f"""
Usage over the sliding windows (local accounting, free tier):

  requests per minute  {st['rpm']:>6} / {LIMIT_RPM}
  requests per day     {st['req_day']:>6} / {LIMIT_RPD}
  audio per hour       {compact(st['sec_hour']):>6} / {compact(LIMIT_ASH)}
  audio per day        {compact(st['sec_day']):>6} / {compact(LIMIT_ASD)}

  left for today       ~{compact(max(0, LIMIT_ASD - st['sec_day']))} of audio,
                       ~{LIMIT_RPD - st['req_day']} requests
""")
