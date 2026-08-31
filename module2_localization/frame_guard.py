import time
import zlib


class FrameHashGuard:
    def __init__(self, enabled=True, stale_after_s=0.4, recovery_frames=3):
        self.enabled = bool(enabled)
        self.stale_after_s = float(stale_after_s)
        self.recovery_frames = max(1, int(recovery_frames))
        self.reset()

    def reset(self):
        self._last_hash = None
        self._same_since = None
        self._stale = False
        self._fresh_frames = 0
        self.last = {
            "enabled": self.enabled,
            "hash": None,
            "repeat_s": 0.0,
            "stale": False,
            "recovering": False,
            "fresh_frames": 0,
        }

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.reset()

    @staticmethod
    def _hash_frame(frame, seed=0):
        if frame is None:
            return seed
        if not frame.flags.c_contiguous:
            frame = frame.copy()
        return zlib.crc32(memoryview(frame), seed) & 0xFFFFFFFF

    def check(self, front_frame, rear_frame, mode="front", now=None):
        if not self.enabled:
            self.last = {
                "enabled": False,
                "hash": None,
                "repeat_s": 0.0,
                "stale": False,
                "recovering": False,
                "fresh_frames": 0,
            }
            return self.last

        now = time.monotonic() if now is None else float(now)
        if mode == "front":
            value = self._hash_frame(front_frame)
        elif mode == "rear":
            value = self._hash_frame(rear_frame)
        else:
            value = self._hash_frame(front_frame)
            value = self._hash_frame(rear_frame, value)

        repeated = self._last_hash is not None and value == self._last_hash
        if repeated:
            self._fresh_frames = 0
            if self._same_since is None:
                self._same_since = now
            repeat_s = max(0.0, now - self._same_since)
            if repeat_s >= self.stale_after_s:
                self._stale = True
        else:
            self._last_hash = value
            self._same_since = None
            repeat_s = 0.0
            if self._stale:
                self._fresh_frames += 1
                if self._fresh_frames >= self.recovery_frames:
                    self._stale = False
                    self._fresh_frames = 0
            else:
                self._fresh_frames = 0

        self.last = {
            "enabled": True,
            "hash": f"{value:08x}",
            "repeated": repeated,
            "repeat_s": round(repeat_s, 4),
            "stale": self._stale,
            "recovering": self._stale and not repeated,
            "fresh_frames": self._fresh_frames,
        }
        return self.last
