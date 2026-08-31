import time


class FrameTimeoutGuard:
    def __init__(self, enabled=True, timeout_s=0.5, recovery_frames=3):
        self.enabled = bool(enabled)
        self.timeout_s = float(timeout_s)
        self.recovery_frames = max(1, int(recovery_frames))
        self.reset()

    def reset(self, now=None):
        now = time.monotonic() if now is None else float(now)
        self._last_token = None
        self._last_fresh_at = now
        self._timed_out = False
        self._fresh_frames = 0
        self.last = self._state(now, fresh=False)

    def set_enabled(self, enabled, now=None):
        self.enabled = bool(enabled)
        self.reset(now)

    def _state(self, now, fresh):
        return {
            "enabled": self.enabled,
            "fresh": bool(fresh),
            "age_s": round(max(0.0, now - self._last_fresh_at), 4),
            "timed_out": bool(self._timed_out),
            "recovering": bool(self._timed_out and self._fresh_frames > 0),
            "fresh_frames": int(self._fresh_frames),
        }

    def observe(self, token, now=None):
        now = time.monotonic() if now is None else float(now)
        fresh = token is not None and token != self._last_token
        if fresh:
            self._last_token = token
            self._last_fresh_at = now
            if self._timed_out:
                self._fresh_frames += 1
                if self._fresh_frames >= self.recovery_frames:
                    self._timed_out = False
                    self._fresh_frames = 0
            else:
                self._fresh_frames = 0
        elif self.enabled and now - self._last_fresh_at >= self.timeout_s:
            self._timed_out = True
            self._fresh_frames = 0
        self.last = self._state(now, fresh)
        return self.last

    def poll(self, now=None):
        return self.observe(self._last_token, now)
