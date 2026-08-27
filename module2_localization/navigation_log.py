import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class NavigationLog:
    def __init__(self, root=None):
        project_root = Path(__file__).resolve().parents[1]
        self.root = Path(root) if root is not None else project_root / "logs"
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
        self.path = self.root / f"navigation_{stamp}_pid{os.getpid()}.jsonl"
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._failed = False
        self.emit("session_started", pid=os.getpid(), cwd=str(Path.cwd()))

    @staticmethod
    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): NavigationLog._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [NavigationLog._jsonable(v) for v in value]
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    def emit(self, event, **fields):
        if self._closed or self._failed:
            return
        with self._lock:
            try:
                self._sequence += 1
                row = {
                    "seq": self._sequence,
                    "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
                    "monotonic_s": round(time.monotonic(), 6),
                    "event": event,
                }
                row.update(self._jsonable(fields))
                self._file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception as exc:
                self._failed = True
                print(f"[log] запись отключена после ошибки: {exc}", file=sys.stderr, flush=True)

    def close(self):
        if self._closed:
            return
        self.emit("session_stopped")
        self._closed = True
        with self._lock:
            self._file.flush()
            self._file.close()


def frame_metrics(frame):
    if frame is None:
        return None
    array = np.asarray(frame)
    sample_step = 16
    sample = array[::sample_step, ::sample_step]
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sample_step": sample_step,
        "mean": round(float(sample.mean()), 3),
        "std": round(float(sample.std()), 3),
        "min": int(sample.min()),
        "max": int(sample.max()),
    }
    if array.ndim == 3 and array.shape[2] >= 3:
        channels = sample.reshape(-1, sample.shape[2]).mean(axis=0)
        result["channel_mean"] = [round(float(value), 3) for value in channels]
    return result
