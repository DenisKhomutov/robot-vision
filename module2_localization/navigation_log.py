import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class NavigationLog:
    DEFAULT_MAX_BYTES = 150 * 1024 * 1024

    def __init__(self, root=None, max_bytes=None):
        project_root = Path(__file__).resolve().parents[1]
        self.root = Path(root) if root is not None else project_root / "logs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes if max_bytes is not None else self.DEFAULT_MAX_BYTES)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes должен быть положительным")
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._failed = False
        self._pid = os.getpid()
        self._day = self._today()
        self.path = self._select_path(self._day)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self.emit("session_started", pid=os.getpid(), cwd=str(Path.cwd()))

    @staticmethod
    def _today():
        return datetime.now().astimezone().strftime("%Y%m%d")

    def _select_path(self, day, force_next=False):
        base = self.root / f"navigation_{day}.jsonl"
        candidates = []
        if base.exists():
            candidates.append((1, base))
        prefix = f"navigation_{day}_"
        for path in self.root.glob(f"{prefix}*.jsonl"):
            suffix = path.stem.removeprefix(prefix)
            if suffix.isdigit():
                candidates.append((int(suffix), path))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            return base
        index, latest = candidates[-1]
        if not force_next and latest.stat().st_size < self.max_bytes:
            return latest
        return self.root / f"navigation_{day}_{index + 1:03d}.jsonl"

    def _rotate_if_needed(self, line_bytes):
        day = self._today()
        day_changed = day != self._day
        size = self.path.stat().st_size if self.path.exists() else 0
        full = size > 0 and size + line_bytes > self.max_bytes
        if not day_changed and not full:
            return
        self._file.flush()
        self._file.close()
        self._day = day
        self.path = self._select_path(day, force_next=full and not day_changed)
        self._file = self.path.open("a", encoding="utf-8", buffering=1)

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
                    "pid": self._pid,
                    "event": event,
                }
                row.update(self._jsonable(fields))
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                self._rotate_if_needed(len(line.encode("utf-8")))
                self._file.write(line)
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
