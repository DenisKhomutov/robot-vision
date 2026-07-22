import argparse
import asyncio
import json
import threading
import time

import cv2

from . import config
from .nats_client import NatsClient
from .services.localization_service import get_localizer


class CameraSource:
    def __init__(self, spec):
        # int -> индекс устройства; строка -> GStreamer-пайплайн
        if isinstance(spec, str) and not spec.isdigit():
            self.cap = cv2.VideoCapture(spec, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(int(spec))
        if not self.cap.isOpened():
            raise RuntimeError(f"не открывается камера: {spec}")
        self._latest = None
        self._stamp = 0
        self.done = False
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                self.done = True
                break
            with self._lock:
                self._latest = frame          # затираем непрочитанный
                self._stamp += 1
        self.cap.release()

    def latest(self):
        with self._lock:
            return self._stamp, self._latest

    def stop(self):
        self._stop = True


class VideoFileSource:

    def __init__(self, path: str, step: int = 2, fps: float = 30.0):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"не открывается видео: {path}")
        self.step = step
        self.dt = 1.0 / fps
        self._latest = None
        self._stamp = 0
        self.done = False

    def start(self):
        asyncio.ensure_future(self._run())

    async def _run(self):
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                self.done = True
                self.cap.release()
                return
            if idx % self.step == 0:
                self._latest = frame
                self._stamp += 1
            idx += 1
            await asyncio.sleep(self.dt)

    def latest(self):
        return self._stamp, self._latest

    def stop(self):
        pass


def to_command(r: dict) -> dict:
    if not r or not r.get("ok") or r["inliers"] < config.MIN_INLIERS:
        return {"valid": False, "ts": round(time.time(), 3),
                "reason": (r or {}).get("reason", f"inliers {(r or {}).get('inliers', 0)}")}
    om = r.get("offset_m")
    dm = r.get("dist_to_route_m")
    return {
        "valid": True,
        "ts": round(time.time(), 3),
        "move_type": r["move_type"],
        "deg": round(r["bearing_deg"], 2),
        "offset": round(r["offset"], 4),
        "dist_to_route": round(r["dist_to_route"], 4),
        "offset_m": round(om, 3) if om is not None else None,
        "dist_to_route_m": round(dm, 3) if dm is not None else None,
        "node": r["node"],
        "inliers": r["inliers"],
    }


async def worker(src, nc, topic: str) -> None:
    loc = get_localizer()          # грузит карту + ALIKED (лениво, один раз)
    loc.steer = config.STEER_MODE
    last_stamp = -1
    while True:
        stamp, frame = src.latest()
        if frame is not None and stamp != last_stamp:
            last_stamp = stamp
            r = loc.locate(frame)
            payload = json.dumps(to_command(r), ensure_ascii=False).encode()
            print(payload.decode(), flush=True)          # всегда в терминал
            if nc:
                await nc.publish(topic, payload)          # и в NATS
        elif src.done:
            break
        else:
            await asyncio.sleep(0.003)


async def main() -> int:
    ap = argparse.ArgumentParser(description="демон локализации: камера -> команда -> терминал+NATS")
    ap.add_argument("--camera", default=None, help="переопределить источник (индекс/GStreamer)")
    ap.add_argument("--video", default=None, help="видеофайл вместо камеры (отладка)")
    ap.add_argument("--source-step", type=int, default=2, help="файл: каждый N-й кадр в буфер")
    ap.add_argument("--no-nats", action="store_true", help="только терминал, без NATS")
    args = ap.parse_args()

    if args.video:
        src = VideoFileSource(args.video, step=args.source_step)
    else:
        src = CameraSource(args.camera or config.CAMERA)

    nc = None
    if not args.no_nats:
        try:
            nc = NatsClient(config.NATS_URL)
            await nc.connect()
        except Exception as e:  # noqa: BLE001
            print(f"[NATS] недоступен ({e}); печатаю только в терминал", flush=True)
            nc = None

    src.start()
    try:
        await worker(src, nc, config.NATS_TOPIC)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        if nc:
            await nc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
