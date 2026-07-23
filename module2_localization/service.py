import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import threading
import time

import cv2
import numpy as np

from . import config
from .nats_client import NatsClient
from .services.localization_service import get_localizer


class ShmSource:
    def __init__(self, socket_path, width, height, fps):
        gst = shutil.which("gst-launch-1.0")
        if gst is None:
            raise RuntimeError("gst-launch-1.0 не найден")
        if not os.path.exists(socket_path):
            raise RuntimeError(f"нет сокета {socket_path}: запущен ли fan-out?")
        self.w, self.h = width, height
        self.frame_size = width * height * 3
        caps = f"video/x-raw,format=I420,width={width},height={height},framerate={fps}/1"
        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)
        cmd = [gst, "-q",
               "shmsrc", f"socket-path={socket_path}", "is-live=true", "do-timestamp=true",
               "!", caps,
               "!", "videoconvert",
               "!", "video/x-raw,format=BGR",
               "!", "fdsink", f"fd={write_fd}"]
        self.proc = subprocess.Popen(cmd, pass_fds=(write_fd,),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(write_fd)
        self.reader = os.fdopen(read_fd, "rb", buffering=0)
        self._latest = None
        self._stamp = 0
        self.done = False
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _read_exactly(self, size):
        buf = bytearray()
        while len(buf) < size and not self._stop:
            chunk = self.reader.read(size - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == size else None

    def _loop(self):
        while not self._stop:
            raw = self._read_exactly(self.frame_size)
            if raw is None:
                self.done = True
                break
            frame = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
            with self._lock:
                self._latest = frame
                self._stamp += 1

    def latest(self):
        with self._lock:
            return self._stamp, self._latest

    def stop(self):
        self._stop = True
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.reader.close()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


class CameraSource:
    def __init__(self, spec):
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
                self._latest = frame
                self._stamp += 1
        self.cap.release()

    def latest(self):
        with self._lock:
            return self._stamp, self._latest

    def stop(self):
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


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


class Smoother:
    def __init__(self, win=5, deadzone=None, hyst=4.0):
        from collections import deque
        self.buf = deque(maxlen=win)
        self.deadzone = config.DEADZONE_DEG if deadzone is None else deadzone
        self.hyst = hyst
        self.mt = "straight"

    def smooth(self, bearing):
        import numpy as np
        self.buf.append(bearing)
        b = float(np.median(self.buf))
        if self.mt == "straight":
            if b > self.deadzone + self.hyst:
                self.mt = "right"
            elif b < -(self.deadzone + self.hyst):
                self.mt = "left"
        else:
            if abs(b) < self.deadzone - self.hyst:
                self.mt = "straight"
            elif b > 0 and self.mt == "left":
                self.mt = "right" if b > self.deadzone + self.hyst else "straight"
            elif b < 0 and self.mt == "right":
                self.mt = "left" if b < -(self.deadzone + self.hyst) else "straight"
        return b, self.mt


def to_command(r: dict, smoother=None) -> dict:
    if not r or not r.get("ok") or r.get("inliers", 0) < config.MIN_INLIERS:
        return {"move_type": "lost", "reason": f"inliers: {(r or {}).get('inliers', 0)}"}
    deg, mt = r["bearing_deg"], r["move_type"]
    if mt == "stop":
        return {"move_type": "stop", "node": r["node"]}
    if smoother is not None and mt != "hold":
        deg, mt = smoother.smooth(deg)
    om = r.get("offset_m")
    dm = r.get("dist_to_route_m")
    return {
        "move_type": mt,
        "deg": round(deg, 2),
        "offset": round(r["offset"], 4),
        "dist_to_route": round(r["dist_to_route"], 4),
        "offset_m": round(om, 3) if om is not None else None,
        "dist_to_route_m": round(dm, 3) if dm is not None else None,
        "node": r["node"],
        "inliers": r["inliers"],
    }


async def worker(src, nc, topic: str, loc, stop_evt=None) -> None:
    from collections import deque

    import numpy as np

    loc.steer = config.STEER_MODE
    loc.deadzone = config.DEADZONE_DEG
    chist = deque(maxlen=5)
    last_cmd = ("straight", 0.0)
    stopped = False
    stop_hits = 0
    last_good = None
    last_stamp = -1
    last_node = None               # узел последней ПРИНЯТОЙ позы
    last_fix_t = None
    rejects = 0
    while not (stop_evt and stop_evt.is_set()):
        stamp, frame = src.latest()
        if frame is not None and stamp != last_stamp:
            last_stamp = stamp
            r = loc.locate(frame)
            jump = None
            if r and r.get("ok") and last_node is not None:
                # вперёд робот не может уехать быстрее физики; назад — сколько угодно
                dt = max(time.monotonic() - last_fix_t, 1e-3)
                allowed = max(config.MIN_NODE_JUMP, int(config.MAX_NODES_PER_SEC * dt))
                if r["node"] - last_node > allowed:
                    rejects += 1
                    if rejects < config.MAX_REJECTS:
                        jump = f"скачок: {r['node'] - last_node} > {allowed}"
                        r = {"ok": False, "inliers": 0}
                    else:
                        rejects = 0   # столько отказов подряд -> мы ошибались, принимаем
                else:
                    rejects = 0
            if r and r.get("ok"):
                last_node, last_fix_t = r["node"], time.monotonic()
                chist.append(r["C"])
                d = (chist[-1] - chist[0]) if len(chist) >= 2 else np.zeros(3)
                moving = float(np.linalg.norm(d)) > config.MOVE_EPS
                if moving:
                    last_cmd = (r["move_type"], r["bearing_deg"])
                else:
                    r["move_type"], r["bearing_deg"] = last_cmd
                if r["move_type"] == "stop" and r["inliers"] >= config.STOP_MIN_INLIERS:
                    stop_hits += 1
                else:
                    stop_hits = 0
                stopped = stopped or stop_hits >= config.STOP_CONFIRM
                if stopped:
                    r["move_type"], r["bearing_deg"] = "stop", 0.0
            cmd = to_command(r)
            if cmd["move_type"] != "lost":
                last_good = cmd
            elif last_good:
                cmd = {"move_type": "lost", "node": last_good["node"],
                       "reason": jump or cmd["reason"]}
            if stopped:
                cmd = {"move_type": "stop", "node": (last_good or {}).get("node")}
            cmd["ts"] = round(time.time(), 3)   # последним полем: по нему мозг ведёт сторожевой таймер
            payload = json.dumps(cmd, ensure_ascii=False).encode()
            print(payload.decode(), flush=True)
            if nc:
                await nc.publish(topic, payload)
            await asyncio.sleep(0)
        elif src.done:
            break
        else:
            await asyncio.sleep(0.003)


async def main() -> int:
    ap = argparse.ArgumentParser(description="демон локализации: камера -> команда -> терминал+NATS")
    ap.add_argument("--camera", default=None, help="переопределить источник (индекс/GStreamer)")
    ap.add_argument("--video", default=None, help="видеофайл вместо камеры (отладка)")
    ap.add_argument("--source-step", type=int, default=2, help="файл: каждый N-й кадр в буфер")
    ap.add_argument("--shm", nargs="?", const=config.CAM_SHM_SOCKET, default=None,
                    help=f"кадры из ветки fan-out (по умолчанию {config.CAM_SHM_SOCKET})")
    ap.add_argument("--no-nats", action="store_true", help="только терминал, без NATS")
    args = ap.parse_args()

    if args.video:
        src = VideoFileSource(args.video, step=args.source_step)
    elif args.shm:
        src = ShmSource(args.shm, config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS)
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

    loc = get_localizer()

    stop_evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_evt.set)

    src.start()
    try:
        await worker(src, nc, config.NATS_TOPIC, loc, stop_evt)
    finally:
        src.stop()
        if nc:
            await nc.close()
        print("Cameras Brain остановлен", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
