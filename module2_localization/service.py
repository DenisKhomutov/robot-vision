import argparse
import asyncio
import json
import signal
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
        # ждём поток: он сам выйдет из cap.read() и освободит устройство. Иначе
        # интерпретатор гасится, пока поток внутри драйвера камеры -> SIGABRT.
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
    """Гасит мигание команд от покадрового шума позы: медиана азимута по последним кадрам
    + гистерезис на move_type (не перекидываем left/right/straight, пока сглаженный азимут
    не выйдет за порог УСТОЙЧИВО). Лёгкое окно (3) — задержка минимальна."""

    def __init__(self, win=5, deadzone=None, hyst=4.0):
        from collections import deque
        self.buf = deque(maxlen=win)
        self.deadzone = config.DEADZONE_DEG if deadzone is None else deadzone
        self.hyst = hyst           # запас за порогом, чтобы не дребезжать у границы
        self.mt = "straight"

    def smooth(self, bearing):
        import numpy as np
        self.buf.append(bearing)
        b = float(np.median(self.buf))
        # гистерезис: входим в поворот при |b|>deadzone+hyst, выходим при |b|<deadzone-hyst
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
    if not r or not r.get("ok") or r["inliers"] < config.MIN_INLIERS:
        return {"valid": False, "ts": round(time.time(), 3),
                "reason": (r or {}).get("reason", f"inliers {(r or {}).get('inliers', 0)}")}
    deg, mt = r["bearing_deg"], r["move_type"]
    if smoother is not None and mt not in ("stop", "hold"):   # эти состояния не сглаживаем
        deg, mt = smoother.smooth(deg)
    om = r.get("offset_m")
    dm = r.get("dist_to_route_m")
    return {
        "valid": True,
        "ts": round(time.time(), 3),
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
    loc.deadzone = config.DEADZONE_DEG   # мёртвая зона азимута из конфига -> в command()
    chist = deque(maxlen=5)        # история позиций: едем/стоим
    last_cmd = ("straight", 0.0)   # держим последнюю команду на месте (на старте — прямо)
    stopped = False                # латч: конец маршрута подтверждён -> команды больше не меняются
    stop_hits = 0                  # сколько подряд кадров говорят «конец»
    last_good = None               # последняя валидная команда: чем дополнять payload после латча
    last_stamp = -1
    while not (stop_evt and stop_evt.is_set()):
        stamp, frame = src.latest()
        if frame is not None and stamp != last_stamp:
            last_stamp = stamp
            r = loc.locate(frame)
            if r and r.get("ok"):
                # команда r из ПОЗЫ (гладкая в движении). Стоим -> держим последнюю, чтобы
                # дрожь позы на месте не крутила команду. Конец -> stop (из command) + латч.
                chist.append(r["C"])
                d = (chist[-1] - chist[0]) if len(chist) >= 2 else np.zeros(3)
                moving = float(np.linalg.norm(d)) > config.MOVE_EPS
                if moving:
                    last_cmd = (r["move_type"], r["bearing_deg"])
                else:
                    r["move_type"], r["bearing_deg"] = last_cmd
                # латч только по ПОДТВЕРЖДЁННОМУ концу: одиночное ложное совпадение у
                # последних узлов иначе останавливает робота навсегда.
                if r["move_type"] == "stop" and r["inliers"] >= config.STOP_MIN_INLIERS:
                    stop_hits += 1
                else:
                    stop_hits = 0
                stopped = stopped or stop_hits >= config.STOP_CONFIRM
                if stopped:
                    r["move_type"], r["bearing_deg"] = "stop", 0.0
            cmd = to_command(r)
            if cmd["valid"]:
                last_good = cmd
            if stopped:   # приехали -> держим STOP, даже если потом потеряемся у стены
                cmd = {**(last_good or {}), "valid": True, "ts": round(time.time(), 3),
                       "move_type": "stop", "deg": 0.0}
            payload = json.dumps(cmd, ensure_ascii=False).encode()
            print(payload.decode(), flush=True)          # всегда в терминал
            if nc:
                await nc.publish(topic, payload)          # и в NATS
            # с камеры кадр готов всегда: без явной уступки планировщику цикл не отдаёт
            # управление, и обработчик сигнала (Ctrl+C) никогда не выполняется.
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

    # Карту грузим ДО установки обработчиков: pycolmap при чтении реконструкции ставит
    # свой обработчик сигналов (glog), затирая наш — тогда Ctrl+C уходит в никуда.
    loc = get_localizer()

    # Ctrl+C ловим сигналом, а не KeyboardInterrupt: иначе прерывание влетает внутрь
    # await nc.close() и в чужие C-потоки (OpenCV) -> SIGABRT вместо выхода.
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
        print("[демон] остановлен", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
