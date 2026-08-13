import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import threading

import cv2
import numpy as np

from . import config
from .core.dualnav import DualNav
from .nats_client import NatsClient
from .services.localization_service import build_localizer, get_localizer


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


def make_control_handler(nav):
    async def on_control(msg):
        try:
            c = json.loads(msg.data.decode())
        except json.JSONDecodeError:
            return
        cmd = c.get("cmd")
        if cmd == "pause":
            nav.pf.pause(); nav.pr.pause()
        elif cmd == "resume":
            nav.pf.resume(); nav.pr.resume()
        elif cmd == "reset":
            nav.pf.reset(); nav.pr.reset()
        elif cmd == "set_mode":
            nav.set_mode(c.get("mode"))
            print(f"[control] режим -> {nav.mode}", flush=True)
            return
        elif cmd == "set_map":
            print(f"[control] set_map пока не реализован (запрошена карта {c.get('map')})", flush=True)
        else:
            print(f"[control] неизвестная команда: {cmd}", flush=True)
            return
        print(f"[control] {cmd}", flush=True)
    return on_control


class TrafficBranch:
    """Ветка светофора: детекция+классификация ВСЕГДА на переднем кадре, но публикуем
    только когда активная навигация в ЗОНЕ. Зону выбирает активная камера DualNav:
    ведёт фронт -> зона фронт-карты, ведёт зад -> зона зад-карты. Последнюю позу помним
    (обе камеры потеряны -> навигация стоп, но светофор продолжает по последней зоне)."""

    def __init__(self, cfg):
        import module1_traffic_light as tl
        from module1_traffic_light import config as tl_config
        tl_config.DET_CONF = getattr(cfg, "TRAFFIC_DET_CONF", tl_config.DET_CONF)
        tl.init_models()
        from module1_traffic_light.services.traffic_light_service import analyze
        self._analyze = analyze
        self.front_zone = getattr(cfg, "FRONT_TRAFFIC_ZONE", None)
        self.rear_zone = getattr(cfg, "REAR_TRAFFIC_ZONE", None)
        self.last_cam, self.last_node = "rear", None
        self.go = True                             # разрешено ехать (нет красного) — латч

    @staticmethod
    def _in(zone, node):
        return zone is None or (node is not None and zone[0] <= node <= zone[1])

    def in_zone(self, cam, node):
        if node is not None:                       # запоминаем последнюю известную позу
            self.last_cam, self.last_node = cam or self.last_cam, node
        cam = cam or self.last_cam
        node = node if node is not None else self.last_node
        zone = self.front_zone if cam == "front" else self.rear_zone
        return self._in(zone, node)

    def update(self, in_zone, frame_bgr):
        """Детекция на переднем кадре в зоне; латч go/stop: красный -> стоп, зелёный -> ехать,
        иначе держим предыдущее. Вне зоны светофор не действует -> go=True."""
        if not in_zone:
            self.go = True
            return None
        from PIL import Image
        r = self._analyze(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        sig = (r.get("signal") or "").lower()
        if "red" in sig or "крас" in sig:
            self.go = False
        elif "green" in sig or "зел" in sig:
            self.go = True
        return r                                   # {"status","signal","confidence"}


async def worker(front_src, rear_src, nc, topic: str, nav, stop_evt=None, traffic=None) -> None:
    if nc:
        await nc.subscribe(config.NATS_CONTROL_TOPIC, make_control_handler(nav))
    last_stamp = -1
    while not (stop_evt and stop_evt.is_set()):
        stamp, rframe = rear_src.latest()               # ведём цикл по задней (всегда есть)
        fframe = front_src.latest()[1] if front_src else None
        if rframe is not None and stamp != last_stamp:
            last_stamp = stamp
            cmd = nav.step(fframe, rframe)
            # светофор: детекция на переднем кадре, КРАСНЫЙ -> демон сам отдаёт stop
            # (мозг только исполняет команду, решение здесь)
            if traffic and fframe is not None:
                inz = traffic.in_zone(cmd.get("cam"), cmd.get("node"))
                tl = traffic.update(inz, fframe)
                if not traffic.go and cmd.get("move_type") != "lost":
                    cmd["move_type"], cmd["deg"] = "stop", 0.0
                    cmd["traffic"] = "red"
                if tl is not None:
                    print(f"[TL] {json.dumps(tl, ensure_ascii=False)} go={traffic.go}", flush=True)
                    if nc:
                        await nc.publish(config.NATS_TRAFFIC_TOPIC,
                                         json.dumps(tl, ensure_ascii=False).encode())
            payload = json.dumps(cmd, ensure_ascii=False).encode()
            print(payload.decode(), flush=True)
            if nc:
                await nc.publish(topic, payload)
            await asyncio.sleep(0)
        elif rear_src.done or (front_src and front_src.done):
            break
        else:
            await asyncio.sleep(0.003)


def _shm(sock):
    return ShmSource(sock, config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS)


async def main() -> int:
    ap = argparse.ArgumentParser(description="демон локализации: камера -> команда -> терминал+NATS")
    ap.add_argument("--camera", default=None, help="переопределить источник (индекс/GStreamer)")
    ap.add_argument("--video", default=None, help="видеофайл вместо камеры (отладка, только зад)")
    ap.add_argument("--source-step", type=int, default=2, help="файл: каждый N-й кадр в буфер")
    ap.add_argument("--shm", nargs="?", const=config.CAM_SHM_SOCKET, default=None,
                    help=f"кадры из ветки fan-out (по умолчанию {config.CAM_SHM_SOCKET})")
    ap.add_argument("--dual", action="store_true",
                    help="двухкамерный режим: грузит фронт+зад локализаторы, можно переключать в viz")
    ap.add_argument("--video-front", default=None, help="dual: видео передней (отладка)")
    ap.add_argument("--video-rear", default=None, help="dual: видео задней (отладка)")
    ap.add_argument("--mode", default=None, choices=["rear", "dual"], help="стартовый режим (по умолч. из конфига)")
    ap.add_argument("--no-nats", action="store_true", help="только терминал, без NATS")
    args = ap.parse_args()

    # dual (фронт-локализатор) — только явно: флаг --dual или NAV_MODE=dual (нужна ГОТОВАЯ фронт-карта)
    dual = args.dual or config.NAV_MODE == "dual"

    # ── ЗАДНИЙ источник (навигация)
    rear_video = args.video_rear or args.video
    if rear_video:
        rear_src = VideoFileSource(rear_video, step=args.source_step)
    elif args.shm:
        rear_src = _shm(args.shm)
    elif dual:
        rear_src = _shm(config.REAR_SHM_SOCKET)
    else:
        rear_src = CameraSource(args.camera or config.CAMERA)

    # ── локализаторы
    front_loc = None
    if dual:
        front_loc = build_localizer(config.FRONT_MAP, config.FRONT_CAM_BACK)
        rear_loc = build_localizer(config.REAR_MAP, config.REAR_CAM_BACK)
    else:
        rear_loc = get_localizer()

    # ── ПЕРЕДНИЙ источник: для dual-навигации И/ИЛИ для светофора (карта фронта не нужна)
    front_src = None
    if dual or getattr(config, "TRAFFIC_LIGHT_ENABLED", False):
        try:
            if args.video_front:
                front_src = VideoFileSource(args.video_front, step=args.source_step)
            else:
                front_src = _shm(config.FRONT_SHM_SOCKET)
        except Exception as e:  # noqa: BLE001
            print(f"[перёд] нет передней камеры ({e})", flush=True)
            if dual:
                return 1

    nav = DualNav(front_loc, rear_loc, config)
    if args.mode:
        nav.set_mode(args.mode)
    print(f"[nav] режим {nav.mode}" + ("  (dual доступен)" if front_loc else ""), flush=True)

    traffic = None
    if getattr(config, "TRAFFIC_LIGHT_ENABLED", False):
        if front_src is None:
            print("[TL] светофор включён, но нет передней камеры (нужен dual/два источника) — пропуск", flush=True)
        else:
            traffic = TrafficBranch(config)
            print("[TL] ветка светофора активна (детекция на переднем кадре, гейт по зоне)", flush=True)

    nc = None
    if not args.no_nats:
        try:
            nc = NatsClient(config.NATS_URL)
            await nc.connect()
        except Exception as e:
            print(f"[NATS] недоступен ({e}); печатаю только в терминал", flush=True)
            nc = None

    stop_evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_evt.set)

    if front_src:
        front_src.start()
    rear_src.start()
    try:
        await worker(front_src, rear_src, nc, config.NATS_TOPIC, nav, stop_evt, traffic)
    finally:
        if front_src:
            front_src.stop()
        rear_src.stop()
        if nc:
            await nc.close()
        print("Cameras Brain остановлен", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
