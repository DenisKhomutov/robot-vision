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
from .services.localization_service import build_runtime_localizer, get_localizer


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


def make_control_handler(nav, traffic=None):
    async def on_control(msg):
        try:
            c = json.loads(msg.data.decode())
        except json.JSONDecodeError:
            return
        cmd = c.get("cmd")
        if cmd == "pause":
            nav.pause()
        elif cmd == "resume":
            nav.resume()
        elif cmd == "reset":
            nav.pf.reset(); nav.pr.reset()
            if traffic:
                traffic.reset()
        elif cmd == "set_mode":
            nav.set_mode(c.get("mode"))
            print(f"[control] режим -> {nav.mode}", flush=True)
            return
        elif cmd == "set_route":
            route = c.get("route")
            if not nav.set_route(route):
                print(f"[control] неизвестный или недоступный маршрут: {route}", flush=True)
                return
            if traffic:
                traffic.reset()
            print(f"[control] маршрут -> {route}; цепочка уже в памяти", flush=True)
            return
        elif cmd == "set_map":
            camera, map_name = c.get("camera"), c.get("map")
            if camera not in ("front", "rear") or not isinstance(map_name, str):
                print("[control] set_map: нужны camera=front|rear и map", flush=True)
                return
            map_dir = config.MAPS_DIR / map_name
            if not (map_dir / "runtime.npz").exists() or not (map_dir / "aliked_bank.npz").exists():
                print(f"[control] set_map: нет runtime-комплекта {map_name}", flush=True)
                return
            print(f"[control] фоновая загрузка {camera} -> {map_name}; текущая карта продолжает вести", flush=True)
            try:
                back_facing = config.FRONT_CAM_BACK if camera == "front" else config.REAR_CAM_BACK
                localizer = await asyncio.to_thread(build_runtime_localizer, map_name, back_facing)
                nav.set_localizer(camera, localizer, map_name)
            except Exception as exc:  # noqa: BLE001
                print(f"[control] set_map ошибка: {exc}", flush=True)
                return
            print(f"[control] карта {camera} -> {map_name}; переключено без остановки", flush=True)
            return
        elif cmd == "set_traffic":
            if traffic is None:
                print("[control] светофорная ветка недоступна: нет передней камеры", flush=True)
                return
            enabled = bool(c.get("enabled"))
            await traffic.set_enabled(enabled)
            print(f"[control] детекция светофора -> {'ON' if enabled else 'OFF'}", flush=True)
            return
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
        self.zones = getattr(cfg, "TRAFFIC_ZONES", {})
        self.last_map, self.last_node = None, None
        # машина состояний в зоне: WAIT_RED (стоим, ждём красный, зелёный игнорим) ->
        # WAIT_GREEN (видели красный, стоим, ждём зелёный) -> GO (защёлка до reset;
        # детектор больше не запускается). Так робот не остановится уже на дороге.
        self.state = "WAIT_RED"
        self.completed = False

    def reset(self):
        self.state = "WAIT_RED"
        self.completed = False
        self.last_map = None
        self.last_node = None

    @staticmethod
    def _in(zone, node):
        return zone is not None and node is not None and zone[0] <= node <= zone[1]

    def in_zone(self, map_name, node):
        if map_name != self.last_map:
            self.last_node = None
            self.last_map = map_name
        if node is not None:                       # запоминаем последнюю известную позу
            self.last_node = node
        node = node if node is not None else self.last_node
        zone = self.zones.get(map_name)
        return self._in(zone, node)

    @property
    def go(self):
        return self.completed

    def update(self, in_zone, frame_bgr):
        """WAIT_RED ignores green; RED arms WAIT_GREEN; next GREEN latches GO."""
        if self.completed:
            return None                            # до reset больше никогда не анализируем
        if not in_zone:
            self.state = "WAIT_RED"
            return None
        from PIL import Image
        r = self._analyze(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        sig = (r.get("signal") or "").lower()
        is_red = "red" in sig or "крас" in sig or "stop" in sig
        is_green = "green" in sig or "зел" in sig or "go" in sig
        if self.state == "WAIT_RED" and is_red:
            self.state = "WAIT_GREEN"
        elif self.state == "WAIT_GREEN" and is_green:
            self.state = "GO"
            self.completed = True
        return r


class TrafficController:
    """Runtime on/off switch with lazy model loading and no navigation pause."""

    def __init__(self, cfg, enabled=False):
        self.cfg = cfg
        self.enabled = False
        self.loading = False
        self.branch = None
        self.start_enabled = enabled

    async def set_enabled(self, enabled):
        if not enabled:
            self.enabled = False
            if self.branch:
                self.branch.reset()
            return
        if self.branch is None:
            self.loading = True
            try:
                self.branch = await asyncio.to_thread(TrafficBranch, self.cfg)
            finally:
                self.loading = False
        self.branch.reset()
        self.enabled = True

    def reset(self):
        if self.branch:
            self.branch.reset()

    @property
    def state(self):
        return None if self.branch is None else self.branch.state

    def process(self, cmd, frame):
        cmd["traffic_enabled"] = self.enabled
        cmd["traffic_loading"] = self.loading
        if not self.enabled or self.branch is None:
            return None
        in_zone = self.branch.in_zone(cmd.get("map"), cmd.get("global_node", cmd.get("node")))
        result = self.branch.update(in_zone, frame)
        cmd["traffic_in_zone"] = in_zone
        cmd["traffic_state"] = self.branch.state
        if in_zone and not self.branch.go and cmd.get("move_type") != "lost":
            cmd["move_type"], cmd["deg"] = "stop", 0.0
        return result


async def worker(front_src, rear_src, nc, topic: str, nav, stop_evt=None, traffic=None) -> None:
    if nc:
        await nc.subscribe(config.NATS_CONTROL_TOPIC, make_control_handler(nav, traffic))
    last_stamp = -1
    while not (stop_evt and stop_evt.is_set()):
        stamp, rframe = rear_src.latest()               # ведём цикл по задней (всегда есть)
        fframe = front_src.latest()[1] if front_src else None
        if rframe is not None and stamp != last_stamp:
            last_stamp = stamp
            cmd = nav.step(fframe, rframe)
            # светофор: детекция на переднем кадре. Решение ПОЛНОСТЬЮ здесь, мозгу
            # отдельно ничего не шлём — только перебиваем команду навигации на stop,
            # пока машина состояний не разрешит ехать (state GO).
            if traffic and fframe is not None:
                tl = traffic.process(cmd, fframe)
                if tl is not None:
                    print(f"[TL] {json.dumps(tl, ensure_ascii=False)} state={traffic.state}", flush=True)
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
    ap.add_argument("--no-recovery", action="store_true",
                    help="отключить full-map recovery шардов (замер чистой скорости по шарду)")
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
    recovery = False if args.no_recovery else None
    front_loc = None
    if dual:
        front_loc = build_runtime_localizer(config.FRONT_MAP, config.FRONT_CAM_BACK, full_recovery=recovery)
        rear_loc = build_runtime_localizer(config.REAR_MAP, config.REAR_CAM_BACK, full_recovery=recovery)
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

    nav = DualNav(front_loc, rear_loc, config, front_map=config.FRONT_MAP, rear_map=config.REAR_MAP)
    if args.mode:
        nav.set_mode(args.mode)
    print(f"[nav] режим {nav.mode}" + ("  (dual доступен)" if front_loc else ""), flush=True)

    traffic = None
    if front_src is None:
        if getattr(config, "TRAFFIC_LIGHT_ENABLED", False):
            print("[TL] светофор включён, но нет передней камеры — пропуск", flush=True)
    else:
        traffic = TrafficController(config)
        if getattr(config, "TRAFFIC_LIGHT_ENABLED", False):
            await traffic.set_enabled(True)
        print(f"[TL] runtime-переключатель готов; старт={'ON' if traffic.enabled else 'OFF'}", flush=True)

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
