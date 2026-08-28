import argparse
import asyncio
import atexit
import gc
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
from .navigation_log import NavigationLog, frame_metrics
from .services.localization_service import build_runtime_localizer


class ShmSource:
    def __init__(self, socket_path, width, height, fps):
        gst = shutil.which("gst-launch-1.0")
        if gst is None:
            raise RuntimeError("gst-launch-1.0 не найден")
        if not os.path.exists(socket_path):
            raise RuntimeError(f"нет сокета {socket_path}: fan-out")
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

    def __init__(self, path: str, step: int = 2, fps: float = 30.0, loop: bool = True):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"не открывается видео: {path}")
        self.step = step
        self.dt = 1.0 / fps
        self.loop = bool(loop)
        self._latest = None
        self._stamp = 0
        self._stop = False
        self.cycle = 0
        self.done = False

    def start(self):
        asyncio.ensure_future(self._run())

    async def _run(self):
        idx = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                if not self.loop:
                    self.done = True
                    self.cap.release()
                    return
                if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.path)
                    if not self.cap.isOpened():
                        self.done = True
                        return
                self.cycle += 1
                idx = 0
                await asyncio.sleep(0)
                continue
            if idx % self.step == 0:
                self._latest = frame
                self._stamp += 1
            idx += 1
            await asyncio.sleep(self.dt)

    def latest(self):
        return self._stamp, self._latest

    def stop(self):
        self._stop = True
        self.cap.release()


def make_control_handler(nav, traffic=None, full_recovery=None, direction=None, route_profile=None,
                         navlog=None):
    async def _ensure_camera(camera, map_name, route):
        """Догрузить карту камеры под конкретный маршрут, если ещё не та."""
        min_shard_index = None
        max_shard_index = None
        if route == "2-1" and route_profile is not None and not route_profile.terminal_maneuvers:
            if map_name == "2-1/front_shard/01_of_25":
                map_name = "2-1/front_shard/02_of_25"
            min_shard_index = getattr(config, "ROUTE_21_NO_MANEUVERS_MIN_SHARD_INDEX", 1)
            max_shard_index = getattr(config, "ROUTE_21_NO_MANEUVERS_MAX_SHARD_INDEX", 23)
        if nav.has_camera(camera) and getattr(nav, f"{camera}_map", None) == map_name:
            return True
        print(f"[control] {camera}: загрузка {map_name} (маршрут {route})...", flush=True)
        try:
            back_facing = config.FRONT_CAM_BACK if camera == "front" else config.REAR_CAM_BACK
            localizer = await asyncio.to_thread(
                build_runtime_localizer, map_name, back_facing, full_recovery,
                min_shard_index, max_shard_index,
                navlog.emit if navlog is not None else None,
            )
            nav.set_localizer(camera, localizer, map_name, route=route)
        except Exception as exc:
            print(f"[control] не удалось загрузить карту {camera}: {exc}", flush=True)
            return False
        return True

    async def on_control(msg):
        try:
            c = json.loads(msg.data.decode())
        except json.JSONDecodeError:
            return
        cmd = c.get("cmd")
        if navlog is not None:
            navlog.emit("control_received", command=c, route=nav.route, mode=nav.mode,
                        front_paused=nav.pf.paused, rear_paused=nav.pr.paused)
        if cmd == "pause":
            nav.pause()
        elif cmd == "resume":
            nav.resume()
        elif cmd == "reset":
            nav.pf.reset(); nav.pr.reset()
            if route_profile:
                route_profile.reset()
            if traffic:
                traffic.reset()
        elif cmd == "reset_traffic":
            if traffic:
                traffic.reset()
            print("[control] сброс светофора: WAIT_RED, детектор снова активен", flush=True)
            return
        elif cmd == "reset_shard":
            nav.reset_shard()
            print("[control] сброс шарда: релокализация по полной карте", flush=True)
            return
        elif cmd == "set_mode":
            if not (nav.pf.paused and nav.pr.paused):
                print("[control] set_mode: сначала ПАУЗА, потом смена режима", flush=True)
                return
            mode = c.get("mode")
            needed = {"front": ("front",), "rear": ("rear",), "dual": ("front", "rear")}.get(mode)
            if needed is None:
                print(f"[control] неизвестный режим: {mode}", flush=True)
                return
            spec = getattr(config, "ROUTES", {}).get(nav.route, {})
            for camera in needed:
                map_name = spec.get(f"{camera}_map")
                if not map_name:
                    print(f"[control] у маршрута {nav.route} нет карты для {camera}", flush=True)
                    return



                if not await _ensure_camera(camera, map_name, nav.route):
                    return
            if not nav.set_mode(mode):
                print(f"[control] режим {mode} недоступен", flush=True)
                return
            print(f"[control] режим -> {nav.mode}", flush=True)
            return
        elif cmd == "set_route":
            if not (nav.pf.paused and nav.pr.paused):
                print("[control] set_route: сначала ПАУЗА, потом смена маршрута", flush=True)
                return
            route = c.get("route")
            spec = getattr(config, "ROUTES", {}).get(route)
            if spec is None:
                print(f"[control] неизвестный маршрут: {route}", flush=True)
                return
            if nav.route != route:
                nav.clear_route()
            nav.loading_route = route
            for camera in ("front", "rear"):
                map_name = spec.get(f"{camera}_map")
                if not map_name:
                    continue
                if not await _ensure_camera(camera, map_name, route):
                    nav.loading_route = None
                    return
            if not nav.set_route(route):
                nav.loading_route = None
                print(f"[control] маршрут {route} недоступен", flush=True)
                return
            if traffic:
                traffic.reset()
            if route_profile:
                route_profile.reset()
            print(f"[control] маршрут -> {route}", flush=True)
            return
        elif cmd == "clear_route":
            if not (nav.pf.paused and nav.pr.paused):
                print("[control] clear_route: сначала ПАУЗА", flush=True)
                return
            nav.clear_route()
            if traffic:
                traffic.reset()
            if route_profile:
                route_profile.reset()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("[control] стоянка: маршрут и карты выгружены", flush=True)
            return
        elif cmd == "set_traffic":
            if traffic is None:
                print("[control] светофорная ветка недоступна: нет передней камеры", flush=True)
                return
            enabled = bool(c.get("enabled"))
            await traffic.set_enabled(enabled)
            print(f"[control] детекция светофора -> {'ON' if enabled else 'OFF'}", flush=True)
            return
        elif cmd == "set_direction":
            if direction is None:
                print("[control] направление недоступно", flush=True)
                return
            enabled = bool(c.get("enabled"))
            direction.set_enabled(enabled)
            print(f"[control] расчёт направления -> {'ON' if enabled else 'OFF'}", flush=True)
            return
        elif cmd == "set_terminal_maneuvers":
            if route_profile is None:
                print("[control] профиль конечных манёвров недоступен", flush=True)
                return
            if not (nav.pf.paused and nav.pr.paused):
                print("[control] set_terminal_maneuvers: сначала ПАУЗА", flush=True)
                return
            if nav.route is not None:
                print("[control] set_terminal_maneuvers: сначала выгрузите маршрут", flush=True)
                return
            route_profile.set_terminal_maneuvers(bool(c.get("enabled")))
            print(f"[control] манёвры начала/конца 2-1 -> {'ON' if route_profile.terminal_maneuvers else 'OFF'}", flush=True)
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
        if node is not None:
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
            return None
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

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = False
        self.loading = False
        self.branch = None

    async def preload(self):
        """Загрузить детектор+классификатор в память заранее, не включая
        детекцию — чтобы set_enabled(True) потом срабатывал мгновенно."""
        if self.branch is not None:
            return
        self.loading = True
        try:
            self.branch = await asyncio.to_thread(TrafficBranch, self.cfg)
        finally:
            self.loading = False

    async def set_enabled(self, enabled):
        if not enabled:
            self.enabled = False
            if self.branch:
                self.branch.reset()
            return
        if self.branch is None:
            await self.preload()
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
            cmd["reason"] = "traffic_light_stop"
        return result


class DirectionController:
    """Runtime on/off (как светофор) для forward/backward: статичные зоны из конфига,
    без модели и пересчёта на лету — просто попадание global_node в диапазон."""

    def __init__(self, cfg, enabled=False):
        self.zones = getattr(cfg, "BACKWARD_ZONES", {})
        self.left_block_zones = getattr(cfg, "BACKWARD_LEFT_BLOCK_ZONES", {})
        self.steering_outlier_guards = getattr(cfg, "STEERING_OUTLIER_GUARDS", {})
        self.deadzone = getattr(cfg, "DEADZONE_DEG", 4.0)
        self.enabled = bool(enabled)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def process(self, cmd):
        cmd["direction_enabled"] = self.enabled
        if not self.enabled:
            return
        ranges = self.zones.get(cmd.get("map"))
        node = cmd.get("global_node", cmd.get("node"))
        backward = bool(ranges and node is not None and any(a <= node <= b for a, b in ranges))
        cmd["direction"] = "backward" if backward else "forward"







        if backward and cmd.get("deg") is not None and cmd.get("move_type") not in ("stop", "lost"):
            deg = ((cmd["deg"] + 180.0 + 180.0) % 360.0) - 180.0
            cmd["deg"] = deg
            cmd["move_type"] = "straight" if abs(deg) < self.deadzone else ("right" if deg > 0 else "left")
        blocked = self.left_block_zones.get(cmd.get("map"))
        block_left = bool(blocked and node is not None and any(a <= node <= b for a, b in blocked))
        if backward and block_left and cmd.get("move_type") == "left":
            cmd["move_type"] = "right"
            cmd["deg"] = abs(float(cmd.get("deg") or self.deadzone))
        guard = self.steering_outlier_guards.get(cmd.get("map"))
        if guard and node is not None and cmd.get("deg") is not None:
            (start, stop), max_deg = guard
            if start <= node <= stop and abs(float(cmd["deg"])) > float(max_deg):
                cmd["deg"] = float(max_deg) if float(cmd["deg"]) > 0 else -float(max_deg)
                cmd["move_type"] = "right" if cmd["deg"] > 0 else "left"


class RouteProfileController:
    def __init__(self, cfg, terminal_maneuvers=None):
        if terminal_maneuvers is None:
            terminal_maneuvers = getattr(cfg, "ROUTE_21_TERMINAL_MANEUVERS_DEFAULT", True)
        self.cfg = cfg
        self.terminal_maneuvers = bool(terminal_maneuvers)
        self.completed = False

    def set_terminal_maneuvers(self, enabled):
        self.terminal_maneuvers = bool(enabled)
        self.reset()

    def reset(self):
        self.completed = False

    def process(self, cmd):
        cmd["terminal_maneuvers"] = self.terminal_maneuvers
        if self.terminal_maneuvers or cmd.get("route") != "2-1":
            return
        node = cmd.get("global_node")
        if node is None:
            if self.completed:
                cmd["move_type"] = "stop"
                cmd["deg"] = 0.0
                cmd["reason"] = "route_complete"
            return
        min_node = int(getattr(self.cfg, "ROUTE_21_NO_MANEUVERS_MIN_NODE", 27))
        stop_node = int(getattr(self.cfg, "ROUTE_21_NO_MANEUVERS_STOP_NODE", 1595))
        if int(node) < min_node:
            cmd["move_type"] = "stop"
            cmd["deg"] = 0.0
            cmd["reason"] = "outside_route_segment"
            return
        if int(node) >= stop_node:
            self.completed = True
        if self.completed:
            cmd["move_type"] = "stop"
            cmd["deg"] = 0.0
            cmd["reason"] = "route_complete"


async def worker(front_src, rear_src, nc, topic: str, nav, stop_evt=None, traffic=None,
                  full_recovery=None, direction=None, route_profile=None, navlog=None) -> None:
    if nc:
        await nc.subscribe(config.NATS_CONTROL_TOPIC, make_control_handler(
            nav, traffic, full_recovery, direction, route_profile, navlog))
    last_stamp = -1
    last_video_cycle = getattr(rear_src, "cycle", None)
    while not (stop_evt and stop_evt.is_set()):
        video_cycle = getattr(rear_src, "cycle", None)
        if video_cycle is not None and video_cycle != last_video_cycle:
            last_video_cycle = video_cycle
            nav.pf.reset()
            nav.pr.reset()
            nav.reset_shard()
            if traffic is not None:
                traffic.reset()
            if route_profile is not None:
                route_profile.reset()
            print(f"[video] новый цикл {video_cycle}: состояние маршрута и светофора сброшено", flush=True)
            if navlog is not None:
                navlog.emit("video_cycle_reset", video_cycle=video_cycle, route=nav.route,
                            mode=nav.mode)
        stamp, rframe = rear_src.latest()
        fframe = front_src.latest()[1] if front_src else None
        if rframe is not None and stamp != last_stamp:
            last_stamp = stamp
            if navlog is not None:
                navlog.emit("frame_received", source_stamp=stamp,
                            front=frame_metrics(fframe), rear=frame_metrics(rframe),
                            route=nav.route, mode=nav.mode)
            cmd = nav.step(fframe, rframe)
            if navlog is not None:
                navlog.emit("command_after_navigation", source_stamp=stamp, command=dict(cmd))
            if route_profile is not None:
                route_profile.process(cmd)
                if navlog is not None:
                    navlog.emit("command_after_route_profile", source_stamp=stamp,
                                command=dict(cmd))
            if direction is not None:
                direction.process(cmd)
                if navlog is not None:
                    navlog.emit("command_after_direction", source_stamp=stamp,
                                command=dict(cmd))



            if traffic and fframe is not None:
                tl = traffic.process(cmd, fframe)
                if navlog is not None:
                    navlog.emit("command_after_traffic", source_stamp=stamp,
                                command=dict(cmd), detection=tl)
                if tl is not None:
                    print(f"[TL] {json.dumps(tl, ensure_ascii=False)} state={traffic.state}", flush=True)
            payload = json.dumps(cmd, ensure_ascii=False).encode()
            print(payload.decode(), flush=True)
            if nc:
                await nc.publish(topic, payload)
            if navlog is not None:
                navlog.emit("command_published", source_stamp=stamp, topic=topic,
                            nats=nc is not None, command=dict(cmd))
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
    ap.add_argument("--video-loop", action=argparse.BooleanOptionalAction, default=True,
                    help="зациклить файловые видеопотоки (по умолчанию включено)")
    ap.add_argument("--mode", default=None, choices=["rear", "dual", "front"], help="стартовый режим (по умолч. из маршрута)")
    ap.add_argument("--route", default=None, choices=list(config.ROUTES),
                    help=f"стартовый маршрут (по умолч. config.DEFAULT_ROUTE={config.DEFAULT_ROUTE!r})")
    ap.add_argument("--no-nats", action="store_true", help="только терминал, без NATS")
    ap.add_argument("--no-recovery", action="store_true",
                    help="выключить ПОСТОЯННЫЕ попытки recovery при LOST (замер чистой "
                         "скорости шарда); полная карта всё равно грузится и используется "
                         "разово для выбора шарда на старте/смене маршрута/reset_shard")
    args = ap.parse_args()
    navlog = NavigationLog()
    atexit.register(navlog.close)
    navlog.emit("arguments", arguments=vars(args))
    print(f"[log] {navlog.path}", flush=True)

    route = args.route if args.route is not None else config.DEFAULT_ROUTE
    route_spec = config.ROUTES[route] if route is not None else {}


    dual = args.dual or config.NAV_MODE in ("dual", "front")


    rear_video = args.video_rear or args.video
    if rear_video:
        rear_src = VideoFileSource(rear_video, step=args.source_step, loop=args.video_loop)
    elif args.shm:
        rear_src = _shm(args.shm)
    elif dual:
        rear_src = _shm(config.REAR_SHM_SOCKET)
    else:
        rear_src = CameraSource(args.camera or config.CAMERA)


    recovery = False if args.no_recovery else None
    front_loc = None
    rear_loc = None
    front_map = route_spec.get("front_map")
    rear_map = route_spec.get("rear_map")



    default_mode = route_spec.get("camera", "front")
    want_front = front_map is not None and (args.mode in ("dual", "front") if args.mode else default_mode != "rear")
    want_rear = rear_map is not None and (args.mode in ("dual", "rear") if args.mode else default_mode == "rear")
    if want_front:
        front_loc = build_runtime_localizer(
            front_map, config.FRONT_CAM_BACK, full_recovery=recovery, event_sink=navlog.emit)
    if want_rear:
        rear_loc = build_runtime_localizer(
            rear_map, config.REAR_CAM_BACK, full_recovery=recovery, event_sink=navlog.emit)


    front_src = None
    if dual or getattr(config, "TRAFFIC_LIGHT_ENABLED", False):
        try:
            if args.video_front:
                front_src = VideoFileSource(args.video_front, step=args.source_step, loop=args.video_loop)
            else:
                front_src = _shm(config.FRONT_SHM_SOCKET)
        except Exception as e:
            print(f"[перёд] нет передней камеры ({e})", flush=True)
            if dual:
                return 1

    nav = DualNav(front_loc, rear_loc, config, front_map=front_map, rear_map=rear_map,
                  route=route, front_route=route if front_loc else None,
                  rear_route=route if rear_loc else None)
    if args.mode:
        nav.set_mode(args.mode)
    elif front_loc is not None and rear_loc is None:
        nav.set_mode("front")
    elif rear_loc is not None and front_loc is None:
        nav.set_mode("rear")
    if route is not None:
        nav.reset_shard()
    print(f"[nav] маршрут {route or 'не выбран'}, режим {nav.mode}" + ("  (dual доступен)" if front_loc and rear_map else ""),
          flush=True)

    traffic = None
    if front_src is None:
        if getattr(config, "TRAFFIC_LIGHT_ENABLED", False):
            print("[TL] светофор включён, но нет передней камеры — пропуск", flush=True)
    else:
        traffic = TrafficController(config)
        if getattr(config, "TRAFFIC_LIGHT_ENABLED", False):
            await traffic.set_enabled(True)
        else:


            await traffic.preload()
        print(f"[TL] runtime-переключатель готов; старт={'ON' if traffic.enabled else 'OFF'}", flush=True)

    direction = DirectionController(config, enabled=getattr(config, "DIRECTION_ENABLED", False))
    print(f"[dir] runtime-переключатель готов; старт={'ON' if direction.enabled else 'OFF'}", flush=True)
    route_profile = RouteProfileController(config)
    print(f"[route-profile] манёвры начала/конца 2-1={'ON' if route_profile.terminal_maneuvers else 'OFF'}", flush=True)

    nc = None
    if not args.no_nats:
        try:
            nc = NatsClient(config.NATS_URL)
            await nc.connect()
        except Exception as e:
            print(f"[NATS] недоступен ({e}); печатаю только в терминал", flush=True)
            nc = None

    if nc is not None:
        from .core import route as route_mod

        async def on_speed(msg):
            try:
                data = json.loads(msg.data.decode())
                pwm = data.get("speed_pwm")
            except (json.JSONDecodeError, AttributeError):
                return
            if pwm is not None:
                route_mod.set_speed_pwm(float(pwm))

        await nc.subscribe(getattr(config, "NATS_SPEED_TOPIC", "ai.nats_speed_topic"), on_speed)

    stop_evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_evt.set)

    if front_src:
        front_src.start()
    rear_src.start()
    try:
        await worker(front_src, rear_src, nc, config.NATS_TOPIC, nav, stop_evt, traffic,
                     recovery, direction, route_profile, navlog)
    except Exception as exc:
        navlog.emit("service_failed", error=repr(exc))
        raise
    finally:
        if front_src:
            front_src.stop()
        rear_src.stop()
        if nc:
            await nc.close()
        navlog.close()
        print("Cameras Brain остановлен", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
