import asyncio
import atexit
import json
import signal

from .. import config
from ..core.camera_navigator import CameraNavigator
from ..services.localizer_factory import create_runtime_localizer
from .frame_sources import CameraSource, ShmSource, VideoFileSource
from .frame_timeout import FrameTimeoutGuard
from .motion_controllers import DirectionController, RouteProfileController
from .nats_gateway import NatsClient
from .navigation_logger import NavigationLog, log_runtime_start
from .navigation_loop import worker
from .traffic_controller import TrafficController


def _shm(sock):
    return ShmSource(sock, config.CAM_WIDTH, config.CAM_HEIGHT, config.CAM_FPS)


async def run(args) -> int:
    navlog = NavigationLog()
    atexit.register(navlog.close)
    log_runtime_start(navlog, args, config)
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
        front_loc = create_runtime_localizer(
            front_map, config.FRONT_CAM_BACK, full_recovery=recovery, event_sink=navlog.emit
        )
    if want_rear:
        rear_loc = create_runtime_localizer(
            rear_map, config.REAR_CAM_BACK, full_recovery=recovery, event_sink=navlog.emit
        )

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

    nav = CameraNavigator(
        front_loc,
        rear_loc,
        config,
        front_map=front_map,
        rear_map=rear_map,
        route=route,
        front_route=route if front_loc else None,
        rear_route=route if rear_loc else None,
    )
    if args.mode:
        nav.set_mode(args.mode)
    elif front_loc is not None and rear_loc is None:
        nav.set_mode("front")
    elif rear_loc is not None and front_loc is None:
        nav.set_mode("rear")
    if route is not None:
        nav.reset_shard()
    print(
        f"[nav] маршрут {route or 'не выбран'}, режим {nav.mode}"
        + ("  (dual доступен)" if front_loc and rear_map else ""),
        flush=True,
    )

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
        from ..core import route_follower as route_mod

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
    frame_guard = FrameTimeoutGuard(
        enabled=getattr(config, "FRAME_TIMEOUT_CHECK_ENABLED", True),
        timeout_s=getattr(config, "FRAME_TIMEOUT_S", 0.5),
        recovery_frames=getattr(config, "FRAME_TIMEOUT_RECOVERY_FRAMES", 3),
    )
    try:
        await worker(
            front_src,
            rear_src,
            nc,
            config.NATS_TOPIC,
            nav,
            stop_evt,
            traffic,
            recovery,
            direction,
            route_profile,
            frame_guard,
            navlog,
        )
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
