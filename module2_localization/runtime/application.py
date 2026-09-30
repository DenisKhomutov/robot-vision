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
    map_name = route_spec.get("map")

    if args.video:
        source = VideoFileSource(args.video, step=args.source_step, loop=args.video_loop)
    elif args.shm:
        source = _shm(args.shm)
    else:
        source = CameraSource(args.camera or config.CAMERA)

    recovery = False if args.no_recovery else None
    localizer = None
    if map_name is not None:
        localizer = create_runtime_localizer(
            map_name,
            getattr(config, "CAMERA_BACK", False),
            full_recovery=recovery,
            event_sink=navlog.emit,
        )

    nav = CameraNavigator(localizer, config, map_name=map_name, route=route)
    if route is not None:
        nav.reset_shard()
    print(f"[nav] маршрут {route or 'не выбран'}, карта {map_name or 'не загружена'}", flush=True)

    traffic = TrafficController(config)
    if getattr(config, "TRAFFIC_LIGHT_ENABLED", True):
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

    if nc is not None:
        from ..core import route_follower as route_mod

        async def on_speed(msg):
            try:
                data = json.loads(msg.data.decode())
                pwm = data.get("speed_pwm")
            except (json.JSONDecodeError, AttributeError):
                return
            if pwm is not None:
                route_mod.set_speed_signal(
                    float(pwm),
                    neutral=getattr(config, "SPEED_NEUTRAL_US", 1500.0),
                    deadzone=getattr(config, "SPEED_DEADZONE_US", 30.0),
                )
                navlog.emit("speed_signal", pulse_us=float(pwm))

        await nc.subscribe(getattr(config, "NATS_SPEED_TOPIC", "gateway.robot.speed"), on_speed)

    stop_evt = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_evt.set)

    source.start()
    frame_guard = FrameTimeoutGuard(
        enabled=getattr(config, "FRAME_TIMEOUT_CHECK_ENABLED", True),
        timeout_s=getattr(config, "FRAME_TIMEOUT_S", 0.5),
        recovery_frames=getattr(config, "FRAME_TIMEOUT_RECOVERY_FRAMES", 3),
    )
    try:
        await worker(
            source,
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
        source.stop()
        if nc:
            await nc.close()
        navlog.close()
        print("Cameras Brain остановлен", flush=True)
    return 0
