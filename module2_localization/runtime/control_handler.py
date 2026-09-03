import asyncio
import gc
import json

from .. import config
from ..services.localizer_factory import create_runtime_localizer


def make_control_handler(
    nav, traffic=None, full_recovery=None, direction=None, route_profile=None, frame_guard=None, navlog=None
):
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
                create_runtime_localizer,
                map_name,
                back_facing,
                full_recovery,
                min_shard_index,
                max_shard_index,
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
            navlog.emit(
                "control_received",
                command=c,
                route=nav.route,
                mode=nav.mode,
                front_paused=nav.pf.paused,
                rear_paused=nav.pr.paused,
            )
        if cmd == "pause":
            nav.pause()
        elif cmd == "resume":
            nav.resume()
        elif cmd == "reset":
            nav.pf.reset()
            nav.pr.reset()
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
        elif cmd in ("set_frame_guard", "set_frame_hash"):
            if frame_guard is None:
                print("[control] контроль потока кадров недоступен", flush=True)
                return
            enabled = bool(c.get("enabled"))
            frame_guard.set_enabled(enabled)
            print(f"[control] контроль потока кадров -> {'ON' if enabled else 'OFF'}", flush=True)
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
            print(
                f"[control] манёвры начала/конца 2-1 -> {'ON' if route_profile.terminal_maneuvers else 'OFF'}",
                flush=True,
            )
            return
        else:
            print(f"[control] неизвестная команда: {cmd}", flush=True)
            return
        print(f"[control] {cmd}", flush=True)

    return on_control
