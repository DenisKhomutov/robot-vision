import asyncio
import gc
import json

from .. import config
from ..localization.factory import create_runtime_localizer


def make_control_handler(nav, traffic=None, full_recovery=None, direction=None, frame_guard=None, navlog=None):
    async def _ensure_map(map_name, route):
        if nav.localizer is not None and nav.map == map_name:
            return True
        print(f"[control] загрузка {map_name} (маршрут {route})...", flush=True)
        try:
            localizer = await asyncio.to_thread(
                create_runtime_localizer,
                map_name,
                getattr(config, "CAMERA_BACK", False),
                full_recovery=full_recovery,
                event_sink=navlog.emit if navlog is not None else None,
            )
            nav.set_localizer(localizer, map_name, route=route)
        except Exception as exc:
            print(f"[control] не удалось загрузить карту: {exc}", flush=True)
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
                paused=nav.pilot.paused,
            )
        if cmd == "pause":
            nav.pause()
        elif cmd == "resume":
            nav.resume()
        elif cmd == "reset":
            nav.reset()
            if traffic:
                traffic.reset()
        elif cmd == "reset_traffic":
            if traffic:
                traffic.reset()
            print("[control] сброс светофора: WAIT_RED, детектор снова активен", flush=True)
            return
        elif cmd == "reset_shard":
            nav.reset_shard()
            print("[control] сброс карты: полная релокализация", flush=True)
            return
        elif cmd == "set_route":
            if not nav.pilot.paused:
                print("[control] set_route: сначала ПАУЗА, потом смена маршрута", flush=True)
                return
            route = c.get("route")
            spec = getattr(config, "ROUTES", {}).get(route)
            if spec is None:
                print(f"[control] неизвестный маршрут: {route}", flush=True)
                return
            map_name = spec.get("map")
            if not map_name:
                print(f"[control] у маршрута {route} не задана map", flush=True)
                return
            if nav.route != route:
                nav.clear_route()
            nav.loading_route = route
            if not await _ensure_map(map_name, route):
                nav.loading_route = None
                return
            if not nav.set_route(route):
                nav.loading_route = None
                print(f"[control] маршрут {route} недоступен", flush=True)
                return
            if traffic:
                traffic.reset()
            print(f"[control] маршрут -> {route}", flush=True)
            return
        elif cmd == "clear_route":
            if not nav.pilot.paused:
                print("[control] clear_route: сначала ПАУЗА", flush=True)
                return
            nav.clear_route()
            if traffic:
                traffic.reset()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("[control] стоянка: маршрут и карта выгружены", flush=True)
            return
        elif cmd == "set_traffic":
            if traffic is None:
                print("[control] светофорная ветка недоступна", flush=True)
                return
            await traffic.set_enabled(bool(c.get("enabled")))
            print(f"[control] детекция светофора -> {'ON' if traffic.enabled else 'OFF'}", flush=True)
            return
        elif cmd == "set_direction":
            if direction is None:
                print("[control] направление недоступно", flush=True)
                return
            direction.set_enabled(bool(c.get("enabled")))
            print(f"[control] расчёт направления -> {'ON' if direction.enabled else 'OFF'}", flush=True)
            return
        elif cmd in ("set_frame_guard", "set_frame_hash"):
            if frame_guard is None:
                print("[control] контроль потока кадров недоступен", flush=True)
                return
            frame_guard.set_enabled(bool(c.get("enabled")))
            print(f"[control] контроль потока кадров -> {'ON' if frame_guard.enabled else 'OFF'}", flush=True)
            return
        else:
            print(f"[control] неизвестная команда: {cmd}", flush=True)
            return
        print(f"[control] {cmd}", flush=True)

    return on_control
