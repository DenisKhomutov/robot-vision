import asyncio
import json

from .. import config
from .control_handler import make_control_handler
from .navigation_logger import frame_metrics


async def worker(
    front_src,
    rear_src,
    nc,
    topic: str,
    nav,
    stop_evt=None,
    traffic=None,
    full_recovery=None,
    direction=None,
    route_profile=None,
    frame_guard=None,
    navlog=None,
) -> None:
    if nc:
        await nc.subscribe(
            config.NATS_CONTROL_TOPIC,
            make_control_handler(nav, traffic, full_recovery, direction, route_profile, frame_guard, navlog),
        )
    last_active_token = None
    last_timeout_publish = 0.0
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
                navlog.emit("video_cycle_reset", video_cycle=video_cycle, route=nav.route, mode=nav.mode)
        rear_stamp, rframe = rear_src.latest()
        if front_src:
            front_stamp, fframe = front_src.latest()
        else:
            front_stamp, fframe = None, None
        if nav.mode == "front":
            active_token = ("front", front_stamp) if fframe is not None else None
        elif nav.mode == "rear":
            active_token = ("rear", rear_stamp) if rframe is not None else None
        else:
            candidate = ("dual", front_stamp, rear_stamp) if fframe is not None and rframe is not None else None
            if (
                candidate is not None
                and last_active_token is not None
                and last_active_token[0] == "dual"
                and (candidate[1] == last_active_token[1] or candidate[2] == last_active_token[2])
            ):
                active_token = last_active_token
            else:
                active_token = candidate
        now = asyncio.get_running_loop().time()
        frame_state = frame_guard.observe(active_token, now) if frame_guard else None
        fresh_active_frame = active_token is not None and active_token != last_active_token
        if fresh_active_frame:
            last_active_token = active_token
            if navlog is not None:
                navlog.emit(
                    "frame_received",
                    source_stamp=active_token,
                    front=frame_metrics(fframe),
                    rear=frame_metrics(rframe),
                    route=nav.route,
                    mode=nav.mode,
                    frame_guard=frame_state,
                )
            if frame_state and frame_state["timed_out"]:
                cmd = {
                    "move_type": "stop",
                    "deg": 0.0,
                    "reason": "camera_timeout",
                    "route_loaded": nav.route is not None,
                    "route": nav.route,
                    "map": nav.front_map if nav.mode == "front" else nav.rear_map,
                    "mode": nav.mode,
                    "cam": nav.mode,
                    "paused": bool(nav.pf.paused and nav.pr.paused),
                }
            else:
                cmd = nav.step(fframe, rframe)
            if frame_state is not None:
                cmd["frame_guard_enabled"] = bool(frame_state["enabled"])
                cmd["frame_timed_out"] = bool(frame_state["timed_out"])
            if navlog is not None:
                navlog.emit("command_after_navigation", source_stamp=active_token, command=dict(cmd))
                navlog.emit("navigation_state", source_stamp=active_token, state=nav.diagnostics())
            if route_profile is not None:
                route_profile.process(cmd)
                if navlog is not None:
                    navlog.emit("command_after_route_profile", source_stamp=active_token, command=dict(cmd))
            if direction is not None:
                direction.process(cmd)
                if navlog is not None:
                    navlog.emit("command_after_direction", source_stamp=active_token, command=dict(cmd))

            if traffic and fframe is not None:
                tl = traffic.process(cmd, fframe)
                if navlog is not None:
                    navlog.emit("command_after_traffic", source_stamp=active_token, command=dict(cmd), detection=tl)
                if tl is not None:
                    print(f"[TL] {json.dumps(tl, ensure_ascii=False)} state={traffic.state}", flush=True)
            payload = json.dumps(cmd, ensure_ascii=False).encode()
            print(payload.decode(), flush=True)
            if nc:
                await nc.publish(topic, payload)
            if navlog is not None:
                navlog.emit(
                    "command_published", source_stamp=active_token, topic=topic, nats=nc is not None, command=dict(cmd)
                )
            await asyncio.sleep(0)
        elif rear_src.done or (front_src and front_src.done):
            break
        elif frame_state and frame_state["timed_out"] and now - last_timeout_publish >= 0.25:
            last_timeout_publish = now
            cmd = {
                "move_type": "stop",
                "deg": 0.0,
                "reason": "camera_timeout",
                "route_loaded": nav.route is not None,
                "route": nav.route,
                "map": nav.front_map if nav.mode == "front" else nav.rear_map,
                "mode": nav.mode,
                "cam": nav.mode,
                "paused": bool(nav.pf.paused and nav.pr.paused),
                "frame_guard_enabled": True,
                "frame_timed_out": True,
            }
            if route_profile is not None:
                route_profile.process(cmd)
            if direction is not None:
                direction.process(cmd)
            if traffic is not None:
                cmd["traffic_enabled"] = traffic.enabled
                cmd["traffic_loading"] = traffic.loading
                cmd["traffic_in_zone"] = False
                cmd["traffic_state"] = traffic.state
            payload = json.dumps(cmd, ensure_ascii=False).encode()
            print(payload.decode(), flush=True)
            if nc:
                await nc.publish(topic, payload)
            if navlog is not None:
                navlog.emit("camera_timeout", source_stamp=active_token, frame_guard=frame_state, command=dict(cmd))
                navlog.emit(
                    "command_published", source_stamp=active_token, topic=topic, nats=nc is not None, command=dict(cmd)
                )
            await asyncio.sleep(0.003)
        else:
            await asyncio.sleep(0.003)
