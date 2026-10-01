import asyncio
import json

from .. import config
from .control_handler import make_control_handler
from .navigation_logger import frame_metrics


async def worker(
    source,
    nc,
    topic: str,
    nav,
    stop_evt=None,
    traffic=None,
    full_recovery=None,
    direction=None,
    frame_guard=None,
    navlog=None,
) -> None:
    if nc:
        await nc.subscribe(
            config.NATS_CONTROL_TOPIC,
            make_control_handler(nav, traffic, full_recovery, direction, frame_guard, navlog),
        )
    last_token = None
    last_timeout_publish = 0.0
    last_video_cycle = getattr(source, "cycle", None)
    while not (stop_evt and stop_evt.is_set()):
        video_cycle = getattr(source, "cycle", None)
        if video_cycle is not None and video_cycle != last_video_cycle:
            last_video_cycle = video_cycle
            nav.reset()
            nav.reset_shard()
            if traffic is not None:
                traffic.reset()
            print(f"[video] новый цикл {video_cycle}: состояние маршрута и светофора сброшено", flush=True)
            if navlog is not None:
                navlog.emit("video_cycle_reset", video_cycle=video_cycle, route=nav.route)

        stamp, frame = source.latest()
        token = stamp if frame is not None else None
        now = asyncio.get_running_loop().time()
        frame_state = frame_guard.observe(token, now) if frame_guard else None
        fresh = token is not None and token != last_token

        if fresh:
            last_token = token
            if navlog is not None:
                navlog.emit(
                    "frame_received",
                    source_stamp=token,
                    frame=frame_metrics(frame),
                    route=nav.route,
                    frame_guard=frame_state,
                )
            if frame_state and frame_state["timed_out"]:
                cmd = _timeout_command(nav)
            else:
                cmd = nav.step(frame)
            if frame_state is not None:
                cmd["frame_guard_enabled"] = bool(frame_state["enabled"])
                cmd["frame_timed_out"] = bool(frame_state["timed_out"])
            if navlog is not None:
                navlog.emit("command_after_navigation", source_stamp=token, command=dict(cmd))
                navlog.emit("navigation_state", source_stamp=token, state=nav.diagnostics())
            if direction is not None:
                direction.process(cmd)
                if navlog is not None:
                    navlog.emit("command_after_direction", source_stamp=token, command=dict(cmd))
            if traffic is not None:
                detection = traffic.process(cmd, frame)
                if navlog is not None:
                    navlog.emit("command_after_traffic", source_stamp=token, command=dict(cmd), detection=detection)
                if detection is not None:
                    print(f"[TL] {json.dumps(detection, ensure_ascii=False)} state={traffic.state}", flush=True)
            await _publish(cmd, token, nc, topic, navlog)
        elif source.done:
            break
        elif frame_state and frame_state["timed_out"] and now - last_timeout_publish >= 0.25:
            last_timeout_publish = now
            cmd = _timeout_command(nav)
            cmd["frame_guard_enabled"] = True
            cmd["frame_timed_out"] = True
            if direction is not None:
                direction.process(cmd)
            if traffic is not None:
                cmd.update(
                    traffic_enabled=traffic.enabled,
                    traffic_loading=traffic.loading,
                    traffic_in_zone=False,
                    traffic_state=traffic.state,
                )
            if navlog is not None:
                navlog.emit("camera_timeout", source_stamp=token, frame_guard=frame_state, command=dict(cmd))
            await _publish(cmd, token, nc, topic, navlog)
        else:
            await asyncio.sleep(0.003)


def _timeout_command(nav):
    return {
        "move_type": "stop",
        "deg": 0.0,
        "reason": "camera_timeout",
        "route_loaded": nav.route is not None,
        "route": nav.route,
        "map": nav.map,
        "paused": bool(nav.pilot.paused),
    }


async def _publish(cmd, token, nc, topic, navlog):
    payload = json.dumps(cmd, ensure_ascii=False).encode()
    print(payload.decode(), flush=True)
    if nc:
        await nc.publish(topic, payload)
    if navlog is not None:
        navlog.emit("command_published", source_stamp=token, topic=topic, nats=nc is not None, command=dict(cmd))
    await asyncio.sleep(0)
