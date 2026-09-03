import logging
import time
from typing import Callable

import numpy as np


def emit_event(sink: Callable[..., None] | None, component: str, event: str, logger: logging.Logger, **fields) -> None:
    if sink is None:
        return
    try:
        sink(event, component=component, **fields)
    except Exception:
        logger.exception("ошибка записи диагностического события %s", event)


def localization_result_summary(result: dict | None) -> dict:
    if not result:
        return {}
    keys = (
        "ok",
        "node",
        "target_node",
        "inliers",
        "pairs",
        "n_pairs",
        "reason",
        "move_type",
        "bearing_deg",
        "reproj_error",
        "error",
        "frame_ms",
        "extract_ms",
        "match_ms",
        "t_ext",
        "t_match",
        "t_pnp",
        "dist_to_route",
        "offset",
        "dist_to_route_m",
        "offset_m",
    )
    summary = {key: result.get(key) for key in keys if key in result}
    if "diagnostics" in result:
        summary["diagnostics"] = result["diagnostics"]
    for key in ("C", "fwd"):
        value = result.get(key)
        if value is not None:
            summary[key] = value
    return summary


def command_filter_diagnostics(command_filter, now=None) -> dict:
    now = time.monotonic() if now is None else now
    return {
        "paused": bool(command_filter.paused),
        "accepted": bool(command_filter.accepted),
        "stopped": bool(command_filter.stopped),
        "stop_hits": int(command_filter.stop_hits),
        "moved_once": bool(command_filter.moved_once),
        "last_node": None if command_filter.last_node is None else int(command_filter.last_node),
        "last_fix_age_s": None
        if command_filter.last_fix_t is None
        else float(max(0.0, now - command_filter.last_fix_t)),
        "jump_rejection": command_filter.jump,
        "consecutive_jump_rejects": int(command_filter.rejects),
        "node_history": [int(value) for value in command_filter.nhist],
        "position_history": [np.asarray(value).tolist() for value in command_filter.chist],
        "last_steering": {
            "move_type": command_filter.last_cmd[0],
            "bearing_deg": float(command_filter.last_cmd[1]),
        },
    }


def camera_navigator_diagnostics(navigator) -> dict:
    return {
        "route": navigator.route,
        "loading_route": navigator.loading_route,
        "route_started": bool(navigator.route_started),
        "mode": navigator.mode,
        "active_camera": navigator.active,
        "front_map": navigator.front_map,
        "rear_map": navigator.rear_map,
        "front_node_offset": int(navigator.front_node_offset),
        "rear_node_offset": int(navigator.rear_node_offset),
        "front_lost_count": int(navigator.front_lost),
        "front_good_count": int(navigator.front_good),
        "front_pilot": navigator.pf.diagnostics(),
        "rear_pilot": navigator.pr.diagnostics(),
    }
