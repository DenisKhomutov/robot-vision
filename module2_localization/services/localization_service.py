import sys
from pathlib import Path
from typing import Any

from .. import config

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

_localizer = None


def build_localizer(map_name, back_facing):
    """Собрать локализатор для конкретной карты и направления камеры (тюнинги — из конфига)."""
    from localizer import AlikedLocalizer
    return AlikedLocalizer(
        map_name,
        kpts=config.QUERY_KPTS,
        det_threshold=config.QUERY_DET_THRESHOLD,
        nms_radius=config.QUERY_NMS_RADIUS,
        max_error=config.MAX_ERROR,
        steer=config.STEER_MODE,
        route_cam=config.ROUTE_CAM,
        route_nodes=config.ROUTE_NODES,
        back_facing=back_facing,
        match_ratio=config.MATCH_RATIO,
        match_topk=config.MATCH_TOPK,
        focal_fallback=config.FOCAL_FALLBACK,
        min_pairs=config.MIN_PAIRS,
        lookahead=config.LOOKAHEAD_NODES,
        lookahead_min=config.LOOKAHEAD_MIN,
        lookahead_adapt=config.LOOKAHEAD_ADAPT,
        deadzone=config.DEADZONE_DEG,
        stanley_k=config.STANLEY_K,
        heading_gate=config.HEADING_GATE,
        stop_end_nodes=config.STOP_END_NODES,
        lag_s=config.NAV_LAG_S,
        lag_adaptive=config.NAV_LAG_ADAPTIVE,
        lead_max=config.NAV_LEAD_MAX,
        lead_smooth=config.NAV_LEAD_SMOOTH,
        win_nodes=getattr(config, "NAV_WIN_NODES", 0),
    )


def get_localizer():
    global _localizer
    if _localizer is None:
        _localizer = build_localizer(config.DEFAULT_MAP, config.CAMERA_BACK)
    return _localizer


def localize(frame) -> dict[str, Any]:
    r = get_localizer().locate(frame)
    if not r or not r.get("ok") or r["inliers"] < config.MIN_INLIERS:
        return {"move_type": "lost", "reason": (r or {}).get("reason", "no fix")}
    return {
        "move_type": r["move_type"],
        "bearing_deg": r["bearing_deg"],
        "offset_m": r.get("offset_m"),
        "dist_to_route_m": r.get("dist_to_route_m"),
        "node": r["node"],
        "inliers": r["inliers"],
    }
