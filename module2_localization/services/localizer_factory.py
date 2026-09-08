import json

from .. import config
from ..core.aliked_localizer import ALIKEDLocalizer


def create_localizer(map_name, back_facing, bank_path=None):
    stop_end_nodes = config.STOP_END_NODES
    shard_metadata = config.MAPS_DIR / map_name / "shard.json"
    if shard_metadata.exists():
        shard = json.loads(shard_metadata.read_text())
        if int(shard["index"]) < int(shard["parts"]) - 1:
            stop_end_nodes = -1
    return ALIKEDLocalizer(
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
        lookahead_speed_div=getattr(config, "LOOKAHEAD_SPEED_DIV", None),
        lookahead_max=getattr(config, "LOOKAHEAD_MAX", None),
        deadzone=config.DEADZONE_DEG,
        stanley_k=config.STANLEY_K,
        heading_gate=config.HEADING_GATE,
        stop_end_nodes=stop_end_nodes,
        lag_s=config.NAV_LAG_S,
        lag_adaptive=config.NAV_LAG_ADAPTIVE,
        lead_max=config.NAV_LEAD_MAX,
        lead_smooth=config.NAV_LEAD_SMOOTH,
        win_nodes=getattr(config, "NAV_WIN_NODES", 0),
        bank_path=bank_path,
    )


def create_runtime_localizer(
    map_name, back_facing, full_recovery=None, min_shard_index=None, max_shard_index=None, event_sink=None
):
    shard = config.MAPS_DIR / map_name / "shard.json"
    if not shard.exists():
        return create_localizer(map_name, back_facing)
    from .sharded_localizer import ShardedLocalizer

    if full_recovery is None:
        full_recovery = getattr(config, "SHARD_FULL_RECOVERY", False)
    return ShardedLocalizer(
        config.MAPS_DIR,
        map_name,
        back_facing,
        create_localizer,
        preload_nodes=config.SHARD_PRELOAD_NODES,
        confirm_fixes=config.SHARD_CONFIRM_FIXES,
        min_inliers=config.MIN_INLIERS,
        preload_all=getattr(config, "SHARD_PRELOAD_ALL", False),
        full_recovery=full_recovery,
        recovery_min_inliers=getattr(config, "SHARD_RECOVERY_MIN_INLIERS", config.MIN_INLIERS),
        min_shard_index=min_shard_index,
        max_shard_index=max_shard_index,
        event_sink=event_sink,
        switch_policies=getattr(config, "SHARD_SWITCH_POLICIES", {}),
    )
