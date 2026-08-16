import time
from collections import deque

import numpy as np


def to_command(r, min_inliers):
    if not r or not r.get("ok") or r.get("inliers", 0) < min_inliers:
        return {"move_type": "lost", "reason": f"inliers: {(r or {}).get('inliers', 0)}"}
    if r["move_type"] == "stop":
        return {"move_type": "stop", "node": r["node"]}
    om, dm = r.get("offset_m"), r.get("dist_to_route_m")
    cmd = {
        "move_type": r["move_type"],
        "deg": round(r["bearing_deg"], 2),
        "offset": round(r["offset"], 4),
        "dist_to_route": round(r["dist_to_route"], 4),
        "offset_m": round(om, 3) if om is not None else None,
        "dist_to_route_m": round(dm, 3) if dm is not None else None,
        "node": r["node"],
        "target_node": r.get("target_node", r["node"]),
        "inliers": r["inliers"],
    }
    C, fwd = r.get("C"), r.get("fwd")
    if C is not None:
        cmd["pos"] = [round(float(C[0]), 4), round(float(C[2]), 4)]
        cmd["head"] = [round(float(fwd[0]), 4), round(float(fwd[2]), 4)]
    return cmd


class Pilot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.chist = deque(maxlen=getattr(cfg, "POSE_HISTORY", 5))
        self.nhist = deque(maxlen=getattr(cfg, "POSE_HISTORY", 5))
        self.last_cmd = ("straight", 0.0)
        self.stopped = False
        self.stop_hits = 0
        self.last_good = None
        self.last_node = None
        self.last_fix_t = None
        self.rejects = 0
        self.accepted = False
        self.jump = None
        self.moved_once = False
        self.paused = True

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def reset(self):
        self.stopped = False
        self.stop_hits = 0
        self.rejects = 0

    def step(self, r, now=None):
        cfg = self.cfg
        now = time.monotonic() if now is None else now
        self.jump = None

        if r and r.get("ok") and self.last_node is not None:
            dt = max(now - self.last_fix_t, 1e-3)
            allowed = max(cfg.MIN_NODE_JUMP, int(cfg.MAX_NODES_PER_SEC * dt))
            if r["node"] - self.last_node > allowed:
                self.rejects += 1
                if self.rejects < cfg.MAX_REJECTS:
                    self.jump = f"скачок: {r['node'] - self.last_node} > {allowed}"
                    r = {"ok": False, "inliers": 0}
                else:
                    self.rejects = 0
            else:
                self.rejects = 0

        if r and r.get("ok"):
            self.last_node, self.last_fix_t = r["node"], now
            self.chist.append(r["C"])
            self.nhist.append(r["node"])
            d = (self.chist[-1] - self.chist[0]) if len(self.chist) >= 2 else np.zeros(3)
            # На плотной уличной карте перемещение за POSE_HISTORY кадров может
            # быть меньше MOVE_EPS, хотя робот действительно едет. Раньше это
            # навсегда защёлкивало первую команду поворота. Изменение ближайшего
            # узла маршрута является независимым признаком движения.
            node_progress = len(self.nhist) >= 2 and len(set(self.nhist)) > 1
            if float(np.linalg.norm(d)) > cfg.MOVE_EPS or node_progress:
                self.moved_once = True
                self.last_cmd = (r["move_type"], r["bearing_deg"])
            elif self.moved_once:
                r["move_type"], r["bearing_deg"] = self.last_cmd
            if r["move_type"] == "stop" and r["inliers"] >= cfg.STOP_MIN_INLIERS:
                self.stop_hits += 1
            else:
                self.stop_hits = 0
            self.stopped = self.stopped or self.stop_hits >= cfg.STOP_CONFIRM
            if self.stopped:
                r["move_type"], r["bearing_deg"] = "stop", 0.0

        cmd = to_command(r, cfg.MIN_INLIERS)
        self.accepted = cmd["move_type"] != "lost"
        if self.accepted:
            self.last_good = cmd
        elif self.last_good:
            cmd = {"move_type": "lost", "node": self.last_good["node"],
                   "reason": self.jump or cmd["reason"]}
        if self.stopped:
            cmd = {"move_type": "stop", "node": (self.last_good or {}).get("node")}
        if self.paused:
            cmd = {"move_type": "stop", "node": (self.last_good or {}).get("node"), "paused": True}
        cmd["ts"] = round(time.time(), 3)
        return cmd
