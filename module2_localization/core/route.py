import numpy as np
from PIL import Image, ExifTags

LOOKAHEAD_NODES = 12
DEADZONE_DEG = 4.0
STANLEY_K = 1.0
LOOKAHEAD_MIN = 5
LOOKAHEAD_ADAPT = 8.0
STOP_END_NODES = 3


class Localizer:

    @staticmethod
    def focal_from_exif(path, width):
        try:
            inv = {v: k for k, v in ExifTags.TAGS.items()}
            ex = Image.open(path)._getexif() or {}
            f35 = ex.get(inv.get("FocalLengthIn35mmFilm"))
            if f35:
                return float(f35) / 36.0 * width
        except Exception:
            pass
        return None

    def command(self, C, fwd, lookahead_nodes=None, mode=None, stanley_k=None):
        lookahead_nodes = getattr(self, "lookahead", LOOKAHEAD_NODES) if lookahead_nodes is None else lookahead_nodes
        mode = getattr(self, "steer", "pursuit") if mode is None else mode
        stanley_k = getattr(self, "stanley_k", STANLEY_K) if stanley_k is None else stanley_k
        d = np.linalg.norm(self.route - C, axis=1)
        f = fwd / (np.linalg.norm(fwd) + 1e-9)
        align = self.route_fwd @ f / (np.linalg.norm(self.route_fwd, axis=1) + 1e-9)
        ok = align > getattr(self, "heading_gate", 0.3)
        k = int(np.argmin(np.where(ok, d, np.inf))) if ok.any() else int(np.argmin(d))
        down = np.array([0, 1.0, 0])
        t = self.route_fwd[k] / (np.linalg.norm(self.route_fwd[k]) + 1e-9)
        v = C - self.route[k]
        e = float(np.dot(np.cross(t, v), down))
        f_flat = fwd - np.dot(fwd, down) * down

        if mode == "stanley":
            j = min(k + lookahead_nodes, len(self.route) - 1)
            t_flat = t - np.dot(t, down) * down
            psi = np.degrees(np.arctan2(np.dot(np.cross(f_flat, t_flat), down),
                                        np.dot(f_flat, t_flat)))
            ang = psi + np.degrees(np.arctan(-stanley_k * e))
        else:
            step = getattr(self, "node_step", 1.0)
            adapt = getattr(self, "lookahead_adapt", LOOKAHEAD_ADAPT)
            lmin = getattr(self, "lookahead_min", LOOKAHEAD_MIN)
            want = max(lookahead_nodes - adapt * abs(e), lmin) * step
            cum = getattr(self, "route_cum", None)
            if cum is not None:
                j = min(int(np.searchsorted(cum, cum[k] + want)), len(self.route) - 1)
            else:
                j = min(k + int(lookahead_nodes), len(self.route) - 1)
            to = self.route[j] - C
            to_flat = to - np.dot(to, down) * down
            ang = np.degrees(np.arctan2(np.dot(np.cross(f_flat, to_flat), down),
                                        np.dot(f_flat, to_flat)))
        dz = getattr(self, "deadzone", None) or DEADZONE_DEG
        mt = "straight" if abs(ang) < dz else ("right" if ang > 0 else "left")
        if k >= len(self.route) - 1 - getattr(self, "stop_end_nodes", STOP_END_NODES):
            mt, ang = "stop", 0.0
        return {"node": k, "target_node": j, "dist_to_route": float(d[k]),
                "offset": e, "bearing_deg": float(ang), "move_type": mt}
