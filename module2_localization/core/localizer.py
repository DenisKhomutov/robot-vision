import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from .route import Localizer


class AlikedLocalizer:
    def __init__(self, map_name="map_office_ref", device=None, kpts=2048, route_range=None,
                 det_threshold=0.2, nms_radius=2, max_error=12.0, steer="pursuit", route_cam=None,
                 route_nodes=None, back_facing=False, match_ratio=0.9, match_topk=8,
                 focal_fallback=1.2, min_pairs=8, pnp_confidence=0.999, pnp_iters=1000,
                 lookahead=12, lookahead_min=5, lookahead_adapt=8.0, deadzone=4.0,
                 stanley_k=1.0, heading_gate=0.3, stop_end_nodes=3,
                 lag_s=0.18, lag_adaptive=False, lead_max=1.5, lead_smooth=5, win_nodes=0,
                 lookahead_speed_div=None, lookahead_max=None, bank_path=None):
        from .hub import use_local_weights
        from lightglue import ALIKED
        use_local_weights()
        self.max_error = max_error
        self.steer = steer
        self.route_cam = route_cam
        self.back_facing = back_facing
        self.match_ratio = match_ratio
        self.match_topk = match_topk
        self.focal_fallback = focal_fallback
        self.min_pairs = min_pairs
        self.pnp_confidence = pnp_confidence
        self.pnp_iters = pnp_iters
        self.lookahead = lookahead
        self.lookahead_min = lookahead_min
        self.lookahead_adapt = lookahead_adapt
        self.lookahead_speed_div = lookahead_speed_div
        self.lookahead_max = lookahead_max
        self.deadzone = deadzone
        self.stanley_k = stanley_k
        self.heading_gate = heading_gate
        self.stop_end_nodes = stop_end_nodes
        self.lag_s = lag_s
        self.lag_adaptive = lag_adaptive
        self.lead_max = lead_max
        self._lead_hist = deque(maxlen=lead_smooth)
        self.win_nodes = win_nodes
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        work = ROOT / "maps" / map_name
        m = np.load(work / "runtime.npz")
        fx, fy, cx, cy = m["K"]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.dist = m["dist"].astype(np.float64)
        self.cam_w, self.cam_h = (int(v) for v in m["size"])
        self.points = m["points"]
        self.ext = ALIKED(max_num_keypoints=kpts, detection_threshold=det_threshold,
                          nms_radius=nms_radius).eval().to(self.dev)
        self.half_ext = self.dev == "cuda"
        if self.half_ext:
            self.ext = self.ext.half()

        sf = work / "scale.json"
        self.scale = json.loads(sf.read_text())["scale_m_per_unit"] if sf.exists() else None

        bank = np.load(Path(bank_path) if bank_path is not None else work / "aliked_bank.npz")
        self.mdesc = torch.from_numpy(bank["desc"].astype(np.float32)).half().to(self.dev)
        self.owner = torch.from_numpy(bank["owner"].astype(np.int64)).to(self.dev)
        self.mxyz = bank["xyz"]
        if "align" in m:
            self.mxyz = self.mxyz @ m["align"].T
        self._bank_owner = bank["owner"]

        pos_by_name = dict(zip(m["names"].tolist(), m["pos"]))
        fwd_by_name = dict(zip(m["names"].tolist(), m["fwd"]))
        order = sorted(pos_by_name)
        if self.route_cam:
            order = [n for n in order if self.route_cam in n]
        if route_range:
            a, b = route_range
            order = [n for n in order if a <= int("".join(filter(str.isdigit, n)) or 0) < b]
        pos = np.array([pos_by_name[n] for n in order])
        if len(pos) > 5:
            seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            thr = 10 * np.median(seg)
            keep = [True] * len(order)
            for k in range(1, len(order) - 1):
                if seg[k - 1] > thr and seg[k] > thr:
                    keep[k] = False
            dropped = len(order) - sum(keep)
            if dropped:
                print(f"[карта] выкинуто выбросов маршрута: {dropped}")
            order = [n for n, kp in zip(order, keep) if kp]
        if route_nodes and len(order) > route_nodes:
            print(f"[карта] маршрут обрезан: {len(order)} -> {route_nodes} узлов")
            order = order[:route_nodes]
        self.route = np.array([pos_by_name[n] for n in order])
        self.route_fwd = np.array([fwd_by_name[n] for n in order])
        if self.back_facing:
            self.route_fwd = -self.route_fwd
        seg = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.route_cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.node_step = float(np.median(seg)) if len(seg) else 1.0

        route = self.route.astype(np.float32)
        pt_node = np.empty(len(self.mxyz), np.int64)
        for i in range(0, len(self.mxyz), 20000):
            seg_pts = self.mxyz[i:i + 20000].astype(np.float32)
            d2 = ((seg_pts[:, None, :] - route[None, :, :]) ** 2).sum(2)
            pt_node[i:i + 20000] = d2.argmin(1)
        self._desc_node = torch.from_numpy(pt_node[self._bank_owner]).to(self.dev)
        self._last_node = None
        print(f"[карта] {len(order)} кадров, {len(self.mxyz)} точек, "
              f"{len(self.owner)} дескрипторов, {self.dev}")

    def _lead(self, C, fwd, frame_ms):

        now = time.perf_counter()
        self._lead_hist.append((np.asarray(C, float), now))
        if len(self._lead_hist) < 2:
            return C
        C0, t0 = self._lead_hist[0]
        dt = now - t0
        if dt < 1e-3:
            return C
        v = float(np.linalg.norm(C - C0)) / dt
        lag = (frame_ms / 1000.0) if self.lag_adaptive else self.lag_s
        lead = min(v * lag, self.lead_max)
        f = fwd / (np.linalg.norm(fwd) + 1e-9)
        return np.asarray(C, float) + lead * f

    def extract_query(self, path):
        from lightglue.utils import load_image
        t0 = time.perf_counter()
        if isinstance(path, np.ndarray):
            rgb = cv2.cvtColor(path, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            im = torch.from_numpy(rgb.transpose(2, 0, 1)).to(self.dev)
        else:
            im = load_image(str(path)).to(self.dev)
        if self.half_ext:
            im = im.half()
        with torch.inference_mode():
            f = self.ext.extract(im)
        return {
            "keypoints": f["keypoints"][0].cpu().numpy(),
            "descriptors": f["descriptors"][0].half(),
            "height": int(im.shape[-2]),
            "width": int(im.shape[-1]),
            "source": None if isinstance(path, np.ndarray) else path,
            "t_ext": (time.perf_counter() - t0) * 1000,
        }

    def locate_features(self, query, chunk=None, exif_focal=None):
        qk = query["keypoints"]
        q = query["descriptors"]
        t_ext = float(query.get("t_ext", 0.0))

        t0 = time.perf_counter()
        n = len(qk)
        if chunk is None:
            chunk = max(4096, int(256e6 / (4 * max(n, 1))))
        neg = torch.finfo(torch.float16).min
        best = torch.full((n,), neg, device=self.dev, dtype=torch.float16)
        bown = torch.full((n,), -1, dtype=torch.long, device=self.dev)
        second = torch.full((n,), neg, device=self.dev, dtype=torch.float16)


        mdesc, owner = self.mdesc, self.owner
        if self._last_node is not None and self.win_nodes > 0:
            lo, hi = self._last_node - self.win_nodes, self._last_node + self.win_nodes
            near = (self._desc_node >= lo) & (self._desc_node <= hi)
            if int(near.sum()) >= 200:
                mdesc, owner = self.mdesc[near], self.owner[near]
        with torch.inference_mode():
            for i in range(0, len(owner), chunk):
                sim = q @ mdesc[i:i + chunk].T
                own = owner[i:i + chunk]
                k = min(self.match_topk, sim.shape[1])
                tv, ti = sim.topk(k, dim=1)
                for c in range(k):
                    v, o = tv[:, c], own[ti[:, c]]
                    nb = v > best
                    second = torch.where(nb & (bown != o) & (bown >= 0),
                                         torch.maximum(second, best), second)
                    second = torch.where(~nb & (o != bown), torch.maximum(second, v), second)
                    bown = torch.where(nb, o, bown)
                    best = torch.where(nb, v, best)
        d1 = torch.sqrt((2 - 2 * best.float()).clamp(min=0))
        d2 = torch.sqrt((2 - 2 * second.float()).clamp(min=0))
        keep = ((d1 / d2.clamp(min=1e-6) < self.match_ratio) & (bown >= 0)).cpu().numpy()
        own_cpu = bown.cpu().numpy()
        t_match = (time.perf_counter() - t0) * 1000

        p2d, p3d = qk[keep], self.mxyz[own_cpu[keep]]
        if len(p2d) < self.min_pairs:
            self._last_node = None
            return {"ok": False, "reason": f"мало пар: {len(p2d)}",
                    "t_ext": t_ext, "t_match": t_match}

        h_img, w_img = query["height"], query["width"]
        K, dist = self.K, self.dist
        if (w_img, h_img) != (self.cam_w, self.cam_h):
            if abs(w_img / h_img - self.cam_w / self.cam_h) < 0.01:
                K = K.copy()
                K[0] *= w_img / self.cam_w
                K[1] *= h_img / self.cam_h
            else:
                source = query.get("source")
                ef = None if source is None else Localizer.focal_from_exif(source, w_img)
                f0 = exif_focal or ef or self.focal_fallback * max(w_img, h_img)
                K = np.array([[f0, 0, w_img / 2], [0, f0, h_img / 2], [0, 0, 1.0]])
                dist = np.zeros(4)

        t0 = time.perf_counter()
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            p3d.astype(np.float64), p2d.astype(np.float64), K, dist,
            reprojectionError=self.max_error, confidence=self.pnp_confidence,
            iterationsCount=self.pnp_iters, flags=cv2.SOLVEPNP_EPNP)
        if ok and inl is not None and len(inl) >= 6:
            idx = inl.ravel()
            rvec, tvec = cv2.solvePnPRefineLM(p3d[idx].astype(np.float64),
                                              p2d[idx].astype(np.float64), K, dist, rvec, tvec)
        t_pnp = (time.perf_counter() - t0) * 1000
        if not ok or inl is None or len(inl) < 6:
            self._last_node = None
            return {"ok": False, "reason": "PnP не сошёлся", "n_pairs": len(p2d),
                    "inliers": 0 if inl is None else len(inl),
                    "t_ext": t_ext, "t_match": t_match, "t_pnp": t_pnp}
        R, _ = cv2.Rodrigues(rvec)
        C = (-R.T @ tvec).ravel()
        fwd = (R.T @ np.array([[0], [0], [1.0]])).ravel()
        if self.back_facing:
            fwd = -fwd
        out = {"ok": True, "C": C, "fwd": fwd, "inliers": len(inl),
               "n_pairs": len(p2d), "t_ext": t_ext, "t_match": t_match, "t_pnp": t_pnp}



        C_lead = self._lead(C, fwd, t_ext + t_match + t_pnp)
        out.update(Localizer.command(self, C_lead, fwd, mode=self.steer))
        out["C_lead"] = C_lead
        self._last_node = out["node"]
        if self.scale:
            out["dist_to_route_m"] = out["dist_to_route"] * self.scale
            out["offset_m"] = out["offset"] * self.scale
        return out

    def locate(self, path, chunk=None, exif_focal=None):
        return self.locate_features(
            self.extract_query(path), chunk=chunk, exif_focal=exif_focal,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--map", default="map_office_ref")
    ap.add_argument("--route-range", default=None, help="эталон только по кадрам a:b (напр. 0:525)")
    args = ap.parse_args()
    rr = tuple(int(x) for x in args.route_range.split(':')) if args.route_range else None
    loc = AlikedLocalizer(args.map, route_range=rr)
    print(f"\n{'фото':<20}{'инл.':>6}{'пар':>7}{'узел':>6}{'до линии':>10}{'азимут':>9}{'команда':>10}{'мс':>7}")
    for p in args.photos:
        r = loc.locate(Path(p))
        n = Path(p).name[:18]
        if r is None:
            print(f"{n:<20} не читается")
            continue
        ms = r["t_ext"] + r["t_match"] + r.get("t_pnp", 0)
        if not r["ok"]:
            print(f"{n:<20} НЕ НАЙДЕНО: {r['reason']:<40}{ms:>7.0f}")
            continue
        d = r.get("dist_to_route_m", r["dist_to_route"])
        u = "м" if "dist_to_route_m" in r else "е"
        print(f"{n:<20}{r['inliers']:>6}{r['n_pairs']:>7}{r['node']:>6}"
              f"{d:>9.2f}{u}{r['bearing_deg']:>+8.1f}°{r['move_type']:>10}{ms:>7.0f}")


if __name__ == "__main__":
    sys.exit(main())
