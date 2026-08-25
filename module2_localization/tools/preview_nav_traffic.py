"""Комбинированный прогон: навигация (зад -> карта) + светофор (перёд -> рамки/класс).

Слева карта с позицией робота, справа передний кадр с детекцией светофора, снизу
телеметрия. Зоны пока не учитываются (детекция на всей трассе) — для наглядности.

    uv run --no-sync python -m module2_localization.tools.preview_nav_traffic \\
        --map map_ns12_rear_full_colored_fixed \\
        --rear  module2_localization/data/new_street/1/camera-rear-720.mkv \\
        --front module2_localization/data/new_street/1/camera-front-720.mkv
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))
from localizer import AlikedLocalizer
from pilot import Pilot
import config as cfg

sys.path.insert(0, str(ROOT.parent))
from module1_traffic_light import init_models
from module1_traffic_light.core.classifier import get_classifier
from module1_traffic_light.core.detector import get_detector

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAP = 720
VID = 900


def tl_color(label):
    lo = label.lower()
    if "red" in lo or "крас" in lo:
        return (0, 0, 255)
    if "green" in lo or "зел" in lo:
        return (0, 200, 0)
    if "yellow" in lo or "жёл" in lo or "жел" in lo:
        return (0, 210, 255)
    return (0, 200, 255)


def build_canvas(loc, size):
    R, xyz = loc.route, loc.points
    lo = np.minimum(np.percentile(xyz[:, [0, 2]], 2, 0), np.percentile(R[:, [0, 2]], 1, 0))
    hi = np.maximum(np.percentile(xyz[:, [0, 2]], 98, 0), np.percentile(R[:, [0, 2]], 99, 0))
    pad = int(size * 0.08)

    def px(p):
        q = (np.atleast_2d(p) - lo) / (hi - lo + 1e-9)
        return np.stack([pad + q[:, 0] * (size - 2 * pad),
                         size - pad - q[:, 1] * (size - 2 * pad)], 1).astype(int)

    img = np.full((size, size, 3), 16, np.uint8)
    for x, y in px(xyz[:, [0, 2]]):
        if 0 <= x < size and 0 <= y < size:
            cv2.circle(img, (x, y), 1, (70, 70, 70), -1)
    rp = px(R[:, [0, 2]])
    med = np.median(np.linalg.norm(np.diff(R, axis=0), axis=1))
    for k in range(len(rp) - 1):
        if np.linalg.norm(R[k + 1] - R[k]) < 30 * med:
            cv2.line(img, tuple(rp[k]), tuple(rp[k + 1]), (200, 170, 60), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(rp[0]), 8, (120, 230, 120), -1)
    cv2.circle(img, tuple(rp[-1]), 8, (60, 60, 240), -1)
    return img, rp, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=cfg.DEFAULT_MAP)
    ap.add_argument("--rear", required=True)
    ap.add_argument("--front", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=3, help="каждый N-й кадр")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--det-conf", type=float, default=0.15)
    args = ap.parse_args()

    loc = AlikedLocalizer(args.map, kpts=cfg.QUERY_KPTS, det_threshold=cfg.QUERY_DET_THRESHOLD,
                          nms_radius=cfg.QUERY_NMS_RADIUS, max_error=cfg.MAX_ERROR,
                          back_facing=cfg.REAR_CAM_BACK, match_ratio=cfg.MATCH_RATIO,
                          match_topk=cfg.MATCH_TOPK, focal_fallback=cfg.FOCAL_FALLBACK,
                          min_pairs=cfg.MIN_PAIRS)
    pilot = Pilot(cfg)
    pilot.resume()
    import module1_traffic_light as tl
    tl.config.DET_CONF = args.det_conf
    init_models()
    det, clsf = get_detector(), get_classifier()

    canvas, rp, px = build_canvas(loc, MAP)
    n = len(rp)
    capR, capF = cv2.VideoCapture(args.rear), cv2.VideoCapture(args.front)
    out_path = Path(args.out) if args.out else ROOT / "out" / f"nav_traffic_{Path(args.rear).stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H = MAP + VID, MAP + 70
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    idx = kept = 0
    last = {"cmd": None, "tl": None}
    while True:
        okR, fr = capR.read()
        okF, ff = capF.read()
        if not okR or not okF:
            break
        if idx % args.step == 0:
            kept += 1
            r = loc.locate(fr)
            cmd = pilot.step(r)
            last["cmd"] = cmd

            pil = Image.fromarray(cv2.cvtColor(ff, cv2.COLOR_BGR2RGB))
            boxes = det.detect(pil)
            sig = None
            for (x1, y1, x2, y2) in boxes:
                label, conf = clsf.classify(pil.crop((x1, y1, x2, y2)))
                c = tl_color(label)
                cv2.rectangle(ff, (x1, y1), (x2, y2), c, 2)
                cv2.putText(ff, f"{label} {conf:.0%}", (x1, max(y1 - 8, 16)), FONT, 0.7, c, 2, cv2.LINE_AA)
                sig = label
            last["tl"] = sig

            img = canvas.copy()
            c = last["cmd"] or {}
            if c.get("move_type") not in (None, "lost") and c.get("node") is not None:
                node = min(max(c["node"], 0), n - 1)
                cv2.circle(img, tuple(rp[node]), 6, (120, 255, 120), 2)
                p = tuple(px([c["pos"][0], c["pos"][1]])[0]) if c.get("pos") else tuple(rp[node])
                cv2.circle(img, p, 9, (60, 60, 255), -1)
                cv2.circle(img, p, 9, (255, 255, 255), 2)
                mt = c["move_type"].upper()
            else:
                mt = "LOST"
            s = min(VID / ff.shape[1], MAP / ff.shape[0])
            fv = cv2.resize(ff, (int(ff.shape[1] * s), int(ff.shape[0] * s)))
            pane = np.full((MAP, VID, 3), 16, np.uint8)
            pane[:fv.shape[0], :fv.shape[1]] = fv
            top = np.hstack([img, pane])
            bar = np.full((70, W, 3), 24, np.uint8)
            node = c.get("node", "-")
            cv2.putText(bar, f"NAV: {mt}  node {node}/{n}  deg {c.get('bearing_deg', c.get('deg', 0)):+.0f}",
                        (16, 30), FONT, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
            tlc = tl_color(last["tl"]) if last["tl"] else (150, 150, 150)
            cv2.putText(bar, f"TRAFFIC: {last['tl'] or 'none'}", (16, 58), FONT, 0.7, tlc, 2, cv2.LINE_AA)
            vw.write(np.vstack([top, bar]))
            if kept % 25 == 0:
                print(f"кадр {idx} | nav {mt} node {node} | светофор {last['tl']}", flush=True)
        idx += 1
    capR.release(); capF.release(); vw.release()
    print(f"\nготово -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
