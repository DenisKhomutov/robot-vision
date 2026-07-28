import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))
from localizer import AlikedLocalizer  # noqa: E402
from pilot import Pilot  # noqa: E402
from route import Localizer  # noqa: E402
import config as cfg  # noqa: E402  (module2_localization/config.py)


def predict_path(loc, C, fwd, mode, steps=45, gain=0.4):
    """Прокатка закона руления вперёд из текущей позы -> предсказанная кривая возврата
    к эталону (у Стэнли — плавный экспоненциальный подход к касательной)."""
    down = np.array([0.0, 1.0, 0.0])
    ds = float(np.median(np.linalg.norm(np.diff(loc.route, axis=0), axis=1)))
    Cs = np.asarray(C, float).copy()
    fs = np.asarray(fwd, float).copy()
    fs = fs - np.dot(fs, down) * down
    fs /= np.linalg.norm(fs) + 1e-9
    pts = [Cs.copy()]
    for _ in range(steps):
        r = Localizer.command(loc, Cs, fs, mode=mode)
        a = float(np.clip(np.radians(r["bearing_deg"]) * gain, -0.5, 0.5))
        ca, sa = np.cos(a), np.sin(a)
        x, z = fs[0], fs[2]
        fs = np.array([ca * x + sa * z, 0.0, -sa * x + ca * z])
        fs /= np.linalg.norm(fs) + 1e-9
        Cs = Cs + ds * fs
        pts.append(Cs.copy())
    return np.array(pts)

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANE = 900
VPANE = 1600
TRAIL = 40
MOVE_EPS = 0.06       # смещение (ед. карты) за окно chist ниже -> считаем, что стоим


def build_canvas(loc, size):
    # эталон — маршрут ЛОКАЛИЗАТОРА (уже отфильтрован: одна камера _c2 + без выбросов),
    # тонкой линией. Раньше рисовались траектории всех 3 камер рига внахлёст -> толсто.
    R = loc.route
    xyz = loc.points
    lo_c, hi_c = np.percentile(xyz[:, [0, 2]], [2, 98], axis=0)
    lo_p, hi_p = np.percentile(R[:, [0, 2]], [1, 99], axis=0)
    lo = np.minimum(lo_c, lo_p)
    hi = np.maximum(hi_c, hi_p)
    pad = int(size * 0.08)

    def px(p):
        q = (np.atleast_2d(p) - lo) / (hi - lo + 1e-9)
        return np.stack([pad + q[:, 0] * (size - 2 * pad),
                         size - pad - q[:, 1] * (size - 2 * pad)], 1).astype(int)

    img = np.full((size, size, 3), 16, np.uint8)
    pp = px(xyz[:, [0, 2]])
    m = (pp[:, 0] >= 0) & (pp[:, 0] < size) & (pp[:, 1] >= 0) & (pp[:, 1] < size)
    for x, y in pp[m]:
        cv2.circle(img, (x, y), 1, (70, 70, 70), -1)

    rp = px(R[:, [0, 2]])
    med = np.median(np.linalg.norm(np.diff(R, axis=0), axis=1))
    for k in range(len(rp) - 1):
        if np.linalg.norm(R[k + 1] - R[k]) < 30 * med:
            cv2.line(img, tuple(rp[k]), tuple(rp[k + 1]), (200, 170, 60), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(rp[0]), 8, (120, 230, 120), -1)
    cv2.circle(img, tuple(rp[-1]), 8, (60, 60, 240), -1)
    return img, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--map", default="map_office_ref")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=3, help="брать каждый N-й кадр")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--kpts", type=int, default=2048)
    ap.add_argument("--q-threshold", type=float, default=0.2, help="порог детекции ЗАПРОСА")
    ap.add_argument("--q-nms", type=int, default=2, help="nms_radius запроса")
    ap.add_argument("--max-error", type=float, default=12.0, help="порог репроекции инлайера PnP, px")
    ap.add_argument("--min-inliers", type=int, default=20)
    ap.add_argument("--min-inlier-ratio", type=float, default=0.0,
                    help="F2: доля инлайеров (инлайеры/пары); ложные фиксы имеют низкую")
    ap.add_argument("--mode", default="pursuit", choices=["pursuit", "stanley"])
    ap.add_argument("--route-cam", default=None, help="риг-карта: маршрут по одной камере, напр. _c2")
    ap.add_argument("--back", action="store_true", help="камера смотрит назад по ходу движения")
    ap.add_argument("--route-nodes", type=int, default=None, help="обрезка маршрута (по умолч. из конфига)")
    args = ap.parse_args()

    loc = AlikedLocalizer(args.map, kpts=args.kpts, det_threshold=args.q_threshold,
                          nms_radius=args.q_nms, max_error=args.max_error, steer=args.mode,
                          route_cam=args.route_cam,
                          route_nodes=args.route_nodes if args.route_nodes is not None else cfg.ROUTE_NODES,
                          back_facing=args.back)
    loc.deadzone = cfg.DEADZONE_DEG   # мёртвая зона азимута из конфига (иначе command берёт 4)
    loc.stop_end_nodes = getattr(cfg, "STOP_END_NODES", 3)
    canvas, px = build_canvas(loc, PANE)
    route_px = px(loc.route[:, [0, 2]])

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = Path(args.out) if args.out else ROOT / "out" / f"{Path(args.video).stem}_localized.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W = PANE + VPANE
    H = PANE + 90
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    tmp = ROOT / "out" / "_frame.jpg"
    trail = deque(maxlen=TRAIL)
    pilot = Pilot(cfg)             # та же логика команд, что у демона
    idx = kept = ok_n = 0
    t_start = time.perf_counter()
    times = []
    last = None

    while True:
        got, frame = cap.read()
        if not got:
            break
        if idx % args.step:
            idx += 1
            continue
        idx += 1
        cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = loc.locate(tmp)
        kept += 1
        ms = r["t_ext"] + r["t_match"] + r.get("t_pnp", 0)
        times.append(ms)

        ratio = r["inliers"] / max(r.get("n_pairs", 0), 1) if r["ok"] else 0.0
        if r["ok"] and (r["inliers"] < args.min_inliers or ratio < args.min_inlier_ratio):
            r = {"ok": False, "inliers": r["inliers"]}
        cmd = pilot.step(r, now=idx / args.fps)
        good = cmd["move_type"] != "lost"
        if pilot.accepted:
            ok_n += 1
            trail.append(px(r["C"][[0, 2]])[0])
            last = r
        elif good and last is not None:
            r = dict(last)                                  # приехали, LOST у стены -> держим STOP
            r["move_type"], r["bearing_deg"] = "stop", 0.0

        m = canvas.copy()
        for k in range(len(trail) - 1):
            a = int(60 + 195 * k / max(len(trail) - 1, 1))
            cv2.line(m, tuple(trail[k]), tuple(trail[k + 1]), (a, a, 255), 2, cv2.LINE_AA)

        if good:
            c = tuple(trail[-1])
            node = r["node"]
            nearest = tuple(route_px[node])
            if args.mode == "pursuit":
                tgt = tuple(route_px[min(r.get("target_node", node + 12), len(route_px) - 1)])
                cv2.line(m, c, tgt, (0, 235, 235), 2, cv2.LINE_AA)   # жёлтый: упреждающая цель
                cv2.circle(m, tgt, 7, (0, 235, 235), 2)
            else:  # stanley: показываем ПОПЕРЕЧНОЕ СМЕЩЕНИЕ (робот -> ближайшая точка)
                cv2.line(m, c, nearest, (0, 170, 255), 2, cv2.LINE_AA)
            cv2.circle(m, nearest, 8, (120, 255, 120), 2)   # ближайшая точка эталона
            pth = px(predict_path(loc, r["C"], r["fwd"], args.mode)[:, [0, 2]])
            for a2, b2 in zip(pth[:-1], pth[1:]):
                cv2.line(m, tuple(a2), tuple(b2), (0, 255, 0), 2, cv2.LINE_AA)   # зелёная: предсказанная траектория
            head = r["fwd"] / (np.linalg.norm(r["fwd"]) + 1e-9)   # курс = поза (гладкая в движении)
            f2 = px((r["C"] + head * 0.6)[[0, 2]])[0]
            cv2.arrowedLine(m, c, tuple(f2), (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.35)
            cv2.circle(m, c, 11, (60, 60, 255), -1)
            cv2.circle(m, c, 11, (255, 255, 255), 2)
        elif last is not None:
            cv2.circle(m, tuple(trail[-1]) if trail else (0, 0), 11, (90, 90, 130), 2)

        s = min(VPANE / frame.shape[1], PANE / frame.shape[0])
        vid = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
        pane = np.full((PANE, VPANE, 3), 16, np.uint8)
        y0 = (PANE - vid.shape[0]) // 2
        x0 = (VPANE - vid.shape[1]) // 2
        pane[y0:y0 + vid.shape[0], x0:x0 + vid.shape[1]] = vid

        top = np.hstack([m, pane])
        bar = np.full((90, W, 3), 24, np.uint8)
        if good:
            dt = (f"{r['dist_to_route_m']:.2f} m" if "dist_to_route_m" in r
                  else f"{r['dist_to_route']:.2f} u")
            txt = (f"node {r['node']:>3}/{len(loc.route)}   to route {dt}   "
                   f"bearing {r['bearing_deg']:+.1f}   inliers {r['inliers']:>4}")
            cv2.putText(bar, txt, (24, 38), FONT, 0.8, (235, 235, 235), 2)
            cv2.putText(bar, r["move_type"].upper(), (24, 76), FONT, 0.9, (120, 255, 160), 2)
        else:
            why = r.get("reason", f"inliers {r.get('inliers', 0)}")
            cv2.putText(bar, f"LOST  ({why})", (24, 50), FONT, 0.9, (90, 90, 240), 2)
        cv2.putText(bar, f"{ms:.0f} ms   frame {idx}/{total}   fix {ok_n}/{kept}",
                    (W - 640, 38), FONT, 0.7, (150, 150, 150), 2)
        cv2.putText(bar, args.map, (W - 640, 74), FONT, 0.6, (110, 110, 110), 1)
        vw.write(np.vstack([top, bar]))

        if kept % 25 == 0:
            el = time.perf_counter() - t_start
            print(f"[{kept}] кадр {idx}/{total} | фиксов {ok_n}/{kept} | "
                  f"{el:.0f}с | ~{el / kept * (total / args.step - kept):.0f}с осталось", flush=True)

    cap.release()
    vw.release()
    tmp.unlink(missing_ok=True)
    print(f"\nкадров {kept}, локализовано {ok_n} ({100 * ok_n / max(kept, 1):.0f}%), "
          f"медиана {np.median(times):.0f} мс -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
