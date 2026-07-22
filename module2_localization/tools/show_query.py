import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from localizer import AlikedLocalizer  # noqa: E402
from route import Localizer  # noqa: E402  (focal_from_exif)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--map", default="map_office_ref")
    ap.add_argument("--out", default=None, help="по умолчанию out/query_<имя>.jpg")
    ap.add_argument("--scale", type=float, default=1.0, help="метров в единице карты")
    ap.add_argument("--max-side", type=int, default=0, help="ужать запрос до N px по длинной стороне")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--images", default="frames_office", help="папка кадров карты для retrieval")
    ap.add_argument("--direct", action="store_true", help="без retrieval, прямой поиск по облаку")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    loc = AlikedLocalizer(args.map)
    xyz = np.array([p.xyz for p in loc.rec.points3D.values()])
    R = loc.route
    lo, hi = np.percentile(np.vstack([xyz[:, [0, 2]], R[:, [0, 2]]]), [1, 99], axis=0)
    pad = int(args.size * 0.07)
    W = H = args.size

    def px(p):
        q = (p - lo) / np.maximum(hi - lo, 1e-9)
        return np.stack([pad + q[:, 0] * (W - 2 * pad), H - pad - q[:, 1] * (H - 2 * pad)], 1).astype(int)

    base = np.full((H, W, 3), 16, np.uint8)
    pp = px(xyz[:, [0, 2]])
    m = (pp[:, 0] >= 0) & (pp[:, 0] < W) & (pp[:, 1] >= 0) & (pp[:, 1] < H)
    for x, y in pp[m]:
        cv2.circle(base, (x, y), 1, (80, 80, 80), -1)
    rp = px(R[:, [0, 2]])
    med = np.median(np.linalg.norm(np.diff(R, axis=0), axis=1))
    for k in range(len(rp) - 1):
        if np.linalg.norm(R[k + 1] - R[k]) < 30 * med:
            cv2.line(base, tuple(rp[k]), tuple(rp[k + 1]), (150, 150, 150), 2, cv2.LINE_AA)
    cv2.circle(base, tuple(rp[0]), 7, (120, 230, 120), -1)
    cv2.circle(base, tuple(rp[-1]), 7, (60, 60, 240), -1)

    f = cv2.FONT_HERSHEY_SIMPLEX
    for p in args.photos:
        src = Path(p)
        qpath = src
        if args.max_side:
            im = cv2.imread(str(src))
            k = args.max_side / max(im.shape[:2])
            if k < 1:
                im = cv2.resize(im, (int(im.shape[1] * k), int(im.shape[0] * k)), interpolation=cv2.INTER_AREA)
            qpath = Path("/tmp") / f"_q_{src.stem}.jpg"
            cv2.imwrite(str(qpath), im, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # EXIF читаем из ОРИГИНАЛА: при пересохранении он теряется
        ef = Localizer.focal_from_exif(src, cv2.imread(str(qpath)).shape[1])
        r = loc.locate(qpath, exif_focal=ef)
        img = base.copy()
        title = src.name
        if r is None or not r["ok"]:
            reason = "не читается" if r is None else r["reason"]
            cv2.putText(img, f"{title}: НЕ НАЙДЕНО ({reason})", (16, 32), f, 0.6, (60, 60, 240), 2)
        else:
            C = px(r["C"][None, [0, 2]])[0]
            fw = r["fwd"][[0, 2]]
            fw = fw / (np.linalg.norm(fw) + 1e-9)
            tip = (int(C[0] + fw[0] * 60), int(C[1] - fw[1] * 60))
            near = rp[r["node"]]
            tgt = rp[r["target_node"]]

            cv2.line(img, tuple(C), tuple(near), (0, 200, 255), 3, cv2.LINE_AA)   # смещение от эталона
            # путь до цели — ПО МАРШРУТУ, а не по прямой: прямая срезает углы и ведёт сквозь стены
            seg = rp[r["node"]:r["target_node"] + 1]
            for a, b in zip(seg[:-1], seg[1:]):
                cv2.line(img, tuple(a), tuple(b), (255, 200, 60), 3, cv2.LINE_AA)
            cv2.circle(img, tuple(near), 6, (0, 200, 255), -1)
            cv2.circle(img, tuple(tgt), 8, (255, 200, 60), 2)                      # цель с упреждением
            cv2.arrowedLine(img, tuple(C), tip, (80, 255, 80), 3, cv2.LINE_AA, tipLength=0.3)
            cv2.circle(img, tuple(C), 10, (80, 255, 80), -1)
            cv2.circle(img, tuple(C), 10, (255, 255, 255), 2)

            cv2.putText(img, title, (16, 30), f, 0.62, (235, 235, 235), 2)
            l2 = (f"inliers {r['inliers']}   node {r['node']}   "
                  f"to_route {r['dist_to_route'] * args.scale:.2f}   offset {r['offset'] * args.scale:+.2f}")
            cv2.putText(img, l2, (16, 56), f, 0.5, (200, 200, 200), 1)
            col = (80, 255, 80) if r["move_type"] == "straight" else (60, 200, 255)
            cv2.putText(img, f"bearing {r['bearing_deg']:+.1f} deg  ->  {r['move_type'].upper()}",
                        (16, 86), f, 0.7, col, 2)
            for i, (c, t) in enumerate((((80, 255, 80), "green = query pose + view dir"),
                                        ((0, 200, 255), "orange = nearest route point"),
                                        ((255, 200, 60), "ring = lookahead target"),
                                        ((150, 150, 150), "grey = reference route"))):
                y = H - 82 + i * 20
                cv2.circle(img, (24, y - 4), 5, c, -1)
                cv2.putText(img, t, (40, y), f, 0.42, (170, 170, 170), 1)

        out = Path(args.out) if args.out else ROOT / "out" / f"query_{src.stem}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print(f"{src.name} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
