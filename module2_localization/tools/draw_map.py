import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map_4f_adaptive_v2")
    ap.add_argument("--out", default=None, help="по умолчанию out/<map>_image.jpg")
    ap.add_argument("--size", type=int, default=1100)
    args = ap.parse_args()

    model = ROOT / "maps" / args.map / "sparse" / "0"
    if not model.exists():
        print(f"нет модели: {model}")
        return 1
    r = pycolmap.Reconstruction(str(model))
    o = sorted(r.images.values(), key=lambda i: i.name)
    P = np.array([(-i.cam_from_world().rotation.matrix().T @ i.cam_from_world().translation) for i in o])
    nums = np.array([int("".join(filter(str.isdigit, i.name)) or 0) for i in o])
    xyz = np.array([p.xyz for p in r.points3D.values()])

    lo_c, hi_c = np.percentile(xyz[:, [0, 2]], [2, 98], axis=0)
    lo_p, hi_p = np.percentile(P[:, [0, 2]], [1, 99], axis=0)
    lo = np.minimum(lo_c, lo_p)
    hi = np.maximum(hi_c, hi_p)
    w = h = args.size
    pad = int(w * 0.064)

    def px(p):
        q = (p - lo) / (hi - lo)
        return np.stack([pad + q[:, 0] * (w - 2 * pad), h - pad - q[:, 1] * (h - 2 * pad)], 1).astype(int)

    img = np.full((h, w, 3), 16, np.uint8)
    pp = px(xyz[:, [0, 2]])
    m = (pp[:, 0] >= 0) & (pp[:, 0] < w) & (pp[:, 1] >= 0) & (pp[:, 1] < h)
    for x, y in pp[m]:
        cv2.circle(img, (x, y), 1, (95, 95, 95), -1)

    tp = px(P[:, [0, 2]])
    med = np.median(np.linalg.norm(np.diff(P, axis=0), axis=1))
    for k in range(len(tp) - 1):
        if np.linalg.norm(P[k + 1] - P[k]) < 30 * med and nums[k + 1] - nums[k] <= 2:
            c = int(255 * k / len(tp))
            cv2.line(img, tuple(tp[k]), tuple(tp[k + 1]), (255 - c, 120, c), 3, cv2.LINE_AA)
    cv2.circle(img, tuple(tp[0]), 9, (120, 230, 120), -1)
    cv2.circle(img, tuple(tp[-1]), 9, (60, 60, 240), -1)

    err = r.compute_mean_reprojection_error()
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, args.map, (20, 36), f, 0.65, (235, 235, 235), 2)
    cv2.putText(img, f"{len(P)} cameras   {len(r.points3D)} points   reproj {err:.2f}px   "
                     f"green=start red=end", (20, 62), f, 0.5, (150, 150, 150), 1)

    out = Path(args.out) if args.out else ROOT / "out" / f"{args.map}_image.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"{len(P)} камер, {len(r.points3D)} точек, ошибка {err:.3f}px -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
