"""
# РЕЗЕРВ: уличная версия (тротуары) — 6-классовая best_seg_v2 + кадры data/photo.
# ВНИМАНИЕ: data/photo снимала ДРУГАЯ камера, эта калибровка к ним НЕ подходит.
# Вернуться, когда откалибруем ту камеру или переснимем тротуар камерой робота.
2D-карта сверху (BEV) + расстояние до краёв тротуара.

Варп как раньше (плотный, до 6 м) — по depth-проекции точек земли, масштаб от
реальной высоты камеры 1.5 м (метры честные). Сверху — замер краёв тротуара.

  1. undistort (K/dist);
  2. depth + сегментация на выправленном кадре;
  3. backproject (реальные fx,fy) + RANSAC-плоскость; SCALE = 1.5м / высота_в_depth;
  4. гомография пиксель->метрическая земля -> плотный варп фото и классов;
  5. кромки тротуара, осевая, смещение робота от осевой.

Запуск:
    uv run python experiment/edge_distance.py data/photo/21.jpg
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from depth_model import infer_depth, load_model
from topdown_map import COLORS, NAMES, fit_ground_plane, segment

OUT_DIR = Path("experiment/edge_out")
INTRINSICS = "experiment/camera_intrinsics.npz"

ALLOWED = {NAMES.index("sidewalk"), NAMES.index("crosswalk")}
CAMERA_HEIGHT_M = 1.50

FWD_MAX_M = 6.0
LAT_HALF_M = 3.0
PPM = 90
GAP_TOL_M = 0.15


def calibrated_backproject(depth, k):
    h, w = depth.shape
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1).reshape(-1, 3), u.ravel(), v.ravel()


def ground_basis(up):
    z = np.array([0.0, 0.0, 1.0])
    fwd = z - (z @ up) * up
    fwd /= np.linalg.norm(fwd)
    lat = np.cross(up, fwd)
    lat /= np.linalg.norm(lat)
    if lat @ np.array([1.0, 0.0, 0.0]) < 0:
        lat = -lat
    return fwd, lat


def clean_allowed(cls_bev):
    """Булева маска проезжей зоны с закрытием дыр — гасит выбросы сегментации (спекл/дырки)."""
    allowed = np.isin(cls_bev, list(ALLOWED)).astype(np.uint8)
    ksize = max(3, int(0.25 * PPM))  # закрываем дыры до ~0.25 м
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    allowed = cv2.morphologyEx(allowed, cv2.MORPH_CLOSE, kernel)
    return allowed.astype(bool)


def analyze_corridor(allowed):
    h, w = allowed.shape
    robot_col = w // 2
    gap_tol = int(GAP_TOL_M * PPM)

    def expand(mrow, step):
        c, gap = robot_col, 0
        while 0 <= c + step < w:
            if mrow[c + step]:
                c, gap = c + step, 0
            elif gap < gap_tol:
                c, gap = c + step, gap + 1
            else:
                break
        return c

    rows, fwd, lc, rc = [], [], [], []
    for r in range(h - 1, -1, -1):
        fwd_m = (h - r) / PPM
        if fwd_m > FWD_MAX_M:
            break
        if not allowed[r, robot_col]:
            continue
        rows.append(r)
        fwd.append(fwd_m)
        lc.append(expand(allowed[r], -1))
        rc.append(expand(allowed[r], +1))

    if len(rows) < int(0.4 * PPM):
        return {"offset_from_center_m": None, "width_m": None, "left_clearance_m": None,
                "right_clearance_m": None, "near_dist_m": None, "valid": False}, None

    rows, fwd = np.array(rows), np.array(fwd)
    lc, rc = np.array(lc, float), np.array(rc, float)
    center = (lc + rc) / 2
    ca, cb = np.polyfit(rows.astype(float), center, 1)  # сглаженная осевая: col = ca*row + cb
    band = fwd <= fwd[0] + 1.0
    left_c = float(np.median(robot_col - lc[band])) / PPM
    right_c = float(np.median(rc[band] - robot_col)) / PPM
    offset = (robot_col - float(np.median(center[band]))) / PPM
    res = {
        "offset_from_center_m": round(offset, 2),
        "width_m": round(left_c + right_c, 2),
        "left_clearance_m": round(left_c, 2),
        "right_clearance_m": round(right_c, 2),
        "near_dist_m": round(float(fwd[0]), 2),
        "valid": True,
    }
    return res, (robot_col, int(rows[0]), rows, lc, rc, ca, cb)


def draw_map(cls_bev, res, mark):
    bev = np.full((*cls_bev.shape, 3), 30, np.uint8)
    known = (cls_bev != 255) & (cls_bev != 0)  # без background (небо/здания — не земля)
    bev[known] = COLORS[cls_bev[known]]
    h, w = bev.shape[:2]
    orange = (0, 165, 255)

    for m in range(1, int(FWD_MAX_M) + 1):
        y = h - int(m * PPM)
        cv2.line(bev, (0, y), (w, y), (90, 90, 90), 1)
        cv2.putText(bev, f"{m}m", (6, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    if res["valid"]:
        robot_col, near_row, _rows, _lc, _rc, ca, cb = mark
        # осевая (предполагаемая линия движения) — от ближней кромки тротуара до 6 м
        y_top = h - int(FWD_MAX_M * PPM)
        cv2.line(bev, (int(ca * near_row + cb), near_row), (int(ca * y_top + cb), y_top), orange, 2)
        cv2.circle(bev, (robot_col, near_row), 7, (0, 0, 255), -1)  # робот на ближней кромке
        white = (255, 255, 255)
        cv2.putText(bev, f"L {res['left_clearance_m']}m    R {res['right_clearance_m']}m",
                    (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 2)
        cv2.putText(bev, f"offset {res['offset_from_center_m']}m    width {res['width_m']}m",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 2)
    return bev


def process(bgr, k, dist, processor, model, device):
    """Кадр BGR -> (панель [оригинал | фото сверху | классы сверху], контракт). Переиспользуется для видео."""
    und = cv2.undistort(bgr, k, dist)
    depth = infer_depth(Image.fromarray(cv2.cvtColor(und, cv2.COLOR_BGR2RGB)), processor, model, device)
    cls_map = segment(und)
    h, w = depth.shape
    if cls_map.shape != (h, w):
        cls_map = cv2.resize(cls_map, (w, h), interpolation=cv2.INTER_NEAREST)

    pts, u, v = calibrated_backproject(depth, k)
    lower = (v > 0.55 * h) & np.isfinite(pts).all(axis=1)
    centroid, up = fit_ground_plane(pts, lower)
    cam_h = abs(float(centroid @ up))
    scale = CAMERA_HEIGHT_M / cam_h if cam_h > 1e-6 else 1.0
    fwd_ax, lat_ax = ground_basis(up)

    height = ((pts - centroid) @ up) * scale
    fwd = (pts @ fwd_ax) * scale
    lat = (pts @ lat_ax) * scale
    g = (np.abs(height) < 0.2) & (fwd > 0.1) & (fwd < FWD_MAX_M * 1.4) & np.isfinite(fwd)
    idx = np.where(g)[0]
    if len(idx) > 6000:
        idx = idx[np.linspace(0, len(idx) - 1, 6000).astype(int)]
    src = np.stack([u[idx], v[idx]], 1).astype(np.float32)
    dst = np.stack([lat[idx], fwd[idx]], 1).astype(np.float32)
    hmg, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.1)

    out_w, out_h = int(2 * LAT_HALF_M * PPM), int(FWD_MAX_M * PPM)
    a = np.array([[PPM, 0, LAT_HALF_M * PPM], [0, -PPM, out_h], [0, 0, 1.0]])
    m = a @ hmg
    cls_bev = cv2.warpPerspective(cls_map, m, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=255)
    photo_bev = cv2.warpPerspective(und, m, (out_w, out_h), flags=cv2.INTER_LINEAR)

    res, mark = analyze_corridor(clean_allowed(cls_bev))
    bev = draw_map(cls_bev, res, mark)
    und_r = cv2.resize(und, (int(und.shape[1] * out_h / und.shape[0]), out_h))
    return np.hstack([und_r, photo_bev, bev]), res


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/photo/21.jpg"
    if not Path(img_path).exists():
        print(f"[ERROR] нет файла: {img_path}")
        return
    intr = np.load(INTRINSICS)
    processor, model, device = load_model()
    panel, res = process(cv2.imread(img_path), intr["K"], intr["dist"], processor, model, device)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    cv2.imwrite(str(OUT_DIR / f"{stem}_edge.jpg"), panel)
    print(f"[INFO] сохранено: {OUT_DIR}/{stem}_edge.jpg  (оригинал | фото сверху | классы сверху)")


if __name__ == "__main__":
    main()
