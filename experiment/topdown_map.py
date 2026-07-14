"""
Реалистичная 2D-карта сверху через IPM (гомография плоскости земли).

Плотный варп кадра (не проекция точек => без дыр и полос):
  1. depth (Depth Anything V2 Metric Outdoor) + RANSAC -> плоскость земли, без калибровки;
  2. по пикселям земли строим гомографию H: кадр (u,v) -> земля (lat, fwd);
  3. cv2.warpPerspective варпает весь кадр -> вид сверху.

Варпаем и фото (реалистичный overhead), и карту классов сегментации (плотная семантика).
Плоская земля отображается точно; приподнятое у горизонта растягивается (природа IPM),
ближняя/средняя зона корректна.

Цвета классов — ровно как в yolo_train/segmentation/predict_video.py.

Запуск:
    uv run python experiment/topdown_map.py data/photo/21.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from depth_model import infer_depth, load_model

OUT_DIR = Path("experiment/topdown_out")

# --- сегментация поверхностей (6 классов) ---
SEG_WEIGHTS = "weights/segmentator_v2/best_seg_v2.pt"
SEG_IMGSZ = 1280
NAMES = ["background", "crosswalk", "curb", "road", "sidewalk", "terrain"]
# BGR, ровно как в predict_video.py (палитра ultralytics + оверрайды 1,3,4)
COLORS = np.array(
    [
        (255, 42, 4),  # 0 background
        (0, 255, 255),  # 1 crosswalk
        (243, 243, 243),  # 2 curb
        (0, 0, 255),  # 3 road
        (144, 238, 144),  # 4 sidewalk
        (221, 111, 255),  # 5 terrain
    ],
    dtype=np.uint8,
)

# --- геометрия карты (в "метрах модели"; масштаб раздут ~x3) ---
HFOV_DEG = 65.0
FWD_MAX = 36.0  # ~12 реальных метров вперёд
LAT_HALF = 15.0  # ~5 реальных метров вбок (±)
# перевод "метров модели" в реальные (для подписей осей). Откалибровано по
# наблюдению 5↔8 (реальные 5 м показывались как 8). Точное значение — рулеткой.
REAL_PER_MODEL = 0.21
CELL = 0.05  # мельче ячейка -> плавнее варп
PX_PER_CELL = 1
GROUND_TOL = 0.5  # пиксель считается землёй, если |высота над плоскостью| < tol

_seg = YOLO(SEG_WEIGHTS)


def segment(image_path: str) -> np.ndarray:
    """6-классовая семантика -> карта классов HxW uint8."""
    r = _seg.predict(image_path, imgsz=SEG_IMGSZ, verbose=False)[0]
    return np.asarray(r.semantic_mask.data).astype(np.uint8)


def seg_overlay(image_bgr: np.ndarray, cls: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(image_bgr, 0.55, COLORS[cls], 0.45, 0)


def backproject(depth: np.ndarray, hfov_deg: float) -> np.ndarray:
    """Глубина -> облако точек (N,3) в координатах камеры (X вправо, Y вниз, Z вперёд)."""
    h, w = depth.shape
    fx = fy = 0.5 * w / np.tan(np.radians(hfov_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1).reshape(-1, 3)


def fit_ground_plane(pts: np.ndarray, lower_mask: np.ndarray, iters: int = 250, tol: float = 0.3):
    """RANSAC: плоскость земли по точкам нижней части кадра -> (точка_на_плоскости, нормаль_вверх)."""
    cand = pts[lower_mask]
    cand = cand[np.isfinite(cand).all(axis=1)]
    if len(cand) > 8000:
        cand = cand[np.linspace(0, len(cand) - 1, 8000).astype(int)]

    best_inliers = None
    rng_idx = np.arange(len(cand))
    for i in range(iters):
        s = cand[(rng_idx * (i + 1) * 2654435761) % len(cand)][:3]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        inliers = np.abs((cand - s[0]) @ n) < tol
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    inlier_pts = cand[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid)
    normal = vh[-1]
    if normal @ (-centroid) < 0:  # ориентируем "вверх", к камере
        normal = -normal
    return centroid, normal


def ground_homography(pts, u, v, centroid, up):
    """Гомография кадр(u,v) -> земля(lat, fwd) по точкам плоскости земли."""
    z_axis = np.array([0.0, 0.0, 1.0])
    forward = z_axis - (z_axis @ up) * up
    forward /= np.linalg.norm(forward)
    lateral = np.cross(up, forward)
    lateral /= np.linalg.norm(lateral)
    if lateral @ np.array([1.0, 0.0, 0.0]) < 0:
        lateral = -lateral

    height = (pts - centroid) @ up
    fwd = pts @ forward
    lat = pts @ lateral

    good = (np.abs(height) < GROUND_TOL) & (fwd > 0) & (fwd < FWD_MAX * 1.5) & np.isfinite(fwd)
    idx = np.where(good)[0]
    if len(idx) > 6000:
        idx = idx[np.linspace(0, len(idx) - 1, 6000).astype(int)]

    src = np.stack([u[idx], v[idx]], axis=1).astype(np.float32)
    dst = np.stack([lat[idx], fwd[idx]], axis=1).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.2)
    return H


def metric_to_px() -> tuple[np.ndarray, int, int]:
    sx = PX_PER_CELL / CELL  # пикселей на метр
    out_w = int(2 * LAT_HALF / CELL) * PX_PER_CELL
    out_h = int(FWD_MAX / CELL) * PX_PER_CELL
    A = np.array([[sx, 0.0, LAT_HALF * sx], [0.0, -sx, out_h], [0.0, 0.0, 1.0]])
    return A, out_w, out_h


def draw_rings(img: np.ndarray) -> None:
    h = img.shape[0]
    for rm in range(2, int(FWD_MAX * REAL_PER_MODEL) + 1, 2):
        y = h - int((rm / REAL_PER_MODEL) / CELL) * PX_PER_CELL
        if 0 <= y < h:
            cv2.line(img, (0, y), (img.shape[1], y), (120, 120, 120), 1)
            cv2.putText(img, f"~{rm}m", (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
    cv2.circle(img, (img.shape[1] // 2, h - 3), 5, (0, 0, 255), -1)


def main() -> None:
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/photo/21.jpg"
    if not Path(img_path).exists():
        print(f"[ERROR] нет файла: {img_path}")
        return

    processor, model, device = load_model()
    image = Image.open(img_path).convert("RGB")
    depth = infer_depth(image, processor, model, device)
    cls_map = segment(img_path)
    h, w = depth.shape
    if cls_map.shape != (h, w):
        cls_map = cv2.resize(cls_map, (w, h), interpolation=cv2.INTER_NEAREST)

    pts = backproject(depth, HFOV_DEG)
    u = np.tile(np.arange(w, dtype=np.float32), h)
    v = np.repeat(np.arange(h, dtype=np.float32), w)
    lower = (v > 0.55 * h) & np.isfinite(pts).all(axis=1)
    centroid, up = fit_ground_plane(pts, lower)

    H = ground_homography(pts, u, v, centroid, up)
    A, out_w, out_h = metric_to_px()
    H_out = A @ H

    orig_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cls_color = COLORS[cls_map]
    td_photo = cv2.warpPerspective(orig_bgr, H_out, (out_w, out_h), flags=cv2.INTER_LINEAR)
    td_seg = cv2.warpPerspective(cls_color, H_out, (out_w, out_h), flags=cv2.INTER_NEAREST)
    draw_rings(td_photo)
    draw_rings(td_seg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    over = seg_overlay(orig_bgr, cls_map)
    scale = out_h / orig_bgr.shape[0]
    orig_r = cv2.resize(orig_bgr, (int(orig_bgr.shape[1] * scale), out_h))
    over_r = cv2.resize(over, (int(over.shape[1] * scale), out_h))
    panel = np.hstack([orig_r, over_r, td_photo, td_seg])
    cv2.imwrite(str(OUT_DIR / f"{stem}_topdown.jpg"), panel)
    print(f"[INFO] сохранено: {OUT_DIR}/{stem}_topdown.jpg  (оригинал | сегментация | вид сверху фото | вид сверху классы)")


if __name__ == "__main__":
    main()
