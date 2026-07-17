"""
Офисный BEV: карта пола сверху + расстояние до краёв прохода.

Сегментация — ADE20K-претрейн (yolo26m-sem-ade20k, 150 классов, floor=3). Обучать
не нужно: понятие «пол» уже выучено на тысячах помещений. best_seg отказались.

  1. undistort (K/dist, калибровка под 1280x720);
  2. ADE20K-семантика -> маска пола;
  3. плоскость пола ИЗ ГЕОМЕТРИИ: K + высота камеры + наклон (depth не нужен);
  4. варп-растяжение (warpPerspective) -> метрический вид сверху;
  5. кромки прохода, смещение робота от центра.

Геометрия подобрана по эталонному коридору 1.70 м: наклон из условия «стены
параллельны», высота — из условия «ширина = 1.70». Проверка: чистые коридорные
кадры дают 1.71-1.76 м.

Уличная (тротуарная) версия — edge_distance_sidewalk.py (там ДРУГАЯ камера!).

Запуск:
    uv run python experiment/edge_distance.py data/route_reference/route_A/forward/front/front13.jpg
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


OUT_DIR = Path("experiment/edge_out")
INTRINSICS = "experiment/camera_intrinsics.npz"
# ADE20K-претрейн (150 классов, видел тысячи помещений) вместо best_seg:
# floor=3 — готовое понятие «пол», обучать не нужно.
SEG_WEIGHTS = "experiment/weights/yolo26m-sem-ade20k.pt"
SEG_IMGSZ = 640
ADE_FLOOR = 3  # ADE20K: 0=wall, 3=floor, 5=ceiling, 12=person, 14=door
CAMERA_HEIGHT_M = 0.485  # подобрано по эталонному коридору 1.70 м (рулетка до корпуса давала 0.55)
PITCH_DEG = 5.5  # камера наклонена вниз; подобрано по условию "стены коридора параллельны"

FWD_MAX_M = 6.0
LAT_HALF_M = 3.0
PPM = 90
GAP_TOL_M = 0.15

_seg = YOLO(SEG_WEIGHTS)


def segment(bgr):
    """ADE20K-семантика -> булева маска пола (проходимая зона)."""
    h, w = bgr.shape[:2]
    r = _seg.predict(bgr, imgsz=SEG_IMGSZ, verbose=False)[0]
    sm = np.asarray(r.semantic_mask.data).astype(np.uint8)
    if sm.shape != (h, w):
        sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST)
    return (sm == ADE_FLOOR).astype(np.uint8)


def ground_basis(up):
    z = np.array([0.0, 0.0, 1.0])
    fwd = z - (z @ up) * up
    fwd /= np.linalg.norm(fwd)
    lat = np.cross(up, fwd)
    lat /= np.linalg.norm(lat)
    if lat @ np.array([1.0, 0.0, 0.0]) < 0:
        lat = -lat
    return fwd, lat


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
    ca, cb = np.polyfit(rows.astype(float), center, 1)
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


SCALE_W = 54  # ширина полосы шкалы слева


def draw_map(cls_bev, res, mark):
    """Карта пола сверху. Только карта — числа вынесены в отдельную панель."""
    h, w = cls_bev.shape
    bev = np.full((h, w, 3), 30, np.uint8)
    bev[cls_bev == 1] = (144, 238, 144)
    robot_col = w // 2
    f = cv2.FONT_HERSHEY_SIMPLEX

    # линии дальности через метр
    for m in range(1, int(FWD_MAX_M) + 1):
        y = h - int(m * PPM)
        cv2.line(bev, (SCALE_W, y), (w, y), (70, 70, 70), 1)

    # шкала слева: сплошная ось с засечками и крупными подписями
    cv2.rectangle(bev, (0, 0), (SCALE_W, h), (12, 12, 12), -1)
    cv2.line(bev, (SCALE_W, 0), (SCALE_W, h), (200, 200, 200), 2)
    for m in range(0, int(FWD_MAX_M) + 1):
        y = h - int(m * PPM)
        cv2.line(bev, (SCALE_W - 12, y), (SCALE_W, y), (200, 200, 200), 2)
        cv2.putText(bev, str(m), (6, y + 6 if m else y - 4), f, 0.6, (235, 235, 235), 2)
    cv2.putText(bev, "m", (6, 18), f, 0.5, (150, 150, 150), 1)

    # робот в нуле — целиком в кадре
    cv2.circle(bev, (robot_col, h - 10), 9, (0, 0, 255), -1)
    cv2.circle(bev, (robot_col, h - 10), 9, (255, 255, 255), 1)
    return bev


def draw_metrics(res, h):
    """Четвёртая панель: только числа, крупно."""
    w = 300
    p = np.full((h, w, 3), 22, np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(p, (0, 0), (0, h), (70, 70, 70), 1)
    if not res["valid"]:
        cv2.putText(p, "no floor", (20, h // 2), f, 1.0, (80, 80, 220), 2)
        return p
    items = [
        ("width", f"{res['width_m']:.2f}", (144, 238, 144)),
        ("left", f"{res['left_clearance_m']:.2f}", (235, 235, 235)),
        ("right", f"{res['right_clearance_m']:.2f}", (235, 235, 235)),
        ("offset", f"{res['offset_from_center_m']:+.2f}", (0, 165, 255)),
    ]
    y = 54
    for label, val, col in items:
        cv2.putText(p, label, (22, y), f, 0.55, (135, 135, 135), 1)
        cv2.putText(p, val, (22, y + 42), f, 1.25, col, 2)
        cv2.putText(p, "m", (22 + 26 * len(val), y + 42), f, 0.6, (110, 110, 110), 1)
        y += 100
    return p


def process(bgr, k, dist):
    """Кадр -> (панель [оригинал | фото сверху | классы сверху], контракт). depth не нужен."""
    und = cv2.undistort(bgr, k, dist)
    cls_map = segment(und)

    # Плоскость пола из ГЕОМЕТРИИ, а не из depth-угадайки: пол горизонтален,
    # камера ровная на CAMERA_HEIGHT_M, K откалибрована => плоскость точна.
    # Варп-растяжение ниже — тот же самый, меняется только источник плоскости.
    th = np.radians(PITCH_DEG)
    up = np.array([0.0, -np.cos(th), np.sin(th)])
    fwd_ax, lat_ax = ground_basis(up)
    kinv = np.linalg.inv(k)
    basis = np.vstack([lat_ax, fwd_ax])
    hmg = np.vstack([-CAMERA_HEIGHT_M * (basis @ kinv), (up @ kinv).reshape(1, 3)])

    out_w, out_h = int(2 * LAT_HALF_M * PPM), int(FWD_MAX_M * PPM)
    a = np.array([[PPM, 0, LAT_HALF_M * PPM], [0, -PPM, out_h], [0, 0, 1.0]])
    m = a @ hmg
    # в BEV варпаем ТОЛЬКО пол: препятствия вертикальные, их IPM-проекция = смаз,
    # который перекрывает пол. Их реальное положение и так = край пола (его и меряем).
    floor = cls_map  # segment() уже вернул маску пола
    cls_bev = cv2.warpPerspective(floor, m, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=255)
    photo_bev = cv2.warpPerspective(und, m, (out_w, out_h), flags=cv2.INTER_LINEAR)

    res, mark = analyze_corridor(cls_bev == 1)
    bev = draw_map(cls_bev, res, mark)
    metrics = draw_metrics(res, out_h)

    # маска сегментации на оригинале — видно, что именно модель считает полом
    layer = und.copy()
    layer[cls_map == 1] = (144, 238, 144)
    masked = cv2.addWeighted(layer, 0.45, und, 0.55, 0)

    scale = out_h / und.shape[0]
    new_w = int(und.shape[1] * scale)
    und_r = cv2.resize(und, (new_w, out_h))
    masked_r = cv2.resize(masked, (new_w, out_h))
    return np.hstack([und_r, masked_r, photo_bev, bev, metrics]), res


def main():
    default = "data/route_reference/route_B/forward/front/front9.jpg"
    img_path = sys.argv[1] if len(sys.argv) > 1 else default
    if not Path(img_path).exists():
        print(f"[ERROR] нет файла: {img_path}")
        return
    intr = np.load(INTRINSICS)
    panel, res = process(cv2.imread(img_path), intr["K"], intr["dist"])
    print(json.dumps(res, ensure_ascii=False, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    cv2.imwrite(str(OUT_DIR / f"{stem}_edge.jpg"), panel)
    print(f"[INFO] сохранено: {OUT_DIR}/{stem}_edge.jpg  (оригинал | маска | вид сверху | карта | метры)")


if __name__ == "__main__":
    main()
