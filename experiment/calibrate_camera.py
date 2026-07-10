"""
Калибровка камеры по шахматке -> матрица K и коэффициенты дисторсии.

Как пользоваться:
1. Распечатай шахматную доску (9x6 ВНУТРЕННИХ углов), замерь размер клетки в мм.
2. Сними 15-25 фото этой доски камерой робота с разных ракурсов
   (доска должна попадать и в центр, и в углы кадра).
3. Сложи фото в CALIB_DIR, выстави BOARD_COLS/ROWS/SQUARE_MM, запусти:
       uv run python experiment/calibrate_camera.py
4. Результат сохранится в OUT_NPZ (K, dist) + напечатается в консоль.

Важно: K привязана к РАЗРЕШЕНИЮ, на котором снимали. Если в матчинге кадр
ресайзишь до 640 по стороне — K надо масштабировать (см. scale_intrinsics).
"""

from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = "experiment/calib_images"
OUT_NPZ = "experiment/camera_intrinsics.npz"
BOARD_COLS = 9  # число ВНУТРЕННИХ углов по горизонтали
BOARD_ROWS = 6  # число ВНУТРЕННИХ углов по вертикали
SQUARE_MM = 25.0  # размер клетки в мм (реальный, линейкой)


def scale_intrinsics(k: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    """Пересчёт K при ресайзе кадра (fx,fy,cx,cy масштабируются линейно)."""
    sx = to_wh[0] / from_wh[0]
    sy = to_wh[1] / from_wh[1]
    k2 = k.copy()
    k2[0, 0] *= sx
    k2[0, 2] *= sx
    k2[1, 1] *= sy
    k2[1, 2] *= sy
    return k2


def main() -> None:
    pattern = (BOARD_COLS, BOARD_ROWS)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    objp = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_MM

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    paths = sorted(p for ext in ("*.jpg", "*.png", "*.jpeg") for p in Path(CALIB_DIR).glob(ext))
    if not paths:
        print(f"[ERROR] Нет фото в {CALIB_DIR}. Сложи туда снимки шахматки.")
        return

    for path in paths:
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (w, h)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            print(f"[SKIP] углы не найдены: {path.name}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        print(f"[OK] {path.name}")

    if len(obj_points) < 5:
        print(f"[ERROR] Слишком мало удачных кадров ({len(obj_points)}). Нужно >= 10.")
        return

    rms, k, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    print(f"\n[RESULT] удачных кадров: {len(obj_points)}, разрешение: {image_size}")
    print(f"[RESULT] reprojection error (RMS, px): {rms:.3f}  (хорошо < 0.5)")
    print(f"[RESULT] K =\n{k}")
    print(f"[RESULT] dist = {dist.ravel()}")

    np.savez(OUT_NPZ, K=k, dist=dist, image_size=np.array(image_size))
    print(f"\n[INFO] сохранено: {OUT_NPZ}")


if __name__ == "__main__":
    main()
