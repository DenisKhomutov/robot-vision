import time
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE0 = "experiment/images/current.jpg"
IMAGE1 = "experiment/images/reference.jpg"
OUTDIR = "experiment/results_xfeat"

MAX_SIDE = 640
TOP_K = 4096
MAX_MATCHES = 0
MAX_DRAW = 25
ARROW_SCALE = 0.15
ARROW_COLOR = (0, 165, 255)
FORCE_CPU = True

USE_STAR = True

MIN_COSSIM = 0.82

HFOV_DEG = 91.8


def resize_keep_aspect(img_bgr: np.ndarray, max_side: int = MAX_SIDE, multiple: int = 8) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = (int(w * scale), int(h * scale)) if scale < 1.0 else (w, h)
    new_w = max(multiple, (new_w // multiple) * multiple)
    new_h = max(multiple, (new_h // multiple) * multiple)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_match_canvas(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    max_draw: int = MAX_DRAW,
) -> np.ndarray:
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0 : w0 + w1] = img1_bgr

    if len(pts0) == 0:
        return canvas

    n = min(len(pts0), max_draw)
    for idx in range(n):
        p0 = (int(pts0[idx][0]), int(pts0[idx][1]))
        p1 = (int(pts1[idx][0]) + w0, int(pts1[idx][1]))
        hue = int(180 * idx / max(n, 1)) 
        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.circle(canvas, p0, 3, color, -1)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

    return canvas


def draw_motion(img_current: np.ndarray, pts0: np.ndarray, pts1: np.ndarray, max_draw: int = MAX_DRAW) -> np.ndarray:
    canvas = img_current.copy()
    for idx in range(min(len(pts0), max_draw)):
        x0, y0 = float(pts0[idx][0]), float(pts0[idx][1])
        x1, y1 = float(pts1[idx][0]), float(pts1[idx][1])
        ex, ey = x0 - (x1 - x0) * ARROW_SCALE, y0 - (y1 - y0) * ARROW_SCALE
        cv2.arrowedLine(canvas, (int(x0), int(y0)), (int(ex), int(ey)), ARROW_COLOR, 1, cv2.LINE_AA, tipLength=0.3)
    return canvas


def similarity_heatmap(img0_bgr: np.ndarray, img1_bgr: np.ndarray, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    def overlay(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        acc = np.zeros((h, w), dtype=np.float32)
        for x, y in pts:
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                acc[yi, xi] += 1.0
        acc = cv2.GaussianBlur(acc, (0, 0), 15)
        peak = float(acc.max())
        if peak > 0:
            acc = acc / peak
        heat = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(img, 0.6, heat, 0.4, 0)

    return np.hstack([overlay(img0_bgr, pts0), overlay(img1_bgr, pts1)])


def ransac_inliers(pts0: np.ndarray, pts1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(pts0) < 8:
        return pts0, pts1
    try:
        _, inliers = cv2.findFundamentalMat(
            pts0, pts1, method=cv2.USAC_MAGSAC, ransacReprojThreshold=0.5, confidence=0.999, maxIters=10000
        )
        if inliers is None:
            return pts0, pts1
        mask = inliers.ravel().astype(bool)
        return pts0[mask], pts1[mask]
    except Exception as e:
        print(f"[WARN] RANSAC failed: {e}")
        return pts0, pts1


def estimate_basic_shift(pts0: np.ndarray, pts1: np.ndarray, w0: int, w1: int) -> dict:
    if len(pts0) == 0:
        return {"horizontal_shift_norm": None, "steering_hint_debug": "unknown"}

    shift = float(np.median(pts0[:, 0] / max(w0, 1) - pts1[:, 0] / max(w1, 1)))
    dead_zone = 0.05
    if shift > dead_zone:
        hint = "right"
    elif shift < -dead_zone:
        hint = "left"
    else:
        hint = "straight"

    yaw_deg = float(np.degrees(np.arctan(2.0 * shift * np.tan(np.radians(HFOV_DEG) / 2.0))))

    return {
        "horizontal_shift_norm": round(shift, 4),
        "yaw_deg_approx": round(yaw_deg, 1),
        "steering_hint_debug": hint,
    }


def main() -> None:
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] device = {device}")

    img0_original = cv2.imread(IMAGE0)
    img1_original = cv2.imread(IMAGE1)
    if img0_original is None:
        raise FileNotFoundError(f"Не удалось прочитать image0: {IMAGE0}")
    if img1_original is None:
        raise FileNotFoundError(f"Не удалось прочитать image1: {IMAGE1}")

    img0 = resize_keep_aspect(img0_original)
    img1 = resize_keep_aspect(img1_original)
    print(f"[INFO] image0: {img0.shape[1]}x{img0.shape[0]}, image1: {img1.shape[1]}x{img1.shape[0]}")

    t0 = time.perf_counter()
    xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=TOP_K)
    xfeat = xfeat.to(device).eval()
    t_load = time.perf_counter() - t0
    print(f"[INFO] XFeat загружен за {t_load:.2f}с")

    t0 = time.perf_counter()
    with torch.inference_mode():
        if USE_STAR:
            pts0, pts1 = xfeat.match_xfeat_star(img0, img1, top_k=TOP_K)
        else:
            pts0, pts1 = xfeat.match_xfeat(img0, img1, top_k=TOP_K, min_cossim=MIN_COSSIM)
    t_match = time.perf_counter() - t0
    print(f"[INFO] матчинг: {t_match * 1000:.0f}мс, matches: {len(pts0)}")

    pts0 = np.asarray(pts0)
    pts1 = np.asarray(pts1)

    if MAX_MATCHES > 0 and len(pts0) > MAX_MATCHES:
        pts0, pts1 = pts0[:MAX_MATCHES], pts1[:MAX_MATCHES]

    pts0_in, pts1_in = ransac_inliers(pts0, pts1)
    print(f"[INFO] inlier matches: {len(pts0_in)}")

    cv2.imwrite(str(outdir / "xfeat_matches_all.jpg"), create_match_canvas(img0, img1, pts0, pts1))
    cv2.imwrite(str(outdir / "xfeat_matches_inliers.jpg"), create_match_canvas(img0, img1, pts0_in, pts1_in))
    cv2.imwrite(str(outdir / "xfeat_motion.jpg"), draw_motion(img0, pts0_in, pts1_in))
    cv2.imwrite(str(outdir / "xfeat_similarity.jpg"), similarity_heatmap(img0, img1, pts0_in, pts1_in))

    print(f"[RESULT] {estimate_basic_shift(pts0_in, pts1_in, img0.shape[1], img1.shape[1])}")
    print(f"[TIMING] load={t_load:.2f}s match={t_match * 1000:.0f}ms")


if __name__ == "__main__":
    main()
