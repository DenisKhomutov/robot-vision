import argparse
from pathlib import Path

import cv2
import kornia.feature as KF
import numpy as np
import torch


def resize_keep_aspect(img_bgr: np.ndarray, max_side: int = 640, multiple: int = 8) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    new_w, new_h = (int(w * scale), int(h * scale)) if scale < 1.0 else (w, h)
    new_w = max(multiple, (new_w // multiple) * multiple)
    new_h = max(multiple, (new_h // multiple) * multiple)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_gray_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(gray).float()[None, None].to(device) / 255.0


def create_match_canvas(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    confidences: np.ndarray | None,
    max_draw: int = 200,
) -> np.ndarray:
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_bgr
    canvas[:h1, w0 : w0 + w1] = img1_bgr

    if len(pts0) == 0:
        return canvas

    order = np.argsort(-confidences) if confidences is not None else np.arange(len(pts0))
    order = order[:max_draw]
    n = len(order)

    for i, idx in enumerate(order):
        p0 = (int(pts0[idx][0]), int(pts0[idx][1]))
        p1 = (int(pts1[idx][0]) + w0, int(pts1[idx][1]))
        hue = int(180 * i / max(n, 1))  # оттенок по спектру, S=V=255 → ярко и разнообразно
        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.circle(canvas, p0, 3, color, -1)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)

    return canvas


def draw_motion(
    img_current: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray | None,
    max_draw: int = 200,
) -> np.ndarray:
    canvas = img_current.copy()
    if len(pts0) == 0:
        return canvas

    order = np.argsort(-conf) if conf is not None else np.arange(len(pts0))
    order = order[:max_draw]
    color = (0, 165, 255)
    scale = 0.5

    for idx in order:
        x0, y0 = float(pts0[idx][0]), float(pts0[idx][1])
        x1, y1 = float(pts1[idx][0]), float(pts1[idx][1])
        ex, ey = x0 - (x1 - x0) * scale, y0 - (y1 - y0) * scale
        cv2.arrowedLine(canvas, (int(x0), int(y0)), (int(ex), int(ey)), color, 1, cv2.LINE_AA, tipLength=0.3)

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


def estimate_basic_shift(pts0: np.ndarray, pts1: np.ndarray, w0: int, w1: int) -> dict:
    if len(pts0) == 0:
        return {"horizontal_shift_norm": None, "steering_hint_debug": "unknown"}

    shift = float(np.median(pts0[:, 0] / max(w0, 1) - pts1[:, 0] / max(w1, 1)))
    dead_zone = 0.04
    if shift > dead_zone:
        hint = "right"
    elif shift < -dead_zone:
        hint = "left"
    else:
        hint = "straight"
    return {"horizontal_shift_norm": shift, "steering_hint_debug": hint}


def load_loftr(weights: str, device: torch.device) -> KF.LoFTR:
    try:
        matcher = KF.LoFTR(pretrained=weights)
    except Exception as e:
        fallback = "indoor" if weights == "indoor_new" else "outdoor"
        print(f"[WARN] weights='{weights}' не загрузились ({e}), fallback='{fallback}'")
        matcher = KF.LoFTR(pretrained=fallback)
    return matcher.to(device).eval()


def filter_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray | None,
    min_conf: float,
    max_matches: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if conf is not None:
        mask = conf >= min_conf
        pts0, pts1, conf = pts0[mask], pts1[mask], conf[mask]
        if max_matches > 0 and len(conf) > max_matches:
            top = np.argsort(-conf)[:max_matches]
            pts0, pts1, conf = pts0[top], pts1[top], conf[top]
    return pts0, pts1, conf


def ransac_inliers(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(pts0) < 8:
        return pts0, pts1, conf
    try:
        _, inliers = cv2.findFundamentalMat(
            pts0, pts1, method=cv2.USAC_MAGSAC, ransacReprojThreshold=0.5, confidence=0.999, maxIters=10000
        )
        if inliers is None:
            return pts0, pts1, conf
        mask = inliers.ravel().astype(bool)
        return pts0[mask], pts1[mask], (conf[mask] if conf is not None else None)
    except Exception as e:
        print(f"[WARN] RANSAC failed: {e}")
        return pts0, pts1, conf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image0", required=True)
    parser.add_argument("--image1", required=True)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--weights", default="outdoor", choices=["indoor_new", "indoor", "outdoor"])
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--max-draw", type=int, default=200)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] device = {device}")

    img0_original = cv2.imread(args.image0)
    img1_original = cv2.imread(args.image1)
    if img0_original is None:
        raise FileNotFoundError(f"Не удалось прочитать image0: {args.image0}")
    if img1_original is None:
        raise FileNotFoundError(f"Не удалось прочитать image1: {args.image1}")

    img0 = resize_keep_aspect(img0_original, max_side=args.max_side)
    img1 = resize_keep_aspect(img1_original, max_side=args.max_side)

    matcher = load_loftr(args.weights, device)
    with torch.inference_mode():
        correspondences = matcher(
            {"image0": image_to_gray_tensor(img0, device), "image1": image_to_gray_tensor(img1, device)}
        )

    pts0 = correspondences["keypoints0"].detach().cpu().numpy()
    pts1 = correspondences["keypoints1"].detach().cpu().numpy()
    conf = correspondences.get("confidence")
    conf = conf.detach().cpu().numpy() if conf is not None else None

    pts0, pts1, conf = filter_matches(pts0, pts1, conf, args.min_conf, args.max_matches)
    print(f"[INFO] matches after filter: {len(pts0)}")

    pts0_in, pts1_in, conf_in = ransac_inliers(pts0, pts1, conf)
    print(f"[INFO] inlier matches: {len(pts0_in)}")

    cv2.imwrite(
        str(outdir / "loftr_matches_all.jpg"),
        create_match_canvas(img0, img1, pts0, pts1, conf, args.max_draw),
    )
    cv2.imwrite(
        str(outdir / "loftr_matches_inliers.jpg"),
        create_match_canvas(img0, img1, pts0_in, pts1_in, conf_in, args.max_draw),
    )
    cv2.imwrite(
        str(outdir / "loftr_motion.jpg"),
        draw_motion(img0, pts0_in, pts1_in, conf_in, args.max_draw),
    )
    cv2.imwrite(
        str(outdir / "loftr_similarity.jpg"),
        similarity_heatmap(img0, img1, pts0_in, pts1_in),
    )

    print(f"[RESULT] {estimate_basic_shift(pts0_in, pts1_in, img0.shape[1], img1.shape[1])}")


if __name__ == "__main__":
    main()
