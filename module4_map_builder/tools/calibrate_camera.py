"""Calibrate a pinhole camera from numbered chessboard photographs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate camera intrinsics and OpenCV distortion")
    parser.add_argument(
        "--images",
        type=Path,
        default=ROOT / "data" / "calibration" / "images",
    )
    parser.add_argument("--cols", type=int, default=9, help="Internal chessboard corners horizontally")
    parser.add_argument("--rows", type=int, default=6, help="Internal chessboard corners vertically")
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--max-view-error", type=float, default=1.0)
    parser.add_argument("--min-images", type=int, default=15)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "configs" / "camera_1280x720_calibrated",
    )
    return parser.parse_args()


def calibrate(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    return cv2.calibrateCamera(object_points, image_points, image_size, None, None)


def view_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
) -> list[float]:
    errors = []
    for obj, observed, rotation, translation in zip(
        object_points, image_points, rotations, translations, strict=True
    ):
        projected, _ = cv2.projectPoints(obj, rotation, translation, camera_matrix, distortion)
        delta = projected.reshape(-1, 2) - observed.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return errors


def main() -> int:
    args = parse_args()
    if args.square_mm <= 0:
        raise SystemExit("--square-mm must be positive")

    template = np.zeros((args.rows * args.cols, 3), dtype=np.float32)
    template[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    template[:, :2] *= args.square_mm

    names: list[str] = []
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    rejected_detection: list[str] = []
    image_size: tuple[int, int] | None = None

    paths = sorted(args.images.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"No JPEG images found in {args.images}")

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            rejected_detection.append(path.name)
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise SystemExit(f"Mixed image sizes: {path.name} is {size}, expected {image_size}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(
            gray,
            (args.cols, args.rows),
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            rejected_detection.append(path.name)
            continue
        names.append(path.name)
        object_points.append(template.copy())
        image_points.append(corners.astype(np.float32))

    if image_size is None or len(names) < args.min_images:
        raise SystemExit(f"Only {len(names)} usable images; at least {args.min_images} required")

    rejected_error: list[dict[str, float | str]] = []
    while True:
        rms, camera_matrix, distortion, rotations, translations = calibrate(
            object_points, image_points, image_size
        )
        errors = view_errors(
            object_points, image_points, camera_matrix, distortion, rotations, translations
        )
        worst_index = int(np.argmax(errors))
        if errors[worst_index] <= args.max_view_error or len(names) <= args.min_images:
            break
        rejected_error.append({"name": names[worst_index], "error_px": errors[worst_index]})
        for values in (names, object_points, image_points):
            values.pop(worst_index)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    coefficients = distortion.reshape(-1).tolist()
    k1, k2, p1, p2, k3 = (coefficients + [0.0] * 5)[:5]

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = prefix.with_suffix(".txt")
    yaml_path = prefix.with_suffix(".yaml")
    json_path = prefix.with_suffix(".json")

    # DPVO's stream reader accepts OpenCV coefficients after fx fy cx cy.
    np.savetxt(txt_path, np.array([[fx, fy, cx, cy, k1, k2, p1, p2, k3]]), fmt="%.12g")
    yaml_data = {
        "camera": {
            "width": image_size[0],
            "height": image_size[1],
            "model": "opencv_pinhole",
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion": {"k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3},
        },
        "chessboard": {"internal_corners": [args.cols, args.rows], "square_mm": args.square_mm},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

    report = {
        "rms_px": float(rms),
        "median_view_error_px": float(np.median(errors)),
        "max_view_error_px": float(max(errors)),
        "used_images": names,
        "rejected_detection": rejected_detection,
        "rejected_error": rejected_error,
        "camera_matrix": camera_matrix.tolist(),
        "distortion": [k1, k2, p1, p2, k3],
        "dpvo_calibration": str(txt_path),
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"used={len(names)} detection_failed={len(rejected_detection)} outliers={len(rejected_error)}")
    print(f"rms={rms:.4f}px median_view={np.median(errors):.4f}px max_view={max(errors):.4f}px")
    print(f"fx={fx:.6f} fy={fy:.6f} cx={cx:.6f} cy={cy:.6f}")
    print(f"dist={k1:.8f} {k2:.8f} {p1:.8f} {p2:.8f} {k3:.8f}")
    print(f"DPVO: {txt_path}")
    print(f"report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
