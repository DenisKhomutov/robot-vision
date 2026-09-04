"""Align a COLMAP reconstruction to a reference through shared image centers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shared", type=int)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")

    reference = pycolmap.Reconstruction(str(args.reference))
    reconstruction = pycolmap.Reconstruction(str(args.input))
    reference_centers = {
        image.name: np.asarray(image.projection_center())
        for image in reference.images.values()
    }
    input_centers = {
        image.name: np.asarray(image.projection_center())
        for image in reconstruction.images.values()
    }
    names = sorted(reference_centers.keys() & input_centers.keys())
    if args.max_shared is not None:
        if args.max_shared < 3:
            raise SystemExit("--max-shared must be at least 3")
        names = names[:args.max_shared]
    if len(names) < 3:
        raise SystemExit(f"only {len(names)} shared images")

    source = np.asarray([input_centers[name] for name in names])
    target = np.asarray([reference_centers[name] for name in names])
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(names)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    variance = np.sum(source_centered * source_centered) / len(names)
    scale = np.trace(np.diag(singular) @ sign) / variance
    translation = target_mean - scale * rotation @ source_mean

    fitted = (scale * (rotation @ source.T)).T + translation
    residuals = np.linalg.norm(fitted - target, axis=1)
    extent = np.linalg.norm(target.max(0) - target.min(0))
    rms = float(np.sqrt(np.mean(residuals**2)))

    reconstruction.transform(pycolmap.Sim3d(np.c_[scale * rotation, translation]))
    args.output.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(args.output))
    print(f"shared={len(names)}")
    print(f"scale={scale:.12g}")
    print(f"rms={rms:.12g}")
    print(f"p90={np.quantile(residuals, 0.9):.12g}")
    print(f"max={residuals.max():.12g}")
    print(f"extent={extent:.12g}")
    print(f"rms_percent={100 * rms / extent:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
