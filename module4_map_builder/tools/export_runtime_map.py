"""Export a DPVO+ALIKED hybrid map in module2 runtime format."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def quat_to_rotation(qx, qy, qz, qw):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = (v / norm for v in (qw, qx, qy, qz))
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def load_poses(path):
    poses = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        timestamp = int(round(float(values[0])))
        center = np.asarray(values[1:4], dtype=np.float64)
        rotation = quat_to_rotation(*map(float, values[4:8]))
        poses[timestamp] = (center, rotation[:, 2])
    return poses


def main():
    args = parse_args()
    bank = np.load(args.bank, allow_pickle=False)
    poses = load_poses(args.trajectory)
    timestamps = bank["tstamps"].astype(np.int64)
    missing = [int(t) for t in timestamps if int(t) not in poses]
    if missing:
        raise SystemExit(f"trajectory lacks timestamps: {missing[:5]}")

    names = bank["image_names"].astype(str)
    positions = np.stack([poses[int(t)][0] for t in timestamps])
    forwards = np.stack([poses[int(t)][1] for t in timestamps])
    if not np.allclose(positions, bank["route"], atol=1e-4):
        raise SystemExit("bank route and DPVO trajectory positions disagree")

    sample = cv2.imread(str(args.images / names[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise SystemExit(f"cannot read sample image: {args.images / names[0]}")
    height, width = sample.shape[:2]
    K = bank["K"].astype(np.float64)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "runtime.npz",
        names=names,
        pos=positions,
        fwd=forwards,
        K=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64),
        dist=bank["dist"].astype(np.float64),
        size=np.array([width, height], dtype=np.int64),
        points=bank["xyz"].astype(np.float32),
    )
    np.savez_compressed(
        args.out_dir / "aliked_bank.npz",
        desc=bank["desc"].astype(np.float16),
        owner=bank["owner"].astype(np.int32),
        xyz=bank["xyz"].astype(np.float32),
    )
    summary = {
        "format": "module2_runtime_map",
        "source_bank": str(args.bank.resolve()),
        "frames": len(names),
        "landmarks": int(len(bank["xyz"])),
        "descriptors": int(len(bank["desc"])),
        "image_size": [width, height],
        "runtime": str((args.out_dir / "runtime.npz").resolve()),
        "aliked_bank": str((args.out_dir / "aliked_bank.npz").resolve()),
    }
    (args.out_dir / "runtime_map.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
