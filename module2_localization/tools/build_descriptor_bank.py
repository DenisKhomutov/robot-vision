"""Build an ALIKED descriptor bank for an already reconstructed map."""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--kpts", type=int, default=4096)
    ap.add_argument("--det-threshold", type=float, default=0.04)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "core"))
    from model_weights import configure_local_model_weights
    from lightglue import ALIKED
    from lightglue.utils import load_image

    configure_local_model_weights()
    work = ROOT / "maps" / args.map
    image_dir = ROOT / "data" / args.images
    rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
    name_of = {image.image_id: image.name for image in rec.images.values()}

    refs = defaultdict(list)
    xyz = []
    for owner, point in enumerate(rec.points3D.values()):
        xyz.append(point.xyz)
        for element in point.track.elements[:12]:
            refs[name_of[element.image_id]].append((owner, element.point2D_idx))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ALIKED(max_num_keypoints=args.kpts,
                       detection_threshold=args.det_threshold).eval().to(device)
    descriptors, owners = [], []
    names = sorted(refs)
    started = time.perf_counter()
    for index, name in enumerate(names, 1):
        with torch.inference_mode():
            features = extractor.extract(load_image(str(image_dir / name)).to(device))
        desc = features["descriptors"][0].cpu().numpy()
        for owner, point2d_index in refs[name]:
            if point2d_index < len(desc):
                descriptors.append(desc[point2d_index])
                owners.append(owner)
        if index % 100 == 0:
            elapsed = time.perf_counter() - started
            print(f"[банк] {index}/{len(names)} | {elapsed:.0f}с", flush=True)

    if not descriptors or not xyz:
        print("[банк] нет дескрипторов или 3D-точек")
        return 1
    np.savez_compressed(work / "aliked_bank.npz",
                        desc=np.stack(descriptors).astype(np.float16),
                        owner=np.asarray(owners, np.int32),
                        xyz=np.stack(xyz))
    print(f"[банк] готово: {len(xyz)} точек, {len(descriptors)} дескрипторов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
