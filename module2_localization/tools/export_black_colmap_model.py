import argparse
from pathlib import Path

import numpy as np
import pycolmap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    reconstruction = pycolmap.Reconstruction(str(args.input))
    black = np.zeros(3, dtype=np.uint8)
    for point in reconstruction.points3D.values():
        point.color = black
    args.output.mkdir(parents=True, exist_ok=True)
    reconstruction.write_binary(str(args.output))
    print(f"images={len(reconstruction.images)} points={len(reconstruction.points3D)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
