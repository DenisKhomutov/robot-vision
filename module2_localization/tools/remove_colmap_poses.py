"""Remove selected registered poses from a COLMAP model without changing DB IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pycolmap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")

    reconstruction = pycolmap.Reconstruction(str(args.input))
    removed: list[dict[str, int | str]] = []
    for image_id in sorted(set(args.image_ids)):
        if image_id not in reconstruction.images:
            raise SystemExit(f"image_id is absent: {image_id}")
        image = reconstruction.images[image_id]
        if not image.has_pose:
            raise SystemExit(f"image_id is not registered: {image_id}")
        removed.append({"image_id": image_id, "name": image.name})
        frame_id = image.frame_id
        reconstruction.deregister_frame(frame_id)
        del reconstruction.images[image_id]
        del reconstruction.frames[frame_id]

    args.output.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(args.output))
    ordered = sorted(reconstruction.images.values(), key=lambda image: image.name)
    manifest = {
        "source": str(args.input.resolve()),
        "removed": removed,
        "registered_images": reconstruction.num_reg_images(),
        "logical_nodes": [
            {"node_id": node_id, "image_id": image.image_id, "name": image.name}
            for node_id, image in enumerate(ordered, 1)
        ],
    }
    (args.output / "pose_removal_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print("removed=" + ",".join(f"{item['image_id']}:{item['name']}" for item in removed))
    print(f"registered_images={reconstruction.num_reg_images()}")
    print(f"points3D={reconstruction.num_points3D()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
