from __future__ import annotations

import argparse
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    images = args.images.resolve()
    reconstruction = pycolmap.Reconstruction(str(model))

    resolved = {}
    missing = []
    for image_id, image in reconstruction.images.items():
        path = images / Path(image.name).name
        if not path.is_file():
            missing.append(str(path))
        resolved[image_id] = path
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(f"Missing {len(missing)} images:\n{preview}")

    for image_id, path in resolved.items():
        reconstruction.images[image_id].name = str(path)
    reconstruction.write_binary(str(model))

    project = model / "project.ini"
    existing = project.read_text(encoding="utf-8").splitlines() if project.exists() else []
    settings = [line for line in existing if not line.startswith("image_path=")]
    settings.append(f"image_path={images}")
    project.write_text("\n".join(settings) + "\n", encoding="utf-8")

    print(f"MODEL={model}")
    print(f"IMAGES={len(resolved)}")
    print(f"IMAGE_PATH={images}")


if __name__ == "__main__":
    main()
