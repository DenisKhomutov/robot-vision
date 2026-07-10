import re
from pathlib import Path

from loguru import logger

base = Path("data/route_reference")


def main() -> None:
    for route in ["route_B", "route_C"]:
        for cam_path in sorted((base / route / "backward").iterdir()):
            camera = cam_path.name
            files = sorted(
                [f for f in cam_path.iterdir() if re.match(rf"^{camera}\d+\.jpg$", f.name)],
                key=lambda p: int(m.group()) if (m := re.search(r"\d+", p.name)) else 0,
            )
            n = len(files)
            if n == 0:
                logger.warning(f"Нет файлов: {cam_path}")
                continue

            for i, f in enumerate(files):
                f.rename(cam_path / f"_tmp{i + 1}.jpg")

            for i in range(n):
                (cam_path / f"_tmp{i + 1}.jpg").rename(cam_path / f"{camera}{n - i}.jpg")

            logger.success(f"Развёрнуто: {route}/backward/{camera} ({n} файлов)")


if __name__ == "__main__":
    main()
