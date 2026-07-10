from pathlib import Path

from loguru import logger

PHOTO_DIR = Path("data/random")
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    images = sorted(
        (p for p in PHOTO_DIR.iterdir() if p.suffix.lower() in SUPPORTED),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )

    if not images:
        logger.error(f"Изображения не найдены в {PHOTO_DIR}")
        return

    logger.info(f"Найдено {len(images)} файлов в {PHOTO_DIR}")

    # two-pass: temp names first to avoid collisions (e.g. 10.jpg → 2.jpg when 2.jpg exists)
    tmp_paths = []
    for idx, path in enumerate(images, start=55):
        tmp = path.with_name(f"__tmp_{idx}{path.suffix.lower()}")
        path.rename(tmp)
        tmp_paths.append((tmp, idx, path.suffix.lower()))

    for tmp, idx, suffix in tmp_paths:
        final = tmp.with_name(f"{idx}{suffix}")
        tmp.rename(final)
        logger.info(f"→  {final.name}")

    logger.success("Переименование завершено")


if __name__ == "__main__":
    main()
