"""CLI вместо ручки POST /segmentation/segment — та же логика, без сервера.

Запуск с терминала:
    uv run --no-sync python module3_segmentation/tools/segment.py <фото...>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image
from module3_segmentation import init_models
from module3_segmentation.services.segmentation_service import analyze


def main():
    ap = argparse.ArgumentParser(description="Сегментация сцены")
    ap.add_argument("images", nargs="+", help="пути к изображениям")
    args = ap.parse_args()
    init_models()
    for p in args.images:
        r = analyze(Image.open(p))
        print(f"{Path(p).name}: {json.dumps(r, ensure_ascii=False, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
