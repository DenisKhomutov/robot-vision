"""CLI вместо ручки POST /traffic_light/detect — та же логика, без сервера.

Запуск с терминала:
    uv run --no-sync python module1_traffic_light/tools/detect.py <фото...>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image
from module1_traffic_light import init_models
from module1_traffic_light.services.traffic_light_service import analyze


def main():
    ap = argparse.ArgumentParser(description="Распознавание сигнала светофора")
    ap.add_argument("images", nargs="+", help="пути к изображениям")
    args = ap.parse_args()
    init_models()
    for p in args.images:
        r = analyze(Image.open(p))
        print(f"{Path(p).name}: {json.dumps(r, ensure_ascii=False, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
