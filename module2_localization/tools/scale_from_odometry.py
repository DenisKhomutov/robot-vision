"""Метрический масштаб карты из ОДОМЕТРИИ (энкодеры робота) вместо кликанья по пикселям.

Робот проехал маршрут, энкодеры дали реальную длину пути (encoder-log.jsonl,
selected.distance_m). Карта построена из того же проезда — её длина в единицах карты
известна. scale = длина_метры / длина_единицы. Пишет scale.json в карту.

    uv run --no-sync python module2_localization/tools/scale_from_odometry.py \
        --map <карта> --log map_for_office/rec/encoder-log.jsonl [--save]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[1]


def odometry_distance(log_path: Path) -> float:
    """Итоговая пройденная дистанция (метры) из последней валидной строки лога."""
    dist = 0.0
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)["encoder"]["selected"]["distance_m"]
        except (json.JSONDecodeError, KeyError):
            continue
        if isinstance(d, (int, float)):
            dist = max(dist, float(d))   # монотонно растёт; берём максимум
    return dist


def route_length_units(map_name: str) -> float:
    rec = pycolmap.Reconstruction(str(ROOT / "maps" / map_name / "sparse" / "0"))
    order = sorted(rec.images.values(), key=lambda i: i.name)
    P = np.array([(-i.cam_from_world().rotation.matrix().T @ i.cam_from_world().translation)
                  for i in order])
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--log", required=True, help="encoder-log.jsonl")
    ap.add_argument("--save", action="store_true", help="записать scale.json в карту")
    args = ap.parse_args()

    dist_m = odometry_distance(Path(args.log))
    arc_u = route_length_units(args.map)
    if arc_u < 1e-6:
        print("длина маршрута ~0, карта пустая?")
        return 1
    scale = dist_m / arc_u
    print(f"одометрия: {dist_m:.2f} м")
    print(f"маршрут карты: {arc_u:.3f} ед.")
    print(f"МАСШТАБ = {scale:.4f} м/ед.")
    print(f"средний шаг между узлами: {arc_u / 1:.3f} ед.  (проверь на глаз потолок-пол)")

    if args.save:
        f = ROOT / "maps" / args.map / "scale.json"
        f.write_text(json.dumps({"scale_m_per_unit": scale, "source": "odometry",
                                 "distance_m": dist_m, "arc_units": arc_u}, indent=2))
        print(f"записано -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
