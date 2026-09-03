import argparse
from pathlib import Path

from ..core.aliked_localizer import ALIKEDLocalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", nargs="+")
    parser.add_argument("--map", default="map_office_ref")
    parser.add_argument("--route-range", default=None, help="эталон только по кадрам a:b (напр. 0:525)")
    args = parser.parse_args()
    route_range = tuple(int(x) for x in args.route_range.split(":")) if args.route_range else None
    localizer = ALIKEDLocalizer(args.map, route_range=route_range)
    print(f"\n{'фото':<20}{'инл.':>6}{'пар':>7}{'узел':>6}{'до линии':>10}{'азимут':>9}{'команда':>10}{'мс':>7}")
    for photo in args.photos:
        result = localizer.locate(Path(photo))
        name = Path(photo).name[:18]
        if result is None:
            print(f"{name:<20} не читается")
            continue
        elapsed_ms = result["t_ext"] + result["t_match"] + result.get("t_pnp", 0)
        if not result["ok"]:
            print(f"{name:<20} НЕ НАЙДЕНО: {result['reason']:<40}{elapsed_ms:>7.0f}")
            continue
        distance = result.get("dist_to_route_m", result["dist_to_route"])
        unit = "м" if "dist_to_route_m" in result else "е"
        print(
            f"{name:<20}{result['inliers']:>6}{result['n_pairs']:>7}{result['node']:>6}"
            f"{distance:>9.2f}{unit}{result['bearing_deg']:>+8.1f}°"
            f"{result['move_type']:>10}{elapsed_ms:>7.0f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
