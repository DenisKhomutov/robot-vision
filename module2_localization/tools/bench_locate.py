"""Замер t_ext/t_match/t_pnp локализатора на реальной карте, БЕЗ требования к
совпадению сцены с картой — годится любой кадр (даже не с маршрута), тайминги
extract/match считаются до PnP и от содержимого сцены не зависят по существу.

    uv run --no-sync python -m module2_localization.tools.bench_locate \\
        --map 2/rear_full --image любое_фото.jpg --n 15
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from localizer import AlikedLocalizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--image", required=True, help="любой кадр той же камеры/разрешения")
    ap.add_argument("--n", type=int, default=15, help="число прогонов (первый — прогрев, не считается)")
    ap.add_argument("--win-nodes", type=int, default=None, help="переопределить окно матчинга (по умолч. из карты)")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"не читается {args.image}")
        return 1

    kw = {}
    if args.win_nodes is not None:
        kw["win_nodes"] = args.win_nodes
    loc = AlikedLocalizer(args.map, **kw)

    times = []
    for i in range(args.n):
        r = loc.locate(frame)
        row = (r.get("t_ext", 0), r.get("t_match", 0), r.get("t_pnp", 0))
        if i == 0:
            print(f"  прогрев: ext {row[0]:.0f}мс match {row[1]:.0f}мс pnp {row[2]:.0f}мс (не считается)")
            continue
        times.append(row)
        print(f"  [{i}/{args.n-1}] ext {row[0]:.0f}мс  match {row[1]:.0f}мс  "
              f"pnp {row[2]:.0f}мс  ok={r.get('ok')}")

    a = np.array(times)
    print(f"\nбанк: {len(loc.owner)} дескрипторов, окно ±{loc.win_nodes} узлов "
          f"({'весь банк' if loc.win_nodes == 0 else 'ограничено'})")
    print(f"ext:   медиана {np.median(a[:,0]):.0f}мс  мин {a[:,0].min():.0f}  макс {a[:,0].max():.0f}")
    print(f"match: медиана {np.median(a[:,1]):.0f}мс  мин {a[:,1].min():.0f}  макс {a[:,1].max():.0f}")
    print(f"pnp:   медиана {np.median(a[:,2]):.0f}мс  мин {a[:,2].min():.0f}  макс {a[:,2].max():.0f}")
    total = a.sum(1)
    print(f"ИТОГО: медиана {np.median(total):.0f}мс/кадр")
    return 0


if __name__ == "__main__":
    sys.exit(main())
