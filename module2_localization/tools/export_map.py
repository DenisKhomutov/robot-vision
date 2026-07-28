"""Экспорт карты в runtime.npz: всё, что нужно рантайму, без pycolmap.

На роботе не остаётся ни COLMAP, ни Ceres — только numpy и OpenCV. Запускается
на машине разработки, где pycolmap есть:

    uv run --no-sync python module2_localization/tools/export_map.py --map map_rig3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def intrinsics(cam):
    """Камера COLMAP -> (fx, fy, cx, cy) и коэффициенты дисторсии в порядке OpenCV."""
    p, name = list(cam.params), cam.model.name
    if name == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return (f, f, cx, cy), [0.0, 0.0, 0.0, 0.0]
    if name == "PINHOLE":
        fx, fy, cx, cy = p
        return (fx, fy, cx, cy), [0.0, 0.0, 0.0, 0.0]
    if name == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        return (f, f, cx, cy), [k, 0.0, 0.0, 0.0]
    if name == "RADIAL":
        f, cx, cy, k1, k2 = p
        return (f, f, cx, cy), [k1, k2, 0.0, 0.0]
    if name == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        return (fx, fy, cx, cy), [k1, k2, p1, p2]
    raise SystemExit(f"модель камеры {name} пока не поддержана экспортом")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    args = ap.parse_args()

    work = ROOT / "maps" / args.map
    rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
    cam = list(rec.cameras.values())[0]
    (fx, fy, cx, cy), dist = intrinsics(cam)

    by_name = {im.name: im for im in rec.images.values()}
    names = sorted(by_name)
    pos, fwd = [], []
    for n in names:
        R = by_name[n].cam_from_world().rotation.matrix()
        pos.append(-R.T @ by_name[n].cam_from_world().translation)
        fwd.append(R.T @ np.array([0, 0, 1.0]))
    pos = np.array(pos, np.float64)
    fwd = np.array(fwd, np.float64)
    points = np.array([p.xyz for p in rec.points3D.values()], np.float64)

    # канон отрисовки: GLOMAP разворачивает карту в XZ на случайный угол.
    # поворачиваем вокруг вертикали (Y) так, чтобы разрыв петли (старт/финиш) смотрел ВВЕРХ
    # (+Z в viz = верх). Команды относительны -> поворот на них не влияет.
    ctr = pos[:, [0, 2]].mean(0)
    d = (pos[0, [0, 2]] + pos[-1, [0, 2]]) / 2 - ctr   # центр -> старт/финиш
    a = np.pi / 2 - np.arctan2(d[1], d[0])             # довернуть до +Z
    c, s = np.cos(a), np.sin(a)
    Ry = np.array([[c, 0, -s], [0, 1.0, 0], [s, 0, c]])
    pos = pos @ Ry.T
    fwd = fwd @ Ry.T
    points = points @ Ry.T

    out = work / "runtime.npz"
    np.savez_compressed(
        out,
        names=np.array(names),
        pos=pos,
        fwd=fwd,
        K=np.array([fx, fy, cx, cy], np.float64),
        dist=np.array(dist, np.float64),
        size=np.array([cam.width, cam.height], np.int64),
        points=points,
        align=Ry,   # тем же поворотом локализатор крутит точки PnP (иначе поза в старой системе)
    )
    print(f"[экспорт] {args.map}: {len(names)} кадров, {rec.num_points3D()} точек, "
          f"камера {cam.model.name} {cam.width}x{cam.height} -> {out.name} "
          f"({out.stat().st_size / 1e6:.1f} МБ)")


main()
