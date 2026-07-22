import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[1]
VIEW = 1600


def observations(rec, im):
    """Точки кадра, у которых есть 3D: пиксель -> xyz."""
    pts, xyz = [], []
    for p2 in im.points2D:
        if p2.has_point3D():
            pts.append(p2.xy)
            xyz.append(rec.points3D[p2.point3D_id].xyz)
    return np.array(pts), np.array(xyz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="map_office_ref")
    ap.add_argument("--images", default="frames_office")
    ap.add_argument("--frame", default=None, help="имя кадра, напр. 00123.jpg")
    ap.add_argument("--dist", type=float, default=None, help="истинное расстояние в МЕТРАХ")
    ap.add_argument("--p1", default=None, help="x,y без окна")
    ap.add_argument("--p2", default=None, help="x,y без окна")
    ap.add_argument("--list", action="store_true", help="показать кадры с числом 3D-точек")
    ap.add_argument("--save", action="store_true", help="записать масштаб в карту")
    ap.add_argument("--max-residual", type=float, default=5.0, help="макс. промах клика, px")
    args = ap.parse_args()

    work = ROOT / "maps" / args.map
    rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
    by_name = {im.name: im for im in rec.images.values()}

    if args.list or not args.frame:
        rows = sorted(((sum(1 for p in im.points2D if p.has_point3D()), n)
                       for n, im in by_name.items()), reverse=True)
        print(f"{'кадр':<14}{'3D-точек':>10}")
        for c, n in rows[:25]:
            print(f"{n:<14}{c:>10}")
        print(f"\nвсего кадров {len(rows)}; запусти с --frame <имя>")
        return 0

    im = by_name.get(args.frame)
    if im is None:
        print(f"нет кадра {args.frame}")
        return 1
    img = cv2.imread(str(ROOT / "data" / args.images / args.frame))
    pts, xyz = observations(rec, im)
    print(f"{args.frame}: {len(pts)} точек с 3D")

    if args.p1 and args.p2:
        clicks = [np.array([float(v) for v in args.p1.split(",")]),
                  np.array([float(v) for v in args.p2.split(",")])]
    else:
        s = min(VIEW / img.shape[1], 1.0)
        disp0 = cv2.resize(img, None, fx=s, fy=s)
        for p in pts * s:
            cv2.circle(disp0, tuple(p.astype(int)), 2, (0, 220, 255), -1)
        clicks = []
        disp = disp0.copy()

        def on_click(ev, x, y, flags, _):
            if ev != cv2.EVENT_LBUTTONDOWN or len(clicks) >= 2:
                return
            q = np.array([x, y]) / s
            j = int(np.argmin(np.linalg.norm(pts - q, axis=1)))
            clicks.append(pts[j])
            cv2.circle(disp, tuple((pts[j] * s).astype(int)), 8, (60, 60, 255), 2)
            if len(clicks) == 2:
                cv2.line(disp, tuple((clicks[0] * s).astype(int)),
                         tuple((clicks[1] * s).astype(int)), (60, 60, 255), 2)
            cv2.imshow("scale", disp)

        cv2.namedWindow("scale")
        cv2.setMouseCallback("scale", on_click)
        cv2.putText(disp, "klikni 2 tochki (zheltye = est 3D), potom Enter",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("scale", disp)
        while len(clicks) < 2 or cv2.waitKey(20) not in (13, 10, 27):
            if cv2.waitKey(20) == 27:
                break
        cv2.destroyAllWindows()
        if len(clicks) < 2:
            print("отменено")
            return 1

    idx = [int(np.argmin(np.linalg.norm(pts - c, axis=1))) for c in clicks]
    res = [float(np.linalg.norm(pts[i] - c)) for i, c in zip(idx, clicks)]
    a, b = xyz[idx[0]], xyz[idx[1]]
    d_map = float(np.linalg.norm(a - b))
    print(f"\nточка 1: пиксель {pts[idx[0]].round(1)}  3D {a.round(3)}  (промах {res[0]:.1f}px)")
    print(f"точка 2: пиксель {pts[idx[1]].round(1)}  3D {b.round(3)}  (промах {res[1]:.1f}px)")
    print(f"расстояние в карте: {d_map:.4f} ед.")

    # промах больше нескольких пикселей = координаты не от этого кадра либо клик мимо;
    # считать масштаб по чужим точкам нельзя, это молча даёт правдоподобное враньё
    if max(res) > args.max_residual:
        print(f"\nОТКАЗ: промах {max(res):.1f}px > {args.max_residual}px. "
              f"Скорее всего пиксели сняты с ДРУГОГО кадра — они привязаны к кадру.")
        return 1

    # доля вдоль силы тяжести: для замеров высоты должна быть близка к 1
    ims = sorted(rec.images.values(), key=lambda i: i.name)
    g = np.mean([i.cam_from_world().rotation.matrix().T @ np.array([0, 1.0, 0]) for i in ims], 0)
    g /= np.linalg.norm(g)
    vert = abs(float(np.dot(a - b, g))) / max(d_map, 1e-9)
    print(f"направление: {vert * 100:.0f}% вертикаль, {np.sqrt(max(1 - vert**2, 0)) * 100:.0f}% горизонталь")

    if args.dist is None:
        print("\nчтобы получить масштаб, повтори командой целиком:")
        print(f"  uv run --no-sync python module2_localization/tools/measure_scale.py "
              f"--map {args.map} --frame {args.frame} \\\n"
              f"      --p1 {pts[idx[0]][0]:.1f},{pts[idx[0]][1]:.1f} "
              f"--p2 {pts[idx[1]][0]:.1f},{pts[idx[1]][1]:.1f} --dist <метры> --save")
        return 0

    scale = args.dist / d_map
    print(f"истинное: {args.dist:.3f} м  ->  МАСШТАБ = {scale:.4f} м/ед.")
    r = np.array([(-i.cam_from_world().rotation.matrix().T @ i.cam_from_world().translation)
                  for i in sorted(rec.images.values(), key=lambda i: i.name)])
    length = float(np.sum(np.linalg.norm(np.diff(r, axis=0), axis=1)))
    print(f"длина маршрута: {length:.2f} ед. = {length * scale:.2f} м")
    print(f"средний шаг между узлами: {length / (len(r) - 1) * scale * 100:.1f} см")

    if args.save:
        f = work / "scale.json"
        f.write_text(json.dumps({"scale_m_per_unit": scale, "frame": args.frame,
                                 "d_map": d_map, "d_real_m": args.dist,
                                 "p1": pts[idx[0]].tolist(), "p2": pts[idx[1]].tolist()}, indent=2))
        print(f"записано -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
