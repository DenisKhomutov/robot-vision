from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAP_SIZE = 720
VIDEO_WIDTH = 1280
BAR_HEIGHT = 130


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--max-dist', type=float, default=5.0)
    return ap.parse_args()


def px_mapper(route, query):
    pts = route if len(query) == 0 else np.vstack([route, query])
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo -= 0.15 * span
    hi += 0.15 * span
    pad = 35
    def px(a):
        a = np.atleast_2d(a)
        q = (a - lo) / (hi - lo + 1e-9)
        return np.stack([pad + q[:,0]*(MAP_SIZE-2*pad), MAP_SIZE-pad-q[:,1]*(MAP_SIZE-2*pad)], 1).astype(int)
    return px


def main():
    args = parse_args()
    bank = np.load(args.bank, allow_pickle=True)
    route = bank['route'][:, [0, 2]].astype(float)
    rows = list(csv.DictReader(args.csv.open()))
    for r in rows:
        for k in ('frame','processed','ok','query_keypoints','pairs','inliers','nearest_node'):
            r[k] = int(float(r[k]))
        for k in ('dist_route','x','y','z','t_ext_ms','t_match_ms'):
            r[k] = float(r[k]) if r[k] not in ('nan','') else np.nan
    valid = [r for r in rows if r['ok'] and np.isfinite(r['x']) and np.isfinite(r['z']) and r['dist_route'] <= args.max_dist]
    query = np.array([[r['x'], r['z']] for r in valid], float) if valid else np.empty((0,2))
    px = px_mapper(route, query)
    route_px = px(route)

    base = np.full((MAP_SIZE, MAP_SIZE, 3), 18, np.uint8)
    for a,b in zip(route_px[:-1], route_px[1:]):
        cv2.line(base, tuple(a), tuple(b), (190,165,65), 2, cv2.LINE_AA)
    cv2.circle(base, tuple(route_px[0]), 7, (80,220,100), -1)
    cv2.circle(base, tuple(route_px[-1]), 7, (60,60,230), -1)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*'mp4v'), fps / args.stride, (MAP_SIZE+VIDEO_WIDTH, MAP_SIZE+BAR_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f'cannot open writer: {args.out}')

    trail = deque(maxlen=80)
    valid_i = 0
    src_idx = -1
    for row in rows:
        frame = None; ok = False
        while src_idx < row['frame']:
            ok, frame = cap.read(); src_idx += 1
            if not ok: break
        if not ok or frame is None: break
        m = base.copy()
        good = row['ok'] and np.isfinite(row['x']) and np.isfinite(row['z']) and row['dist_route'] <= args.max_dist
        if good:
            qp = tuple(px(np.array([[row['x'], row['z']]], float))[0])
            trail.append(qp)
            node = int(np.clip(row['nearest_node'], 0, len(route_px)-1))
            rp = tuple(route_px[node])
            cv2.circle(m, rp, 8, (100,255,120), 2)
            cv2.line(m, qp, rp, (0,180,255), 2, cv2.LINE_AA)
            cv2.circle(m, qp, 10, (40,40,255), -1)
            cv2.circle(m, qp, 10, (255,255,255), 2)
            valid_i += 1
        for a,b in zip(list(trail)[:-1], list(trail)[1:]):
            cv2.line(m, a, b, (60,60,255), 2, cv2.LINE_AA)

        frame = cv2.resize(frame, (VIDEO_WIDTH, MAP_SIZE))
        top = np.hstack([m, frame])
        bar = np.full((BAR_HEIGHT, MAP_SIZE+VIDEO_WIDTH, 3), 24, np.uint8)
        status = 'OK' if good else ('PNP_OUTLIER' if row['ok'] else 'LOST')
        color = (90,240,120) if good else ((0,180,255) if row['ok'] else (70,70,255))
        cv2.putText(bar, f'DPVO + ALIKED localization  frame={row["frame"]}/{total}  status={status}', (18,34), FONT, 0.78, color, 2)
        cv2.putText(bar, f'node={row["nearest_node"]}  dist={row["dist_route"]:.3f}  pairs={row["pairs"]}  inliers={row["inliers"]}  kpts={row["query_keypoints"]}', (18,72), FONT, 0.72, (220,220,220), 2)
        cv2.putText(bar, f'x={row["x"]:.3f} y={row["y"]:.3f} z={row["z"]:.3f}  ext={row["t_ext_ms"]:.0f}ms match={row["t_match_ms"]:.0f}ms  max_dist_filter={args.max_dist}', (18,110), FONT, 0.62, (170,200,255), 2)
        writer.write(np.vstack([top, bar]))
    cap.release(); writer.release()
    print(args.out)
    print('rows', len(rows), 'valid_after_filter', valid_i)

if __name__ == '__main__':
    main()
