"""Localize a video against a DPVO+ALIKED hybrid bank."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--calib', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--kpts', type=int, default=4096)
    ap.add_argument('--det-threshold', type=float, default=0.04)
    ap.add_argument('--match-ratio', type=float, default=0.9)
    ap.add_argument('--match-topk', type=int, default=8)
    ap.add_argument('--min-pairs', type=int, default=12)
    ap.add_argument('--min-inliers', type=int, default=40)
    ap.add_argument('--max-error', type=float, default=12.0)
    ap.add_argument('--max-frames', type=int, default=0)
    return ap.parse_args()


def read_calib(path: Path):
    vals = np.loadtxt(path, dtype=float).reshape(-1)
    fx, fy, cx, cy = vals[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
    dist = vals[4:].astype(np.float64) if len(vals) > 4 else np.zeros(4, dtype=np.float64)
    return K, dist


def extract_frame_tensor(frame, device: str, half: bool):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)
    return t.half() if half else t


def match_descriptors(q: torch.Tensor, mdesc: torch.Tensor, owner: torch.Tensor, match_ratio: float, topk: int):
    n = q.shape[0]
    neg = torch.finfo(torch.float16).min
    best = torch.full((n,), neg, device=q.device, dtype=torch.float16)
    second = torch.full((n,), neg, device=q.device, dtype=torch.float16)
    bown = torch.full((n,), -1, dtype=torch.long, device=q.device)
    chunk = max(4096, int(256e6 / (4 * max(n, 1))))
    with torch.inference_mode():
        for i in range(0, len(owner), chunk):
            sim = q @ mdesc[i:i+chunk].T
            own = owner[i:i+chunk]
            k = min(topk, sim.shape[1])
            tv, ti = sim.topk(k, dim=1)
            for c in range(k):
                v, o = tv[:, c], own[ti[:, c]]
                nb = v > best
                second = torch.where(nb & (bown != o) & (bown >= 0), torch.maximum(second, best), second)
                second = torch.where(~nb & (o != bown), torch.maximum(second, v), second)
                bown = torch.where(nb, o, bown)
                best = torch.where(nb, v, best)
    d1 = torch.sqrt((2 - 2 * best.float()).clamp(min=0))
    d2 = torch.sqrt((2 - 2 * second.float()).clamp(min=0))
    ratio = d1 / d2.clamp(min=1e-6)
    keep = (ratio < match_ratio) & (bown >= 0)
    return keep.cpu().numpy(), bown.cpu().numpy(), ratio.cpu().numpy()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path.cwd()))
    from lightglue import ALIKED
    from module2_localization.core.model_weights import configure_local_model_weights

    configure_local_model_weights()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = device == 'cuda'
    extractor = ALIKED(max_num_keypoints=args.kpts, detection_threshold=args.det_threshold).eval().to(device)
    if half:
        extractor = extractor.half()

    bank = np.load(args.bank, allow_pickle=True)
    mdesc = torch.from_numpy(bank['desc'].astype(np.float32)).half().to(device)
    owner = torch.from_numpy(bank['owner'].astype(np.int64)).to(device)
    xyz = bank['xyz'].astype(np.float64)
    node = bank['node'].astype(np.int64)
    route = bank['route'].astype(np.float64)
    K, dist = read_calib(args.calib)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f'cannot open video: {args.video}')
    rows=[]
    frame_id=-1
    processed=0
    while True:
        ok=False; frame=None
        for _ in range(args.stride):
            ok, frame = cap.read(); frame_id += 1
            if not ok: break
        if not ok or frame is None: break
        if args.max_frames and processed >= args.max_frames: break
        t0=time.perf_counter()
        image = extract_frame_tensor(frame, device, half)
        with torch.inference_mode():
            f = extractor.extract(image)
        qk = f['keypoints'][0].detach().cpu().numpy().astype(np.float64)
        qd = f['descriptors'][0].half()
        t_ext=(time.perf_counter()-t0)*1000
        t1=time.perf_counter()
        if len(qk):
            keep, own, ratio = match_descriptors(qd, mdesc, owner, args.match_ratio, args.match_topk)
        else:
            keep=np.zeros((0,), bool); own=np.zeros((0,), int); ratio=np.zeros((0,), float)
        t_match=(time.perf_counter()-t1)*1000
        p2d=qk[keep]
        p3d=xyz[own[keep]] if len(own) else np.empty((0,3))
        ok_pnp=False; inliers=0; nearest_node=-1; dist_route=np.nan
        C=[np.nan,np.nan,np.nan]
        if len(p2d) >= args.min_pairs:
            ok_pnp, rvec, tvec, inl = cv2.solvePnPRansac(
                p3d, p2d, K, dist, reprojectionError=args.max_error,
                confidence=0.999, iterationsCount=1000, flags=cv2.SOLVEPNP_EPNP)
            if ok_pnp and inl is not None:
                inliers=int(len(inl))
                R,_=cv2.Rodrigues(rvec)
                C=(-R.T @ tvec).reshape(3)
                d2=((route-C[None,:])**2).sum(axis=1)
                nearest_node=int(np.argmin(d2))
                dist_route=float(np.sqrt(d2[nearest_node]))
        rows.append({
            'frame': frame_id,
            'processed': processed,
            'ok': int(bool(ok_pnp and inliers >= args.min_inliers)),
            'query_keypoints': int(len(qk)),
            'pairs': int(len(p2d)),
            'inliers': int(inliers),
            'nearest_node': nearest_node,
            'dist_route': dist_route,
            'x': float(C[0]), 'y': float(C[1]), 'z': float(C[2]),
            't_ext_ms': t_ext, 't_match_ms': t_match,
        })
        processed += 1
        if processed % 25 == 0:
            print(f'{processed}: ok={sum(r["ok"] for r in rows)}/{len(rows)} last_pairs={len(p2d)} inl={inliers}', flush=True)
    cap.release()

    csv_path=args.out_dir/'dpvo_aliked_localization.csv'
    with csv_path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['frame'])
        w.writeheader(); w.writerows(rows)
    ok_count=sum(r['ok'] for r in rows)
    summary={
        'bank': str(args.bank), 'video': str(args.video), 'frames': len(rows), 'ok': ok_count,
        'ok_ratio': ok_count / max(len(rows),1),
        'median_pairs': float(np.median([r['pairs'] for r in rows])) if rows else 0,
        'median_inliers_ok': float(np.median([r['inliers'] for r in rows if r['ok']])) if ok_count else 0,
        'median_dist_ok': float(np.median([r['dist_route'] for r in rows if r['ok']])) if ok_count else None,
        'min_pairs': args.min_pairs,
        'min_inliers': args.min_inliers,
        'csv': str(csv_path),
    }
    (args.out_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
