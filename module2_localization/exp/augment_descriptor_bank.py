import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from module2_localization import config
from module2_localization.services.localization_service import build_localizer


def match_query(localizer, query, chunk=None):
    q = query["descriptors"]
    n = len(query["keypoints"])
    if chunk is None:
        chunk = max(4096, int(256e6 / (4 * max(n, 1))))
    neg = torch.finfo(torch.float16).min
    best = torch.full((n,), neg, device=localizer.dev, dtype=torch.float16)
    second = torch.full((n,), neg, device=localizer.dev, dtype=torch.float16)
    best_owner = torch.full((n,), -1, device=localizer.dev, dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, len(localizer.owner), chunk):
            similarity = q @ localizer.mdesc[start:start + chunk].T
            owners = localizer.owner[start:start + chunk]
            count = min(localizer.match_topk, similarity.shape[1])
            values, indices = similarity.topk(count, dim=1)
            for column in range(count):
                value = values[:, column]
                owner = owners[indices[:, column]]
                newer = value > best
                second = torch.where(
                    newer & (best_owner != owner) & (best_owner >= 0),
                    torch.maximum(second, best), second,
                )
                second = torch.where(
                    ~newer & (owner != best_owner), torch.maximum(second, value), second,
                )
                best_owner = torch.where(newer, owner, best_owner)
                best = torch.where(newer, value, best)
    d1 = torch.sqrt((2 - 2 * best.float()).clamp(min=0))
    d2 = torch.sqrt((2 - 2 * second.float()).clamp(min=0))
    return (
        best_owner.cpu().numpy(),
        best.float().cpu().numpy(),
        (d1 / d2.clamp(min=1e-6)).cpu().numpy(),
    )


def camera_model(localizer, query):
    width, height = query["width"], query["height"]
    K = localizer.K.copy()
    dist = localizer.dist
    if (width, height) != (localizer.cam_w, localizer.cam_h):
        if abs(width / height - localizer.cam_w / localizer.cam_h) >= 0.01:
            focal = localizer.focal_fallback * max(width, height)
            K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
            dist = np.zeros(4)
        else:
            K[0] *= width / localizer.cam_w
            K[1] *= height / localizer.cam_h
    return K, dist


def estimate_pose(localizer, query, owner, ratio, min_inliers):
    valid = (owner >= 0) & (ratio < localizer.match_ratio)
    points2d = query["keypoints"][valid]
    points3d = localizer.mxyz[owner[valid]]
    if len(points2d) < localizer.min_pairs:
        return None
    K, dist = camera_model(localizer, query)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d.astype(np.float64), points2d.astype(np.float64), K, dist,
        reprojectionError=localizer.max_error,
        confidence=localizer.pnp_confidence,
        iterationsCount=localizer.pnp_iters,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None
    selected = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(
        points3d[selected].astype(np.float64),
        points2d[selected].astype(np.float64), K, dist, rvec, tvec,
    )
    return rvec, tvec, K, dist, int(len(inliers))


def geometric_candidates(localizer, query, owner, similarity, ratio, pose,
                         max_reprojection, min_similarity, max_ratio):
    rvec, tvec, K, dist, _ = pose
    projected, _ = cv2.projectPoints(
        localizer.mxyz.astype(np.float64), rvec, tvec, K, dist,
    )
    projected = projected.reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec)
    depth = (R @ localizer.mxyz.T + tvec)[2]
    valid_owner = owner >= 0
    safe_owner = np.maximum(owner, 0)
    error = np.linalg.norm(query["keypoints"] - projected[safe_owner], axis=1)
    return np.flatnonzero(
        valid_owner
        & (depth[safe_owner] > 0)
        & (similarity >= min_similarity)
        & (ratio <= max_ratio)
        & (error <= max_reprojection)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--map", default="1-2/front_full")
    parser.add_argument("--output", default=None)
    parser.add_argument("--frame-step", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--max-reprojection", type=float, default=3.0)
    parser.add_argument("--min-similarity", type=float, default=0.65)
    parser.add_argument("--max-ratio", type=float, default=0.98)
    parser.add_argument("--max-per-owner", type=int, default=4)
    parser.add_argument("--duplicate-similarity", type=float, default=0.985)
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / "artifacts" / args.map / "sunny_bank"
    )
    output.mkdir(parents=True, exist_ok=True)
    localizer = build_localizer(args.map, False)
    source_dir = config.MAPS_DIR / args.map
    bank = np.load(source_dir / "aliked_bank.npz")
    original_desc = bank["desc"]
    original_owner = bank["owner"]
    order = np.argsort(original_owner, kind="stable")
    sorted_owner = original_owner[order]

    additions_desc = []
    additions_owner = []
    accepted_by_owner = {}
    report = {"map": args.map, "video": args.video, "frames": []}
    capture = cv2.VideoCapture(args.video)
    source_index = used = 0
    started = time.perf_counter()
    while used < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % args.frame_step:
            source_index += 1
            continue
        if (frame.shape[1], frame.shape[0]) != (1280, 720):
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        query = localizer.extract_query(frame)
        owner, similarity, ratio = match_query(localizer, query)
        pose = estimate_pose(localizer, query, owner, ratio, args.min_inliers)
        frame_report = {"frame": source_index, "accepted": False, "added": 0}
        if pose is not None:
            frame_report["inliers"] = pose[-1]
            candidates = geometric_candidates(
                localizer, query, owner, similarity, ratio, pose,
                args.max_reprojection, args.min_similarity, args.max_ratio,
            )
            descriptors = query["descriptors"].float().cpu().numpy()
            added = 0
            for query_index in candidates:
                point_owner = int(owner[query_index])
                already = accepted_by_owner.setdefault(point_owner, [])
                if len(already) >= args.max_per_owner:
                    continue
                lo = np.searchsorted(sorted_owner, point_owner, side="left")
                hi = np.searchsorted(sorted_owner, point_owner, side="right")
                existing = original_desc[order[lo:hi]].astype(np.float32)
                descriptor = descriptors[query_index]
                if len(existing) and float((existing @ descriptor).max()) >= args.duplicate_similarity:
                    continue
                if already and float((np.stack(already) @ descriptor).max()) >= args.duplicate_similarity:
                    continue
                already.append(descriptor)
                additions_desc.append(descriptor)
                additions_owner.append(point_owner)
                added += 1
            frame_report.update(accepted=True, candidates=int(len(candidates)), added=added)
        report["frames"].append(frame_report)
        print(json.dumps(frame_report, ensure_ascii=False), flush=True)
        used += 1
        source_index += 1
    capture.release()

    if additions_desc:
        augmented_desc = np.concatenate([
            original_desc, np.asarray(additions_desc, dtype=np.float16),
        ])
        augmented_owner = np.concatenate([
            original_owner, np.asarray(additions_owner, dtype=np.int32),
        ])
    else:
        augmented_desc, augmented_owner = original_desc, original_owner
    np.savez_compressed(
        output / "aliked_bank.npz", desc=augmented_desc,
        owner=augmented_owner, xyz=bank["xyz"],
    )
    shutil.copy2(source_dir / "runtime.npz", output / "runtime.npz")
    if (source_dir / "scale.json").exists():
        shutil.copy2(source_dir / "scale.json", output / "scale.json")
    report.update({
        "original_descriptors": int(len(original_desc)),
        "added_descriptors": int(len(additions_desc)),
        "result_descriptors": int(len(augmented_desc)),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "output": str(output),
    })
    (output / "augmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
