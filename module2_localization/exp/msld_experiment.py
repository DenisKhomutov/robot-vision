import argparse
import csv
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import config
from ..core.aliked_localizer import ALIKEDLocalizer
from .export_msld_gui import export_gui_map


def read_frames(path, indices):
    cap = cv2.VideoCapture(str(path))
    wanted, frames = set(indices), {}
    for number in range(max(indices) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if number in wanted:
            frames[number] = frame
    cap.release()
    if len(frames) != len(indices):
        raise RuntimeError(f"прочитано {len(frames)}/{len(indices)} кадров из {path}")
    return frames


def to_cpu(query, number):
    return {
        **query,
        "descriptors": query["descriptors"].cpu(),
        "source": None,
        "frame_number": number,
    }


def to_device(query, device):
    return {**query, "descriptors": query["descriptors"].to(device)}


def extract(localizer, sessions):
    result = {}
    for name, (video, indices) in sessions.items():
        frames = read_frames(video, indices)
        result[name] = []
        for index, number in enumerate(indices, 1):
            result[name].append(to_cpu(localizer.extract_query(frames[number]), number))
            print(f"[extract] {name} {index}/{len(indices)}", flush=True)
    return result


def matched_owners(localizer, query):
    q = query["descriptors"]
    count = len(query["keypoints"])
    chunk = max(4096, int(256e6 / (4 * max(count, 1))))
    neg = torch.finfo(torch.float16).min
    best = torch.full((count,), neg, device=localizer.dev, dtype=torch.float16)
    second = torch.full_like(best, neg)
    owner1 = torch.full((count,), -1, device=localizer.dev, dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, len(localizer.owner), chunk):
            similarity = q @ localizer.mdesc[start:start + chunk].T
            owners = localizer.owner[start:start + chunk]
            k = min(localizer.match_topk, similarity.shape[1])
            values, indices = similarity.topk(k, dim=1)
            for column in range(k):
                value, owner = values[:, column], owners[indices[:, column]]
                better = value > best
                second = torch.where(
                    better & (owner1 != owner) & (owner1 >= 0),
                    torch.maximum(second, best), second,
                )
                second = torch.where(
                    ~better & (owner != owner1), torch.maximum(second, value), second,
                )
                owner1 = torch.where(better, owner, owner1)
                best = torch.where(better, value, best)
    d1 = torch.sqrt((2 - 2 * best.float()).clamp(min=0))
    d2 = torch.sqrt((2 - 2 * second.float()).clamp(min=0))
    keep = (d1 / d2.clamp(min=1e-6) < localizer.match_ratio) & (owner1 >= 0)
    return np.flatnonzero(keep.cpu().numpy()), owner1[keep].cpu().numpy()


def verified_owners(localizer, query, result):
    if not result.get("ok") or result.get("inliers", 0) < 6:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    query_indices, owners = matched_owners(localizer, query)
    pnp = result["diagnostics"]["pnp"]
    rvec = np.asarray(pnp["rvec"], np.float64)
    tvec = np.asarray(pnp["tvec"], np.float64)
    K = np.asarray(result["diagnostics"]["K"], np.float64)
    dist = np.asarray(result["diagnostics"]["dist"], np.float64)
    points = localizer.mxyz[owners].astype(np.float64)
    projected, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
    error = np.linalg.norm(projected.reshape(-1, 2) - query["keypoints"][query_indices], axis=1)
    depth = (cv2.Rodrigues(rvec)[0] @ points.T + tvec.reshape(3, 1))[2]
    keep = (error <= localizer.max_error) & (depth > 0)
    return owners, owners[keep], error[keep]


def statistics(localizer, queries):
    n = len(localizer.mxyz)
    matched = np.zeros(n, np.int32)
    inliers = np.zeros(n, np.int32)
    sessions = np.zeros(n, np.int16)
    error_sum = np.zeros(n, np.float64)
    error_count = np.zeros(n, np.int32)
    for session, items in queries.items():
        seen = set()
        for index, stored in enumerate(items, 1):
            query = to_device(stored, localizer.dev)
            result = localizer.locate_features(query)
            owners, verified, errors = verified_owners(localizer, query, result)
            np.add.at(matched, np.unique(owners), 1)
            if len(verified):
                unique, first = np.unique(verified, return_index=True)
                np.add.at(inliers, unique, 1)
                np.add.at(error_sum, unique, errors[first])
                np.add.at(error_count, unique, 1)
                seen.update(unique.tolist())
            print(f"[stats] {session} {index}/{len(items)} inliers={result.get('inliers', 0)}", flush=True)
        if seen:
            sessions[np.fromiter(seen, np.int64)] += 1
    mean_error = np.full(n, 50.0)
    valid = error_count > 0
    mean_error[valid] = error_sum[valid] / error_count[valid]
    precision = inliers / np.maximum(matched, 1)
    score = 1000 * sessions + 20 * np.log1p(inliers) + 10 * precision - mean_error
    return {"matched_frames": matched, "inlier_frames": inliers,
            "session_hits": sessions, "mean_error": mean_error,
            "precision": precision, "score": score}


def spatial_bins(localizer, count):
    nearest = np.empty(len(localizer.mxyz), np.int32)
    route = localizer.route.astype(np.float32)
    for start in range(0, len(localizer.mxyz), 5000):
        points = localizer.mxyz[start:start + 5000].astype(np.float32)
        d2 = ((points[:, None] - route[None]) ** 2).sum(2)
        nearest[start:start + len(points)] = d2.argmin(1)
    return np.minimum(nearest * count // len(route), count - 1)


def select(score, bins, fraction, stable, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for bin_id in range(bins.max() + 1):
        candidates = np.flatnonzero(bins == bin_id)
        quota = max(1, round(len(candidates) * fraction))
        if stable:
            order = np.argsort(-(score[candidates] + rng.random(len(candidates)) * 1e-7))
            chosen = candidates[order[:quota]]
        else:
            chosen = rng.choice(candidates, quota, replace=False)
        selected.extend(chosen.tolist())
    return np.asarray(sorted(selected), np.int64)


def save_map(source, runtime, selected, destination, metadata):
    destination.mkdir(parents=True)
    with np.load(source) as bank:
        owners = bank["owner"]
        keep = np.isin(owners, selected)
        remap = np.full(len(bank["xyz"]), -1, np.int32)
        remap[selected] = np.arange(len(selected), dtype=np.int32)
        np.savez_compressed(destination / "aliked_bank.npz",
                            desc=bank["desc"][keep], owner=remap[owners[keep]],
                            xyz=bank["xyz"][selected])
    shutil.copy2(runtime, destination / "runtime.npz")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def evaluate(map_name, bank_path, queries):
    localizer = ALIKEDLocalizer(map_name, bank_path=bank_path)
    rows, started = [], time.perf_counter()
    for session, items in queries.items():
        localizer._last_node = None
        for stored in items:
            result = localizer.locate_features(to_device(stored, localizer.dev))
            rows.append({"session": session, "frame": stored["frame_number"],
                         "node": result.get("node"), "inliers": int(result.get("inliers", 0)),
                         "pairs": int(result.get("n_pairs", 0)),
                         "frame_ms": float(result.get("frame_ms", 0))})
    elapsed = time.perf_counter() - started
    values = np.asarray([row["inliers"] for row in rows])
    times = np.asarray([row["frame_ms"] for row in rows])
    longest = current = 0
    for value in values:
        current = current + 1 if value < 15 else 0
        longest = max(longest, current)
    nodes = [row["node"] for row in rows if row["node"] is not None]
    with np.load(bank_path) as bank:
        point_count, descriptor_count = len(bank["xyz"]), len(bank["desc"])
    summary = {"points": point_count, "descriptors": descriptor_count,
               "bank_mib": round(bank_path.stat().st_size / 2**20, 2), "frames": len(rows),
               "recall10": round(float(np.mean(values >= 10)), 4),
               "recall15": round(float(np.mean(values >= 15)), 4),
               "recall20": round(float(np.mean(values >= 20)), 4),
               "inliers_min": int(values.min()), "inliers_p10": round(float(np.percentile(values, 10)), 2),
               "inliers_median": float(np.median(values)),
               "longest_lost15": longest,
               "node_jumps_gt10": sum(abs(b - a) > 10 for a, b in zip(nodes, nodes[1:])),
               "frame_ms_median": round(float(np.median(times)), 2),
               "frame_ms_p90": round(float(np.percentile(times, 90)), 2),
               "wall_s": round(elapsed, 2)}
    del localizer
    torch.cuda.empty_cache()
    return rows, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="быстрый A/B-тест MSLD")
    parser.add_argument("--map", default="1-2/front_shard/03_of_25")
    parser.add_argument("--normal-video", type=Path,
                        default=Path("module2_localization/data/street_video/camera-front-1-720.mkv"))
    parser.add_argument("--sun-video", type=Path, default=Path("module2_localization/data/test_neg.mkv"))
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.75, 0.5, 0.25])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")
    output = args.out or Path("module2_localization/exp/artifacts/msld") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    map_dir = config.MAPS_DIR / args.map
    source_bank, runtime = map_dir / "aliked_bank.npz", map_dir / "runtime.npz"
    train = {"normal_train": (args.normal_video, list(range(2441, 3090, 32))),
             "sun_train": (args.sun_video, list(range(0, 120, 6)))}
    validation = {"normal_eval": (args.normal_video, list(range(2457, 3090, 32))),
                  "sun_eval": (args.sun_video, list(range(120, 239, 6)))}
    base = ALIKEDLocalizer(args.map)
    train_queries, validation_queries = extract(base, train), extract(base, validation)
    stats = statistics(base, train_queries)
    np.savez_compressed(output / "landmark_statistics.npz", **stats)
    bins = spatial_bins(base, 12)
    variants = [("baseline", source_bank, "baseline", 1.0)]
    for fraction in args.fractions:
        for stable in (True, False):
            kind = "stable" if stable else "random"
            name = f"{kind}_{round(fraction * 100)}"
            chosen = select(stats["score"], bins, fraction, stable, 42)
            destination = output / "maps" / name
            save_map(source_bank, runtime, chosen, destination,
                     {"source_map": args.map, "kind": kind, "fraction": fraction})
            export_gui_map(destination)
            variants.append((name, destination / "aliked_bank.npz", kind, fraction))
    del base
    torch.cuda.empty_cache()
    summaries = []
    diagnostics = output / "diagnostics"
    diagnostics.mkdir()
    for name, bank, kind, fraction in variants:
        print(f"[evaluate] {name}", flush=True)
        rows, summary = evaluate(args.map, bank, validation_queries)
        summary = {"variant": name, "kind": kind, "fraction": fraction, **summary}
        summaries.append(summary)
        with (diagnostics / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    write_csv(output / "summary.csv", summaries)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"RESULT={output}", flush=True)


if __name__ == "__main__":
    main()
