import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from module2_localization.services.localization_service import build_localizer


def evaluate(video, map_name, bank_path, frame_offset, frame_step, max_frames, quiet=False):
    localizer = build_localizer(map_name, False, bank_path=bank_path)
    capture = cv2.VideoCapture(video)
    rows = []
    source_index = 0
    started = time.perf_counter()
    while len(rows) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        selected = source_index >= frame_offset and (
            (source_index - frame_offset) % frame_step == 0
        )
        if selected:
            if (frame.shape[1], frame.shape[0]) != (1280, 720):
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            result = localizer.locate(frame)
            rows.append({
                "frame": source_index,
                "ok": bool(result.get("ok")),
                "node": result.get("node"),
                "inliers": int(result.get("inliers", 0)),
                "pairs": int(result.get("n_pairs", 0)),
            })
            if not quiet:
                print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        source_index += 1
    capture.release()
    inliers = [row["inliers"] for row in rows]
    return {
        "bank": str(bank_path),
        "frames": rows,
        "count": len(rows),
        "accepted_20": sum(value >= 20 for value in inliers),
        "accepted_15": sum(value >= 15 for value in inliers),
        "median_inliers": float(np.median(inliers)) if inliers else 0.0,
        "min_inliers": min(inliers, default=0),
        "max_inliers": max(inliers, default=0),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--map", default="1-2/front_full")
    parser.add_argument("--augmented-bank", required=True)
    parser.add_argument("--frame-offset", type=int, default=10)
    parser.add_argument("--frame-step", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    original = Path("module2_localization/maps") / args.map / "aliked_bank.npz"
    reports = {
        "original": evaluate(
            args.video, args.map, original, args.frame_offset,
            args.frame_step, args.max_frames, args.quiet,
        ),
        "augmented": evaluate(
            args.video, args.map, Path(args.augmented_bank), args.frame_offset,
            args.frame_step, args.max_frames, args.quiet,
        ),
    }
    before = reports["original"]
    after = reports["augmented"]
    reports["delta"] = {
        "accepted_20": after["accepted_20"] - before["accepted_20"],
        "accepted_15": after["accepted_15"] - before["accepted_15"],
        "median_inliers": after["median_inliers"] - before["median_inliers"],
    }
    text = json.dumps(reports, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
