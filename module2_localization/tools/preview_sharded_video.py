"""Render a real sharded-localization run with map, commands and traffic state."""

# ruff: noqa: E501, T201

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .. import config
from ..core.pilot import Pilot
from ..service import TrafficBranch
from ..services.localization_service import build_runtime_localizer

MAP_SIZE = 720
VIDEO_WIDTH = 1280
BAR_HEIGHT = 120
FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_canvas(map_name: str) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    data = np.load(config.MAPS_DIR / map_name / "runtime.npz")
    route = data["pos"]
    points = data["points"]
    if len(points) > 60_000:
        points = points[np.linspace(0, len(points) - 1, 60_000, dtype=int)]
    all_xz = np.vstack((route[:, [0, 2]], points[:, [0, 2]]))
    lo, hi = np.percentile(all_xz, [1, 99], axis=0)
    pad = 35

    def px(value: np.ndarray) -> np.ndarray:
        q = (np.atleast_2d(value) - lo) / (hi - lo + 1e-9)
        return np.stack((pad + q[:, 0] * (MAP_SIZE - 2 * pad),
                         MAP_SIZE - pad - q[:, 1] * (MAP_SIZE - 2 * pad)), axis=1).astype(int)

    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), 15, np.uint8)
    cloud_px = px(points[:, [0, 2]])
    valid = ((cloud_px[:, 0] >= 0) & (cloud_px[:, 0] < MAP_SIZE)
             & (cloud_px[:, 1] >= 0) & (cloud_px[:, 1] < MAP_SIZE))
    canvas[cloud_px[valid, 1], cloud_px[valid, 0]] = (65, 65, 65)
    route_px = px(route[:, [0, 2]])
    for a, b in zip(route_px[:-1], route_px[1:], strict=True):
        cv2.line(canvas, tuple(a), tuple(b), (205, 170, 55), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(route_px[0]), 7, (80, 220, 100), -1)
    cv2.circle(canvas, tuple(route_px[-1]), 7, (60, 60, 230), -1)
    return canvas, route_px, px


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--map", default=config.FRONT_MAP)
    parser.add_argument("--full-map", default="route12_front_colmap_full_1701_fullopencv_cpu10_final")
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--traffic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-frames", type=int, default=None, help="ограничение для короткой проверки")
    parser.add_argument("--diagnostics", default=None, help="JSONL с результатом каждого обработанного кадра")
    args = parser.parse_args()

    localizer = build_runtime_localizer(args.map, False)
    pilot = Pilot(config)
    pilot.resume()
    traffic = None
    traffic_completed = False
    canvas, route_px, px = build_canvas(args.full_map)
    trail = deque(maxlen=80)

    capture = cv2.VideoCapture(args.video)
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = args.fps or source_fps / args.step
    width, height = MAP_SIZE + VIDEO_WIDTH, MAP_SIZE + BAR_HEIGHT
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"не удалось открыть writer {output}")

    source_index = processed = fixed = 0
    diagnostics_path = Path(args.diagnostics) if args.diagnostics else output.with_suffix(".jsonl")
    diagnostics = diagnostics_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    timings = []
    active_map = args.map
    last_good = None
    switch_banner = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % args.step:
            source_index += 1
            continue
        source_index += 1
        tick = time.perf_counter()
        result = localizer.locate(frame)
        elapsed_ms = (time.perf_counter() - tick) * 1000
        timings.append(elapsed_ms)
        processed += 1

        map_name = result.pop("_map_name", active_map)
        offset = int(result.pop("_node_offset", 0))
        switched = bool(result.pop("_map_switched", False))
        if switched:
            was_paused = pilot.paused
            pilot = Pilot(config)
            if not was_paused:
                pilot.resume()
            active_map = map_name
            switch_banner = max(1, int(output_fps * 2))
        command = pilot.step(result, now=source_index / source_fps)
        global_node = None if command.get("node") is None else int(command["node"]) + offset
        command["global_node"] = global_node
        command["map"] = map_name
        diagnostics.write(json.dumps({
            "processed": processed, "source_frame": source_index, "map": map_name,
            "local_node": command.get("node"), "global_node": global_node,
            "ok": bool(result.get("ok")), "accepted": bool(pilot.accepted),
            "inliers": result.get("inliers"), "pairs": result.get("pairs"),
            "reason": result.get("reason"), "switched": switched,
            "full_map_recovery": bool(result.get("_full_map_recovery")),
            "recovery_map": result.get("_recovery_map"), "pilot_reject": pilot.jump,
            "elapsed_ms": round(elapsed_ms, 2), "command": command.get("move_type"),
        }, ensure_ascii=False) + "\n")

        zone = config.TRAFFIC_ZONES.get(map_name)
        in_zone = zone is not None and global_node is not None and zone[0] <= global_node <= zone[1]
        traffic_label = "OFF"
        if args.traffic:
            traffic_label = "ARMED / OUTSIDE ZONE"
            if traffic_completed:
                traffic_label = "GO LATCHED / DETECTOR OFF"
            elif in_zone:
                if traffic is None:
                    traffic = TrafficBranch(config)
                detection = traffic.update(True, frame)
                traffic_label = traffic.state
                if not traffic.go and command.get("move_type") != "lost":
                    command["move_type"], command["deg"] = "stop", 0.0
                if traffic.go:
                    traffic_completed = True
                    traffic_label = "GO LATCHED / DETECTOR OFF"
                if detection is not None:
                    signal = detection.get("signal") or "none"
                    traffic_label += f" / {signal}"

        good = result.get("ok") and pilot.accepted
        if good:
            fixed += 1
            last_good = result
            trail.append(px(result["C"][[0, 2]])[0])

        map_image = canvas.copy()
        for a, b in zip(list(trail)[:-1], list(trail)[1:], strict=True):
            cv2.line(map_image, tuple(a), tuple(b), (80, 80, 255), 2, cv2.LINE_AA)
        if good and global_node is not None:
            node = min(max(global_node, 0), len(route_px) - 1)
            target = command.get("target_node")
            target = node if target is None else min(max(int(target) + offset, 0), len(route_px) - 1)
            position = tuple(px(result["C"][[0, 2]])[0])
            cv2.circle(map_image, tuple(route_px[node]), 7, (100, 255, 120), 2)
            cv2.line(map_image, position, tuple(route_px[target]), (0, 230, 230), 2, cv2.LINE_AA)
            cv2.circle(map_image, position, 10, (50, 50, 255), -1)
        elif last_good is not None and trail:
            cv2.circle(map_image, tuple(trail[-1]), 10, (90, 90, 180), 2)

        resized = cv2.resize(frame, (VIDEO_WIDTH, MAP_SIZE))
        top = np.hstack((map_image, resized))
        bar = np.full((BAR_HEIGHT, width, 3), 24, np.uint8)
        preload = getattr(localizer, "preload_status", {"target": None, "ready": False})
        cv2.putText(bar, f"MAP: {map_name}", (18, 30), FONT, 0.65, (215, 215, 215), 2)
        cv2.putText(bar, f"node local={command.get('node')} global={global_node}  command={command.get('move_type','lost').upper()}  deg={command.get('deg',0):+.1f}",
                    (18, 62), FONT, 0.68, (100, 255, 150) if command.get("move_type") != "lost" else (80, 80, 240), 2)
        cv2.putText(bar, f"TRAFFIC: {traffic_label}", (18, 96), FONT, 0.65, (80, 210, 255), 2)
        preload_text = "none" if preload["target"] is None else f"{preload['target']} ({'READY' if preload['ready'] else 'LOADING'})"
        cv2.putText(bar, f"preload: {preload_text}", (900, 30), FONT, 0.58, (180, 180, 180), 1)
        cv2.putText(bar, f"{elapsed_ms:.0f} ms  source {source_index}/{total or '?'}  fixes {fixed}/{processed}",
                    (900, 64), FONT, 0.58, (180, 180, 180), 1)
        if switch_banner:
            cv2.putText(bar, f"SHARD SWITCH -> {active_map}", (900, 102), FONT, 0.7, (0, 235, 235), 2)
            switch_banner -= 1
        writer.write(np.vstack((top, bar)))

        if args.max_frames is not None and processed >= args.max_frames:
            break

        if processed % 25 == 0:
            run = time.perf_counter() - started
            expected = total / args.step if total > 0 else 0
            remaining = run / processed * max(expected - processed, 0) if expected else 0
            print(f"[{processed}] source={source_index}/{total or '?'} fixes={fixed} map={active_map} "
                  f"{run:.0f}s elapsed ~{remaining:.0f}s left", flush=True)

    capture.release()
    writer.release()
    diagnostics.close()
    if hasattr(localizer, "close"):
        localizer.close()
    median = float(np.median(timings)) if timings else 0.0
    print(f"готово: {output} | frames={processed} fixes={fixed} ({100*fixed/max(processed,1):.1f}%) "
          f"median={median:.0f}ms total={time.perf_counter()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
