"""Run DPVO and save the internal map state.

This is intentionally separate from DPVO's demo.py.  demo.py exports viewer
artifacts; this tool additionally writes ``dpvo_state.pt`` for future
localization experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from module3_map_builder.dpvo_state import save_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DPVO_DIR = ROOT / "external" / "DPVO"
DEFAULT_OUT_ROOT = ROOT / "out" / "dpvo"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--dpvo-dir", type=Path, default=DEFAULT_DPVO_DIR)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--network", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--opts", nargs="+", default=[])
    ap.add_argument("--save-ply", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    dpvo_dir = args.dpvo_dir.resolve()
    out_dir = (args.out_root / args.name).resolve()
    if out_dir.exists() and not args.force:
        raise SystemExit(f"output already exists: {out_dir} (use --force)")
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(dpvo_dir))
    os.chdir(dpvo_dir)

    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    from dpvo.plot_utils import save_ply
    from dpvo.stream import video_stream

    config = args.config or (dpvo_dir / "config" / "default.yaml")
    network = args.network or (dpvo_dir / "dpvo.pth")
    cfg.merge_from_file(str(config))
    if args.opts:
        cfg.merge_from_list(args.opts)

    queue: Queue = Queue(maxsize=8)
    reader = Process(target=video_stream, args=(queue, str(args.video.resolve()), str(args.calib.resolve()), args.stride, args.skip))
    reader.start()

    slam = None
    last_intrinsics = None
    while True:
        timestamp, image_np, intrinsics_np = queue.get()
        if timestamp < 0:
            break
        image = torch.from_numpy(image_np).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics_np).cuda()
        last_intrinsics = intrinsics_np
        if slam is None:
            _, height, width = image.shape
            slam = DPVO(cfg, str(network), ht=height, wd=width, viz=False)
        slam(timestamp, image, intrinsics)

    reader.join()
    if slam is None or last_intrinsics is None:
        raise SystemExit("DPVO did not process any frames")

    points = slam.pg.points_.detach().cpu().numpy()[: slam.m]
    colors = slam.pg.colors_.view(-1, 3).detach().cpu().numpy()[: slam.m]
    poses, tstamps = slam.terminate()

    trajectory = PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=np.asarray(tstamps, dtype=np.float64),
    )
    trajectory_path = out_dir / "trajectory_tum.txt"
    file_interface.write_tum_trajectory_file(str(trajectory_path), trajectory)

    if args.save_ply:
        save_ply(str(out_dir / "map_points"), points, colors)

    summary = save_state(
        slam,
        out_dir / "dpvo_state.pt",
        source={
            "name": args.name,
            "video": str(args.video.resolve()),
            "calib": str(args.calib.resolve()),
            "stride": args.stride,
            "skip": args.skip,
            "config": str(config.resolve()),
            "network": str(network.resolve()),
        },
    )
    summary["trajectory"] = str(trajectory_path)
    summary["video"] = str(args.video.resolve())
    summary["calib"] = str(args.calib.resolve())
    (out_dir / "dpvo_state_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
