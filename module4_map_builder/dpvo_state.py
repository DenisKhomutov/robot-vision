"""Serialize the real DPVO map state.

PLY/COLMAP exports are viewer artifacts.  They are not enough for a DPVO-based
runtime because DPVO local tracking depends on the PatchGraph tensors and the
network feature memories.  This module stores those tensors in a single
``dpvo_state.pt`` file so experiments can later be loaded by a localization
runtime instead of being rebuilt from video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone()


def _serialize_delta(delta: dict[int, tuple[int, Any]]) -> dict[int, tuple[int, torch.Tensor]]:
    """Serialize removed-frame relative poses.

    DPVO stores each removed frame as ``timestamp -> (source_timestamp, SE3)``.
    The SE3 object itself is not a stable portable format, but its 7D tensor is.
    """

    out: dict[int, tuple[int, torch.Tensor]] = {}
    for timestamp, (source_timestamp, pose_delta) in delta.items():
        out[int(timestamp)] = (int(source_timestamp), _cpu_tensor(pose_delta.data))
    return out


def build_state_dict(slam: Any, *, source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a CPU-only, torch.save-compatible DPVO state dictionary."""

    pg = slam.pg
    n = int(pg.n)
    m = int(pg.m)

    return {
        "format": "robot_vision.module4.dpvo_state",
        "format_version": 1,
        "source": source or {},
        "camera": {
            "height": int(slam.ht),
            "width": int(slam.wd),
            "res": int(slam.RES),
            "patch_size": int(slam.P),
            "patches_per_frame": int(slam.M),
            "buffer_size": int(slam.N),
            "patch_memory": int(slam.pmem),
            "feature_memory": int(slam.mem),
            "descriptor_dim": int(slam.DIM),
        },
        "runtime": {
            "is_initialized": bool(slam.is_initialized),
            "counter": int(slam.counter),
            "tlist": list(slam.tlist),
            "ran_global_ba": np.asarray(slam.ran_global_ba).copy(),
        },
        "graph": {
            "n": n,
            "m": m,
            "tstamps": np.asarray(pg.tstamps_[:n]).copy(),
            "poses": _cpu_tensor(pg.poses_[:n]),
            "patches": _cpu_tensor(pg.patches_[:n]),
            "intrinsics": _cpu_tensor(pg.intrinsics_[:n]),
            "points": _cpu_tensor(pg.points_[:m]),
            "colors": _cpu_tensor(pg.colors_[:n]),
            "index": _cpu_tensor(pg.index_[: n + 1]),
            "index_map": _cpu_tensor(pg.index_map_[: n + 1]),
            "delta": _serialize_delta(pg.delta),
            "active_edges": {
                "net": _cpu_tensor(pg.net),
                "ii": _cpu_tensor(pg.ii),
                "jj": _cpu_tensor(pg.jj),
                "kk": _cpu_tensor(pg.kk),
                "target": _cpu_tensor(pg.target) if hasattr(pg, "target") else None,
                "weight": _cpu_tensor(pg.weight) if hasattr(pg, "weight") else None,
            },
            "inactive_edges": {
                "ii": _cpu_tensor(pg.ii_inac),
                "jj": _cpu_tensor(pg.jj_inac),
                "kk": _cpu_tensor(pg.kk_inac),
                "target": _cpu_tensor(pg.target_inac),
                "weight": _cpu_tensor(pg.weight_inac),
            },
        },
        "feature_memory": {
            "imap": _cpu_tensor(slam.imap_),
            "gmap": _cpu_tensor(slam.gmap_),
            "fmap1": _cpu_tensor(slam.fmap1_),
            "fmap2": _cpu_tensor(slam.fmap2_),
        },
    }


def save_state(slam: Any, path: Path, *, source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Save DPVO state and return a compact summary."""

    state = build_state_dict(slam, source=source)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    graph = state["graph"]
    features = state["feature_memory"]
    return {
        "path": str(path),
        "frames": int(graph["n"]),
        "patches": int(graph["m"]),
        "points": int(graph["points"].shape[0]),
        "active_edges": int(graph["active_edges"]["ii"].numel()),
        "inactive_edges": int(graph["inactive_edges"]["ii"].numel()),
        "imap_shape": list(features["imap"].shape),
        "gmap_shape": list(features["gmap"].shape),
        "fmap1_shape": list(features["fmap1"].shape),
        "fmap2_shape": list(features["fmap2"].shape),
    }



def _to_device(value: torch.Tensor, device: torch.device | str, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = value.to(device=device)
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value


def load_state(path: Path) -> dict[str, Any]:
    """Load a DPVO state saved by :func:`save_state`."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "robot_vision.module4.dpvo_state":
        raise ValueError(f"unsupported DPVO state format: {state.get('format')!r}")
    if int(state.get("format_version", -1)) != 1:
        raise ValueError(f"unsupported DPVO state version: {state.get('format_version')!r}")
    return state


def load_state_into_slam(slam: Any, state: dict[str, Any]) -> dict[str, int]:
    """Restore saved tensors into an already constructed ``DPVO`` object.

    The network still has to be constructed normally from weights. This fills
    the map/graph/feature state so the next call to ``slam(...)`` continues from
    the saved map instead of starting empty.
    """

    from dpvo.lietorch import SE3

    graph = state["graph"]
    runtime = state["runtime"]
    features = state["feature_memory"]
    pg = slam.pg
    device = pg.poses_.device

    n = int(graph["n"])
    m = int(graph["m"])
    if n > pg.N:
        raise ValueError(f"state has {n} frames, but current DPVO buffer is {pg.N}")
    if m > pg.N * pg.M:
        raise ValueError(f"state has {m} patches, but current DPVO buffer holds {pg.N * pg.M}")

    pg.n = n
    pg.m = m
    slam.tlist = list(runtime["tlist"])
    slam.counter = int(runtime["counter"])
    slam.is_initialized = bool(runtime["is_initialized"])
    slam.ran_global_ba[:] = np.asarray(runtime["ran_global_ba"], dtype=bool)

    pg.tstamps_[:n] = np.asarray(graph["tstamps"], dtype=np.int64)
    pg.poses_[:n] = _to_device(graph["poses"], device, torch.float32)
    pg.patches_[:n] = _to_device(graph["patches"], device, torch.float32)
    pg.intrinsics_[:n] = _to_device(graph["intrinsics"], device, torch.float32)
    pg.points_[:m] = _to_device(graph["points"], device, torch.float32)
    pg.colors_[:n] = _to_device(graph["colors"], device, torch.uint8)
    pg.index_[: n + 1] = _to_device(graph["index"], device, torch.long)
    pg.index_map_[: n + 1] = _to_device(graph["index_map"], device, torch.long)

    pg.delta = {
        int(timestamp): (int(source_timestamp), SE3(_to_device(delta_pose, device, torch.float32)))
        for timestamp, (source_timestamp, delta_pose) in graph["delta"].items()
    }

    active = graph["active_edges"]
    pg.net = _to_device(active["net"], device, pg.net.dtype)
    pg.ii = _to_device(active["ii"], device, torch.long)
    pg.jj = _to_device(active["jj"], device, torch.long)
    pg.kk = _to_device(active["kk"], device, torch.long)
    if active["target"] is not None:
        pg.target = _to_device(active["target"], device, torch.float32)
    if active["weight"] is not None:
        pg.weight = _to_device(active["weight"], device, torch.float32)

    inactive = graph["inactive_edges"]
    pg.ii_inac = _to_device(inactive["ii"], device, torch.long)
    pg.jj_inac = _to_device(inactive["jj"], device, torch.long)
    pg.kk_inac = _to_device(inactive["kk"], device, torch.long)
    pg.target_inac = _to_device(inactive["target"], device, torch.float32)
    pg.weight_inac = _to_device(inactive["weight"], device, torch.float32)

    slam.imap_[:] = _to_device(features["imap"], device, slam.imap_.dtype)
    slam.gmap_[:] = _to_device(features["gmap"], device, slam.gmap_.dtype)
    slam.fmap1_[:] = _to_device(features["fmap1"], device, slam.fmap1_.dtype)
    slam.fmap2_[:] = _to_device(features["fmap2"], device, slam.fmap2_.dtype)

    return {
        "frames": n,
        "patches": m,
        "active_edges": int(pg.ii.numel()),
        "inactive_edges": int(pg.ii_inac.numel()),
    }
