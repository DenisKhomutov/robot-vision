import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class RuntimeMap:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    camera_size: tuple[int, int]
    points: np.ndarray
    scale: float | None
    descriptors: torch.Tensor
    owners: torch.Tensor
    points3d: np.ndarray
    descriptor_nodes: torch.Tensor
    route: np.ndarray
    route_forward: np.ndarray
    route_cumulative: np.ndarray
    node_step: float
    image_count: int


def load_runtime_map(
    root: Path,
    map_name: str,
    device: str,
    route_cam=None,
    route_range=None,
    route_nodes=None,
    back_facing=False,
    bank_path=None,
):
    work = root / "maps" / map_name
    runtime = np.load(work / "runtime.npz")
    fx, fy, cx, cy = runtime["K"]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    distortion = runtime["dist"].astype(np.float64)
    camera_size = tuple(int(value) for value in runtime["size"])
    points = runtime["points"]
    scale_file = work / "scale.json"
    scale = json.loads(scale_file.read_text())["scale_m_per_unit"] if scale_file.exists() else None

    bank = np.load(Path(bank_path) if bank_path is not None else work / "aliked_bank.npz")
    descriptors = torch.from_numpy(bank["desc"].astype(np.float32)).half().to(device)
    owners = torch.from_numpy(bank["owner"].astype(np.int64)).to(device)
    points3d = bank["xyz"]
    if "align" in runtime:
        points3d = points3d @ runtime["align"].T

    positions = dict(zip(runtime["names"].tolist(), runtime["pos"]))
    forwards = dict(zip(runtime["names"].tolist(), runtime["fwd"]))
    order = sorted(positions)
    if route_cam:
        order = [name for name in order if route_cam in name]
    if route_range:
        start, stop = route_range
        order = [name for name in order if start <= int("".join(filter(str.isdigit, name)) or 0) < stop]
    position_array = np.array([positions[name] for name in order])
    if len(position_array) > 5:
        segments = np.linalg.norm(np.diff(position_array, axis=0), axis=1)
        threshold = 10 * np.median(segments)
        keep = [True] * len(order)
        for index in range(1, len(order) - 1):
            if segments[index - 1] > threshold and segments[index] > threshold:
                keep[index] = False
        dropped = len(order) - sum(keep)
        if dropped:
            print(f"[карта] выкинуто выбросов маршрута: {dropped}")
        order = [name for name, retain in zip(order, keep) if retain]
    if route_nodes and len(order) > route_nodes:
        print(f"[карта] маршрут обрезан: {len(order)} -> {route_nodes} узлов")
        order = order[:route_nodes]

    route = np.array([positions[name] for name in order])
    route_forward = np.array([forwards[name] for name in order])
    if back_facing:
        route_forward = -route_forward
    segments = np.linalg.norm(np.diff(route, axis=0), axis=1)
    route_cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    node_step = float(np.median(segments)) if len(segments) else 1.0

    nearest_node = np.empty(len(points3d), np.int64)
    route32 = route.astype(np.float32)
    for offset in range(0, len(points3d), 20000):
        segment_points = points3d[offset : offset + 20000].astype(np.float32)
        distances = ((segment_points[:, None, :] - route32[None, :, :]) ** 2).sum(2)
        nearest_node[offset : offset + 20000] = distances.argmin(1)
    descriptor_nodes = torch.from_numpy(nearest_node[bank["owner"]]).to(device)

    return RuntimeMap(
        camera_matrix,
        distortion,
        camera_size,
        points,
        scale,
        descriptors,
        owners,
        points3d,
        descriptor_nodes,
        route,
        route_forward,
        route_cumulative,
        node_step,
        len(order),
    )
