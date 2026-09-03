import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config


def log_catalog(logs_dir: Path) -> list[dict[str, object]]:
    if not logs_dir.exists():
        return []
    result = []
    for path in logs_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        result.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    result.sort(key=lambda item: int(item["mtime_ns"]), reverse=True)
    for item in result:
        item.pop("mtime_ns")
    return result


def resolve_log(logs_dir: Path, name: str) -> Path | None:
    if not name or name != Path(name).name:
        return None
    candidate = logs_dir / name
    if candidate.is_symlink():
        return None
    try:
        resolved_root = logs_dir.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    if resolved.parent != resolved_root or not resolved.is_file():
        return None
    return resolved


def map_payload(name: str, cloud_limit: int = 12_000) -> dict[str, object]:
    path = config.MAPS_DIR / name / "runtime.npz"
    data = np.load(path)
    pos = data["pos"][:, [0, 2]]
    points = data["points"][:, [0, 2]]
    if len(points) > cloud_limit:
        indices = np.linspace(0, len(points) - 1, cloud_limit, dtype=int)
        points = points[indices]
    return {"name": name, "route": pos.round(4).tolist(), "cloud": points.round(4).tolist()}


def map_catalog() -> list[dict[str, object]]:
    result = []

    for runtime_file in sorted(config.MAPS_DIR.rglob("runtime.npz")):
        path = runtime_file.parent
        if not (path / "aliked_bank.npz").exists():
            continue
        name = path.relative_to(config.MAPS_DIR).as_posix()
        camera = "front" if "front" in name else "rear" if "rear" in name else "unknown"
        item: dict[str, object] = {"name": name, "camera": camera, "shard": False}
        metadata = path / "shard.json"
        if metadata.exists():
            shard = json.loads(metadata.read_text())
            item.update(
                {
                    "shard": True,
                    "number": shard["number"],
                    "parts": shard["parts"],
                    "start": shard["global_node_start"],
                    "stop": shard["global_node_stop_exclusive"],
                    "core_start": shard["core_global_start"],
                    "core_stop": shard["core_global_stop_exclusive"],
                }
            )
        result.append(item)
    return result
