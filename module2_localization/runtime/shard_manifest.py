import json
from pathlib import Path


def load_shard_chain(maps_dir: Path, start_map: str) -> list[dict]:
    for manifest in maps_dir.glob("**/manifest.json"):
        data = json.loads(manifest.read_text())
        shards = data.get("shards", [])
        if any(item.get("map") == start_map for item in shards):
            return shards
    raise RuntimeError(f"для шарда {start_map} не найден manifest")


def recovery_map_name(shards: list[dict]) -> str:
    name = str(shards[0].get("source_map", ""))
    if not name:
        raise RuntimeError("manifest шардов не содержит source_map для recovery")
    return name


def validate_runtime_bundle(maps_dir: Path, map_name: str) -> None:
    required = (
        maps_dir / map_name / "runtime.npz",
        maps_dir / map_name / "aliked_bank.npz",
    )
    if not all(path.exists() for path in required):
        raise RuntimeError(f"нет полного runtime-комплекта recovery: {map_name}")


def shard_index_for_global_node(shards: list[dict], global_node: int, min_index: int, max_index: int) -> int:
    for index, item in enumerate(shards):
        if global_node < int(item["core_global_stop_exclusive"]):
            return min(max(index, min_index), max_index)
    return max_index
