from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


RUNTIME_KEYS = {"names", "pos", "fwd", "K", "dist", "size", "points"}
BANK_KEYS = {"desc", "owner", "xyz"}
SHARD_KEYS = {
    "source_map", "index", "number", "parts", "global_node_start",
    "global_node_stop_exclusive", "core_global_start",
    "core_global_stop_exclusive", "local_core_start",
    "local_core_stop_exclusive", "overlap_nodes", "track_visibility",
    "route_nodes", "points", "descriptors",
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.files = 0
        self.shards = 0
        self.nodes = 0
        self.points = 0
        self.descriptors = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)


def finite(audit: Audit, value: np.ndarray, label: str) -> None:
    if np.issubdtype(value.dtype, np.number):
        audit.require(bool(np.isfinite(value).all()), f"{label}: есть NaN/Inf")


def load_npz(audit: Audit, path: Path, required: set[str]) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        audit.errors.append(f"нет файла: {path}")
        return None
    audit.require(path.stat().st_size > 0, f"пустой файл: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            audit.require(not missing, f"{path}: нет ключей {sorted(missing)}")
            result = {key: archive[key] for key in archive.files}
    except Exception as exc:
        audit.errors.append(f"{path}: архив не читается полностью: {exc}")
        return None
    audit.files += 1
    for key, value in result.items():
        finite(audit, value, f"{path}:{key}")
    return result


def validate_pair(audit: Audit, directory: Path, label: str) -> tuple[dict, dict] | None:
    runtime = load_npz(audit, directory / "runtime.npz", RUNTIME_KEYS)
    bank = load_npz(audit, directory / "aliked_bank.npz", BANK_KEYS)
    if runtime is None or bank is None:
        return None
    names = runtime.get("names", np.empty(0))
    pos = runtime.get("pos", np.empty((0, 3)))
    fwd = runtime.get("fwd", np.empty((0, 3)))
    points = runtime.get("points", np.empty((0, 3)))
    k = runtime.get("K", np.empty(0))
    dist = runtime.get("dist", np.empty(0))
    size = runtime.get("size", np.empty(0))
    desc = bank.get("desc", np.empty((0, 0)))
    owner = bank.get("owner", np.empty(0))
    xyz = bank.get("xyz", np.empty((0, 3)))
    audit.require(names.ndim == 1 and len(names) > 0, f"{label}: names должен быть непустым 1D")
    audit.require(pos.shape == (len(names), 3), f"{label}: pos {pos.shape}, ожидалось ({len(names)}, 3)")
    audit.require(fwd.shape == (len(names), 3), f"{label}: fwd {fwd.shape}, ожидалось ({len(names)}, 3)")
    audit.require(points.ndim == 2 and points.shape[1:] == (3,) and len(points) > 0,
                  f"{label}: неверный points {points.shape}")
    audit.require(k.shape == (4,) and bool((k[:2] > 0).all()), f"{label}: неверный K {k}")
    audit.require(dist.ndim == 1, f"{label}: dist должен быть 1D, получено {dist.shape}")
    audit.require(size.shape == (2,) and bool((size > 0).all()), f"{label}: неверный size {size}")
    audit.require(len(set(map(str, names.tolist()))) == len(names), f"{label}: повторяющиеся names")
    if len(fwd):
        norms = np.linalg.norm(fwd, axis=1)
        audit.require(bool(np.allclose(norms, 1.0, atol=1e-3)),
                      f"{label}: fwd не единичные, диапазон {norms.min():.6g}..{norms.max():.6g}")
    audit.require(desc.ndim == 2 and len(desc) > 0, f"{label}: неверный desc {desc.shape}")
    audit.require(owner.ndim == 1 and len(owner) == len(desc),
                  f"{label}: owner {owner.shape} не соответствует desc {desc.shape}")
    audit.require(xyz.shape == points.shape, f"{label}: xyz {xyz.shape} != points {points.shape}")
    xyz_runtime = xyz @ runtime["align"].T if "align" in runtime and runtime["align"].shape == (3, 3) else xyz
    if xyz_runtime.shape == points.shape:
        audit.require(bool(np.array_equal(xyz_runtime, points) or np.allclose(xyz_runtime, points, rtol=0, atol=1e-7)),
                      f"{label}: преобразованный xyz descriptor bank не совпадает с runtime points")
    if len(owner):
        audit.require(np.issubdtype(owner.dtype, np.integer), f"{label}: owner не integer: {owner.dtype}")
        audit.require(int(owner.min()) >= 0 and int(owner.max()) < len(points),
                      f"{label}: owner вне диапазона 0..{len(points) - 1}: {owner.min()}..{owner.max()}")
    if "align" in runtime:
        align = runtime["align"]
        audit.require(align.shape == (3, 3), f"{label}: align имеет форму {align.shape}")
        if align.shape == (3, 3):
            audit.require(bool(np.allclose(align @ align.T, np.eye(3), atol=1e-5)),
                          f"{label}: align не ортонормальна")
            audit.require(abs(float(np.linalg.det(align)) - 1.0) < 1e-5,
                          f"{label}: det(align) != 1")
    return runtime, bank


def read_json(audit: Audit, path: Path) -> dict | None:
    if not path.is_file():
        audit.errors.append(f"нет файла: {path}")
        return None
    audit.require(path.stat().st_size > 0, f"пустой файл: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        audit.errors.append(f"{path}: неверный JSON: {exc}")
        return None
    audit.files += 1
    audit.require(isinstance(value, dict), f"{path}: корень JSON не object")
    return value if isinstance(value, dict) else None


def validate_route(audit: Audit, maps: Path, route: str) -> None:
    print(f"\n=== {route} ===", flush=True)
    shard_root = maps / route / "front_shard"
    manifest = read_json(audit, shard_root / "manifest.json")
    if manifest is None:
        return
    source_name = manifest.get("source_map")
    shards = manifest.get("shards")
    audit.require(isinstance(source_name, str) and bool(source_name), f"{route}: нет source_map")
    audit.require(isinstance(shards, list) and bool(shards), f"{route}: shards пуст или не list")
    if not isinstance(source_name, str) or not isinstance(shards, list) or not shards:
        return
    source_dir = maps / source_name
    source_pair = validate_pair(audit, source_dir, f"{route}:full")
    expected_dirs = {str(item.get("map", "")).split("/")[-1] for item in shards}
    actual_dirs = {p.name for p in shard_root.iterdir() if p.is_dir()}
    audit.require(expected_dirs == actual_dirs,
                  f"{route}: папки не совпадают с manifest; нет={sorted(expected_dirs-actual_dirs)}, лишние={sorted(actual_dirs-expected_dirs)}")
    previous_core_stop = None
    previous_global_stop = None
    for index, item in enumerate(shards):
        prefix = f"{route}[{index + 1}/{len(shards)}]"
        audit.require(isinstance(item, dict), f"{prefix}: запись manifest не object")
        if not isinstance(item, dict):
            continue
        missing = SHARD_KEYS - set(item)
        audit.require(not missing, f"{prefix}: в manifest нет {sorted(missing)}")
        map_name = item.get("map")
        if not isinstance(map_name, str):
            continue
        directory = maps / map_name
        metadata = read_json(audit, directory / "shard.json")
        pair = validate_pair(audit, directory, prefix)
        if metadata is None or pair is None:
            continue
        runtime, bank = pair
        for key in SHARD_KEYS:
            if key in item and key in metadata:
                audit.require(item[key] == metadata[key], f"{prefix}: {key} manifest={item[key]!r}, shard.json={metadata[key]!r}")
        audit.require(item.get("source_map") == source_name, f"{prefix}: другой source_map")
        audit.require(item.get("index") == index, f"{prefix}: index={item.get('index')}")
        audit.require(item.get("number") == index + 1, f"{prefix}: number={item.get('number')}")
        audit.require(item.get("parts") == len(shards), f"{prefix}: parts={item.get('parts')}")
        start = int(item.get("global_node_start", -1))
        stop = int(item.get("global_node_stop_exclusive", -1))
        core_start = int(item.get("core_global_start", -1))
        core_stop = int(item.get("core_global_stop_exclusive", -1))
        local_start = int(item.get("local_core_start", -1))
        local_stop = int(item.get("local_core_stop_exclusive", -1))
        audit.require(0 <= start <= core_start < core_stop <= stop, f"{prefix}: неверные диапазоны")
        audit.require(local_start == core_start - start and local_stop == core_stop - start,
                      f"{prefix}: неверные локальные core-границы")
        audit.require(int(item.get("route_nodes", -1)) == stop - start == len(runtime["names"]),
                      f"{prefix}: route_nodes/диапазон/names расходятся")
        audit.require(int(item.get("points", -1)) == len(runtime["points"]), f"{prefix}: points не совпадает")
        audit.require(int(item.get("descriptors", -1)) == len(bank["desc"]), f"{prefix}: descriptors не совпадает")
        if previous_core_stop is not None:
            audit.require(core_start == previous_core_stop,
                          f"{prefix}: разрыв/наложение core: {previous_core_stop} -> {core_start}")
            audit.require(start < int(previous_global_stop), f"{prefix}: нет перекрытия с предыдущим шардом")
        previous_core_stop = core_stop
        previous_global_stop = stop
        if source_pair is not None:
            full_runtime, _ = source_pair
            full_names = full_runtime["names"]
            audit.require(stop <= len(full_names), f"{prefix}: диапазон выходит за full map")
            if 0 <= start < stop <= len(full_names):
                audit.require(bool(np.array_equal(runtime["names"], full_names[start:stop])),
                              f"{prefix}: names не являются срезом full map")
                audit.require(bool(np.allclose(runtime["pos"], full_runtime["pos"][start:stop], rtol=0, atol=1e-10)),
                              f"{prefix}: pos не совпадает с full map")
                audit.require(bool(np.allclose(runtime["fwd"], full_runtime["fwd"][start:stop], rtol=0, atol=1e-10)),
                              f"{prefix}: fwd не совпадает с full map")
            for key in ("K", "dist", "size", "align"):
                if key in runtime or key in full_runtime:
                    audit.require(key in runtime and key in full_runtime and np.array_equal(runtime[key], full_runtime[key]),
                                  f"{prefix}: {key} не совпадает с full map")
        allowed = {"runtime.npz", "aliked_bank.npz", "shard.json", "scale.json"}
        actual_files = {p.name for p in directory.iterdir() if p.is_file()}
        audit.warn(actual_files <= allowed, f"{prefix}: лишние файлы {sorted(actual_files-allowed)}")
        audit.shards += 1
        audit.nodes += len(runtime["names"])
        audit.points += len(runtime["points"])
        audit.descriptors += len(bank["desc"])
        print(f"[{index + 1:02d}/{len(shards):02d}] {map_name}: nodes={len(runtime['names'])}, points={len(runtime['points'])}, desc={len(bank['desc'])} OK", flush=True)
    if source_pair is not None:
        full_nodes = len(source_pair[0]["names"])
        audit.require(int(shards[0].get("core_global_start", -1)) == 0, f"{route}: core начинается не с 0")
        audit.require(int(shards[-1].get("core_global_stop_exclusive", -1)) == full_nodes,
                      f"{route}: core заканчивается не на конце full map ({full_nodes})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-dir", type=Path, default=Path("module2_localization/maps"))
    parser.add_argument("--routes", nargs="+", default=["1-2", "2-1", "3-1"])
    args = parser.parse_args()
    audit = Audit()
    for route in args.routes:
        validate_route(audit, args.maps_dir, route)
    print("\n=== ИТОГ ===")
    print(f"routes={len(args.routes)} shards={audit.shards} files={audit.files} nodes_with_overlap={audit.nodes} points={audit.points} descriptors={audit.descriptors}")
    for warning in audit.warnings:
        print(f"WARNING: {warning}")
    for error in audit.errors:
        print(f"ERROR: {error}")
    if audit.errors:
        print(f"FAIL: ошибок {len(audit.errors)}, предупреждений {len(audit.warnings)}")
        return 1
    print(f"OK: структурных ошибок нет, предупреждений {len(audit.warnings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
