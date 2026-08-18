"""Split a runtime localization map into smaller overlapping map directories."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("runtime-shards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Разделить runtime.npz + aliked_bank.npz на перекрывающиеся шарды",
    )
    parser.add_argument("--map", required=True, help="исходная карта внутри module2_localization/maps")
    parser.add_argument("--parts", type=int, required=True, help="число равных основных частей")
    parser.add_argument("--overlap-nodes", type=int, default=25, help="перекрытие с каждой стороны")
    parser.add_argument("--prefix", default=None, help="префикс выходных каталогов")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.parts < 1 or args.overlap_nodes < 0:
        raise SystemExit("нужно parts >= 1 и overlap-nodes >= 0")

    maps = ROOT / "maps"
    source = maps / args.map
    runtime_path = source / "runtime.npz"
    bank_path = source / "aliked_bank.npz"
    if not runtime_path.exists() or not bank_path.exists():
        raise SystemExit(f"нет полного runtime-комплекта в {source}")

    runtime = np.load(runtime_path)
    bank = np.load(bank_path)
    names = runtime["names"]
    pos = runtime["pos"]
    fwd = runtime["fwd"]
    points = runtime["points"]
    xyz = bank["xyz"]
    desc = bank["desc"]
    owner = bank["owner"].astype(np.int64, copy=False)

    if len(points) != len(xyz) or len(desc) != len(owner):
        raise SystemExit("runtime и descriptor bank несовместимы")
    if len(owner) and (owner.min() < 0 or owner.max() >= len(points)):
        raise SystemExit("owner содержит неверные индексы 3D-точек")
    if args.parts > len(names):
        raise SystemExit("частей больше, чем узлов маршрута")

    # Each point belongs to its closest route node. Chunking avoids an Npoints*Nnodes
    # allocation for the 1701-node front map.
    point_node = np.empty(len(points), np.int32)
    route = pos.astype(np.float32, copy=False)
    for start in range(0, len(points), 10_000):
        block = points[start : start + 10_000].astype(np.float32, copy=False)
        distance2 = ((block[:, None, :] - route[None, :, :]) ** 2).sum(axis=2)
        point_node[start : start + len(block)] = distance2.argmin(axis=1)

    edges = np.linspace(0, len(names), args.parts + 1, dtype=int)
    prefix = args.prefix or f"{args.map}_shard"
    created: list[dict[str, object]] = []
    for index in range(args.parts):
        core_start, core_stop = int(edges[index]), int(edges[index + 1])
        start = max(0, core_start - args.overlap_nodes)
        stop = min(len(names), core_stop + args.overlap_nodes)
        point_keep = (point_node >= start) & (point_node < stop)
        old_point_ids = np.flatnonzero(point_keep)
        remap = np.full(len(points), -1, np.int64)
        remap[old_point_ids] = np.arange(len(old_point_ids), dtype=np.int64)
        descriptor_keep = point_keep[owner]
        shard_owner = remap[owner[descriptor_keep]].astype(np.int32)

        name = f"{prefix}_{index + 1:02d}_of_{args.parts:02d}"
        output = maps / name
        if output.exists():
            if not args.overwrite:
                raise SystemExit(f"уже существует {output}; нужен --overwrite")
            shutil.rmtree(output)
        output.mkdir(parents=True)

        runtime_data = {key: runtime[key] for key in runtime.files}
        runtime_data["names"] = names[start:stop]
        runtime_data["pos"] = pos[start:stop]
        runtime_data["fwd"] = fwd[start:stop]
        runtime_data["points"] = points[old_point_ids]
        np.savez_compressed(output / "runtime.npz", **runtime_data)
        np.savez_compressed(
            output / "aliked_bank.npz",
            desc=desc[descriptor_keep],
            owner=shard_owner,
            xyz=xyz[old_point_ids],
        )

        metadata = {
            "source_map": args.map,
            "index": index,
            "number": index + 1,
            "parts": args.parts,
            "global_node_start": start,
            "global_node_stop_exclusive": stop,
            "core_global_start": core_start,
            "core_global_stop_exclusive": core_stop,
            "local_core_start": core_start - start,
            "local_core_stop_exclusive": core_stop - start,
            "overlap_nodes": args.overlap_nodes,
            "route_nodes": stop - start,
            "points": int(len(old_point_ids)),
            "descriptors": int(descriptor_keep.sum()),
        }
        (output / "shard.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        created.append({"map": name, **metadata})
        LOGGER.info(
            "[%d/%d] %s: global [%d,%d), core [%d,%d), nodes=%d, points=%d, desc=%d",
            index + 1,
            args.parts,
            name,
            start,
            stop,
            core_start,
            core_stop,
            stop - start,
            len(old_point_ids),
            descriptor_keep.sum(),
        )

    manifest = maps / f"{prefix}_manifest.json"
    manifest.write_text(json.dumps({"source_map": args.map, "shards": created}, ensure_ascii=False, indent=2) + "\n")
    LOGGER.info("manifest: %s", manifest)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
