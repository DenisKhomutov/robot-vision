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
    parser.add_argument("--overlap-nodes", type=int, default=25,
                        help="запасной край для точек без COLMAP-трека (или fallback без sparse/0)")
    parser.add_argument("--prefix", default=None, help="префикс выходных каталогов")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-track-visibility", action="store_true",
                        help="не использовать COLMAP-треки (sparse/0); резать только по ближайшему узлу")
    parser.add_argument("--balance", choices=["descriptors", "nodes"], default="descriptors",
                        help="границы шардов: равный вес по дескрипторам (по умолчанию) или равные отрезки маршрута")
    return parser.parse_args()


def balanced_edges(weight_per_node: np.ndarray, parts: int) -> np.ndarray:
    """Границы шардов так, чтобы суммарный вес (число дескрипторов) на шард был
    примерно одинаковым — иначе густые участки маршрута дают тяжёлые шарды и
    неравномерную скорость выдачи команд/предзагрузки."""
    cum = np.concatenate(([0.0], np.cumsum(weight_per_node)))
    total = cum[-1]
    if total <= 0:
        return np.linspace(0, len(weight_per_node), parts + 1, dtype=int)
    targets = np.linspace(0, total, parts + 1)
    edges = np.searchsorted(cum, targets, side="left")
    edges = np.clip(edges, 0, len(weight_per_node))
    edges[0], edges[-1] = 0, len(weight_per_node)


    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    edges[-1] = len(weight_per_node)
    return edges


def load_point_track_nodes(source: Path, names: np.ndarray, num_points: int) -> list[set] | None:
    """Для каждой 3D-точки — множество узлов (индексов кадров), которые её реально
    наблюдали по COLMAP-треку. Точнее, чем «ближайший узел»: дальний ориентир
    (фасад, столб) виден с десятков кадров подряд, а не с одного ближайшего."""
    sparse = source / "sparse" / "0"
    if not sparse.exists():
        return None
    import pycolmap
    rec = pycolmap.Reconstruction(str(sparse))
    name_to_node = {name: i for i, name in enumerate(names.tolist())}
    image_id_to_node = {}
    for image in rec.images.values():
        node = name_to_node.get(image.name)
        if node is not None:
            image_id_to_node[image.image_id] = node
    if len(image_id_to_node) != len(names):
        LOGGER.warning(
            "сопоставилось %d/%d кадров COLMAP<->runtime.npz — возможно, разные модели",
            len(image_id_to_node), len(names),
        )
    track_nodes: list[set] = []
    for point in rec.points3D.values():
        nodes = {image_id_to_node[e.image_id] for e in point.track.elements if e.image_id in image_id_to_node}
        track_nodes.append(nodes)
    if len(track_nodes) != num_points:
        raise SystemExit(
            f"число точек COLMAP ({len(track_nodes)}) != число точек runtime.npz ({num_points}); "
            "sparse/0 не соответствует этой карте"
        )
    return track_nodes


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

    track_nodes = None if args.no_track_visibility else load_point_track_nodes(source, names, len(points))
    if track_nodes is not None:
        LOGGER.info("нарезка по видимости COLMAP-трека (sparse/0 найден)")
    else:
        LOGGER.info("нарезка по ближайшему узлу (нет sparse/0 или --no-track-visibility)")



    point_node = np.empty(len(points), np.int32)
    route = pos.astype(np.float32, copy=False)
    for start in range(0, len(points), 10_000):
        block = points[start : start + 10_000].astype(np.float32, copy=False)
        distance2 = ((block[:, None, :] - route[None, :, :]) ** 2).sum(axis=2)
        point_node[start : start + len(block)] = distance2.argmin(axis=1)

    if args.balance == "descriptors":


        desc_weight = np.bincount(point_node[owner], minlength=len(names)).astype(np.float64)
        edges = balanced_edges(desc_weight, args.parts)
        LOGGER.info("границы по дескрипторам: %s", edges.tolist())
    else:
        edges = np.linspace(0, len(names), args.parts + 1, dtype=int)
    prefix = args.prefix or f"{args.map}_shard"
    created: list[dict[str, object]] = []
    for index in range(args.parts):
        core_start, core_stop = int(edges[index]), int(edges[index + 1])
        start = max(0, core_start - args.overlap_nodes)
        stop = min(len(names), core_stop + args.overlap_nodes)
        nearest_keep = (point_node >= start) & (point_node < stop)
        if track_nodes is not None:
            track_keep = np.fromiter(
                (any(start <= n < stop for n in nodes) for nodes in track_nodes),
                dtype=bool, count=len(track_nodes),
            )
            point_keep = nearest_keep | track_keep
        else:
            point_keep = nearest_keep
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
            "track_visibility": track_nodes is not None,
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
