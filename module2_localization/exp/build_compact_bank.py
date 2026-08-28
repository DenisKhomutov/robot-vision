from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from .. import config


EXP_ROOT = Path(__file__).resolve().parent


def artifact_dir(map_name: str) -> Path:
    return EXP_ROOT / "artifacts" / Path(map_name)


def normalize(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float32, copy=False)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def medoid_near_center(rows: np.ndarray) -> np.ndarray:
    center = rows.mean(axis=0)
    center /= max(float(np.linalg.norm(center)), 1e-12)
    return rows[int(np.argmax(rows @ center))]


def point_prototypes(rows: np.ndarray, count: int) -> list[np.ndarray]:
    rows = normalize(rows)
    if count == 1 or len(rows) < 4:
        return [medoid_near_center(rows)]
    center = medoid_near_center(rows)
    other = rows[int(np.argmin(rows @ center))]
    centers = np.stack((center, other))
    labels = np.zeros(len(rows), dtype=np.int8)
    for _ in range(5):
        labels = np.argmax(rows @ centers.T, axis=1)
        if labels.min() == labels.max():
            return [medoid_near_center(rows)]
        centers = np.stack([
            normalize(rows[labels == cluster].mean(axis=0, keepdims=True))[0]
            for cluster in range(2)
        ])
    result = []
    for cluster in range(2):
        members = rows[labels == cluster]
        result.append(members[int(np.argmax(members @ centers[cluster]))])
    return result


def import_faiss():
    try:
        return importlib.import_module("faiss")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "FAISS не установлен. compact_bank.npz уже можно использовать с torch-exact; "
            "для индекса установите совместимую с этой машиной сборку FAISS."
        ) from exc


def build_faiss_indices(desc: np.ndarray, output: Path, mode: str,
                        nlist: int, train_samples: int, seed: int) -> list[str]:
    faiss = import_faiss()
    vectors = normalize(desc)
    dimension = vectors.shape[1]
    created = []
    if mode in ("flat", "both"):
        flat = faiss.IndexFlatIP(dimension)
        flat.add(vectors)
        faiss.write_index(flat, str(output / "compact_flat.faiss"))
        created.append("compact_flat.faiss")
    if mode in ("ivf", "both"):
        effective_nlist = min(int(nlist), max(1, len(vectors) // 40))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, effective_nlist, faiss.METRIC_INNER_PRODUCT,
        )
        rng = np.random.default_rng(seed)
        if len(vectors) > train_samples:
            sample = vectors[rng.choice(len(vectors), train_samples, replace=False)]
        else:
            sample = vectors
        index.train(np.ascontiguousarray(sample))
        index.add(vectors)
        faiss.write_index(index, str(output / "compact_ivf.faiss"))
        created.append("compact_ivf.faiss")
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Построить компактный ALIKED-банк полной карты")
    parser.add_argument("--map", required=True, help="карта относительно module2_localization/maps")
    parser.add_argument("--prototypes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--faiss", choices=("none", "flat", "ivf", "both"), default="none")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--train-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = config.MAPS_DIR / args.map
    bank_path = source / "aliked_bank.npz"
    runtime_path = source / "runtime.npz"
    if not bank_path.exists() or not runtime_path.exists():
        raise SystemExit(f"нет полного runtime-комплекта: {source}")
    output = args.output or artifact_dir(args.map)
    output.mkdir(parents=True, exist_ok=True)
    compact_path = output / "compact_bank.npz"
    metadata_path = output / "metadata.json"
    if not args.overwrite and (compact_path.exists() or metadata_path.exists()):
        raise SystemExit(f"результат уже существует: {output}; нужен --overwrite")

    started = time.perf_counter()
    bank = np.load(bank_path)
    desc = bank["desc"]
    owner = bank["owner"].astype(np.int64, copy=False)
    xyz = bank["xyz"].astype(np.float32, copy=False)
    if len(desc) != len(owner) or (len(owner) and (owner.min() < 0 or owner.max() >= len(xyz))):
        raise SystemExit("исходный aliked_bank.npz повреждён")

    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    boundaries = np.flatnonzero(np.diff(sorted_owner)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(order)]))
    prototypes = []
    prototype_owner = []
    for number, (start, stop) in enumerate(zip(starts, stops), 1):
        point_id = int(sorted_owner[start])
        rows = desc[order[start:stop]]
        selected = point_prototypes(rows, args.prototypes)
        prototypes.extend(selected)
        prototype_owner.extend([point_id] * len(selected))
        if number % 25_000 == 0:
            print(f"[compact] точки {number}/{len(starts)}, прототипы {len(prototypes)}", flush=True)

    compact_desc = normalize(np.asarray(prototypes, dtype=np.float32)).astype(np.float16)
    compact_owner = np.asarray(prototype_owner, dtype=np.int32)
    np.savez_compressed(
        compact_path,
        desc=compact_desc,
        owner=compact_owner,
        xyz=xyz,
    )
    indices = []
    if args.faiss != "none":
        indices = build_faiss_indices(
            compact_desc, output, args.faiss, args.nlist, args.train_samples, args.seed,
        )

    elapsed = time.perf_counter() - started
    metadata = {
        "source_map": args.map,
        "source_bank": str(bank_path.resolve()),
        "prototype_method": "spherical_two_medoid" if args.prototypes == 2 else "center_medoid",
        "prototypes_requested": args.prototypes,
        "source_descriptors": int(len(desc)),
        "points": int(len(xyz)),
        "compact_descriptors": int(len(compact_desc)),
        "descriptor_dimension": int(compact_desc.shape[1]),
        "compression_ratio": float(len(desc) / len(compact_desc)),
        "compact_dtype": str(compact_desc.dtype),
        "faiss_indices": indices,
        "nlist_requested": args.nlist,
        "build_seconds": elapsed,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[готово] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
