from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .build_compact_bank import artifact_dir, build_faiss_indices


def main() -> int:
    parser = argparse.ArgumentParser(description="Построить FAISS-индекс готового компактного банка")
    parser.add_argument("--map", required=True)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--mode", choices=("flat", "ivf", "both"), default="both")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--train-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.artifact or artifact_dir(args.map)
    bank_path = output / "compact_bank.npz"
    if not bank_path.exists():
        raise SystemExit(f"сначала постройте компактный банк: {bank_path}")
    targets = []
    if args.mode in ("flat", "both"):
        targets.append(output / "compact_flat.faiss")
    if args.mode in ("ivf", "both"):
        targets.append(output / "compact_ivf.faiss")
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"индексы уже существуют: {existing}; нужен --overwrite")

    bank = np.load(bank_path)
    created = build_faiss_indices(
        bank["desc"], output, args.mode, args.nlist, args.train_samples, args.seed,
    )
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["faiss_indices"] = sorted(set(metadata.get("faiss_indices", [])) | set(created))
    metadata["faiss_nlist_requested"] = args.nlist
    metadata["faiss_train_samples"] = args.train_samples
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"artifact": str(output), "created": created}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
