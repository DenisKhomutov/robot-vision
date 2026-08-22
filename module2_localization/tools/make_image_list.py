"""Write a deterministic COLMAP image list from a dataset directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--exclude", action="append", default=[],
                        help="image name to omit; may be repeated")
    args = parser.parse_args()

    all_names = sorted(path.name for path in args.images.glob("*.jpg"))[:args.first]
    if len(all_names) != args.first:
        raise SystemExit(f"requested first {args.first} images, found {len(all_names)}")
    excluded = set(args.exclude)
    unknown = excluded - set(all_names)
    if unknown:
        raise SystemExit(f"excluded names are outside selected range: {sorted(unknown)}")
    names = [name for name in all_names if name not in excluded]
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(names) + "\n")
    print(f"wrote {len(names)} names: {names[0]} .. {names[-1]}; excluded={sorted(excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
