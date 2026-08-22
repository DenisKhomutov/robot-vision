"""Merge independently extracted video parts into one continuously numbered dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from .extract_adaptive import displacement, grid_points

ROOT = Path(__file__).resolve().parents[1]


def parse_part(value: str) -> tuple[str, int]:
    name, sep, start = value.partition(":")
    return name, int(start) if sep else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Объединить части датасета с общей нумерацией")
    parser.add_argument("--part", action="append", required=True,
                        help="каталог внутри data и минимальный source frame: DIR[:START]")
    parser.add_argument("--out", required=True, help="итоговый каталог внутри data")
    parser.add_argument("--boundary-min-move", type=float, default=2.0,
                        help="удалять начальные кадры новой части ближе этого порога к предыдущему")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = ROOT / "data" / args.out
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"{output} уже существует; нужен --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    manifest: dict[str, object] = {
        "output": str(output.resolve()),
        "boundary_min_move_px": args.boundary_min_move,
        "parts": [],
    }
    output_index = 0
    previous_gray = None

    for part_number, raw in enumerate(args.part, 1):
        name, minimum_source_frame = parse_part(raw)
        directory = ROOT / "data" / name
        extraction_path = directory / "extraction.json"
        extraction = json.loads(extraction_path.read_text()) if extraction_path.exists() else {}
        files = [p for p in sorted(directory.glob("*.jpg")) if int(p.stem) >= minimum_source_frame]
        if not files:
            raise SystemExit(f"нет подходящих кадров в {directory} начиная с {minimum_source_frame}")

        skipped_boundary: list[int] = []
        if previous_gray is not None:
            points = grid_points(previous_gray.shape[1], previous_gray.shape[0])
            while files:
                gray = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
                shift = displacement(previous_gray, gray, points)
                if shift is None or shift >= args.boundary_min_move:
                    break
                skipped_boundary.append(int(files.pop(0).stem))
            if not files:
                raise SystemExit(f"вся часть {name} исключена как неподвижный стык")

        first_output = output_index
        source_frames = []
        for source in files:
            shutil.copy2(source, output / f"{output_index:06d}.jpg")
            source_frames.append(int(source.stem))
            output_index += 1
        previous_gray = cv2.imread(str(files[-1]), cv2.IMREAD_GRAYSCALE)
        manifest["parts"].append({
            "part": part_number,
            "directory": name,
            "source_video": extraction.get("source"),
            "minimum_source_frame": minimum_source_frame,
            "skipped_boundary_frames": skipped_boundary,
            "output_start": first_output,
            "output_stop_exclusive": output_index,
            "kept": len(files),
            "first_source_frame": source_frames[0],
            "last_source_frame": source_frames[-1],
        })

    manifest["total_frames"] = output_index
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"готово: {output_index} кадров -> {output}")
    for part in manifest["parts"]:
        print(f"  part {part['part']}: output [{part['output_start']},{part['output_stop_exclusive']}) "
              f"source [{part['first_source_frame']},{part['last_source_frame']}], "
              f"стык удалён {part['skipped_boundary_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
