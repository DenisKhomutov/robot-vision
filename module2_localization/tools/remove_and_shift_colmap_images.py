import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remove-first", type=int, required=True)
    parser.add_argument("--remove-last", type=int, required=True)
    parser.add_argument("--shift-from", type=int, required=True)
    parser.add_argument("--shift", type=int, required=True)
    parser.add_argument("--min-track-length", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    with tempfile.TemporaryDirectory(prefix="colmap_remove_shift_") as temporary:
        root = Path(temporary)
        text = root / "text"
        binary = root / "binary"
        text.mkdir()
        binary.mkdir()
        subprocess.run([
            "colmap", "model_converter", "--input_path", str(args.input),
            "--output_path", str(text), "--output_type", "TXT",
        ], check=True)

        image_lines = (text / "images.txt").read_text().splitlines()
        old_to_new = {}
        removed = []
        rewritten = []
        data_index = 0
        keep_observations = False
        for line in image_lines:
            if not line or line.startswith("#"):
                rewritten.append(line)
                continue
            if data_index % 2 == 0:
                fields = line.split()
                old_id = int(fields[0])
                if args.remove_first <= old_id <= args.remove_last:
                    removed.append((old_id, fields[-1]))
                    keep_observations = False
                else:
                    new_id = old_id + args.shift if old_id >= args.shift_from else old_id
                    if new_id in old_to_new.values():
                        raise RuntimeError(f"duplicate resulting image_id={new_id}")
                    old_to_new[old_id] = new_id
                    fields[0] = str(new_id)
                    rewritten.append(" ".join(fields))
                    keep_observations = True
            elif keep_observations:
                rewritten.append(line)
            data_index += 1
        (text / "images.txt").write_text("\n".join(rewritten) + "\n")

        point_lines = (text / "points3D.txt").read_text().splitlines()
        rewritten = []
        kept_points = 0
        for line in point_lines:
            if not line or line.startswith("#"):
                rewritten.append(line)
                continue
            fields = line.split()
            track = []
            for index in range(8, len(fields), 2):
                old_id = int(fields[index])
                if old_id in old_to_new:
                    track.extend((str(old_to_new[old_id]), fields[index + 1]))
            if len(track) < 2 * args.min_track_length:
                continue
            rewritten.append(" ".join(fields[:8] + track))
            kept_points += 1
        (text / "points3D.txt").write_text("\n".join(rewritten) + "\n")

        subprocess.run([
            "colmap", "model_converter", "--input_path", str(text),
            "--output_path", str(binary), "--output_type", "BIN",
        ], check=True)
        shutil.copytree(binary, args.output)

    print(f"removed_images={len(removed)}")
    print(f"remaining_images={len(old_to_new)}")
    print(f"remaining_points={kept_points}")
    print(f"id_range={min(old_to_new.values())}..{max(old_to_new.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
