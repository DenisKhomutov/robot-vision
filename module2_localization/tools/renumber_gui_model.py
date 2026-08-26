import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime = np.load(args.runtime, allow_pickle=True)
    names = [Path(str(name)).name for name in runtime["names"]]
    new_id_by_name = {name: node + 1 for node, name in enumerate(names)}
    source = Path(args.input)
    output = Path(args.output)

    with tempfile.TemporaryDirectory(prefix="colmap_gui_ids_") as temp_name:
        temp = Path(temp_name)
        text = temp / "text"
        binary = temp / "binary"
        text.mkdir()
        binary.mkdir()
        subprocess.run([
            "colmap", "model_converter", "--input_path", str(source),
            "--output_path", str(text), "--output_type", "TXT",
        ], check=True)

        image_lines = (text / "images.txt").read_text().splitlines()
        old_to_new = {}
        rewritten = []
        data_line = 0
        for line in image_lines:
            if not line or line.startswith("#"):
                rewritten.append(line)
                continue
            if data_line % 2 == 0:
                fields = line.split()
                old_id = int(fields[0])
                name = Path(fields[-1]).name
                if name not in new_id_by_name:
                    raise RuntimeError(f"нет runtime-узла для {name}")
                new_id = new_id_by_name[name]
                old_to_new[old_id] = new_id
                fields[0] = str(new_id)
                line = " ".join(fields)
            rewritten.append(line)
            data_line += 1
        if len(old_to_new) != len(names):
            raise RuntimeError(f"модель={len(old_to_new)}, runtime={len(names)}")
        (text / "images.txt").write_text("\n".join(rewritten) + "\n")

        point_lines = (text / "points3D.txt").read_text().splitlines()
        rewritten = []
        for line in point_lines:
            if not line or line.startswith("#"):
                rewritten.append(line)
                continue
            fields = line.split()
            for index in range(8, len(fields), 2):
                fields[index] = str(old_to_new[int(fields[index])])
            rewritten.append(" ".join(fields))
        (text / "points3D.txt").write_text("\n".join(rewritten) + "\n")

        subprocess.run([
            "colmap", "model_converter", "--input_path", str(text),
            "--output_path", str(binary), "--output_type", "BIN",
        ], check=True)
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(binary, output)
    print(f"готово: {output}; image_id = runtime_node + 1; изображений={len(names)}")


if __name__ == "__main__":
    main()
