from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "external" / "DPVO"
DPVO_REPO = "https://github.com/princeton-vl/DPVO.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone the official Princeton DPVO repository into module3.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--repo", default=DPVO_REPO)
    parser.add_argument("--depth", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target
    if target.exists():
        print(f"DPVO already exists: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--recursive",
        "--depth",
        str(args.depth),
        args.repo,
        str(target),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
