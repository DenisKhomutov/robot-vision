from pathlib import Path

import torch

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


def use_local_weights():
    ckpt = WEIGHTS / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    for p in WEIGHTS.glob("*.pth"):
        link = ckpt / p.name
        if not link.exists():
            link.symlink_to(p.resolve())
    torch.hub.set_dir(str(WEIGHTS))
