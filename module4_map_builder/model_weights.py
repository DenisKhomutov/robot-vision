"""Local model-weight cache used only by the offline map builder."""
from pathlib import Path
import torch

WEIGHTS = Path(__file__).resolve().parent / "weights"


def configure_local_model_weights() -> None:
    checkpoint_dir = WEIGHTS / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for weight in WEIGHTS.glob("*.pth"):
        link = checkpoint_dir / weight.name
        if not link.exists():
            link.symlink_to(weight.resolve())
    torch.hub.set_dir(str(WEIGHTS))
