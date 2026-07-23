"""Локальные веса ALIKED/LightGlue вместо загрузки из сети.

Обе модели тянут веса через torch.hub, то есть качают с github в общий кэш
~/.cache/torch. На роботе сети может не быть, а кэш — не часть проекта: чистая
машина молча уходит в интернет и падает без него. Уводим hub в weights/ модуля,
где файлы уже лежат рядом с кодом.

Веса в weights/ лежат плоско, а hub ищет их в подпапке checkpoints/ — поэтому
раскладываем симлинки. Чего нет локально, hub скачает туда же (не в общий кэш).
"""
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
