from datetime import datetime
from typing import Any

from PIL import Image

from .. import config
from ..core.classifier import get_classifier
from ..core.detector import get_detector


def _save_crop(crop: Image.Image, label: str) -> None:
    config.CROPS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{label}.jpg"
    crop.convert("RGB").save(config.CROPS_DIR / filename)


def analyze(image: Image.Image) -> dict[str, Any]:
    boxes = get_detector().detect(image)

    if not boxes:
        return {"status": "no_traffic_light", "signal": None, "confidence": None, "box": None}

    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    crop = image.crop(box)
    label, conf = get_classifier().classify(crop)
    if config.SAVE_CROPS:
        _save_crop(crop, label)

    if label == config.UNKNOWN_LABEL or conf < config.CLS_CONF:
        return {"status": "light_unclassified", "signal": None, "confidence": round(conf, 4), "box": box}

    return {"status": "light_classified", "signal": label, "confidence": round(conf, 4), "box": box}
