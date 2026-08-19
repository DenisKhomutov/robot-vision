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
    # светофор не найден
    if not boxes:
        return {"status": "no_traffic_light", "signal": None, "confidence": None, "box": None}
    box = boxes[0]
    crop = image.crop(box)
    label, conf = get_classifier().classify(crop)
    _save_crop(crop, label)
    # светофор найден, сигнал не распознан
    if label == config.UNKNOWN_LABEL or conf < config.CLS_CONF:
        return {"status": "light_unclassified", "signal": None, "confidence": round(conf, 4), "box": box}
    # светофор найден и сигнал распознан
    return {"status": "light_classified", "signal": label, "confidence": round(conf, 4), "box": box}
