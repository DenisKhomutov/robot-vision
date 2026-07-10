from loguru import logger
from PIL import Image
from ultralytics import YOLO

from .. import config


class SignalClassifier:
    def __init__(self) -> None:
        self.model = YOLO(config.CLASSIFIER_WEIGHTS)

    def classify(self, crop: Image.Image) -> tuple[str, float]:
        square = crop.convert("RGB").resize((224, 224))
        results = self.model(square, verbose=False)
        probs = results[0].probs
        label = str(results[0].names[int(probs.top1)])
        conf = float(probs.top1conf)
        return label, conf


_classifier: SignalClassifier | None = None


def init_classifier() -> None:
    global _classifier
    if _classifier is not None:
        return
    logger.info("Загрузка классификатора сигнала")
    _classifier = SignalClassifier()
    logger.success("Классификатор сигнала загружен")


def get_classifier() -> SignalClassifier:
    if _classifier is None:
        raise RuntimeError("Classifier не инициализирован")
    return _classifier
