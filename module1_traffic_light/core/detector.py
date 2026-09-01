from loguru import logger
from PIL import Image
from ultralytics import YOLO

from .. import config

Box = tuple[int, int, int, int]


class TrafficLightDetector:
    def __init__(self) -> None:
        self.model = YOLO(config.DETECTOR_WEIGHTS)

    def detect(self, image: Image.Image) -> list[Box]:
        results = self.model(
            image, 
            conf=config.DET_CONF, 
            imgsz = config.DET_IMGSZ, 
            verbose=False)
        
        boxes = results[0].boxes

        paired = sorted(
            zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True),
            key=lambda t: t[1],
            reverse=True,
        )
        
        return [(int(x1), int(y1), int(x2), int(y2)) 
                for (x1, y1, x2, y2), _ in paired]


_detector: TrafficLightDetector | None = None


def init_detector() -> None:
    global _detector
    if _detector is not None:
        return
    logger.info("Загрузка детектора светофора")
    _detector = TrafficLightDetector()
    logger.success("Детектор светофора загружен")


def get_detector() -> TrafficLightDetector:
    if _detector is None:
        raise RuntimeError("Detector не инициализирован")
    return _detector
