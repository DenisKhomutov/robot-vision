from datetime import datetime
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from ultralytics import YOLO

from .. import config


def _mask_color(class_name: str) -> tuple[int, int, int]:
    return config.COLOR_ALLOWED if class_name == config.ALLOWED_PATH_CLASS else config.COLOR_OBSTACLE


class Segmentator:
    def __init__(self) -> None:
        self.model = YOLO(config.SEGMENTATION_WEIGHTS)
        self.use_half = torch.cuda.is_available()

    def segment(self, img: Image.Image) -> list[dict[str, Any]]:
        results = self.model(img, conf=config.SEG_CONF, imgsz=config.IMGSZ, half=self.use_half, verbose=False)
        r = results[0]
        if r.masks is None:
            return []

        names = r.names
        classes = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        polys = r.masks.xy


        poly_points: list[list[tuple[float, float]]] = [[(float(x), float(y)) for x, y in poly] for poly in polys]
        class_names = [str(names[int(c)]) for c in classes]
        conf_values = [float(c) for c in confs]

        self._save_visualization(img, class_names, conf_values, poly_points)


        return [
            {"class": cn, "confidence": round(cv, 4)}
            for cn, cv in zip(class_names, conf_values, strict=True)
            if cn == config.ALLOWED_PATH_CLASS
        ]

    def _save_visualization(
        self,
        img: Image.Image,
        class_names: list[str],
        conf_values: list[float],
        poly_points: list[list[tuple[float, float]]],
    ) -> None:
        config.SEG_VIS_DIR.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]

        overlay = image.copy()
        drawn: list[tuple[Any, str, float, tuple[int, int, int]]] = []
        for class_name, conf, pts in zip(class_names, conf_values, poly_points, strict=True):
            if len(pts) < 3:
                continue
            color = _mask_color(class_name)
            np_pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [np_pts], color)
            drawn.append((np_pts, class_name, conf, color))

        blended = cv2.addWeighted(overlay, config.SEG_ALPHA, image, 1 - config.SEG_ALPHA, 0)

        for np_pts, class_name, conf, color in drawn:
            cv2.polylines(blended, [np_pts], True, color, 2)
            x, y = int(np_pts[0][0]), int(np_pts[0][1])
            label = f"{class_name} {conf:.0%}"
            cv2.putText(blended, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        trapezoid = np.array([(int(px * w), int(py * h)) for px, py in config.MOTION_TRAPEZOID], dtype=np.int32)
        cv2.polylines(blended, [trapezoid], True, config.COLOR_TRAPEZOID, 3)

        filename = f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(config.SEG_VIS_DIR / filename), blended)


_segmentator: Segmentator | None = None


def init_segmentator() -> None:
    global _segmentator
    if _segmentator is not None:
        return
    logger.info("Загрузка сегментатора сцены")
    _segmentator = Segmentator()
    logger.success("Сегментатор сцены загружен")


def get_segmentator() -> Segmentator:
    if _segmentator is None:
        raise RuntimeError("Segmentator не инициализирован")
    return _segmentator
