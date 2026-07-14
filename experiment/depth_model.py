"""Общая загрузка Depth Anything V2 Metric Outdoor Small из локальных весов.

Веса лежат в weights/depth_anything_v2_metric_outdoor_small (скачаны один раз).
Если папки нет — падает с понятной подсказкой (не лезет молча в интернет).
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# размер модели: DEPTH_MODEL_SIZE=small|base (по умолчанию base — чище геометрия)
_SIZE = os.getenv("DEPTH_MODEL_SIZE", "base").lower()
_DIRS = {
    "small": Path("weights/depth_anything_v2_metric_outdoor_small"),
    "base": Path("weights/depth_anything_v2_metric_outdoor_base"),
    "large": Path("weights/depth_anything_v2_metric_outdoor_large"),
}


def load_model() -> tuple[object, object, str]:
    model_dir = _DIRS.get(_SIZE, _DIRS["base"])
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Нет локальных весов: {model_dir}\n"
            f"Скачай один раз (repo: Depth-Anything-V2-Metric-Outdoor-{_SIZE.capitalize()}-hf):\n"
            "  from huggingface_hub import snapshot_download\n"
            f"  snapshot_download('depth-anything/Depth-Anything-V2-Metric-Outdoor-{_SIZE.capitalize()}-hf',\n"
            f"      local_dir='{model_dir}')"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir).to(device).eval()
    print(f"[INFO] depth-модель: {model_dir.name}")
    return processor, model, device


def infer_depth(image: Image.Image, processor: object, model: object, device: str) -> np.ndarray:
    """Метрическая глубина (метры) в разрешении исходного кадра, HxW float32."""
    w, h = image.size
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])[0]
    return result["predicted_depth"].squeeze().float().cpu().numpy()
