from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DETECTOR_WEIGHTS = str(_WEIGHTS_DIR / "best_det.pt")
CLASSIFIER_WEIGHTS = str(_WEIGHTS_DIR / "best_cls.pt")

DET_IMGSZ = 960

CROPS_DIR = Path(__file__).resolve().parent / "data" / "crops"

DET_CONF = 0.25
CLS_CONF = 0.75

UNKNOWN_LABEL = "unknown"
