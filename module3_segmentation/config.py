from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
SEGMENTATION_WEIGHTS = str(_WEIGHTS_DIR / "best_seg.pt")

SEG_CONF = 0.5
IMGSZ = 640

ALLOWED_PATH_CLASS = "allowed_path"

SEG_VIS_DIR = Path(__file__).resolve().parent / "data" / "seg_photos"

COLOR_ALLOWED = (235, 206, 135)
COLOR_OBSTACLE = (62, 151, 228)
COLOR_TRAPEZOID = (0, 0, 255)
SEG_ALPHA = 0.15

MOTION_TRAPEZOID = [(0.40, 0.75), (0.60, 0.75), (0.95, 1.0), (0.05, 1.0)]
