from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parent
MAPS_DIR = _MODULE_ROOT / "maps"

DEFAULT_MAP = "map_office_ref"

QUERY_KPTS = 8192            # потолок точек ALIKED
QUERY_DET_THRESHOLD = 0.05   
QUERY_NMS_RADIUS = 2
MAX_ERROR = 12.0             

MIN_INLIERS = 20             

LOOKAHEAD_NODES = 12         # упреждение цели в узлах эталона
DEADZONE_DEG = 8.0           # азимут меньше -> straight
STEER_MODE = "pursuit"       # "pursuit" | "stanley"

NATS_URL = "nats://127.0.0.1:4222"
NATS_TOPIC = "robot.vision.localization"

CAMERA = "0"
