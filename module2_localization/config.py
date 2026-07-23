from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parent
MAPS_DIR = _MODULE_ROOT / "maps"

DEFAULT_MAP = "map_rig3"
ROUTE_CAM = "_c2"            # риг-карта: маршрут по ЦЕНТРАЛЬНОЙ камере (робот смотрит одну)

QUERY_KPTS = 8192            # потолок точек ALIKED
QUERY_DET_THRESHOLD = 0.05   
QUERY_NMS_RADIUS = 2
MAX_ERROR = 12.0             

MIN_INLIERS = 20
MOVE_EPS = 0.06              # смещение (ед. карты) за окно ниже -> стоим (курс держим, команда hold)

LOOKAHEAD_NODES = 12         # упреждение цели в узлах эталона
DEADZONE_DEG = 4.0           # азимут меньше -> straight
STEER_MODE = "pursuit"       # "pursuit" | "stanley"

# демон (ноутбук-командир) шлёт в свой локальный NATS (127.0.0.1);
# viz на ЭТОЙ машине цепляется к NATS командира по его IP.
NATS_HOST = "192.168.30.133"
NATS_URL = "nats://127.0.0.1:4222"
NATS_TOPIC = "robot.vision.localization"

CAMERA = "0"
