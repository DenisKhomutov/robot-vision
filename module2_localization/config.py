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

ROUTE_NODES = 310            # хвост карты у стены разваливается: режем маршрут по этот узел

# отбраковка неправдоподобных поз: ВПЕРЁД можно не быстрее этого, назад — без ограничения
MAX_NODES_PER_SEC = 40       # узлов маршрута в секунду (узел ~5 см -> ~2 м/с)
MIN_NODE_JUMP = 10           # запас на редкие кадры и рывки, узлов
MAX_REJECTS = 5              # столько отказов подряд -> верим кадру (иначе застрянем навсегда)

STOP_CONFIRM = 3             # столько кадров подряд про конец маршрута -> латч STOP
STOP_MIN_INLIERS = 40        # и не меньше стольких инлайеров в каждом

LOOKAHEAD_NODES = 12         # упреждение цели в узлах эталона
DEADZONE_DEG = 4.0           # азимут меньше -> straight
STEER_MODE = "pursuit"       # "pursuit" | "stanley"

NATS_HOST = "192.168.30.133"
NATS_URL = "nats://127.0.0.1:4222"
NATS_TOPIC = "robot.vision.localization"

CAMERA = "0"

# ветка fan-out (tee -> shmsink) для этого модуля; 1280x720 = разрешение камеры карты
CAM_SHM_SOCKET = "/tmp/cam_raw"
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
