from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parent
MAPS_DIR = _MODULE_ROOT / "maps"

ROUTE_CAM = None
ROUTE_NODES = None
SHARD_PRELOAD_NODES = 25
SHARD_CONFIRM_FIXES = 2
SHARD_PRELOAD_ALL = False
SHARD_FULL_RECOVERY = True

SHARD_RECOVERY_MIN_INLIERS = 35

QUERY_KPTS = 1024
QUERY_DET_THRESHOLD = 0.05
QUERY_NMS_RADIUS = 2

MATCH_RATIO = 0.9
MATCH_TOPK = 8
FOCAL_FALLBACK = 1.2

MAX_ERROR = 12.0
MIN_PAIRS = 8
MIN_INLIERS = 20

MOVE_EPS = 0.06
POSE_HISTORY = 5
MAX_NODES_PER_SEC = 40
MIN_NODE_JUMP = 10
MAX_REJECTS = 5

STOP_CONFIRM = 3
STOP_MIN_INLIERS = 40
STOP_END_NODES = 5

NAV_LAG_S = 0.08
NAV_LAG_ADAPTIVE = False
NAV_LEAD_MAX = 0.08
NAV_LEAD_SMOOTH = 7
NAV_WIN_NODES = 0

STEER_MODE = "pursuit"
LOOKAHEAD_NODES = 10
LOOKAHEAD_MIN = 4
LOOKAHEAD_ADAPT = 30.0

LOOKAHEAD_SPEED_DIV = 17.0
LOOKAHEAD_MAX = 20.0
NATS_SPEED_TOPIC = "ai.nats_speed_topic"
DEADZONE_DEG = 5.5
STANLEY_K = 1.0
HEADING_GATE = 0.3

CAMERA = "0"
CAMERA_BACK = True
CAM_SHM_SOCKET = "/tmp/cam_raw"
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30

NAV_MODE = "front"
FRONT_CAM_BACK = False
REAR_CAM_BACK = True

ROUTES = {
    "1-2": {"label": "Маршрут 1-2", "camera": "front",
            "front_map": "1-2/front_shard/01_of_25", "rear_map": None},
    "2": {"label": "Маршрут 2 (3)", "camera": "rear",
          "front_map": None, "rear_map": "2/rear_shard/01_of_10"},
    "2-1": {"label": "Маршрут 2-1", "camera": "front",
            "front_map": "2-1/front_shard/01_of_25", "rear_map": None},
    "3-1": {"label": "Маршрут 3-1", "camera": "front",
            "front_map": "3-1/front_shard/01_of_25", "rear_map": None},
    "office": {"label": "Маршрут офис", "camera": "front",
               "front_map": "office/front", "rear_map": "office/rear"},
}

DEFAULT_ROUTE = None
FRONT_MAP = ROUTES["1-2"]["front_map"]
REAR_MAP = ROUTES["2"]["rear_map"]
DEFAULT_MAP = REAR_MAP
FRONT_SHM_SOCKET = "/tmp/cam_front_raw"
REAR_SHM_SOCKET = "/tmp/cam_raw"
DUAL_LOST_HOLD = 3
DUAL_BACK_HOLD = 5

TRAFFIC_LIGHT_ENABLED = True

ROUTE_2_TRAFFIC_ZONE = (410, 430)
ROUTE_12_TRAFFIC_ZONE = (877, 886)
ROUTE_21_TRAFFIC_ZONE = (714, 718)
ROUTE_31_TRAFFIC_ZONE = (897, 901)
TRAFFIC_ZONES = {

    "2/rear_full": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/01_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/02_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/03_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/04_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/05_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/06_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/07_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/08_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/09_of_10": ROUTE_2_TRAFFIC_ZONE,
    "2/rear_shard/10_of_10": ROUTE_2_TRAFFIC_ZONE,


    "1-2/front_full": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/01_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/02_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/03_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/04_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/05_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/06_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/07_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/08_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/09_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/10_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/11_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/12_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/13_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/14_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/15_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/16_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/17_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/18_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/19_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/20_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/21_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/22_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/23_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/24_of_25": ROUTE_12_TRAFFIC_ZONE,
    "1-2/front_shard/25_of_25": ROUTE_12_TRAFFIC_ZONE,

    "2-1/front_full": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/01_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/02_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/03_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/04_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/05_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/06_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/07_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/08_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/09_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/10_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/11_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/12_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/13_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/14_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/15_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/16_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/17_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/18_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/19_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/20_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/21_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/22_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/23_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/24_of_25": ROUTE_21_TRAFFIC_ZONE,
    "2-1/front_shard/25_of_25": ROUTE_21_TRAFFIC_ZONE,

    "3-1/front_full": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/01_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/02_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/03_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/04_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/05_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/06_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/07_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/08_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/09_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/10_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/11_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/12_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/13_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/14_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/15_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/16_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/17_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/18_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/19_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/20_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/21_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/22_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/23_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/24_of_25": ROUTE_31_TRAFFIC_ZONE,
    "3-1/front_shard/25_of_25": ROUTE_31_TRAFFIC_ZONE,


}
TRAFFIC_DET_CONF = 0.15

DIRECTION_ENABLED = True
ROUTE_21_BACKWARD_ZONES = [(0, 26), (1596, 1632)]
ROUTE_31_BACKWARD_ZONES = [(0, 28), (1776, 1813)]
BACKWARD_ZONES = {

    "2-1/front_full": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/01_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/02_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/03_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/04_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/05_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/06_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/07_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/08_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/09_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/10_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/11_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/12_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/13_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/14_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/15_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/16_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/17_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/18_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/19_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/20_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/21_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/22_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/23_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/24_of_25": ROUTE_21_BACKWARD_ZONES,
    "2-1/front_shard/25_of_25": ROUTE_21_BACKWARD_ZONES,



    "3-1/front_full": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/01_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/02_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/03_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/04_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/05_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/06_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/07_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/08_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/09_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/10_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/11_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/12_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/13_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/14_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/15_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/16_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/17_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/18_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/19_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/20_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/21_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/22_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/23_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/24_of_25": ROUTE_31_BACKWARD_ZONES,
    "3-1/front_shard/25_of_25": ROUTE_31_BACKWARD_ZONES,
}
BACKWARD_LEFT_BLOCK_ZONES = {
    name: ROUTE_21_BACKWARD_ZONES
    for name in BACKWARD_ZONES
    if name.startswith("2-1/")
}

NATS_HOST = "192.168.40.48"
NATS_URL = "nats://127.0.0.1:4222"
NATS_TOPIC = "robot.vision.localization"
NATS_CONTROL_TOPIC = "robot.vision.control"
NATS_TRAFFIC_TOPIC = "robot.vision.traffic_light"
