from pathlib import Path


_MODULE_ROOT = Path(__file__).resolve().parent
MAPS_DIR = _MODULE_ROOT / "maps"

# Query localization
QUERY_KPTS = 1024
QUERY_DET_THRESHOLD = 0.05
QUERY_NMS_RADIUS = 2
MATCH_RATIO = 0.9
MATCH_TOPK = 8
FOCAL_FALLBACK = 1.2
MAX_ERROR = 12.0
MIN_PAIRS = 8
MIN_INLIERS = 15

# Route following and command filtering
ROUTE_CAM = None
ROUTE_NODES = None
STEER_MODE = "pursuit"
LOOKAHEAD_NODES = 5
LOOKAHEAD_MIN = 4
LOOKAHEAD_MAX = 20.0
LOOKAHEAD_ADAPT = 30.0
LOOKAHEAD_SPEED_DIV = 45.0
DEADZONE_DEG = 5.5
STANLEY_K = 1.0
HEADING_GATE = 0.3
NAV_LAG_S = 0.08
NAV_LAG_ADAPTIVE = False
NAV_LEAD_MAX = 0.08
NAV_LEAD_SMOOTH = 7
NAV_WIN_NODES = 0

MOVE_EPS = 0.06
POSE_HISTORY = 5
MAX_NODES_PER_SEC = 40
MIN_NODE_JUMP = 10
MAX_REJECTS = 5
STOP_CONFIRM = 3
STOP_MIN_INLIERS = 40
STOP_END_NODES = 3

# Compatibility defaults for the current runtime loader. There are no shard
# policies in the new stack; these matter only if a map has shard.json
SHARD_PRELOAD_NODES = 25
SHARD_CONFIRM_FIXES = 2
SHARD_PRELOAD_ALL = False
SHARD_FULL_RECOVERY = True
SHARD_RECOVERY_MIN_INLIERS = 20
SHARD_SWITCH_POLICIES = {}

# Cameras
CAMERA = "0"
CAM_SHM_SOCKET = "/tmp/cam_raw"
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
FRONT_SHM_SOCKET = "/tmp/cam_front_raw"
REAR_SHM_SOCKET = "/tmp/cam_raw"
FRONT_CAM_BACK = False
REAR_CAM_BACK = True

FRAME_TIMEOUT_CHECK_ENABLED = True
FRAME_TIMEOUT_S = 0.5
FRAME_TIMEOUT_RECOVERY_FRAMES = 1

# The new office runtime map will be placed under maps/office/front. It is not
# selected automatically until the new localizer is integrated and validated
ROUTES = {
    "office": {
        "label": "Офис",
        "camera": "front",
        "front_map": "office/front",
        "rear_map": None,
    },
}
DEFAULT_ROUTE = None
FRONT_MAP = ROUTES["office"]["front_map"]
REAR_MAP = None
DEFAULT_MAP = FRONT_MAP
NAV_MODE = "front"
DUAL_LOST_HOLD = 3
DUAL_BACK_HOLD = 5

# Route-specific field behavior is disabled for office development
TRAFFIC_LIGHT_ENABLED = False
TRAFFIC_ZONES = {}
TRAFFIC_DET_CONF = 0.15
DIRECTION_ENABLED = False
BACKWARD_ZONES = {}
BACKWARD_MAPS = set()
BACKWARD_RIGHT_ONLY_MAPS = set()
STEERING_OUTLIER_GUARDS = {}

# Messaging
NATS_URL = "nats://127.0.0.1:4222"
NATS_TOPIC = "robot.vision.localization"
NATS_CONTROL_TOPIC = "robot.vision.control"
NATS_TRAFFIC_TOPIC = "robot.vision.traffic_light"
NATS_SPEED_TOPIC = "gateway.robot.speed"
