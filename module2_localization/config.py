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

SPEED_LOOKAHEAD_ENABLED = False
LOOKAHEAD_SPEED_DIV = 45.0
SPEED_NEUTRAL_US = 1500.0
SPEED_DEADZONE_US = 30.0

DEADZONE_DEG = 5.0
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
CAMERA_BACK = False

FRAME_TIMEOUT_CHECK_ENABLED = True
FRAME_TIMEOUT_S = 0.5
FRAME_TIMEOUT_RECOVERY_FRAMES = 1

# No map or route is selected by default. Runtime maps are deployment artifacts
# and are registered here only after field validation.
# Schema: {"route-id": {"label": "Display name", "map": "relative/map/path"}}
ROUTES = {}
DEFAULT_ROUTE = None
DEFAULT_MAP = None

# Traffic-light models are always loaded with the runtime. Detection remains
# inactive outside map-specific zones.
TRAFFIC_LIGHT_ENABLED = True
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
