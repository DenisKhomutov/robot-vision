"""Окружение конкретного развёртывания робота."""

from pathlib import Path


_MODULE_ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = _MODULE_ROOT / "maps"

# Camera and frame transport
CAMERA = "0"
CAM_SHM_SOCKET = "/tmp/cam_raw"
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
CAMERA_BACK = False

# Messaging
NATS_URL = "nats://127.0.0.1:4222"
NATS_TOPIC = "robot.vision.localization"
NATS_CONTROL_TOPIC = "robot.vision.control"
NATS_TRAFFIC_TOPIC = "robot.vision.traffic_light"
NATS_SPEED_TOPIC = "gateway.robot.speed"


__all__ = [name for name in globals() if name.isupper()]

