import os

from dotenv import load_dotenv

# подгружаем .env (для standalone-запусков; в Docker env инжектит compose)
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "place_recognition"

DINOV2_MODEL = "dinov2_vitl14"
VECTOR_DIM = 2048

ROUTE_DIR = "data/route_reference"

TOP_K = 5
COSINE_THRESHOLD = 0.5
P_COEF = 3
MAX_JUMP = 5

# веса камер в голосовании: боковые смотрят в однообразные заборы/стены
# (перцептивный алиасинг), поэтому доверяем им меньше. Неизвестная камера → 1.0
CAMERA_WEIGHTS = {"front": 1.0, "back": 1.0, "left": 0.5, "right": 0.5}

EMBED_BATCH_SIZE = 8

# деплой-значения (адреса, секреты) — из .env
CONTROL_URL = os.getenv("CONTROL_URL") or None
CAMERA_URL = os.getenv("CAMERA_URL") or None
CAMERA_TOKEN = os.getenv("CAMERA_TOKEN") or None
