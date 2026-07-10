from loguru import logger
from qdrant_client import QdrantClient

from .. import config

client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

logger.success(f"Qdrant подключён: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
