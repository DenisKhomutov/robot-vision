import time
import uuid

import httpx
from loguru import logger

from .. import config

STATUS_URL = "http://127.0.0.1:8000/localization/status"
SLEEP_BETWEEN = 0.5


def trigger_camera() -> None:
    if not config.CAMERA_URL:
        raise RuntimeError("CAMERA_URL не задан в .env")
    httpx.post(
        config.CAMERA_URL,
        headers={"X-Capture-Token": config.CAMERA_TOKEN or ""},
        json={"request_id": str(uuid.uuid4())},
        timeout=5,
    )


def main() -> None:
    while True:
        try:
            busy = httpx.get(STATUS_URL, timeout=5).json()["busy"]
            if not busy:
                trigger_camera()
                logger.info("Модель свободна — дёрнул камеры")
        except httpx.HTTPError as e:
            logger.error(f"Ошибка: {e}")

        time.sleep(SLEEP_BETWEEN)


if __name__ == "__main__":
    main()
