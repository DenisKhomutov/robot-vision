import nats
from loguru import logger


class NatsClient:
    def __init__(self, url: str):
        self.url = url
        self.nc = None

    async def connect(self):
        async def disconnected_cb():
            logger.warning("NATS: Соединение разорвано! Пытаюсь переподключиться...")

        async def reconnected_cb():
            logger.success(
                f"NATS: Успешное переподключение к {self.nc.connected_url.netloc}"
            )

        async def error_cb(e):
            logger.error(f"NATS ошибка: {e}")

        async def closed_cb():
            logger.info("NATS: Соединение окончательно закрыто.")

        try:
            self.nc = await nats.connect(
                servers=[self.url],
                reconnected_cb=reconnected_cb,
                disconnected_cb=disconnected_cb,
                error_cb=error_cb,
                closed_cb=closed_cb,
                max_reconnect_attempts=-1,
            )
            logger.success(f"Успешное подключение к NATS: {self.url}")
        except Exception as e:
            logger.exception(f"Критическая ошибка подключения к NATS: {e}")
            raise

    async def publish(self, subject: str, payload: bytes, headers: dict = None):
        if not self.nc:
            logger.error("Попытка publish до инициализации подключения к NATS!")
            return

        await self.nc.publish(subject, payload, headers=headers)
        logger.debug(f"NATS -> Опубликовано в '{subject}'")

    async def subscribe(self, subject: str, callback, queue: str = ""):
        if not self.nc:
            logger.error("Попытка subscribe до инициализации подключения к NATS!")
            return

        sub = await self.nc.subscribe(subject, queue=queue, cb=callback)
        queue_msg_log = f" (Очередь балансировки: '{queue}')" if queue else ""
        logger.info(f"NATS <- Подкаст настроен на топик '{subject}'{queue_msg_log}")
        return sub

    async def request(self, subject: str, payload: bytes, timeout: float = 1.0):
        if not self.nc:
            logger.error("Попытка request до инициализации подключения к NATS!")
            return None
        return await self.nc.request(subject, payload, timeout=timeout)

    async def close(self):
        if self.nc:
            await self.nc.close()
