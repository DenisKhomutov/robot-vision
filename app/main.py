from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from module1_traffic_light import init_models as init_traffic_light
from module1_traffic_light.api.routes import traffic_light
from module2_localization import init_models as init_localization
from module2_localization.api.routes import localization
from module3_segmentation import init_models as init_segmentation
from module3_segmentation.api.routes import segment

from .logging_setup import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_localization()
    init_traffic_light()
    init_segmentation()
    logger.success("API запущен. Модели и Qdrant готовы к работе.")
    yield
    logger.info("API остановлен.")


app = FastAPI(title="Robot Vision API", lifespan=lifespan)

app.include_router(localization.router)
app.include_router(traffic_light.router)
app.include_router(segment.router)
