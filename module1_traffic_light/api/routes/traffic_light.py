import io

from fastapi import APIRouter, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image

from ...services.traffic_light_service import analyze
from ..schemas import TrafficLightResponse

router = APIRouter(prefix="/traffic_light", tags=["traffic_light"])


@router.post("/detect", summary="Распознавание сигнала светофора")
async def detect_endpoint(file: UploadFile) -> TrafficLightResponse:
    image = Image.open(io.BytesIO(await file.read()))
    result = await run_in_threadpool(analyze, image)
    logger.info(f"Распознавание светофора: '{file.filename}': статус={result['status']} / сигнал={result['signal']}")
    return TrafficLightResponse(**result)
