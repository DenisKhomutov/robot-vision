import io

from fastapi import APIRouter, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image

from ...services.segmentation_service import analyze
from ..schemas import SegmentationResponse

router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@router.post("/segment", summary="Сегментация сцены")
async def segment_endpoint(file: UploadFile) -> SegmentationResponse:
    image = Image.open(io.BytesIO(await file.read()))
    result = await run_in_threadpool(analyze, image)
    logger.info(f"Сегментация '{file.filename}': классов={len(result['segments'])} путь={result['path_present']}")
    return SegmentationResponse(**result)
