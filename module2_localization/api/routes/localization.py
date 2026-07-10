import io
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image

from ... import config
from ...services.place_service import build_routes, fuse, search, search_batch
from ..schemas import (
    BatchResponse,
    BuildRoutesResponse,
    HealthResponse,
    SearchResponse,
    StatusResponse,
)

router = APIRouter(prefix="/localization", tags=["localization"])

_busy = False


def _push_decision(decision: dict[str, Any]) -> None:
    if not config.CONTROL_URL:
        return
    try:
        httpx.post(config.CONTROL_URL, json=decision, timeout=5)
    except httpx.HTTPError as e:
        logger.error(f"Не удалось отправить decision на управление: {e}")


@router.get("/health", summary="Статус сервиса")
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=config.DINOV2_MODEL, collection=config.COLLECTION_NAME)


@router.get("/status", summary="Статус занятости модели")
def status() -> StatusResponse:
    return StatusResponse(busy=_busy)


@router.post("/search", summary="Поиск по одному изображению")
async def search_endpoint(file: UploadFile, top_k: int = config.TOP_K) -> SearchResponse:
    image = Image.open(io.BytesIO(await file.read()))
    results = await run_in_threadpool(search, image, top_k)
    logger.info(f"Поиск '{file.filename}': найдено {len(results)} результатов")
    return SearchResponse.model_validate({"query": file.filename, "found": len(results), "results": results})


@router.post("/search_batch", summary="Поиск по батчу изображений")
async def search_batch_endpoint(
    front: UploadFile,
    back: UploadFile,
    left: UploadFile,
    right: UploadFile,
    top_k: int = config.TOP_K,
    route_id: str | None = None,
    direction: str | None = None,
    prev_point: int | None = None,
    debug: bool = False,
) -> BatchResponse:
    global _busy
    _busy = True
    try:
        images = {
            "front": Image.open(io.BytesIO(await front.read())),
            "back": Image.open(io.BytesIO(await back.read())),
            "left": Image.open(io.BytesIO(await left.read())),
            "right": Image.open(io.BytesIO(await right.read())),
        }
        files = {"front": front, "back": back, "left": left, "right": right}
        per_camera = await run_in_threadpool(search_batch, images, top_k, route_id, direction)
        decision = fuse(per_camera, prev_point)
        logger.info(
            f"Батч-поиск [{route_id}/{direction}]: "
            f"точка={decision['route_point']} релокализация={decision['relocalized']}"
        )
        _push_decision(decision)
        if debug:
            return BatchResponse.model_validate(
                {
                    "decision": decision,
                    "per_camera": {f"{c} ({files[c].filename})": hits for c, hits in per_camera.items()},
                }
            )
        return BatchResponse.model_validate({"decision": decision})
    finally:
        _busy = False


@router.post("/build_routes", summary="Преобразование в векторы")
def build_routes_endpoint(route_dir: str = config.ROUTE_DIR) -> BuildRoutesResponse:
    try:
        return BuildRoutesResponse.model_validate(build_routes(route_dir))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
