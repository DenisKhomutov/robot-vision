import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from loguru import logger
from PIL import Image
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, SearchParams, VectorParams
from tqdm import tqdm

from .. import config
from ..core.embedder import get_embedder
from ..core.qdrant_connection import client

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def search(image: Image.Image | str | Path, top_k: int = config.TOP_K) -> list[dict[str, Any]]:
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    vector = get_embedder().embed(image)

    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        limit=top_k,
        score_threshold=config.COSINE_THRESHOLD,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )

    return [
        {
            "score": round(r.score, 4),
            "point_id": r.id,
            **{
                k: (r.payload or {}).get(k)
                for k in ("image_path", "route_id", "route_point", "distance_m", "direction", "camera")
            },
        }
        for r in results.points
    ]


def search_batch(
    images: Mapping[str, Image.Image],
    top_k: int = config.TOP_K,
    route_id: str | None = None,
    direction: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    names = list(images.keys())
    vectors = get_embedder().embed_batch(list(images.values()))

    base_conditions = []
    if route_id is not None:
        base_conditions.append(FieldCondition(key="route_id", match=MatchValue(value=route_id)))
    if direction is not None:
        base_conditions.append(FieldCondition(key="direction", match=MatchValue(value=direction)))

    return {
        name: [
            {
                "score": round(r.score, 4),
                "point_id": r.id,
                **{
                    k: (r.payload or {}).get(k)
                    for k in ("image_path", "route_id", "route_point", "direction", "camera")
                },
            }
            for r in client.query_points(
                collection_name=config.COLLECTION_NAME,
                query=vector,
                query_filter=Filter(
                    must=[*base_conditions, FieldCondition(key="camera", match=MatchValue(value=name))]
                ),
                limit=top_k,
                score_threshold=config.COSINE_THRESHOLD,
                with_payload=True,
                search_params=SearchParams(hnsw_ef=128, exact=False),
            ).points
        ]
        for name, vector in zip(names, vectors, strict=True)
    }


def fuse(per_camera: dict[str, list[dict[str, Any]]], prev_point: int | None = None) -> dict[str, Any]:
    # топ-1 по каждой камере (Qdrant уже сортирует по убыванию score)
    top1: dict[str, dict[str, Any]] = {cam: results[0] for cam, results in per_camera.items() if results}

    if not top1:
        return {"route_point": prev_point, "route_id": None, "direction": None, "relocalized": False, "cameras_used": 0}

    # фильтр по MAX_JUMP
    relocalized = False
    if prev_point is not None:
        valid = {cam: r for cam, r in top1.items() if abs(int(r["route_point"]) - prev_point) <= config.MAX_JUMP}
        if not valid:
            # все камеры не прошли — релокализация: берём лучшую по взвешенному score, обрезаем прыжок
            best = max(top1.items(), key=lambda kv: config.CAMERA_WEIGHTS.get(kv[0], 1.0) * kv[1]["score"])[1]
            step = config.MAX_JUMP if int(best["route_point"]) > prev_point else -config.MAX_JUMP
            return {
                "route_point": prev_point + step,
                "route_id": best.get("route_id"),
                "direction": best.get("direction"),
                "relocalized": True,
                "cameras_used": 0,
            }
        top1 = valid

    # голосование с весами камер: группируем взвешенный вклад по route_point
    weighted_votes: dict[int, float] = {}  # сумма весов = «взвешенное число голосов»
    weighted_scores: dict[int, float] = {}  # сумма weight * score
    for cam, r in top1.items():
        pt = int(r["route_point"])
        w = config.CAMERA_WEIGHTS.get(cam, 1.0)
        weighted_votes[pt] = weighted_votes.get(pt, 0.0) + w
        weighted_scores[pt] = weighted_scores.get(pt, 0.0) + w * r["score"]

    # ранжирование: больше взвешенных голосов → выше взвешенный score → меньший номер
    winner_point = min(
        weighted_votes,
        key=lambda pt: (-weighted_votes[pt], -weighted_scores[pt], pt),
    )

    # маршрут и направление берём из точки-победителя
    winner = next(r for r in top1.values() if int(r["route_point"]) == winner_point)

    return {
        "route_point": winner_point,
        "route_id": winner.get("route_id"),
        "direction": winner.get("direction"),
        "relocalized": relocalized,
        "cameras_used": len(top1),
    }


def build_routes(route_dir: str = config.ROUTE_DIR) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    base = project_root / route_dir

    if not base.exists():
        raise FileNotFoundError(f"Директория не найдена: {route_dir}")

    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)

    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=config.VECTOR_DIM, distance=Distance.COSINE),
    )
    logger.info(f"Коллекция '{config.COLLECTION_NAME}' создана (dim={config.VECTOR_DIM})")

    points = []
    point_id = 0
    for route_path in sorted(base.iterdir()):
        route_id = route_path.name
        for dir_path in sorted(route_path.iterdir()):
            direction = dir_path.name
            for cam_path in sorted(dir_path.iterdir()):
                camera = cam_path.name
                image_paths = sorted(
                    (p for p in cam_path.iterdir() if p.suffix.lower() in SUPPORTED),
                    key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"),
                )
                logger.info(f"Векторизация {route_id}/{direction}/{camera} ({len(image_paths)} фото)")
                for chunk_start in tqdm(
                    range(0, len(image_paths), config.EMBED_BATCH_SIZE),
                    desc=f"{route_id}/{direction}/{camera}",
                    file=sys.stdout,
                ):
                    chunk = image_paths[chunk_start : chunk_start + config.EMBED_BATCH_SIZE]
                    pil_images = cast(list[Image.Image], [Image.open(p) for p in chunk])
                    vectors = get_embedder().embed_batch(pil_images)
                    for i, (img_path, vector) in enumerate(zip(chunk, vectors, strict=True)):
                        points.append(
                            PointStruct(
                                id=point_id,
                                vector=vector,
                                payload={
                                    "route_id": route_id,
                                    "route_point": chunk_start + i + 1,
                                    "direction": direction,
                                    "camera": camera,
                                    "image_path": str(img_path.relative_to(project_root)).replace("\\", "/"),
                                },
                            )
                        )
                        point_id += 1

    client.upsert(collection_name=config.COLLECTION_NAME, points=points)

    info = client.get_collection(config.COLLECTION_NAME)
    logger.success(f"Проиндексировано {len(points)} точек, статус: {info.status}")

    return {"indexed": len(points), "status": str(info.status)}
