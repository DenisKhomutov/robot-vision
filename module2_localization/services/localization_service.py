"""Сервисный слой локализации: кадр -> поза на эталоне + команда контроллеру.

Каркас. Пока localize() принимает ПУТЬ к кадру (как в офлайн-инструментах).
Онлайн (WebRTC -> numpy-кадр, слияние с сегментацией, гейтинг по прошлой позе)
нашивается позже — см. архитектуру в памяти проекта.
"""
import sys
from pathlib import Path
from typing import Any

from .. import config

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

_localizer = None


def get_localizer():
    global _localizer
    if _localizer is None:
        from localizer import AlikedLocalizer
        _localizer = AlikedLocalizer(
            config.DEFAULT_MAP,
            kpts=config.QUERY_KPTS,
            det_threshold=config.QUERY_DET_THRESHOLD,
            nms_radius=config.QUERY_NMS_RADIUS,
            max_error=config.MAX_ERROR,
            route_cam=getattr(config, "ROUTE_CAM", None),
        )
    return _localizer


def localize(frame) -> dict[str, Any]:
    """frame -> команда. Возвращает valid=False, если позе нельзя доверять."""
    r = get_localizer().locate(frame)
    if not r or not r.get("ok") or r["inliers"] < config.MIN_INLIERS:
        return {"valid": False, "reason": (r or {}).get("reason", "no fix")}
    return {
        "valid": True,
        "move_type": r["move_type"],
        "bearing_deg": r["bearing_deg"],
        "offset_m": r.get("offset_m"),
        "dist_to_route_m": r.get("dist_to_route_m"),
        "node": r["node"],
        "inliers": r["inliers"],
    }
