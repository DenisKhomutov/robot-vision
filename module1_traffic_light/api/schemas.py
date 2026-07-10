from typing import Literal

from pydantic import BaseModel


class TrafficLightResponse(BaseModel):
    status: Literal["light_classified", "light_unclassified", "no_traffic_light"]
    signal: str | None = None
    confidence: float | None = None
    box: list[int] | None = None
