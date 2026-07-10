from pydantic import BaseModel, ConfigDict


class SearchHit(BaseModel):
    score: float
    point_id: int
    image_path: str | None = None
    route_id: str | None = None
    route_point: int | None = None
    distance_m: float | None = None
    direction: str | None = None
    camera: str | None = None


class Decision(BaseModel):
    route_point: int | None = None
    route_id: str | None = None
    direction: str | None = None
    relocalized: bool
    cameras_used: int


class SearchResponse(BaseModel):
    query: str | None = None
    found: int
    results: list[SearchHit]


class BatchResponse(BaseModel):
    decision: Decision
    per_camera: dict[str, list[SearchHit]] | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model: str
    collection: str


class StatusResponse(BaseModel):
    busy: bool


class BuildRoutesResponse(BaseModel):
    indexed: int
    status: str
