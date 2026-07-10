from pydantic import BaseModel, ConfigDict, Field


class Segment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    confidence: float


class SegmentationResponse(BaseModel):
    segments: list[Segment]
    path_present: bool
