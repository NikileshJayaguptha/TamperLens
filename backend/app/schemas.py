import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RegionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    index: int
    bbox: list[int]
    polygons: list[list[list[int]]]
    area_frac: float
    noise_score: float
    classifier_score: float | None
    score: float
    details: dict
    label: str | None


class AnalysisSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: str
    error: str | None
    width: int
    height: int
    overall_score: float | None
    model_version: str | None
    created_at: datetime


class AnalysisOut(AnalysisSummary):
    regions: list[RegionOut]


class LabelIn(BaseModel):
    label: Literal["real", "ai"] | None


class LabelStats(BaseModel):
    real: int
    ai: int
    unlabeled: int


class Health(BaseModel):
    status: str
    sam_loaded: bool
    classifier_loaded: bool
    model_version: str | None
