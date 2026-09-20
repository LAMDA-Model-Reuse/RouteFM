from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RoutingRecord:
    """One canonical (query, model action, outcome) observation."""

    source: str
    dataset_id: str
    query_id: str
    model_id: str
    query_embedding: list[float]
    score: float
    query_text: str = ""
    cost: float = 0.0
    budget: float | None = None
    num_observations: int = 1
    score_distribution: list[float] | None = None
    domain: str = "unknown"
    modality: str = "text"
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoutingRecord":
        required = {"source", "dataset_id", "query_id", "model_id", "query_embedding", "score"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"canonical record is missing fields: {sorted(missing)}")
        record = cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})
        if not 0.0 <= record.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {record.score}")
        if not record.query_embedding:
            raise ValueError("query_embedding must not be empty")
        return record
