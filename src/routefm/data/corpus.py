from __future__ import annotations

import json
import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from .schema import RoutingRecord
from .filtering import entry_allows_model, entry_allows_query


def query_partition(
    source: str, eval_name: str, query_id: str, validation_fraction: float, partition_seed: int
) -> str:
    """Return a stable fit/val assignment independent of Python's hash seed."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    key = f"{partition_seed}\0{source}\0{eval_name}\0{query_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return "val" if bucket < validation_fraction else "fit"


class CanonicalCorpus:
    """Sparse routing observations indexed by dataset, model and query."""

    def __init__(self, records: Iterable[RoutingRecord]):
        records = list(records)
        if not records:
            raise ValueError("the corpus contains no records")
        self.embedding_dim = len(records[0].query_embedding)
        self.tasks: dict[str, list[RoutingRecord]] = defaultdict(list)
        self.task_sources: dict[str, str] = {}
        for record in records:
            if len(record.query_embedding) != self.embedding_dim:
                raise ValueError("all query embeddings must have the same dimension")
            task_key = f"{record.source}::{record.dataset_id}"
            self.tasks[task_key].append(record)
            self.task_sources[task_key] = record.source
        self.tasks = dict(self.tasks)

    @classmethod
    def from_jsonl(cls, paths: str | Path | Iterable[str | Path]) -> "CanonicalCorpus":
        if isinstance(paths, (str, Path)):
            paths = [paths]
        records: list[RoutingRecord] = []
        for path in paths:
            with Path(path).open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        record = RoutingRecord.from_dict(payload)
                        record.metadata = dict(record.metadata)
                        record.metadata.setdefault("cost_available", "cost" in payload)
                        records.append(record)
                    except Exception as exc:
                        raise ValueError(f"invalid record at {path}:{line_no}: {exc}") from exc
        return cls(records)

    @classmethod
    def from_jsonl_entries(cls, entries: Iterable[dict]) -> "CanonicalCorpus":
        records: list[RoutingRecord] = []
        for entry in entries:
            with Path(entry["path"]).open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        record = RoutingRecord.from_dict(payload)
                        record.metadata = dict(record.metadata)
                        record.metadata.setdefault(
                            "cost_available", "cost" in payload
                        )
                    except Exception as exc:
                        raise ValueError(f"invalid record at {entry['path']}:{line_no}: {exc}") from exc
                    requested_split = entry.get("split")
                    if requested_split is not None and record.split != requested_split:
                        continue
                    if not entry_allows_query(entry, record.dataset_id, payload.get("query_text", "")):
                        continue
                    if not entry_allows_model(entry, record.model_id):
                        continue
                    records.append(record)
        return cls(records)

    @classmethod
    def from_dense_artifacts(cls, entries: Iterable[dict]) -> "CanonicalCorpus":
        """Load ORBIT-style dense tables while sharing each query embedding in memory."""
        try:
            import numpy as np
            import pandas as pd  # noqa: F401 - required by pandas pickle reconstruction
        except ImportError as exc:
            raise RuntimeError("dense artifacts require numpy and pandas") from exc
        records: list[RoutingRecord] = []
        for entry in entries:
            root = Path(entry["path"])
            split = entry.get("split", "train")
            source = entry.get("source", root.name)
            requested_partition = entry.get("query_partition", "all")
            if requested_partition not in {"all", "fit", "val"}:
                raise ValueError("query_partition must be one of: all, fit, val")
            validation_fraction = float(entry.get("validation_fraction", 0.1))
            partition_seed = int(entry.get("partition_seed", 17_001))
            dataset_grouping = entry.get("dataset_grouping", "eval_name")
            if dataset_grouping not in {"eval_name", "source", "eval_prefix"}:
                raise ValueError(
                    f"unsupported dataset_grouping={dataset_grouping!r} in {root}; "
                    "expected eval_name, source, or eval_prefix"
                )
            frame = pd.read_pickle(root / f"{split}.pkl")
            embeddings = np.load(root / f"{split}_embeddings.npy", mmap_mode="r")
            models = json.loads((root / "models.json").read_text(encoding="utf-8"))
            artifact_metadata = {"artifact_path": str(root)}
            if len(frame) != len(embeddings):
                raise ValueError(f"row/embedding mismatch in {root}: {len(frame)} != {len(embeddings)}")
            # Model/lineage filtering depends only on the entry and model ID,
            # not the query. Preserve column order while avoiding repeated
            # partition hashes for every query in large candidate pools.
            allowed_models = [
                (model_index, model_name)
                for model_index, model_name in enumerate(models)
                if entry_allows_model(entry, str(model_name))
            ]
            for position, (row_index, row) in enumerate(frame.iterrows()):
                eval_name = str(row.get("eval_name", source))
                query_id = str(row.get("sample_id", f"{split}_{row_index}"))
                query_text = str(row.get("query_text", row.get("prompt", "")))
                if not entry_allows_query(entry, eval_name, query_text):
                    continue
                if requested_partition != "all" and query_partition(
                    source, eval_name, query_id, validation_fraction, partition_seed
                ) != requested_partition:
                    continue
                if dataset_grouping == "source":
                    dataset_id = source
                elif dataset_grouping == "eval_prefix":
                    dataset_id = eval_name.split("/", 1)[0]
                else:
                    dataset_id = eval_name
                embedding = embeddings[position]
                for model_index, model_name in allowed_models:
                    score = row.get(f"model_{model_index}_performance")
                    if score is None or not math.isfinite(float(score)):
                        continue
                    cost_key = f"model_{model_index}_cost"
                    raw_cost = row.get(cost_key)
                    cost_available = raw_cost is not None and math.isfinite(float(raw_cost))
                    cost = raw_cost if cost_available else 0.0
                    cost = 0.0 if cost is None or not math.isfinite(float(cost)) else float(cost)
                    records.append(
                        RoutingRecord(
                            source=source,
                            dataset_id=dataset_id,
                            query_id=query_id,
                            model_id=str(model_name),
                            query_embedding=embedding,
                            score=min(1.0, max(0.0, float(score))),
                            query_text=query_text,
                            cost=cost,
                            split=split,
                            metadata={
                                **artifact_metadata,
                                "cost_available": cost_available,
                            },
                        )
                    )
        return cls(records)


class SyntheticCorpus(CanonicalCorpus):
    """Deterministic latent-factor corpus for smoke tests."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_datasets: int = 3,
        num_queries: int = 160,
        num_models: int = 8,
        latent_dim: int = 12,
        seed: int = 7,
        query_seed: int | None = None,
        capability_seed: int | None = None,
        num_worlds: int = 1,
        function_type: str = "bilinear",
        nonlinear_strength: float = 1.0,
        outcome_noise: float = 0.10,
        repeated_outcomes: int = 8,
    ):
        if function_type not in {"bilinear", "nonlinear"}:
            raise ValueError("function_type must be 'bilinear' or 'nonlinear'")
        if num_worlds < 1:
            raise ValueError("num_worlds must be positive")
        if not 0.0 <= nonlinear_strength <= 1.0:
            raise ValueError("nonlinear_strength must be in [0, 1]")
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(latent_dim, embedding_dim, generator=generator) / math.sqrt(latent_dim)
        capability_generator = generator
        if capability_seed is not None:
            capability_generator = torch.Generator().manual_seed(capability_seed)
        capability_dim = latent_dim if function_type == "bilinear" else 3 * latent_dim
        capability_banks = [
            torch.randn(num_models, capability_dim, generator=capability_generator) for _ in range(num_worlds)
        ]
        model_bias = torch.linspace(-0.6, 0.6, num_models)
        if query_seed is not None:
            # Keep the routing world fixed (embedding map + model capabilities),
            # while generating genuinely unseen queries and stochastic outcomes.
            generator.manual_seed(query_seed)
        rng = random.Random(seed if query_seed is None else query_seed)
        records: list[RoutingRecord] = []
        for world_idx, capabilities in enumerate(capability_banks):
            for dataset_idx in range(num_datasets):
                domain_shift = torch.randn(latent_dim, generator=generator) * 0.3
                for query_idx in range(num_queries):
                    latent = torch.randn(latent_dim, generator=generator) + domain_shift
                    embedding = (latent @ projection + 0.05 * torch.randn(embedding_dim, generator=generator)).tolist()
                    if function_type == "nonlinear":
                        # Continuously increase task complexity during curriculum training.
                        # Keeping the linear block unscaled preserves a learnable bridge
                        # between the bilinear and fully nonlinear routing worlds.
                        capability_features = torch.cat(
                            (
                                latent,
                                nonlinear_strength * (latent.square() - 1.0),
                                nonlinear_strength * torch.sin(latent),
                            )
                        )
                    else:
                        capability_features = latent
                    for model_idx in range(num_models):
                        logit = (capability_features * capabilities[model_idx]).sum() / math.sqrt(capability_dim)
                        mean = torch.sigmoid(logit + model_bias[model_idx]).item()
                        # A smooth empirical distribution emulates repeated DARS generations.
                        draws = [
                            min(1.0, max(0.0, rng.gauss(mean, outcome_noise)))
                            for _ in range(repeated_outcomes)
                        ]
                        hist = [0.0] * 21
                        for draw in draws:
                            hist[min(20, round(draw * 20))] += 1.0 / len(draws)
                        cost = 0.002 + 0.001 * model_idx + 0.0002 * abs(latent[0].item())
                        world_prefix = f"w{world_idx}_" if num_worlds > 1 else ""
                        records.append(
                            RoutingRecord(
                                source=f"synthetic_w{world_idx}" if num_worlds > 1 else "synthetic",
                                dataset_id=f"{world_prefix}synthetic_{dataset_idx}",
                                query_id=f"{world_prefix}q_{dataset_idx}_{query_idx}",
                                model_id=f"{world_prefix}model_{model_idx}",
                                query_embedding=embedding,
                                score=sum(draws) / len(draws),
                                score_distribution=hist,
                                cost=cost,
                                num_observations=len(draws),
                                metadata={
                                    "synthetic_world": world_idx,
                                    "function_type": function_type,
                                    "nonlinear_strength": nonlinear_strength,
                                },
                            )
                        )
        super().__init__(records)
