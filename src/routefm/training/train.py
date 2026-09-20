"""Data adapters for the published RouteFM pretraining recipe."""
from __future__ import annotations

from pathlib import Path

import torch

from routefm.data import CanonicalCorpus
from routefm.data.episode_sampler import EpisodeSampler


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _corpus(config: dict) -> CanonicalCorpus:
    data = config["data"]
    if data.get("type") != "mixed":
        raise ValueError("the published pretraining recipe expects mixed real-data sources")

    def partition(entries: list[dict]) -> list[dict]:
        value = data.get("model_partition")
        if value is None:
            return entries
        return [{
            **entry,
            "model_partition": entry.get("model_partition", value),
            "model_holdout_fraction": entry.get(
                "model_holdout_fraction", data.get("model_holdout_fraction", 0.2)
            ),
            "model_partition_seed": entry.get(
                "model_partition_seed", data.get("model_partition_seed", 29004)
            ),
        } for entry in entries]

    records = []
    dense_entries = [
        entry for entry in data.get("dense_entries", [])
        if entry.get("enabled", True)
        and (Path(entry["path"]) / f"{entry.get('split', 'train')}.pkl").exists()
    ]
    if dense_entries:
        dense = CanonicalCorpus.from_dense_artifacts(partition(dense_entries))
        records.extend(record for rows in dense.tasks.values() for record in rows)
    for entry in partition(data.get("jsonl_entries", [])):
        if entry.get("enabled", True):
            corpus = CanonicalCorpus.from_jsonl_entries([entry])
            records.extend(record for rows in corpus.tasks.values() for record in rows)
    if not records:
        raise ValueError("no pretraining records were loaded; check data paths")
    return CanonicalCorpus(records)


def _sampler(corpus: CanonicalCorpus, config: dict, seed: int) -> EpisodeSampler:
    value = config["episodes"]
    if value.get("type") != "task_local":
        raise ValueError("the published recipe expects task_local episodes")
    return EpisodeSampler(
        corpus,
        min_models=value.get("min_models", 6),
        max_models=value.get("max_models", 24),
        context_sizes=tuple(value.get("context_sizes", [8, 16, 32, 64])),
        context_size_weights=value.get("context_size_weights"),
        min_targets=value.get("min_targets", 16),
        max_targets=value.get("max_targets", 32),
        bins=config["model"].get("quality_bins", 21),
        max_episode_attempts=value.get("max_episode_attempts", 32),
        max_task_resamples=value.get("max_task_resamples", 64),
        opportunity_margin=value.get("opportunity_margin", 0.03),
        opportunity_min_switch_fraction=value.get("opportunity_min_switch_fraction", 0.25),
        opportunity_min_oracle_gain=value.get("opportunity_min_oracle_gain", 0.03),
        opportunity_min_winner_fraction=value.get("opportunity_min_winner_fraction", 0.10),
        boundary_min_fraction=value.get("boundary_min_fraction", 0.25),
        boundary_q=value.get("boundary_q", 0.25),
        boundary_threshold_min=value.get("boundary_threshold_min", 0.02),
        boundary_threshold_max=value.get("boundary_threshold_max", 0.08),
        aligned_sparse_coverages=tuple(value.get("aligned_sparse_coverages", [0.4, 0.6, 0.8])),
        unaligned_shared_fractions=tuple(value.get("unaligned_shared_fractions", [0.0, 0.1, 0.25])),
        episode_type_weights=value.get("episode_type_weights"),
        context_layout_weights=value.get("context_layout_weights"),
        sources_per_batch=value.get("sources_per_batch", 2),
        tasks_per_batch=value.get("tasks_per_batch", 8),
        context_observation_schema=config["model"].get("observation_schema", "extended7"),
        seed=seed,
    )
