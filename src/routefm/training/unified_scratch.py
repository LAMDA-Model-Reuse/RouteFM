"""One-pass, multi-scale RouteFM training from random initialization.

The training run has no expert checkpoints, branch selection, or
post-hoc learned aggregation.  One RouteFM instance and one optimizer are used
for the complete run.  Scale-homogeneous batches avoid padding a K=8 episode to
K=1024 while retaining one fixed batch size.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from routefm.config import load_config
from routefm.data import CanonicalCorpus
from routefm.data.sampler import collate_episodes
from routefm.models import RouteFM, RouteFMConfig
from routefm.training.deployment_agnostic import (
    deployment_agnostic_loss,
    multi_output_point_loss,
)
from routefm.training.train import _corpus, _move, _sampler


class WeightedCycle:
    """Deterministic shuffled cycles with exact long-run integer proportions."""

    def __init__(self, weights: dict[str, float], rng: random.Random, units: int = 20):
        positive = {str(k): float(v) for k, v in weights.items() if float(v) > 0}
        if not positive:
            raise ValueError("weighted cycle requires at least one positive weight")
        total = sum(positive.values())
        raw = {k: units * v / total for k, v in positive.items()}
        counts = {k: max(1, int(math.floor(v))) for k, v in raw.items()}
        while sum(counts.values()) < units:
            key = max(raw, key=lambda k: raw[k] - counts[k])
            counts[key] += 1
        while sum(counts.values()) > units:
            choices = [k for k in counts if counts[k] > 1]
            if not choices:
                break
            key = min(choices, key=lambda k: raw[k] - counts[k])
            counts[key] -= 1
        self.template = [k for k in sorted(counts) for _ in range(counts[k])]
        self.rng = rng
        self.pending: list[str] = []

    def next(self) -> str:
        if not self.pending:
            self.pending = list(self.template)
            self.rng.shuffle(self.pending)
        return self.pending.pop()

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.getstate(), "pending": list(self.pending)}

    def load_state_dict(self, value: dict[str, Any]) -> None:
        self.rng.setstate(value["rng"])
        self.pending = list(value["pending"])


def routing_regret_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Differentiable expected regret under the deployed quality-only policy."""
    valid = mask.any(-1)
    logits = score.masked_fill(~mask, -1e4) / max(float(temperature), 1e-4)
    policy = logits.softmax(-1)
    expected = (policy * target * mask).sum(-1)
    oracle = target.masked_fill(~mask, -1e4).max(-1).values
    return ((oracle - expected) * valid).sum() / valid.sum().clamp_min(1)


def _records_by_source(corpus: CanonicalCorpus) -> dict[str, list]:
    result: dict[str, list] = defaultdict(list)
    for rows in corpus.tasks.values():
        for record in rows:
            result[record.source].append(record)
    return dict(result)


def _validation_groups(config: dict) -> list[str]:
    weights = config["training"]["validation"].get("scale_weights")
    return [
        group for group in config["training"]["scale_groups"]
        if weights is None or float(weights.get(group, 0.0)) > 0.0
    ]


def _make_samplers(
    config: dict,
    split: str,
    seed: int,
    *,
    candidate_ranges: dict[str, tuple[int, int]] | None = None,
) -> tuple[dict, list[str]]:
    """Build independent streams, keeping validation on its declared range.

    The empty range name preserves stable (group, source) keys and seeds.
    Additional ranges use their own RNG, cycles, and accepted-pool caches: a
    pool accepted with four candidates must never leak into an M >= 6 phase.
    """
    if candidate_ranges is not None and split != "train":
        raise ValueError("phase candidate ranges apply only to training")
    cfg = copy.deepcopy(config)
    if split != "train":
        from routefm.evaluation.evaluate import _apply_data_split, _apply_evaluation_suite

        # Data-source comparisons must be selected on the same held-out queries
        # AND candidate pool, not each arm's differently sized training pool.
        reference_data = config["training"]["validation"].get("data")
        if reference_data is not None:
            cfg["data"] = copy.deepcopy(reference_data)
        _apply_data_split(cfg, split)
        suite = cfg["training"]["validation"].get("evaluation_suite")
        if suite:
            _apply_evaluation_suite(cfg, suite)
    corpus = _corpus(cfg)
    by_source = _records_by_source(corpus)
    configured_sources = (
        config["data"]["source_weights"]
        if split == "train"
        else config["training"]["validation"]["source_weights"]
    )
    sources = [name for name in configured_sources if name in by_source]
    groups = config["training"]["scale_groups"]
    samplers = {}
    ranges = candidate_ranges or {"": _phase_model_range(config, {})}
    source_corpora = {source: CanonicalCorpus(by_source[source]) for source in sources}
    for range_index, (range_name, model_range) in enumerate(ranges.items()):
        range_seed = seed if not range_name else seed + 1_000_000 * (range_index + 1)
        for group_index, (group, group_spec) in enumerate(groups.items()):
            # Keep the original enumeration index for validation
            # seeds even when a curriculum appends a training-only K group.
            if split != "train" and group not in _validation_groups(config):
                continue
            for source_index, source in enumerate(sources):
                source_cfg = copy.deepcopy(cfg)
                source_cfg["episodes"]["context_sizes"] = list(group_spec["context_sizes"])
                if candidate_ranges is not None:
                    source_cfg["episodes"]["min_models"], source_cfg["episodes"]["max_models"] = model_range
                key = (range_name, group, source) if range_name else (group, source)
                try:
                    samplers[key] = _sampler(
                        source_corpora[source], source_cfg,
                        range_seed + 10_000 * group_index + 1_000 * source_index,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"cannot construct sampler source={source}, group={group}, "
                        f"candidate_range={model_range}: {error}"
                    ) from error
    return samplers, sources


def _phase_model_range(config: dict, phase: dict) -> tuple[int, int]:
    episodes = config["episodes"]
    minimum = int(phase.get("min_models", episodes.get("min_models", 6)))
    maximum = int(phase.get("max_models", episodes.get("max_models", 24)))
    if minimum < 2 or maximum < minimum:
        raise ValueError(
            f"invalid candidate range ({minimum}, {maximum}) in phase "
            f"{phase.get('name', 'default')}"
        )
    return minimum, maximum


def _phase_range_name(config: dict, phase: dict) -> str:
    model_range = _phase_model_range(config, phase)
    if model_range == _phase_model_range(config, {}):
        return ""
    return f"models_{model_range[0]}_{model_range[1]}"


def _training_candidate_ranges(config: dict, phases: list[dict]) -> dict:
    return {
        _phase_range_name(config, phase): _phase_model_range(config, phase)
        for phase in phases
    }


def _training_sampler_key(config: dict, phase: dict, group: str, source: str) -> tuple:
    schema = str(phase.get("context_schema", "score_cost"))
    range_name = _phase_range_name(config, phase)
    return (schema, range_name, group, source) if range_name else (schema, group, source)


def _candidate_availability(samplers: dict) -> dict:
    """Report proposal ceilings; rejection may change the accepted M prior."""
    result = {}
    for key, sampler in samplers.items():
        counts = [len(sampler._task_models[task]) for task in sampler._valid_tasks]
        result["::".join(key)] = {
            "min_models": sampler.min_models,
            "max_models": sampler.max_models,
            "eligible_tasks": len(counts),
            "available_models_per_task_histogram": dict(sorted(Counter(counts).items())),
            "effective_max_models_per_task_histogram": dict(sorted(Counter(
                min(sampler.max_models, count) for count in counts
            ).items())),
        }
    return result


def _record_candidate_counts(counts: dict, phase: dict, sources: list[str], episodes: list) -> None:
    phase_counts = counts.setdefault(phase["name"], {})
    for source, episode in zip(sources, episodes, strict=True):
        histogram = phase_counts.setdefault(source, {})
        key = str(len(episode.models))
        histogram[key] = histogram.get(key, 0) + 1


def _loss(
    model: RouteFM,
    batch: dict,
    weights: dict,
    mode: str = "multi_output_point_regret",
) -> tuple[torch.Tensor, dict]:
    output = model(batch)
    if mode == "deployment_agnostic":
        distribution_keys = {
            key: weights[key]
            for key in (
                "distribution_weight", "mean_weight", "pairwise_weight",
                "regret_weight", "gap_weight", "mean_huber_beta",
                "gap_huber_beta", "pairwise_target_temperature",
                "pairwise_prediction_temperature", "pairwise_reliability_prior",
                "routing_temperature",
            )
            if key in weights
        }
        return deployment_agnostic_loss(output, batch, **distribution_keys)
    if mode not in {"multi_output_point", "multi_output_point_regret"}:
        raise ValueError(f"unknown curriculum loss mode: {mode}")
    point_keys = {
        key: weights[key]
        for key in (
            "score_weight", "score_pairwise_weight", "cost_weight",
            "cost_pairwise_weight", "score_huber_beta", "cost_huber_beta",
            "pairwise_target_temperature", "pairwise_prediction_temperature",
        )
    }
    loss, metrics = multi_output_point_loss(output, batch, **point_keys)
    if mode == "multi_output_point":
        return loss, metrics
    regret = routing_regret_loss(
        output["score_mean"], batch["target_score"], batch["target_mask"],
        float(weights["routing_temperature"]),
    )
    loss = loss + float(weights["score_regret_weight"]) * regret
    metrics = dict(metrics)
    metrics["routing_regret_loss"] = regret.detach()
    metrics["loss"] = loss.detach()
    return loss, metrics


def _point_normalizers(batch: dict) -> dict[str, float]:
    """Counts for exact masked-loss accumulation, not mean-of-microbatch means."""
    mask = batch["target_mask"].bool()
    counts = mask.sum(-1)
    models = mask.shape[-1]
    upper = torch.ones(models, models, dtype=torch.bool, device=mask.device).triu(1)
    pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2) & upper
    result = {
        "cells": float(counts.sum()),
        "pairs": float((counts * (counts - 1) / 2).sum()),
        "targets": float(mask.any(-1).sum()),
    }
    for name, key in (("score", "target_score"), ("cost", "target_cost_normalized")):
        target = batch[key]
        non_tied = (target.unsqueeze(-1) - target.unsqueeze(-2)).abs() > 1e-6
        result[name + "_pairs"] = float((pairs & non_tied).sum())
    return result


def _accumulation_loss(model, batch, weights, mode, totals):
    if mode not in {"multi_output_point", "multi_output_point_regret"}:
        raise ValueError("microbatch accumulation supports point score/cost losses only")
    counts = _point_normalizers(batch)
    ratios = {key: counts[key] / max(totals[key], 1.0) for key in counts}
    scaled = dict(weights)
    for key, count in (
        ("score_weight", "cells"), ("cost_weight", "cells"),
        ("score_pairwise_weight", "pairs"), ("cost_pairwise_weight", "pairs"),
        ("score_regret_weight", "targets"),
    ):
        if key in scaled:
            scaled[key] = float(scaled[key]) * ratios[count]
    loss, values = _loss(model, batch, scaled, mode)
    metric_counts = {
        "score_huber_loss": "cells", "cost_huber_loss": "cells",
        "score_mae": "cells", "cost_mae": "cells",
        "score_pairwise_loss": "pairs", "cost_pairwise_loss": "pairs",
        "routing_regret_loss": "targets", "routing_accuracy": "targets",
        "score_pairwise_accuracy": "score_pairs", "cost_pairwise_accuracy": "cost_pairs",
    }
    # AMP rounds targets inside the original point loss, so these last two
    # monitoring accuracies are microbatch-weighted estimates under AMP.
    values = {key: value * ratios[metric_counts[key]] if key in metric_counts else value
              for key, value in values.items()}
    return loss, values


def collate_training_episodes(episodes: list) -> dict:
    """Collate Routing episodes while retaining its point-cost supervision fields."""
    batch = collate_episodes(episodes)
    batch["episode_types"] = [episode.episode_type for episode in episodes]
    batch["context_layouts"] = [episode.context_layout for episode in episodes]
    batch["context_sizes"] = torch.tensor(
        [episode.context_size for episode in episodes], dtype=torch.long
    )
    target_count, model_count = batch["target_score"].shape[1:]
    for key, dtype in (
        ("target_cost", torch.float32),
        ("target_num_observations", torch.float32),
        ("target_cost_normalized", torch.float32),
        ("target_stability_mask", torch.bool),
    ):
        batch[key] = torch.zeros(
            len(episodes), target_count, model_count, dtype=dtype
        )
    for index, episode in enumerate(episodes):
        targets, models = episode.target_score.shape
        for key in (
            "target_cost", "target_num_observations",
            "target_cost_normalized", "target_stability_mask",
        ):
            value = getattr(episode, key)
            if value is None:
                raise ValueError(f"Routing episode is missing {key}")
            batch[key][index, :targets, :models] = value
    return batch


def _weighted(values: dict[str, float], weights: dict[str, float]) -> float:
    present = [(values[k], float(w)) for k, w in weights.items() if k in values]
    if not present:
        raise ValueError("no validation cells available for configured weights")
    return sum(value * weight for value, weight in present) / sum(weight for _, weight in present)


def _curriculum(training: dict, steps: int) -> list[dict]:
    """Resolve and validate a predeclared single-run curriculum."""
    configured = training.get("curriculum")
    if not configured:
        return [{
            "name": "joint",
            "end_step": steps,
            "scale_weights": {
                key: value["weight"] for key, value in training["scale_groups"].items()
            },
            "loss_mode": str(training.get("loss_mode", "multi_output_point_regret")),
            "effective_batch_size": int(training.get("effective_batch_size", 1)),
            "loss_weights": {},
        }]
    phases = copy.deepcopy(configured)
    previous = 0
    names = set()
    known_groups = set(training["scale_groups"])
    for phase in phases:
        name = str(phase["name"])
        end = int(phase["end_step"])
        if name in names:
            raise ValueError(f"duplicate curriculum phase: {name}")
        if end <= previous:
            raise ValueError("curriculum end_step values must be strictly increasing")
        weights = {str(k): float(v) for k, v in phase["scale_weights"].items()}
        if not set(weights) <= known_groups or not any(value > 0 for value in weights.values()):
            raise ValueError(f"invalid scale weights in curriculum phase {name}")
        phase["name"] = name
        phase["end_step"] = end
        phase["scale_weights"] = weights
        phase["loss_mode"] = str(
            phase.get("loss_mode", training.get("loss_mode", "multi_output_point_regret"))
        )
        phase["effective_batch_size"] = int(
            phase.get("effective_batch_size", training.get("effective_batch_size", 1))
        )
        if phase["effective_batch_size"] < 1:
            raise ValueError(f"invalid effective_batch_size in curriculum phase {name}")
        for field in ("min_models", "max_models"):
            if field in phase:
                value = phase[field]
                if isinstance(value, bool) or int(value) != float(value) or int(value) < 2:
                    raise ValueError(f"invalid {field} in curriculum phase {name}")
                phase[field] = int(value)
        if ("min_models" in phase and "max_models" in phase
                and phase["max_models"] < phase["min_models"]):
            raise ValueError(f"invalid min_models/max_models in curriculum phase {name}")
        names.add(name)
        previous = end
    if previous != steps:
        raise ValueError("the final curriculum end_step must equal training.steps")
    return phases


def _phase_at_step(phases: list[dict], step: int) -> tuple[int, dict, int]:
    """Return phase index/spec/start for a zero-based optimizer step."""
    start = 0
    for index, phase in enumerate(phases):
        if step < int(phase["end_step"]):
            return index, phase, start
        start = int(phase["end_step"])
    last_start = 0 if len(phases) == 1 else int(phases[-2]["end_step"])
    return len(phases) - 1, phases[-1], last_start


def _curriculum_lr_scale(
    step: int, phases: list[dict], default_warmup: int
) -> float:
    """Piecewise cosine schedule within one optimizer trajectory."""
    index, phase, start = _phase_at_step(phases, step)
    end = int(phase["end_step"])
    start_scale = float(phase.get("lr_start_scale", 1.0))
    end_scale = float(phase.get("lr_end_scale", 0.1))
    warmup = int(phase.get("warmup_steps", default_warmup if index == 0 else 0))
    local = max(0, step - start)
    if warmup and local < warmup:
        return max(1e-3, start_scale * (local + 1) / warmup)
    decay_steps = max(
        1, int(phase.get("lr_decay_steps", end - start - warmup))
    )
    progress = min(1.0, max(0.0, (local - warmup) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_scale + (start_scale - end_scale) * cosine


@torch.no_grad()
def validate(
    model: RouteFM,
    config: dict,
    device: torch.device,
    samplers: dict | None = None,
    sources: list[str] | None = None,
) -> dict:
    spec = config["training"]["validation"]
    if samplers is None or sources is None:
        samplers, sources = _make_samplers(
            config, spec.get("data_split", "val"), int(spec["evaluation_seed"])
        )
    totals = defaultdict(lambda: {"quality": 0.0, "context_mean": 0.0, "count": 0})
    model.eval()
    for group in _validation_groups(config):
        for source in sources:
            sampler = samplers[(group, source)]
            for _ in range(int(spec["episodes_per_cell"])):
                try:
                    batch = _move(sampler.sample_batch(1), device)
                except RuntimeError:
                    # Some validation sources cannot form K=1024 episodes. The
                    # final metric renormalizes over cells fixed by this rule.
                    break
                output = model(batch)
                mask = batch["target_mask"]
                valid = mask.any(-1)
                choice = output["score_mean"].masked_fill(~mask, -1e4).argmax(-1)
                quality = batch["target_score"].gather(-1, choice[..., None]).squeeze(-1)
                context_mask = batch["context_mask"].float()
                context_mean = (
                    (batch["context_features"][..., 0] * context_mask).sum(-1)
                    / context_mask.sum(-1).clamp_min(1)
                )
                baseline_choice = context_mean.masked_fill(~batch["candidate_mask"], -1e4).argmax(-1)
                baseline_choice = baseline_choice[:, None].expand_as(valid)
                baseline = batch["target_score"].gather(-1, baseline_choice[..., None]).squeeze(-1)
                cell = totals[(group, source)]
                cell["quality"] += float(quality.masked_select(valid).sum())
                cell["context_mean"] += float(baseline.masked_select(valid).sum())
                cell["count"] += int(valid.sum())
    cells = {
        f"{group}::{source}": {
            "quality": values["quality"] / values["count"],
            "context_mean": values["context_mean"] / values["count"],
            "count": values["count"],
        }
        for (group, source), values in totals.items() if values["count"]
    }
    group_quality, group_baseline = {}, {}
    for group in _validation_groups(config):
        q = {s: cells[f"{group}::{s}"]["quality"] for s in sources if f"{group}::{s}" in cells}
        b = {s: cells[f"{group}::{s}"]["context_mean"] for s in sources if f"{group}::{s}" in cells}
        group_quality[group] = _weighted(q, spec["source_weights"])
        group_baseline[group] = _weighted(b, spec["source_weights"])
    quality = _weighted(group_quality, spec["scale_weights"])
    baseline = _weighted(group_baseline, spec["scale_weights"])
    model.train()
    return {
        "quality_balanced": quality,
        "context_mean_balanced": baseline,
        "gain_over_context_mean": quality - baseline,
        "by_scale": group_quality,
        "context_mean_by_scale": group_baseline,
        "cells": cells,
    }


def _sampler_state(samplers: dict) -> dict:
    result = {}
    for key, sampler in samplers.items():
        result["::".join(key)] = {
            "rng": sampler.rng.getstate(),
            "episode_cycle": list(sampler._episode_cycle),
            "layout_cycle": list(sampler._layout_cycle),
            "accepted_pool_cache": dict(sampler._accepted_pool_cache),
        }
    return result


def _load_sampler_state(samplers: dict, state: dict) -> None:
    for key, sampler in samplers.items():
        value = state["::".join(key)]
        sampler.rng.setstate(value["rng"])
        sampler._episode_cycle.clear()
        sampler._episode_cycle.extend(value["episode_cycle"])
        sampler._layout_cycle.clear()
        sampler._layout_cycle.extend(value["layout_cycle"])
        sampler._accepted_pool_cache.clear()
        sampler._accepted_pool_cache.update(value["accepted_pool_cache"])


def save_checkpoint(path: Path, model, optimizer, scheduler, step, config, runtime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "format": "routefm_unified_scratch",
        "release_version": str(config.get("release_version", "0.28")),
        "step": step,
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "runtime": runtime,
    }, temporary)
    temporary.replace(path)


def _truncate_jsonl_after_step(path: Path, completed_step: int) -> None:
    """Drop unsnapshotted log rows before an exact-state resume.

    A process can advance beyond the most recent periodic checkpoint before an
    infrastructure interruption. Those rows do not belong to the restored
    optimizer/RNG trajectory and must not coexist with the replayed steps.
    """
    if not path.exists():
        return
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["step"]) <= completed_step:
            rows.append(line)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + ("\n" if rows else ""))
    temporary.replace(path)


def train(config: dict, device_name: str | None = None, steps_override: int | None = None) -> Path:
    training = config["training"]
    if training.get("init_from") is not None:
        raise ValueError("unified-from-scratch training forbids init_from")
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name or training.get("device", "cuda"))
    candidate_ranges = _training_candidate_ranges(
        config, training.get("curriculum") or [{}]
    )
    score_cost_samplers, sources = _make_samplers(
        config, "train", seed + 101, candidate_ranges=candidate_ranges
    )
    samplers = {
        ("score_cost", *key): value for key, value in score_cost_samplers.items()
    }
    configured_schemas = {
        str(phase.get("context_schema", "score_cost"))
        for phase in training.get("curriculum", ())
    }
    if "extended7" in configured_schemas:
        extended_config = copy.deepcopy(config)
        extended_config["model"]["observation_schema"] = "extended7"
        extended_config["model"]["observation_features"] = 7
        extended_samplers, extended_sources = _make_samplers(
            extended_config, "train", seed + 101, candidate_ranges=candidate_ranges
        )
        if extended_sources != sources:
            raise ValueError("extended-feature and score/cost training sources differ")
        samplers.update({
            ("extended7", *key): value for key, value in extended_samplers.items()
        })
    model = RouteFM(RouteFMConfig(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["lr"]), weight_decay=float(training["weight_decay"]))
    steps = int(steps_override or training["steps"])
    warmup = max(1, round(steps * float(training["warmup_fraction"])))

    schedule_training = training
    if steps_override is not None and training.get("curriculum") and steps < int(training["steps"]):
        schedule_training = copy.deepcopy(training)
        shortened = []
        for phase in schedule_training["curriculum"]:
            phase = copy.deepcopy(phase)
            phase["end_step"] = min(int(phase["end_step"]), steps)
            shortened.append(phase)
            if phase["end_step"] == steps:
                break
        schedule_training["curriculum"] = shortened
    phases = _curriculum(schedule_training, steps)

    def lr_scale(step: int) -> float:
        if training.get("curriculum"):
            return _curriculum_lr_scale(step, phases, warmup)
        if step < warmup:
            return max(1e-3, (step + 1) / warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    output = Path(training["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    scale_cycles = {
        phase["name"]: WeightedCycle(
            phase["scale_weights"], random.Random(seed + 201 + 10_000 * index),
            units=20 if training.get("curriculum") else 5,
        )
        for index, phase in enumerate(phases)
    }
    source_cycle = WeightedCycle(config["data"]["source_weights"], random.Random(seed + 202), units=20)
    start, best = 0, -float("inf")
    candidate_counts = {}
    resume = training.get("resume")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        runtime = saved["runtime"]
        start, best = int(saved["step"]), float(runtime["best_validation"])
        candidate_counts = copy.deepcopy(runtime.get("candidate_counts", {}))
        if "scale_cycles" in runtime:
            for name, state in runtime["scale_cycles"].items():
                scale_cycles[name].load_state_dict(state)
        else:
            # Exact resume for checkpoints with an earlier schedule.
            scale_cycles[phases[0]["name"]].load_state_dict(runtime["scale_cycle"])
        source_cycle.load_state_dict(runtime["source_cycle"])
        _load_sampler_state(samplers, runtime["samplers"])
        random.setstate(runtime["python_rng"])
        torch.set_rng_state(runtime["torch_rng"])
        if device.type == "cuda" and runtime.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(runtime["cuda_rng"])
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    metrics_path = output / "metrics.jsonl"
    validation_path = output / "validation_metrics.jsonl"
    if resume:
        _truncate_jsonl_after_step(metrics_path, start)
        _truncate_jsonl_after_step(validation_path, start)
    print(json.dumps({"event": "start", "parameters": sum(p.numel() for p in model.parameters()), "start_step": start, "steps": steps, "device": str(device), "sources": sources, "curriculum": phases}), flush=True)
    candidate_availability = _candidate_availability(samplers)
    (output / "candidate_availability.json").write_text(
        json.dumps(candidate_availability, indent=2) + "\n"
    )
    print(json.dumps({"event": "candidate_availability", "streams": candidate_availability}), flush=True)

    def runtime_state() -> dict:
        return {
            "best_validation": best,
            "scale_cycles": {name: cycle.state_dict() for name, cycle in scale_cycles.items()},
            "source_cycle": source_cycle.state_dict(),
            "samplers": _sampler_state(samplers),
            "candidate_counts": copy.deepcopy(candidate_counts),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }

    model.train()
    for step in range(start, steps):
        phase_index, phase, phase_start = _phase_at_step(phases, step)
        optimizer_state_reset = (
            step == phase_start
            and phase_index > 0
            and bool(phase.get("reset_optimizer_state", False))
        )
        if optimizer_state_reset:
            # Preserve the one model/optimizer object while matching a staged
            # protocol that deliberately clears Adam moments at a boundary.
            optimizer.state.clear()
        group = scale_cycles[phase["name"]].next()
        context_schema = str(phase.get("context_schema", "score_cost"))
        loss_weights = dict(training["loss_weights"])
        loss_weights.update(phase.get("loss_weights", {}))
        optimizer.zero_grad(set_to_none=True)
        batch_sources = []
        episodes = []
        for _ in range(int(phase["effective_batch_size"])):
            source = source_cycle.next()
            episodes.append(samplers[_training_sampler_key(config, phase, group, source)].sample())
            batch_sources.append(source)
        minimum_models, maximum_models = _phase_model_range(config, phase)
        if not all(minimum_models <= len(episode.models) <= maximum_models for episode in episodes):
            raise RuntimeError(f"sampled candidate count outside declared range in {phase['name']}")
        _record_candidate_counts(candidate_counts, phase, batch_sources, episodes)
        microbatch_size = int(training.get("microbatch_size", len(episodes)))
        if microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        accumulate = microbatch_size < len(episodes)
        totals: dict[str, float] = defaultdict(float)
        if accumulate:
            for episode in episodes:
                counts = _point_normalizers({
                    "target_mask": episode.target_mask,
                    "target_score": episode.target_score,
                    "target_cost_normalized": episode.target_cost_normalized,
                })
                for key, value in counts.items():
                    totals[key] += value
        values: dict = {}
        for offset in range(0, len(episodes), microbatch_size):
            batch = _move(collate_training_episodes(episodes[offset:offset + microbatch_size]), device)
            batch["_context_schema"] = context_schema
            if phase.get("disable_local_context", False):
                batch["_disable_local_context"] = True
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if accumulate:
                    loss, micro_values = _accumulation_loss(
                        model, batch, loss_weights, str(phase["loss_mode"]), totals
                    )
                else:
                    loss, micro_values = _loss(model, batch, loss_weights, mode=str(phase["loss_mode"]))
            loss.backward()
            for key, value in micro_values.items():
                values[key] = values.get(key, 0.0) + value
            del batch, loss, micro_values
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        completed = step + 1
        if completed == 1 or completed % int(training["log_every"]) == 0:
            row = {
                "step": completed, "phase": phase["name"], "phase_index": phase_index,
                "loss_mode": phase["loss_mode"],
                "context_schema": context_schema,
                "local_context_enabled": (
                    model.local_context_blocks is not None
                    and model.config.local_context_mode != "disabled"
                    and not phase.get("disable_local_context", False)
                ),
                "optimizer_state_reset": optimizer_state_reset,
                "group": group, "sources": batch_sources,
                "context_sizes": [episode.context_size for episode in episodes],
                "context_layouts": [episode.context_layout for episode in episodes],
                "context_grid_coverages": [
                    episode.sampling_metrics.get("context_grid_coverage") for episode in episodes
                ],
                "microbatch_size": min(microbatch_size, len(episodes)),
                "gradient_accumulation_steps": math.ceil(len(episodes) / microbatch_size),
                "episode_types": [episode.episode_type for episode in episodes],
                "effective_batch_size": int(phase["effective_batch_size"]),
                "min_models": minimum_models, "max_models": maximum_models,
                "candidate_counts": [len(episode.models) for episode in episodes],
                "candidate_histogram_by_source": candidate_counts[phase["name"]],
                "lr": scheduler.get_last_lr()[0],
                **{key: float(value) for key, value in values.items()},
            }
            print(json.dumps(row), flush=True)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        validation_spec = training["validation"]
        validate_now = completed % int(validation_spec["every"]) == 0
        save_now = completed % int(training["save_every"]) == 0
        if validate_now:
            result = validate(model, config, device)
            result["step"] = completed
            improved = result["quality_balanced"] > best
            result["best"] = improved
            if improved:
                best = result["quality_balanced"]
            print(json.dumps({"validation": result}), flush=True)
            with validation_path.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            if improved:
                save_checkpoint(output / "checkpoint_best.pt", model, optimizer, scheduler, completed, config, runtime_state())
        if save_now:
            save_checkpoint(output / f"checkpoint_{completed}.pt", model, optimizer, scheduler, completed, config, runtime_state())
    final = output / "checkpoint_last.pt"
    save_checkpoint(final, model, optimizer, scheduler, steps, config, runtime_state())
    print(json.dumps({"event": "complete", "checkpoint": str(final), "best_validation": best}), flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.resume:
        config["training"]["resume"] = args.resume
    if args.steps is not None and args.steps < int(config["training"]["steps"]):
        config["training"]["validation"]["every"] = args.steps + 1
        config["training"]["save_every"] = args.steps + 1
    train(config, args.device, args.steps)


if __name__ == "__main__":
    main()
