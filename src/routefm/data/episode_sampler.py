from __future__ import annotations

import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from typing import Any

import torch

from .corpus import CanonicalCorpus
from .sampler import (
    Episode,
    _cost_available,
    _distribution,
    _score_uncertainty,
    collate_episodes,
)
from .schema import RoutingRecord


EPISODE_CYCLE = ("natural",) * 5 + ("opportunity",) * 3 + ("boundary",) * 2
LAYOUT_CYCLE = (
    ("aligned_dense",) * 4
    + ("aligned_sparse",) * 3
    + ("unaligned_sparse",) * 3
)


def _integer_cycle(
    weights: dict[str, float] | None,
    defaults: tuple[str, ...],
    allowed: set[str],
) -> tuple[str, ...]:
    if weights is None:
        return defaults
    unknown = set(weights) - allowed
    if unknown:
        raise ValueError(f"unknown cycle values: {sorted(unknown)}")
    positive = {
        name: Fraction(str(value)).limit_denominator(100)
        for name, value in weights.items() if float(value) > 0.0
    }
    if not positive:
        raise ValueError("cycle weights must contain a positive value")
    denominator = math.lcm(*(value.denominator for value in positive.values()))
    counts = {
        name: value.numerator * (denominator // value.denominator)
        for name, value in positive.items()
    }
    divisor = reduce(math.gcd, counts.values())
    return tuple(
        name for name, count in counts.items()
        for _ in range(count // divisor)
    )


@dataclass
class RoutingEpisode(Episode):
    """A task-local routing episode plus auditable sampling metadata."""

    episode_type: str = "natural"
    context_layout: str = "aligned_dense"
    context_size: int = 0
    context_query_ids: list[list[str]] = field(default_factory=list)
    target_query_ids: list[str] = field(default_factory=list)
    target_texts: list[str] = field(default_factory=list)
    sampling_metrics: dict[str, float] = field(default_factory=dict)
    target_cost: torch.Tensor | None = None
    target_cost_normalized: torch.Tensor | None = None
    target_num_observations: torch.Tensor | None = None
    target_stability_mask: torch.Tensor | None = None


@dataclass
class EpisodeAudit:
    task: str
    source: str
    episode_type: str
    context_layout: str
    sampling_metrics: dict[str, float]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty list")
    values = sorted(float(value) for value in values)
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight


class EpisodeSampler:
    """Task-local natural/opportunity/boundary routing episodes.

    The sampler deliberately separates proposal from acceptance. Candidate
    models, targets, Context size, and Context layout are sampled before any
    target outcomes are inspected. Opportunity and boundary modes reject or
    retain that complete proposal; they never rank, crop, or replace targets.
    Natural mode never consults target labels to decide whether to accept.
    """

    def __init__(
        self,
        corpus: CanonicalCorpus,
        min_models: int = 6,
        max_models: int = 24,
        context_sizes: tuple[int, ...] = (8, 16, 32, 64),
        context_size_weights: dict[int, float] | None = None,
        min_targets: int = 16,
        max_targets: int = 32,
        bins: int = 21,
        max_episode_attempts: int = 32,
        max_task_resamples: int = 64,
        opportunity_margin: float = 0.03,
        opportunity_min_switch_fraction: float = 0.25,
        opportunity_min_oracle_gain: float = 0.03,
        opportunity_min_winner_fraction: float = 0.10,
        boundary_min_fraction: float = 0.25,
        boundary_q: float = 0.25,
        boundary_threshold_min: float = 0.02,
        boundary_threshold_max: float = 0.08,
        aligned_sparse_coverages: tuple[float, ...] = (0.4, 0.6, 0.8),
        unaligned_shared_fractions: tuple[float, ...] = (0.0, 0.1, 0.25),
        episode_type_weights: dict[str, float] | None = None,
        context_layout_weights: dict[str, float] | None = None,
        sources_per_batch: int = 2,
        tasks_per_batch: int = 8,
        context_observation_schema: str = "extended7",
        seed: int = 0,
    ):
        if min_models < 2 or max_models < min_models:
            raise ValueError("invalid min_models/max_models")
        if min_targets < 16 or max_targets < min_targets:
            raise ValueError("task-local routing requires 16 <= min_targets <= max_targets")
        if not context_sizes or min(context_sizes) < 1:
            raise ValueError("context_sizes must contain positive integers")
        if max_episode_attempts < 1 or max_task_resamples < 1:
            raise ValueError("resampling limits must be positive")
        if not 0.0 <= boundary_min_fraction <= 1.0:
            raise ValueError("boundary_min_fraction must be in [0, 1]")
        if not 0.0 < boundary_q < 1.0:
            raise ValueError("boundary_q must be in (0, 1)")
        if boundary_threshold_min > boundary_threshold_max:
            raise ValueError("invalid boundary threshold clamp")
        if not aligned_sparse_coverages or not all(
            0.0 < value <= 1.0 for value in aligned_sparse_coverages
        ):
            raise ValueError("aligned sparse coverages must be in (0, 1]")
        if not unaligned_shared_fractions or not all(
            0.0 <= value <= 1.0 for value in unaligned_shared_fractions
        ):
            raise ValueError("unaligned shared fractions must be in [0, 1]")
        if context_observation_schema not in {"extended7", "score_cost"}:
            raise ValueError("unknown Context observation schema")

        self.corpus = corpus
        self.min_models = int(min_models)
        self.max_models = int(max_models)
        self.context_sizes = tuple(int(value) for value in context_sizes)
        self.context_size_weights = {
            int(key): float(value)
            for key, value in (context_size_weights or {}).items()
        }
        if set(self.context_size_weights) - set(self.context_sizes):
            raise ValueError("context_size_weights contains an unknown size")
        if self.context_size_weights and not any(
            value > 0.0 for value in self.context_size_weights.values()
        ):
            raise ValueError("context_size_weights must contain a positive weight")
        self.min_targets = int(min_targets)
        self.max_targets = int(max_targets)
        self.bins = int(bins)
        self.max_episode_attempts = int(max_episode_attempts)
        self.max_task_resamples = int(max_task_resamples)
        self.opportunity_margin = float(opportunity_margin)
        self.opportunity_min_switch_fraction = float(
            opportunity_min_switch_fraction
        )
        self.opportunity_min_oracle_gain = float(opportunity_min_oracle_gain)
        self.opportunity_min_winner_fraction = float(
            opportunity_min_winner_fraction
        )
        self.boundary_min_fraction = float(boundary_min_fraction)
        self.boundary_q = float(boundary_q)
        self.boundary_threshold_min = float(boundary_threshold_min)
        self.boundary_threshold_max = float(boundary_threshold_max)
        self.aligned_sparse_coverages = tuple(
            float(value) for value in aligned_sparse_coverages
        )
        self.unaligned_shared_fractions = tuple(
            float(value) for value in unaligned_shared_fractions
        )
        self.episode_cycle = _integer_cycle(
            episode_type_weights,
            EPISODE_CYCLE,
            {"natural", "opportunity", "boundary"},
        )
        self.layout_cycle = _integer_cycle(
            context_layout_weights,
            LAYOUT_CYCLE,
            {"aligned_dense", "aligned_sparse", "unaligned_sparse"},
        )
        self.sources_per_batch = max(2, int(sources_per_batch))
        self.tasks_per_batch = max(2, int(tasks_per_batch))
        self.context_observation_schema = context_observation_schema
        self.rng = random.Random(seed)
        self._episode_cycle: deque[str] = deque()
        self._layout_cycle: deque[str] = deque()

        self._by_task: dict[str, dict[str, dict[str, RoutingRecord]]] = {}
        self._task_models: dict[str, list[str]] = {}
        self._task_queries: dict[str, list[str]] = {}
        self._global_scores: dict[str, dict[str, float]] = {}
        self._boundary_thresholds: dict[str, float] = {}
        for task, records in corpus.tasks.items():
            by_model: dict[str, dict[str, RoutingRecord]] = defaultdict(dict)
            by_query: dict[str, dict[str, RoutingRecord]] = defaultdict(dict)
            for record in records:
                by_model[record.model_id][record.query_id] = record
                by_query[record.query_id][record.model_id] = record
            self._by_task[task] = dict(by_model)
            self._task_models[task] = sorted(by_model)
            self._task_queries[task] = sorted(by_query)
            self._global_scores[task] = {
                model: sum(float(record.score) for record in rows.values()) / len(rows)
                for model, rows in by_model.items()
            }
            positive_gaps = []
            for values in by_query.values():
                scores = sorted(
                    (float(record.score) for record in values.values()), reverse=True
                )
                if len(scores) >= 2 and scores[0] > scores[1]:
                    positive_gaps.append(scores[0] - scores[1])
            raw = _quantile(positive_gaps, self.boundary_q) if positive_gaps else 0.0
            self._boundary_thresholds[task] = min(
                self.boundary_threshold_max,
                max(self.boundary_threshold_min, raw),
            )

        self._valid_tasks = [
            task for task in self._by_task
            if len(self._task_models[task]) >= self.min_models
            and len(self._task_queries[task]) >= self.min_targets
        ]
        if not self._valid_tasks:
            raise ValueError("no task can form a task-local routing episode")
        # The confirmed task prior is proportional to eligible query count.
        self._task_weights = [
            float(len(self._task_queries[task])) for task in self._valid_tasks
        ]
        self._tasks_by_source: dict[str, list[str]] = defaultdict(list)
        for task in self._valid_tasks:
            self._tasks_by_source[self.corpus.task_sources[task]].append(task)
        self._stats: Counter[str] = Counter()
        self._task_stats: Counter[str] = Counter()
        self._source_stats: Counter[str] = Counter()
        self._reason_stats: Counter[str] = Counter()
        self._metric_sums: Counter[str] = Counter()
        self._metric_counts: Counter[str] = Counter()
        self._accepted_pool_cache: dict[tuple[str, str], list[tuple[str, ...]]] = (
            defaultdict(list)
        )

    def _next_from_cycle(self, queue: deque[str], values: tuple[str, ...]) -> str:
        if not queue:
            shuffled = list(values)
            self.rng.shuffle(shuffled)
            queue.extend(shuffled)
        return queue.popleft()

    def _sample_task(self) -> str:
        return self.rng.choices(
            self._valid_tasks, weights=self._task_weights, k=1
        )[0]

    def _sample_context_size(self) -> int:
        if not self.context_size_weights:
            return self.rng.choice(self.context_sizes)
        return self.rng.choices(
            self.context_sizes,
            weights=[
                max(0.0, self.context_size_weights.get(size, 0.0))
                for size in self.context_sizes
            ],
            k=1,
        )[0]

    @staticmethod
    def _query_embedding(records: list[RoutingRecord]) -> torch.Tensor:
        return torch.tensor(records[0].query_embedding, dtype=torch.float32)

    @staticmethod
    def _point_cost(record: RoutingRecord) -> float:
        """Return the observed cost when available."""
        if not _cost_available(record) or not math.isfinite(float(record.cost)):
            return 0.0
        return max(0.0, float(record.cost))

    def _target_statistics(
        self,
        task: str,
        models: list[str],
        target_ids: list[str],
    ) -> dict[str, Any]:
        by_model = self._by_task[task]
        winners: list[str] = []
        top_model_sets: list[set[str]] = []
        margins: list[float] = []
        oracle_scores: list[float] = []
        masks: list[list[bool]] = []
        score_rows: list[list[float]] = []
        for query_id in target_ids:
            scored = sorted(
                (
                    (float(by_model[model][query_id].score), model)
                    for model in models if query_id in by_model[model]
                ),
                key=lambda value: (-value[0], value[1]),
            )
            if len(scored) < 2:
                return {"valid": False, "reason": "target_coverage_lt2"}
            winners.append(scored[0][1])
            top_score = scored[0][0]
            top_model_sets.append({
                model for score, model in scored
                if math.isclose(score, top_score, abs_tol=1e-12)
            })
            margins.append(scored[0][0] - scored[1][0])
            oracle_scores.append(scored[0][0])
            masks.append([query_id in by_model[model] for model in models])
            score_rows.append([
                float(by_model[model][query_id].score)
                if query_id in by_model[model] else float("nan")
                for model in models
            ])

        full_models = [
            index for index in range(len(models))
            if all(mask[index] for mask in masks)
        ]
        constant_model = None
        constant_quality = float("-inf")
        for index in full_models:
            quality = sum(row[index] for row in score_rows) / len(score_rows)
            if quality > constant_quality:
                constant_quality = quality
                constant_model = models[index]
        winner_counts = Counter(winners)
        stable_winner_counts = Counter(
            winner for winner, margin in zip(winners, margins)
            if margin >= self.opportunity_margin
        )
        switch_fraction = (
            sum(winner != constant_model for winner in winners) / len(winners)
            if constant_model is not None else 0.0
        )
        oracle_quality = sum(oracle_scores) / len(oracle_scores)
        return {
            "valid": True,
            "winners": winners,
            "top_model_sets": top_model_sets,
            "winner_counts": winner_counts,
            "stable_winner_counts": stable_winner_counts,
            "margins": margins,
            "constant_model": constant_model,
            "switch_fraction": switch_fraction,
            "oracle_gain": (
                oracle_quality - constant_quality
                if constant_model is not None else float("-inf")
            ),
        }

    def _accept(
        self,
        episode_type: str,
        task: str,
        stats: dict[str, Any],
    ) -> tuple[bool, str, dict[str, float]]:
        # Natural acceptance is intentionally label-blind. Coverage was checked
        # structurally while constructing the complete proposal.
        if episode_type == "natural":
            return True, "accepted", {}
        if not stats["valid"]:
            return False, str(stats["reason"]), {}
        target_count = len(stats["winners"])
        if episode_type == "opportunity":
            stable_counts = sorted(stats["stable_winner_counts"].values(), reverse=True)
            if len(stable_counts) < 2:
                return False, "opportunity_lt2_stable_winners", {}
            required_support = max(
                2, math.ceil(self.opportunity_min_winner_fraction * target_count)
            )
            if stable_counts[1] < required_support:
                return False, "opportunity_second_winner_support", {}
            if stats["constant_model"] is None:
                return False, "opportunity_no_full_constant", {}
            if stats["switch_fraction"] < self.opportunity_min_switch_fraction:
                return False, "opportunity_switch_fraction", {}
            if stats["oracle_gain"] < self.opportunity_min_oracle_gain:
                return False, "opportunity_oracle_gain", {}
            return True, "accepted", {
                "switch_fraction": float(stats["switch_fraction"]),
                "oracle_gain": float(stats["oracle_gain"]),
                "distinct_winners": float(len(stats["winner_counts"])),
            }
        if episode_type == "boundary":
            threshold = self._boundary_thresholds[task]
            # An exact tie is the limiting decision boundary. Pairwise losses
            # consume its equal scores as a soft 0.5 preference.
            boundary = [0.0 <= gap <= threshold for gap in stats["margins"]]
            boundary_fraction = sum(boundary) / target_count
            boundary_winners = set().union(*(
                top_models for top_models, keep
                in zip(stats["top_model_sets"], boundary) if keep
            )) if any(boundary) else set()
            if boundary_fraction < self.boundary_min_fraction:
                return False, "boundary_fraction", {}
            if len(boundary_winners) < 2:
                return False, "boundary_lt2_winners", {}
            return True, "accepted", {
                "boundary_fraction": float(boundary_fraction),
                "boundary_threshold": float(threshold),
                "distinct_boundary_winners": float(len(boundary_winners)),
            }
        raise ValueError(f"unknown episode type: {episode_type}")

    def _context_records(
        self,
        task: str,
        models: list[str],
        target_ids: list[str],
        context_size: int,
        layout: str,
    ) -> tuple[list[list[RoutingRecord | None]], dict[str, float]] | None:
        by_model = self._by_task[task]
        excluded = set(target_ids)
        available = {
            model: [query_id for query_id in by_model[model] if query_id not in excluded]
            for model in models
        }
        if any(len(values) < context_size for values in available.values()):
            return None
        common = sorted(set.intersection(*(set(available[model]) for model in models)))
        if layout == "aligned_dense":
            if len(common) < context_size:
                return None
            shared = self.rng.sample(common, context_size)
            records = [[by_model[model][query_id] for query_id in shared] for model in models]
            return records, {
                "context_overlap": 1.0,
                "context_grid_coverage": 1.0,
                **self._context_alignment_metrics(records),
            }

        if layout == "aligned_sparse":
            coverage = self.rng.choice(self.aligned_sparse_coverages)
            grid_size = math.ceil(context_size / coverage)
            if len(common) < grid_size:
                return None
            grid = self.rng.sample(common, grid_size)
            # Allocate the M*K cells evenly over the shared grid. Given the
            # configured M>=6 and coverage>=0.4, every grid query can be
            # observed by at least two models. This avoids singleton grid rows.
            if len(models) * context_size < 2 * grid_size:
                return None
            observer_counts = {query_id: 0 for query_id in grid}
            selected_sets: list[set[str]] = []
            for _ in models:
                shuffled = list(grid)
                self.rng.shuffle(shuffled)
                chosen = set(sorted(
                    shuffled, key=lambda query_id: observer_counts[query_id]
                )[:context_size])
                for query_id in chosen:
                    observer_counts[query_id] += 1
                selected_sets.append(chosen)
            if min(observer_counts.values()) < 2:
                return None
            records: list[list[RoutingRecord | None]] = []
            for model, chosen in zip(models, selected_sets):
                records.append([
                    by_model[model][query_id] if query_id in chosen else None
                    for query_id in grid
                ])
            overlap = len(set.intersection(*selected_sets)) / context_size
            return records, {
                "context_overlap": float(overlap),
                "context_grid_coverage": float(context_size / grid_size),
                "context_grid_min_observers": float(min(observer_counts.values())),
                **self._context_alignment_metrics(records),
            }

        if layout != "unaligned_sparse":
            raise ValueError(f"unknown Context layout: {layout}")
        requested_shared = self.rng.choice(self.unaligned_shared_fractions)
        shared_count = round(context_size * requested_shared)
        if len(common) < shared_count:
            return None
        shared = self.rng.sample(common, shared_count)
        globally_used = set(shared)
        query_lists: list[list[str]] = []
        for model in models:
            private_count = context_size - shared_count
            candidates = [
                query_id for query_id in available[model]
                if query_id not in globally_used
            ]
            if len(candidates) < private_count:
                # Keep exactly K observations per model. Reuse a query only when
                # the task cannot provide enough pairwise-private IDs.
                candidates.extend(
                    query_id for query_id in available[model]
                    if query_id not in shared and query_id not in candidates
                )
            if len(candidates) < private_count:
                return None
            private = self.rng.sample(candidates, private_count)
            globally_used.update(private)
            query_lists.append(shared + private)
        selected_sets = [set(values) for values in query_lists]
        overlap = len(set.intersection(*selected_sets)) / context_size
        records = [
            [by_model[model][query_id] for query_id in query_ids]
            for model, query_ids in zip(models, query_lists)
        ]
        return records, {
            "context_overlap": float(overlap),
            "context_requested_shared_fraction": float(requested_shared),
            **self._context_alignment_metrics(records),
        }

    def _context_alignment_metrics(
        self,
        records: list[list[RoutingRecord | None]],
    ) -> dict[str, float]:
        """Describe alignment after layout construction without sampling again.

        Pair overlap is the intersection size for each unordered model pair;
        Jaccard divides that intersection by the pair's union. The observer
        histogram is over unique query IDs and includes explicit zero-valued
        bins through ``max_models`` so aggregation does not drop absent bins.
        These metrics are audit-only and never participate in acceptance.
        """
        selected_sets = [
            {record.query_id for record in model_records if record is not None}
            for model_records in records
        ]
        pair_overlaps: list[float] = []
        pair_jaccards: list[float] = []
        for left_index, left in enumerate(selected_sets):
            for right in selected_sets[left_index + 1:]:
                intersection = len(left & right)
                union = len(left | right)
                pair_overlaps.append(float(intersection))
                pair_jaccards.append(float(intersection / union if union else 1.0))

        observer_counts: Counter[str] = Counter()
        for selected in selected_sets:
            observer_counts.update(selected)
        observer_values = list(observer_counts.values())
        unique_queries = len(observer_values)
        metrics = {
            "context_pair_overlap_count_mean": (
                sum(pair_overlaps) / len(pair_overlaps) if pair_overlaps else 0.0
            ),
            "context_pair_overlap_count_min": min(pair_overlaps, default=0.0),
            "context_pair_overlap_count_max": max(pair_overlaps, default=0.0),
            "context_pair_jaccard_mean": (
                sum(pair_jaccards) / len(pair_jaccards) if pair_jaccards else 0.0
            ),
            "context_pair_jaccard_min": min(pair_jaccards, default=0.0),
            "context_pair_jaccard_max": max(pair_jaccards, default=0.0),
            "context_unique_query_count": float(unique_queries),
            "context_query_observer_count_mean": (
                sum(observer_values) / unique_queries if unique_queries else 0.0
            ),
            "context_query_observer_count_min": float(
                min(observer_values, default=0)
            ),
            "context_query_observer_count_max": float(
                max(observer_values, default=0)
            ),
        }
        observer_histogram = Counter(observer_values)
        for observer_count in range(1, self.max_models + 1):
            metrics[f"context_query_observer_count_{observer_count}_fraction"] = (
                observer_histogram[observer_count] / unique_queries
                if unique_queries else 0.0
            )
        return metrics

    def _build_episode(
        self,
        task: str,
        episode_type: str,
        layout: str,
        audit_only: bool = False,
    ) -> tuple[RoutingEpisode | EpisodeAudit | None, str]:
        by_model = self._by_task[task]
        max_models = min(self.max_models, len(by_model))
        cached = self._accepted_pool_cache[(episode_type, task)]
        if episode_type != "natural" and cached and self.rng.random() < 0.90:
            models = list(self.rng.choice(cached))
            # Cache only accelerates candidate-set discovery; slot order remains
            # freshly anonymous on every episode.
            self.rng.shuffle(models)
            model_count = len(models)
            self._stats[f"cached_pool_proposals/{episode_type}"] += 1
        else:
            model_count = self.rng.randint(self.min_models, max_models)
            models = self.rng.sample(self._task_models[task], model_count)
            self.rng.shuffle(models)

        # Targets are proposed uniformly from the task independently of model
        # quality and of the sampled anonymous candidate identities.
        query_ids = self._task_queries[task]
        max_target_count = min(self.max_targets, len(query_ids))
        if max_target_count < self.min_targets:
            return None, "task_target_count"
        target_count = self.rng.randint(self.min_targets, max_target_count)
        target_ids = self.rng.sample(query_ids, target_count)
        if any(
            sum(query_id in by_model[model] for model in models) < 2
            for query_id in target_ids
        ):
            return None, "target_coverage_lt2"

        context_size = self._sample_context_size()
        context_result = self._context_records(
            task, models, target_ids, context_size, layout
        )
        if context_result is None:
            return None, f"context_infeasible_{layout}"
        contexts, context_metrics = context_result
        # Natural proposals are accepted before outcome statistics are read.
        # Statistics are computed afterward only to make dry-run audits useful.
        target_stats: dict[str, Any] = {}
        if episode_type == "natural":
            accepted, reason, acceptance_metrics = True, "accepted", {}
        else:
            target_stats = self._target_statistics(task, models, target_ids)
            accepted, reason, acceptance_metrics = self._accept(
                episode_type, task, target_stats
            )
        if not accepted:
            return None, reason
        if episode_type != "natural":
            pool = tuple(sorted(models))
            pool_cache = self._accepted_pool_cache[(episode_type, task)]
            if pool not in pool_cache:
                # A small cache preserves diversity without unbounded growth.
                if len(pool_cache) >= 16:
                    pool_cache.pop(0)
                pool_cache.append(pool)
        else:
            # Computed only after unconditional natural acceptance, solely for
            # auditing; these outcomes cannot affect the proposal.
            target_stats = self._target_statistics(task, models, target_ids)
        target_coverage = sum(
            query_id in by_model[model]
            for query_id in target_ids for model in models
        ) / (target_count * model_count)
        metrics = {
            **context_metrics,
            **acceptance_metrics,
            "target_count": float(target_count),
            "model_count": float(model_count),
            "context_size": float(context_size),
            "target_coverage": float(target_coverage),
            "distinct_winners": float(
                len(target_stats.get("winner_counts", ()))
            ),
        }
        if audit_only:
            return EpisodeAudit(
                task=task,
                source=self.corpus.task_sources[task],
                episode_type=episode_type,
                context_layout=layout,
                sampling_metrics=metrics,
            ), "accepted"

        dim = self.corpus.embedding_dim
        max_context_width = max(len(records) for records in contexts)
        context_query = torch.zeros(model_count, max_context_width, dim)
        feature_count = 2 if self.context_observation_schema == "score_cost" else 7
        context_features = torch.zeros(model_count, max_context_width, feature_count)
        context_mask = torch.zeros(model_count, max_context_width, dtype=torch.bool)
        context_query_ids: list[list[str]] = []
        for model_index, records in enumerate(contexts):
            context_query_ids.append([
                record.query_id if record is not None else "" for record in records
            ])
            for context_index, record in enumerate(records):
                if record is None:
                    continue
                context_query[model_index, context_index] = torch.tensor(
                    record.query_embedding, dtype=torch.float32
                )
                if self.context_observation_schema == "score_cost":
                    context_features[model_index, context_index] = torch.tensor([
                        float(record.score), math.log1p(self._point_cost(record))
                    ])
                else:
                    # Cost and cost-available are excluded from the quality Context.
                    context_features[model_index, context_index] = torch.tensor([
                        float(record.score),
                        0.0,
                        _score_uncertainty(record, self.bins),
                        math.log1p(max(1, record.num_observations)),
                        0.0,
                        float(record.score_distribution is not None),
                        1.0,
                    ])
                context_mask[model_index, context_index] = True

        context_cost_low = context_features.new_zeros(())
        context_cost_high = context_features.new_zeros(())
        if self.context_observation_schema == "score_cost":
            observed_context_cost = context_features[..., 1].masked_select(context_mask)
            context_cost_low = observed_context_cost.min()
            context_cost_high = observed_context_cost.max()
            span = context_cost_high - context_cost_low
            if span > 0:
                normalized = (context_features[..., 1] - context_cost_low) / span
                context_features[..., 1] = normalized.masked_fill(~context_mask, 0.0)
            else:
                context_features[..., 1].zero_()

        target_query = torch.zeros(target_count, dim)
        target_distribution = torch.zeros(target_count, model_count, self.bins)
        target_score = torch.zeros(target_count, model_count)
        target_cost = torch.zeros(target_count, model_count)
        target_cost_normalized = torch.zeros(target_count, model_count)
        target_cost_log = torch.zeros(target_count, model_count)
        target_cost_mask = torch.zeros(target_count, model_count, dtype=torch.bool)
        target_uncertainty = torch.zeros(target_count, model_count)
        target_num_observations = torch.zeros(target_count, model_count)
        target_stability_mask = torch.zeros(
            target_count, model_count, dtype=torch.bool
        )
        target_mask = torch.zeros(target_count, model_count, dtype=torch.bool)
        target_texts: list[str] = []
        for target_index, query_id in enumerate(target_ids):
            records = [
                by_model[model][query_id]
                for model in models if query_id in by_model[model]
            ]
            target_texts.append(str(records[0].query_text))
            target_query[target_index] = self._query_embedding(records)
            for model_index, model in enumerate(models):
                record = by_model[model].get(query_id)
                if record is None:
                    continue
                target_distribution[target_index, model_index] = _distribution(
                    record, self.bins
                )
                target_score[target_index, model_index] = float(record.score)
                point_cost = (
                    self._point_cost(record)
                    if self.context_observation_schema == "score_cost"
                    else max(0.0, float(record.cost))
                )
                target_cost[target_index, model_index] = point_cost
                target_cost_log[target_index, model_index] = math.log1p(point_cost)
                target_cost_mask[target_index, model_index] = (
                    True if self.context_observation_schema == "score_cost"
                    else _cost_available(record)
                )
                target_uncertainty[target_index, model_index] = _score_uncertainty(
                    record, self.bins
                )
                target_num_observations[target_index, model_index] = max(
                    1, int(record.num_observations)
                )
                target_stability_mask[target_index, model_index] = (
                    record.score_distribution is not None
                    or int(record.num_observations) > 1
                )
                target_mask[target_index, model_index] = True

        if self.context_observation_schema == "score_cost":
            span = context_cost_high - context_cost_low
            if span > 0:
                target_cost_normalized = (
                    (target_cost_log - context_cost_low) / span
                ).clamp(0.0, 1.0)
                target_cost_normalized.masked_fill_(~target_mask, 0.0)

        routing_score = target_score.clone()
        return RoutingEpisode(
            context_query=context_query,
            context_features=context_features,
            context_mask=context_mask,
            target_query=target_query,
            target_distribution=target_distribution,
            target_score=target_score,
            target_routing_score=routing_score,
            target_cost_log=target_cost_log,
            target_cost_mask=target_cost_mask,
            target_uncertainty=target_uncertainty,
            target_mask=target_mask,
            task=task,
            source=self.corpus.task_sources[task],
            models=models,
            global_model_score=torch.tensor([
                self._global_scores[task][model] for model in models
            ]),
            context_policy=layout,
            target_sampling_mode=episode_type,
            episode_type=episode_type,
            context_layout=layout,
            context_size=context_size,
            context_query_ids=context_query_ids,
            target_query_ids=target_ids,
            target_texts=target_texts,
            sampling_metrics=metrics,
            target_cost=target_cost,
            target_cost_normalized=target_cost_normalized,
            target_num_observations=target_num_observations,
            target_stability_mask=target_stability_mask,
        ), "accepted"

    def _record_episode(self, episode: RoutingEpisode | EpisodeAudit) -> None:
        self._stats["accepted"] += 1
        self._stats[f"accepted_type/{episode.episode_type}"] += 1
        self._stats[f"accepted_layout/{episode.context_layout}"] += 1
        self._task_stats[episode.task] += 1
        self._source_stats[episode.source] += 1
        for key, value in episode.sampling_metrics.items():
            if math.isfinite(float(value)):
                metric_key = f"{episode.episode_type}/{key}"
                self._metric_sums[metric_key] += float(value)
                self._metric_counts[metric_key] += 1
                layout_metric_key = f"{episode.context_layout}/{key}"
                self._metric_sums[f"layout/{layout_metric_key}"] += float(value)
                self._metric_counts[f"layout/{layout_metric_key}"] += 1

    def sample(
        self,
        task: str | None = None,
        episode_type: str | None = None,
        context_layout: str | None = None,
    ) -> RoutingEpisode:
        if task is not None and task not in self._valid_tasks:
            raise ValueError(f"task {task!r} cannot form a task-local routing episode")
        if episode_type is None:
            episode_type = self._next_from_cycle(
                self._episode_cycle, self.episode_cycle
            )
        if episode_type not in {"natural", "opportunity", "boundary"}:
            raise ValueError(f"unknown episode type: {episode_type}")
        if context_layout is None:
            context_layout = self._next_from_cycle(
                self._layout_cycle, self.layout_cycle
            )
        if context_layout not in {
            "aligned_dense", "aligned_sparse", "unaligned_sparse"
        }:
            raise ValueError(f"unknown Context layout: {context_layout}")

        for task_attempt in range(self.max_task_resamples):
            selected_task = task or self._sample_task()
            for _ in range(self.max_episode_attempts):
                self._stats["proposals"] += 1
                self._stats[f"proposals_type/{episode_type}"] += 1
                episode, reason = self._build_episode(
                    selected_task, episode_type, context_layout, audit_only=False
                )
                if episode is not None:
                    self._record_episode(episode)
                    return episode
                self._stats["rejected"] += 1
                self._reason_stats[f"{episode_type}/{reason}"] += 1
            self._stats[f"task_resamples/{episode_type}"] += 1
            if task is not None:
                break
        raise RuntimeError(
            f"could not sample {episode_type}/{context_layout} after "
            f"{self.max_episode_attempts} proposals per task"
        )

    def sample_audit(
        self,
        task: str | None = None,
        episode_type: str | None = None,
        context_layout: str | None = None,
    ) -> EpisodeAudit:
        """Run identical proposal/rejection logic without allocating tensors."""
        if task is not None and task not in self._valid_tasks:
            raise ValueError(f"task {task!r} cannot form a task-local routing episode")
        if episode_type is None:
            episode_type = self._next_from_cycle(
                self._episode_cycle, self.episode_cycle
            )
        if context_layout is None:
            context_layout = self._next_from_cycle(
                self._layout_cycle, self.layout_cycle
            )
        if episode_type not in {"natural", "opportunity", "boundary"}:
            raise ValueError(f"unknown episode type: {episode_type}")
        if context_layout not in {
            "aligned_dense", "aligned_sparse", "unaligned_sparse"
        }:
            raise ValueError(f"unknown Context layout: {context_layout}")
        for _ in range(self.max_task_resamples):
            selected_task = task or self._sample_task()
            for _ in range(self.max_episode_attempts):
                self._stats["proposals"] += 1
                self._stats[f"proposals_type/{episode_type}"] += 1
                episode, reason = self._build_episode(
                    selected_task, episode_type, context_layout, audit_only=True
                )
                if episode is not None:
                    assert isinstance(episode, EpisodeAudit)
                    self._record_episode(episode)
                    return episode
                self._stats["rejected"] += 1
                self._reason_stats[f"{episode_type}/{reason}"] += 1
            self._stats[f"task_resamples/{episode_type}"] += 1
            if task is not None:
                break
        raise RuntimeError(
            f"could not audit {episode_type}/{context_layout} after "
            f"{self.max_episode_attempts} proposals per task"
        )

    def _batch_tasks(self, batch_size: int) -> list[str]:
        """Plan independent episodes while covering sources/tasks in a batch."""
        if batch_size < 2 or len(self._tasks_by_source) < 2:
            return [self._sample_task() for _ in range(batch_size)]
        sources = list(self._tasks_by_source)
        self.rng.shuffle(sources)
        source_count = min(batch_size, self.sources_per_batch, len(sources))
        chosen_sources: list[str] = []
        while sources and len(chosen_sources) < source_count:
            weights = [
                sum(len(self._task_queries[task]) for task in self._tasks_by_source[source])
                for source in sources
            ]
            source = self.rng.choices(sources, weights=weights, k=1)[0]
            sources.remove(source)
            chosen_sources.append(source)
        tasks = [
            self.rng.choices(
                self._tasks_by_source[source],
                weights=[len(self._task_queries[task]) for task in self._tasks_by_source[source]],
                k=1,
            )[0]
            for source in chosen_sources
        ]
        distinct_target = min(batch_size, self.tasks_per_batch, len(self._valid_tasks))
        remaining = [task for task in self._valid_tasks if task not in tasks]
        while remaining and len(tasks) < distinct_target:
            task = self.rng.choices(
                remaining,
                weights=[len(self._task_queries[value]) for value in remaining],
                k=1,
            )[0]
            remaining.remove(task)
            tasks.append(task)
        while len(tasks) < batch_size:
            tasks.append(self._sample_task())
        self.rng.shuffle(tasks)
        return tasks

    def _sample_for_task_or_source(
        self, task: str, episode_type: str, context_layout: str
    ) -> RoutingEpisode:
        source = self.corpus.task_sources[task]
        alternatives = [
            value for value in self._tasks_by_source[source] if value != task
        ]
        self.rng.shuffle(alternatives)
        for candidate in [task, *alternatives]:
            try:
                return self.sample(
                    task=candidate,
                    episode_type=episode_type,
                    context_layout=context_layout,
                )
            except RuntimeError:
                continue
        # Preserve the requested episode type/layout; broaden only the task.
        return self.sample(
            episode_type=episode_type, context_layout=context_layout
        )

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor | list]:
        episodes = []
        for task in self._batch_tasks(batch_size):
            episode_type = self._next_from_cycle(
                self._episode_cycle, self.episode_cycle
            )
            context_layout = self._next_from_cycle(
                self._layout_cycle, self.layout_cycle
            )
            episodes.append(self._sample_for_task_or_source(
                task, episode_type, context_layout
            ))
        batch = collate_episodes(episodes)
        batch["episode_types"] = [episode.episode_type for episode in episodes]
        batch["context_layouts"] = [episode.context_layout for episode in episodes]
        batch["context_sizes"] = torch.tensor([
            episode.context_size for episode in episodes
        ], dtype=torch.long)
        max_targets = batch["target_score"].shape[1]
        max_models = batch["target_score"].shape[2]
        batch["target_cost"] = torch.zeros(
            len(episodes), max_targets, max_models
        )
        batch["target_num_observations"] = torch.zeros(
            len(episodes), max_targets, max_models
        )
        batch["target_stability_mask"] = torch.zeros(
            len(episodes), max_targets, max_models, dtype=torch.bool
        )
        batch["target_cost_normalized"] = torch.zeros(
            len(episodes), max_targets, max_models
        )
        for index, episode in enumerate(episodes):
            targets, models = episode.target_score.shape
            assert episode.target_cost is not None
            assert episode.target_num_observations is not None
            assert episode.target_stability_mask is not None
            assert episode.target_cost_normalized is not None
            batch["target_cost"][index, :targets, :models] = episode.target_cost
            batch["target_num_observations"][index, :targets, :models] = (
                episode.target_num_observations
            )
            batch["target_stability_mask"][index, :targets, :models] = (
                episode.target_stability_mask
            )
            batch["target_cost_normalized"][index, :targets, :models] = (
                episode.target_cost_normalized
            )
        return batch

    def diagnostics(self) -> dict[str, Any]:
        proposals = self._stats["proposals"]
        accepted = self._stats["accepted"]
        result: dict[str, Any] = {
            "proposals": proposals,
            "accepted": accepted,
            "rejected": self._stats["rejected"],
            "acceptance_rate": accepted / proposals if proposals else 0.0,
            "accepted_types": {
                name: self._stats[f"accepted_type/{name}"]
                for name in ("natural", "opportunity", "boundary")
            },
            "accepted_layouts": {
                name: self._stats[f"accepted_layout/{name}"]
                for name in ("aligned_dense", "aligned_sparse", "unaligned_sparse")
            },
            "proposal_acceptance_by_type": {},
            "task_resamples": {
                name: self._stats[f"task_resamples/{name}"]
                for name in ("natural", "opportunity", "boundary")
            },
            "rejections": dict(sorted(self._reason_stats.items())),
            "accepted_tasks": dict(self._task_stats.most_common()),
            "accepted_sources": dict(self._source_stats.most_common()),
            "task_boundary_thresholds": {
                task: self._boundary_thresholds[task] for task in self._valid_tasks
            },
            "accepted_pool_cache_sizes": {
                f"{episode_type}/{task}": len(values)
                for (episode_type, task), values
                in sorted(self._accepted_pool_cache.items()) if values
            },
            "metric_means": {
                key: self._metric_sums[key] / self._metric_counts[key]
                for key in sorted(self._metric_sums)
                if not key.startswith("layout/")
            },
            "metric_means_by_layout": {
                layout: {
                    key.removeprefix(f"layout/{layout}/"): (
                        self._metric_sums[key] / self._metric_counts[key]
                    )
                    for key in sorted(self._metric_sums)
                    if key.startswith(f"layout/{layout}/")
                }
                for layout in (
                    "aligned_dense", "aligned_sparse", "unaligned_sparse"
                )
            },
        }
        for name in ("natural", "opportunity", "boundary"):
            type_proposals = self._stats[f"proposals_type/{name}"]
            type_accepted = self._stats[f"accepted_type/{name}"]
            result["proposal_acceptance_by_type"][name] = (
                type_accepted / type_proposals if type_proposals else 0.0
            )
        return result
