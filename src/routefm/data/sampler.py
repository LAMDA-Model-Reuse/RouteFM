from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import torch

from .corpus import CanonicalCorpus
from .schema import RoutingRecord


@dataclass
class Episode:
    context_query: torch.Tensor
    context_features: torch.Tensor
    context_mask: torch.Tensor
    target_query: torch.Tensor
    target_distribution: torch.Tensor
    target_score: torch.Tensor
    target_routing_score: torch.Tensor
    target_cost_log: torch.Tensor
    target_cost_mask: torch.Tensor
    target_uncertainty: torch.Tensor
    target_mask: torch.Tensor
    task: str
    source: str
    models: list[str]
    global_model_score: torch.Tensor
    context_policy: str = "random"
    target_sampling_mode: str = "uniform"


def _distribution(record: RoutingRecord, bins: int) -> torch.Tensor:
    if record.score_distribution is not None:
        value = torch.tensor(record.score_distribution, dtype=torch.float32)
        if value.numel() != bins:
            raise ValueError(f"score_distribution has {value.numel()} bins; expected {bins}")
        return value / value.sum().clamp_min(1e-8)
    value = torch.zeros(bins)
    value[min(bins - 1, round(record.score * (bins - 1)))] = 1.0
    return value


def _cost_available(record: RoutingRecord) -> bool:
    return bool(record.metadata.get("cost_available", record.cost > 0.0))


def _score_uncertainty(record: RoutingRecord, bins: int) -> float:
    """Outcome uncertainty, or a conservative posterior proxy for point labels."""
    if record.score_distribution is not None:
        distribution = _distribution(record, bins)
        positions = torch.linspace(0.0, 1.0, bins)
        mean = (distribution * positions).sum()
        variance = (distribution * (positions - mean).square()).sum()
        return float(variance.clamp_min(0.0).sqrt().item())
    observations = max(1, int(record.num_observations))
    alpha = 1.0 + float(record.score) * observations
    beta = 1.0 + (1.0 - float(record.score)) * observations
    total = alpha + beta
    variance = alpha * beta / (total * total * (total + 1.0))
    return math.sqrt(max(0.0, variance))


class EpisodeSampler:
    """Samples training examples as routing episodes, never flattened cells."""

    def __init__(
        self,
        corpus: CanonicalCorpus,
        min_models: int = 4,
        max_models: int = 16,
        context_sizes: tuple[int, ...] = (4, 8, 16, 32, 64, 128),
        min_targets: int = 8,
        max_targets: int = 32,
        context_dropout: tuple[float, float] = (0.2, 0.7),
        bins: int = 21,
        source_weights: dict[str, float] | None = None,
        aligned_context: bool = False,
        cost_dropout: tuple[float, float] = (0.0, 0.0),
        normalize_cost_per_episode: bool = False,
        target_sampling: str = "uniform",
        target_sampling_weights: dict[str, float] | None = None,
        target_sampling_pool_size: int | None = None,
        min_distinct_target_winners: int = 2,
        min_opportunity_fraction: float = 0.5,
        episode_resample_attempts: int = 24,
        target_score_source: str = "observed",
        min_winner_margin: float = 0.0,
        context_sampling_weights: dict[str, float] | None = None,
        context_size_weights: dict[int, float] | None = None,
        context_sampling_pool_size: int | None = None,
        simulated_new_model_fraction: float = 0.0,
        simulated_new_model_context_sizes: tuple[int, ...] | None = None,
        global_model_scores: dict[str, dict[str, float]] | None = None,
        seed: int = 0,
    ):
        self.corpus = corpus
        self.min_models = min_models
        self.max_models = max_models
        self.context_sizes = context_sizes
        self.min_targets = min_targets
        self.max_targets = max_targets
        self.context_dropout = context_dropout
        self.bins = bins
        self.source_weights = source_weights or {}
        self.aligned_context = aligned_context
        self.cost_dropout = cost_dropout
        self.normalize_cost_per_episode = bool(normalize_cost_per_episode)
        allowed_target_sampling = {
            "uniform", "disagreement", "opportunity", "opportunity_switch"
        }
        if target_sampling not in allowed_target_sampling:
            raise ValueError(
                "target_sampling must be uniform, disagreement, opportunity, or opportunity_switch"
            )
        self.target_sampling = target_sampling
        self.target_sampling_weights = target_sampling_weights or {target_sampling: 1.0}
        unknown_target_sampling = set(self.target_sampling_weights) - allowed_target_sampling
        if unknown_target_sampling or not any(
            float(value) > 0 for value in self.target_sampling_weights.values()
        ):
            raise ValueError(
                "invalid target sampling weights; "
                f"unknown={sorted(unknown_target_sampling)}"
            )
        self.target_sampling_pool_size = (
            int(target_sampling_pool_size) if target_sampling_pool_size else None
        )
        self.min_distinct_target_winners = int(min_distinct_target_winners)
        self.min_opportunity_fraction = float(min_opportunity_fraction)
        self.episode_resample_attempts = int(episode_resample_attempts)
        if target_score_source not in {"observed", "latent_probability"}:
            raise ValueError("target_score_source must be observed or latent_probability")
        self.target_score_source = target_score_source
        self.min_winner_margin = float(min_winner_margin)
        if self.min_distinct_target_winners < 2:
            raise ValueError("min_distinct_target_winners must be at least 2")
        if not 0.0 <= self.min_opportunity_fraction <= 1.0:
            raise ValueError("min_opportunity_fraction must be in [0, 1]")
        if self.episode_resample_attempts < 1:
            raise ValueError("episode_resample_attempts must be positive")
        if not 0.0 <= self.min_winner_margin <= 1.0:
            raise ValueError("min_winner_margin must be in [0, 1]")
        self.context_sampling_weights = context_sampling_weights or {"random": 1.0}
        allowed_context_policies = {"random", "semantic", "informative", "adversarial"}
        unknown = set(self.context_sampling_weights) - allowed_context_policies
        if unknown or not any(float(value) > 0 for value in self.context_sampling_weights.values()):
            raise ValueError(f"invalid context sampling weights; unknown={sorted(unknown)}")
        self.context_sampling_pool_size = int(context_sampling_pool_size or 256)
        self.simulated_new_model_fraction = float(simulated_new_model_fraction)
        self.simulated_new_model_context_sizes = tuple(
            int(size) for size in (simulated_new_model_context_sizes or ())
        )
        if not 0.0 <= self.simulated_new_model_fraction < 1.0:
            raise ValueError("simulated_new_model_fraction must be in [0, 1)")
        if self.simulated_new_model_context_sizes and min(
            self.simulated_new_model_context_sizes
        ) < 1:
            raise ValueError("simulated_new_model_context_sizes must be positive")
        self.context_size_weights = {
            int(size): float(weight) for size, weight in (context_size_weights or {}).items()
        }
        unknown_context_sizes = set(self.context_size_weights) - set(self.context_sizes)
        if unknown_context_sizes or (
            self.context_size_weights
            and not any(weight > 0 for weight in self.context_size_weights.values())
        ):
            raise ValueError(
                "invalid context size weights; "
                f"unknown={sorted(unknown_context_sizes)}"
            )
        if self.target_sampling_pool_size is not None and self.target_sampling_pool_size < max_targets:
            raise ValueError("target_sampling_pool_size must be >= max_targets")
        self.rng = random.Random(seed)
        self._indexed = {task: self._index(records) for task, records in corpus.tasks.items()}
        inferred_global_model_scores = {
            task: {
                model: sum(self._routing_score(record) for record in rows.values()) / len(rows)
                for model, rows in index.items()
            }
            for task, index in self._indexed.items()
        }
        self._global_model_scores = global_model_scores or inferred_global_model_scores
        self._valid_tasks = []
        self._invalid_task_reasons: dict[str, str] = {}
        for task, index in self._indexed.items():
            if len(index) < min_models:
                self._invalid_task_reasons[task] = f"only {len(index)} models"
                continue
            union_queries = set().union(*(set(rows) for rows in index.values()))
            if len(union_queries) < min_targets + 1:
                self._invalid_task_reasons[task] = (
                    f"only {len(union_queries)} queries; need at least {min_targets + 1} "
                    "for context plus targets"
                )
                continue
            if aligned_context:
                common_queries = set.intersection(*(set(rows) for rows in index.values()))
                if len(common_queries) < min_targets + 1:
                    self._invalid_task_reasons[task] = (
                        f"only {len(common_queries)} shared queries; need at least {min_targets + 1}"
                    )
                    continue
            self._valid_tasks.append(task)
        if not self._valid_tasks:
            examples = list(self._invalid_task_reasons.items())[:5]
            raise ValueError(
                f"no dataset can form an episode with >= {min_models} models and "
                f">= {min_targets + 1} queries; examples={examples}"
            )

    def _sample_task(self) -> str:
        weights = [max(0.0, float(self.source_weights.get(self.corpus.task_sources[task], 1.0))) for task in self._valid_tasks]
        if not any(weights):
            raise ValueError("all configured source weights are zero")
        # Divide a source's probability across its tasks; source weights therefore
        # control episodes instead of being multiplied by the number of datasets.
        counts: dict[str, int] = defaultdict(int)
        for task in self._valid_tasks:
            counts[self.corpus.task_sources[task]] += 1
        weights = [weight / counts[self.corpus.task_sources[task]] for task, weight in zip(self._valid_tasks, weights)]
        return self.rng.choices(self._valid_tasks, weights=weights, k=1)[0]

    @staticmethod
    def _index(records: list[RoutingRecord]) -> dict[str, dict[str, RoutingRecord]]:
        result: dict[str, dict[str, RoutingRecord]] = defaultdict(dict)
        for record in records:
            result[record.model_id][record.query_id] = record
        return dict(result)

    def _routing_score(self, record: RoutingRecord) -> float:
        if self.target_score_source == "latent_probability":
            value = record.metadata.get("latent_probability")
            if value is not None:
                return float(value)
        return float(record.score)

    def _query_routing_statistics(
        self,
        query_id: str,
        by_model: dict[str, dict[str, RoutingRecord]],
        models: list[str],
        global_model: str | None = None,
    ) -> tuple[str, float, float, float]:
        present = [model for model in models if query_id in by_model[model]]
        scored = sorted(
            ((self._routing_score(by_model[model][query_id]), model) for model in present),
            reverse=True,
        )
        if not scored:
            raise ValueError(f"query {query_id!r} has no candidate records")
        values = [value for value, _ in scored]
        mean = sum(values) / len(values)
        disagreement = math.sqrt(
            sum((value - mean) ** 2 for value in values) / len(values)
        ) if len(values) >= 2 else 0.0
        margin = values[0] - values[1] if len(values) >= 2 else 0.0
        opportunity = 0.0
        if global_model is not None and global_model in by_model and query_id in by_model[global_model]:
            opportunity = values[0] - self._routing_score(by_model[global_model][query_id])
        return scored[0][1], disagreement, opportunity, margin

    def sample(
        self,
        task: str | None = None,
        target_sampling_override: str | None = None,
    ) -> Episode:
        requested_task = task
        if requested_task is not None and requested_task not in self._valid_tasks:
            raise ValueError(f"task {requested_task!r} cannot form an episode")
        if target_sampling_override is not None:
            if target_sampling_override not in self.target_sampling_weights:
                raise ValueError(
                    f"target sampling override {target_sampling_override!r} is not configured"
                )
            target_sampling = target_sampling_override
        else:
            target_sampling = self.rng.choices(
                list(self.target_sampling_weights),
                weights=[
                    max(0.0, float(self.target_sampling_weights[name]))
                    for name in self.target_sampling_weights
                ],
                k=1,
            )[0]
        for _ in range(self.episode_resample_attempts):
            task = requested_task or self._sample_task()
            by_model = self._indexed[task]
            max_models = min(self.max_models, len(by_model))
            n_models = self.rng.randint(self.min_models, max_models)
            models = self.rng.sample(sorted(by_model), n_models)
            self.rng.shuffle(models)
            common_queries = set.intersection(*(set(by_model[m]) for m in models))
            if len(common_queries) < self.min_targets + 1:
                # Sparse corpora may not have a fully dense pool; retain models with the most overlap.
                union = set.union(*(set(by_model[m]) for m in models))
                query_ids = sorted(union)
            else:
                query_ids = sorted(common_queries)
            if target_sampling != "opportunity_switch":
                break
            task_global_scores = self._global_model_scores.get(task, {})
            global_model = max(models, key=lambda model: task_global_scores.get(model, 0.0))
            stable_statistics = [
                self._query_routing_statistics(query_id, by_model, models, global_model)
                for query_id in query_ids
                if any(query_id in by_model[model] for model in models)
            ]
            winners = {winner for winner, _, _, margin in stable_statistics if margin >= self.min_winner_margin}
            stable_queries = sum(margin >= self.min_winner_margin for _, _, _, margin in stable_statistics)
            stable_opportunities = sum(
                margin >= self.min_winner_margin and winner != global_model
                for winner, _, _, margin in stable_statistics
            )
            required_opportunities = math.ceil(self.min_targets * self.min_opportunity_fraction)
            if (
                len(winners) >= self.min_distinct_target_winners
                and stable_queries >= self.min_targets
                and stable_opportunities >= required_opportunities
            ):
                break
        else:
            raise RuntimeError(
                "could not sample an opportunity episode with "
                f">={self.min_distinct_target_winners} distinct query winners"
            )
        n_targets = min(len(query_ids) - 1, self.rng.randint(self.min_targets, self.max_targets))
        if n_targets < 1:
            raise RuntimeError(f"task {task!r} has too few queries for an episode")
        if target_sampling in {"disagreement", "opportunity", "opportunity_switch"}:
            weighted_query_ids = query_ids
            if self.target_sampling_pool_size and len(query_ids) > self.target_sampling_pool_size:
                # Large real tasks may contain thousands of queries. Scoring
                # every query independently for every episode makes large-batch
                # training CPU-bound. A fresh uniform proposal pool followed by
                # exact weighted-without-replacement sampling retains hard-query
                # emphasis without traversing the complete task each time.
                weighted_query_ids = self.rng.sample(query_ids, self.target_sampling_pool_size)
            weighted: list[tuple[float, str]] = []
            query_statistics: dict[str, tuple[str, float, float, float]] = {}
            task_global_scores = self._global_model_scores.get(task, {})
            global_model = max(models, key=lambda model: task_global_scores.get(model, 0.0))
            for query_id in weighted_query_ids:
                winner, disagreement, opportunity, winner_margin = (
                    self._query_routing_statistics(query_id, by_model, models, global_model)
                )
                if target_sampling == "opportunity_switch" and winner_margin < self.min_winner_margin:
                    continue
                query_statistics[query_id] = (winner, disagreement, opportunity, winner_margin)
                if target_sampling == "opportunity":
                    weight = max(0.0, opportunity) + 1e-4
                elif target_sampling == "opportunity_switch":
                    weight = disagreement + 2.0 * max(0.0, opportunity) + 1e-4
                else:
                    weight = disagreement + 1e-4
                # Weighted sampling without replacement (Efraimidis-Spirakis key).
                weighted.append((math.log(max(self.rng.random(), 1e-12)) / weight, query_id))
            if target_sampling == "opportunity_switch":
                by_winner: dict[str, list[str]] = defaultdict(list)
                for query_id, (winner, _, _, _) in query_statistics.items():
                    by_winner[winner].append(query_id)
                ranked_groups = sorted(
                    by_winner.values(),
                    key=lambda group: max(
                        query_statistics[query_id][1] + 2.0 * query_statistics[query_id][2]
                        for query_id in group
                    ),
                    reverse=True,
                )
                selected = [
                    max(
                        group,
                        key=lambda query_id: (
                            query_statistics[query_id][1] + 2.0 * query_statistics[query_id][2]
                        ),
                    )
                    for group in ranked_groups[: min(self.min_distinct_target_winners, n_targets)]
                ]
                desired_opportunities = min(
                    n_targets, math.ceil(n_targets * self.min_opportunity_fraction)
                )
                opportunity_candidates = sorted(
                    (
                        query_id for query_id, (winner, _, _, _) in query_statistics.items()
                        if winner != global_model and query_id not in selected
                    ),
                    key=lambda query_id: query_statistics[query_id][2],
                    reverse=True,
                )
                current_opportunities = sum(
                    query_statistics[query_id][0] != global_model for query_id in selected
                )
                selected.extend(
                    opportunity_candidates[: max(0, desired_opportunities - current_opportunities)]
                )
                selected_set = set(selected)
                selected.extend(
                    query_id for _, query_id in sorted(weighted, reverse=True)
                    if query_id not in selected_set
                )
                target_ids = set(selected[:n_targets])
            else:
                target_ids = {query_id for _, query_id in sorted(weighted, reverse=True)[:n_targets]}
        else:
            target_ids = set(self.rng.sample(query_ids, n_targets))

        contexts: list[list[RoutingRecord]] = []
        max_k = 1
        policies = list(self.context_sampling_weights)
        policy_weights = [max(0.0, float(self.context_sampling_weights[name])) for name in policies]
        context_policy = self.rng.choices(policies, weights=policy_weights, k=1)[0]
        if self.aligned_context:
            available = sorted(set.intersection(*(set(by_model[m]) for m in models)) - target_ids)
            if not available:
                raise RuntimeError(f"task {task!r} has no shared context queries outside its targets")
            wanted = self._sample_context_size()
            keep_ratio = 1.0 - self.rng.uniform(*self.context_dropout)
            k = max(1, min(len(available), round(wanted * keep_ratio)))
            shared_ids = self._select_context_ids(
                context_policy, available, k, by_model, models, sorted(target_ids)
            )
            contexts = [[by_model[model][query_id] for query_id in shared_ids] for model in models]
            max_k = k
        else:
            for model in models:
                candidates = [r for q, r in by_model[model].items() if q not in target_ids]
                if not candidates:
                    candidates = list(by_model[model].values())
                wanted = self._sample_context_size()
                keep_ratio = 1.0 - self.rng.uniform(*self.context_dropout)
                k = max(1, min(len(candidates), round(wanted * keep_ratio)))
                selected = self.rng.sample(candidates, k)
                contexts.append(selected)
                max_k = max(max_k, k)

        # Simulate deployment-time model addition while training: selected fit
        # candidates have only a small performance history, while the remaining
        # candidates retain full context. The router never receives model IDs.
        if self.simulated_new_model_context_sizes and self.simulated_new_model_fraction:
            count = min(
                n_models - 1,
                max(1, round(n_models * self.simulated_new_model_fraction)),
            )
            for index in self.rng.sample(range(n_models), count):
                budget = self.rng.choice(self.simulated_new_model_context_sizes)
                if len(contexts[index]) > budget:
                    contexts[index] = self.rng.sample(contexts[index], budget)

        dim = self.corpus.embedding_dim
        cost_drop_probability = self.rng.uniform(*self.cost_dropout)
        cq = torch.zeros(n_models, max_k, dim)
        cf = torch.zeros(n_models, max_k, 7)
        cmask = torch.zeros(n_models, max_k, dtype=torch.bool)
        for mi, records in enumerate(contexts):
            for ki, record in enumerate(records):
                cq[mi, ki] = torch.tensor(record.query_embedding)
                keep_cost = (
                    _cost_available(record)
                    and self.rng.random() >= cost_drop_probability
                )
                observed_cost = record.cost if keep_cost else 0.0
                cf[mi, ki] = torch.tensor(
                    [
                        record.score,
                        math.log1p(max(0.0, observed_cost)),
                        _score_uncertainty(record, self.bins),
                        math.log1p(max(1, record.num_observations)),
                        float(keep_cost),
                        float(record.score_distribution is not None),
                        1.0,
                    ]
                )
                cmask[mi, ki] = True

        if self.normalize_cost_per_episode:
            available_cost = (cf[..., 4] > 0.0) & cmask
            low = cf[..., 1].masked_fill(~available_cost, float("inf")).amin()
            high = cf[..., 1].masked_fill(~available_cost, float("-inf")).amax()
            if available_cost.any() and high > low:
                cf[..., 1] = ((cf[..., 1] - low) / (high - low)).masked_fill(
                    ~available_cost, 0.0
                )
            else:
                cf[..., 1].zero_()

        target_ids = sorted(target_ids)
        tq = torch.zeros(n_targets, dim)
        td = torch.zeros(n_targets, n_models, self.bins)
        ts = torch.zeros(n_targets, n_models)
        trs = torch.zeros(n_targets, n_models)
        tc = torch.zeros(n_targets, n_models)
        tcmask = torch.zeros(n_targets, n_models, dtype=torch.bool)
        tu = torch.zeros(n_targets, n_models)
        tmask = torch.zeros(n_targets, n_models, dtype=torch.bool)
        for ti, query_id in enumerate(target_ids):
            any_record = next((by_model[m].get(query_id) for m in models if query_id in by_model[m]), None)
            if any_record is None:
                continue
            tq[ti] = torch.tensor(any_record.query_embedding)
            for mi, model in enumerate(models):
                record = by_model[model].get(query_id)
                if record is None:
                    continue
                td[ti, mi] = _distribution(record, self.bins)
                ts[ti, mi] = record.score
                trs[ti, mi] = self._routing_score(record)
                tc[ti, mi] = math.log1p(max(0.0, record.cost))
                tcmask[ti, mi] = _cost_available(record)
                tu[ti, mi] = _score_uncertainty(record, self.bins)
                tmask[ti, mi] = True
        task_global_scores = self._global_model_scores.get(task, {})
        global_scores = torch.tensor(
            [task_global_scores.get(model, self._global_model_scores.get("*", {}).get(model, 0.0)) for model in models]
        )
        return Episode(
            cq, cf, cmask, tq, td, ts, trs, tc, tcmask, tu, tmask,
            task, self.corpus.task_sources[task], models, global_scores, context_policy,
            target_sampling,
        )

    def _sample_context_size(self) -> int:
        if not self.context_size_weights:
            return self.rng.choice(self.context_sizes)
        return self.rng.choices(
            self.context_sizes,
            weights=[max(0.0, self.context_size_weights.get(size, 0.0)) for size in self.context_sizes],
            k=1,
        )[0]

    def _select_context_ids(
        self,
        policy: str,
        available: list[str],
        k: int,
        by_model: dict[str, dict[str, RoutingRecord]],
        models: list[str],
        target_ids: list[str],
    ) -> list[str]:
        if policy == "random" or len(available) <= k:
            return self.rng.sample(available, k)
        proposal = available
        if len(proposal) > self.context_sampling_pool_size:
            proposal = self.rng.sample(proposal, self.context_sampling_pool_size)

        def record_for(query_id: str) -> RoutingRecord:
            return next(by_model[model][query_id] for model in models if query_id in by_model[model])

        target_embeddings = torch.stack([
            torch.tensor(record_for(query_id).query_embedding, dtype=torch.float32)
            for query_id in target_ids
        ])
        target_embeddings = torch.nn.functional.normalize(target_embeddings, dim=-1)

        semantic_values: dict[str, float] = {}
        if policy in {"semantic", "adversarial"}:
            proposal_embeddings = torch.stack([
                torch.tensor(record_for(query_id).query_embedding, dtype=torch.float32)
                for query_id in proposal
            ])
            proposal_embeddings = torch.nn.functional.normalize(proposal_embeddings, dim=-1)
            maxima = (target_embeddings @ proposal_embeddings.T).max(dim=0).values
            semantic_values = {
                query_id: float(value) for query_id, value in zip(proposal, maxima.tolist())
            }

        def semantic_score(query_id: str) -> float:
            return semantic_values[query_id]

        def disagreement(query_id: str) -> float:
            values = [by_model[model][query_id].score for model in models if query_id in by_model[model]]
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

        if policy == "semantic":
            ranked = sorted(proposal, key=semantic_score, reverse=True)
        elif policy == "informative":
            ranked = sorted(proposal, key=disagreement, reverse=True)
        else:
            # A hard context is semantically close to the target set but suggests
            # a different best model. It tests whether the router aggregates a
            # profile instead of copying one attractive observation.
            target_votes: dict[str, int] = defaultdict(int)
            for query_id in target_ids:
                present = [model for model in models if query_id in by_model[model]]
                if present:
                    target_votes[max(present, key=lambda model: by_model[model][query_id].score)] += 1
            target_best = max(target_votes, key=target_votes.get) if target_votes else models[0]

            def adversarial_score(query_id: str) -> tuple[int, float, float]:
                present = [model for model in models if query_id in by_model[model]]
                context_best = max(present, key=lambda model: by_model[model][query_id].score)
                return (int(context_best != target_best), semantic_score(query_id), disagreement(query_id))

            ranked = sorted(proposal, key=adversarial_score, reverse=True)
        selected = ranked[:k]
        if len(selected) < k:
            selected.extend(self.rng.sample([value for value in available if value not in selected], k - len(selected)))
        return selected

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        return collate_episodes([self.sample() for _ in range(batch_size)])


def collate_episodes(episodes: list[Episode]) -> dict[str, torch.Tensor]:
    batch = len(episodes)
    max_m = max(e.context_query.shape[0] for e in episodes)
    max_k = max(e.context_query.shape[1] for e in episodes)
    max_t = max(e.target_query.shape[0] for e in episodes)
    dim = episodes[0].context_query.shape[-1]
    bins = episodes[0].target_distribution.shape[-1]
    result = {
        "context_query": torch.zeros(batch, max_m, max_k, dim),
        "context_features": torch.zeros(
            batch, max_m, max_k, episodes[0].context_features.shape[-1]
        ),
        "context_mask": torch.zeros(batch, max_m, max_k, dtype=torch.bool),
        "candidate_mask": torch.zeros(batch, max_m, dtype=torch.bool),
        "target_query": torch.zeros(batch, max_t, dim),
        "target_distribution": torch.zeros(batch, max_t, max_m, bins),
        "target_score": torch.zeros(batch, max_t, max_m),
        "target_routing_score": torch.zeros(batch, max_t, max_m),
        "target_cost_log": torch.zeros(batch, max_t, max_m),
        "target_cost_mask": torch.zeros(batch, max_t, max_m, dtype=torch.bool),
        "target_uncertainty": torch.zeros(batch, max_t, max_m),
        "target_mask": torch.zeros(batch, max_t, max_m, dtype=torch.bool),
        "global_model_score": torch.zeros(batch, max_m),
        "tasks": [episode.task for episode in episodes],
        "sources": [episode.source for episode in episodes],
        "models": [episode.models for episode in episodes],
        "context_policies": [episode.context_policy for episode in episodes],
        "target_sampling_modes": [episode.target_sampling_mode for episode in episodes],
    }
    # Routing carries raw Target text for optional trainable text encoders. Keep
    # the base Episode API backward compatible by only adding this field when
    # the episode type exposes it. Empty strings represent padded Targets.
    if any(hasattr(episode, "target_texts") for episode in episodes):
        result["target_texts"] = [[""] * max_t for _ in episodes]
    if any(hasattr(episode, "context_query_ids") for episode in episodes):
        result["context_query_ids"] = [
            [[""] * max_k for _ in range(max_m)] for _ in episodes
        ]
        result["target_query_ids"] = [[""] * max_t for _ in episodes]
    for bi, episode in enumerate(episodes):
        if episode.context_features.shape[-1] != result["context_features"].shape[-1]:
            raise ValueError("all episodes in a batch must use one observation schema")
        m, k = episode.context_mask.shape
        t = episode.target_query.shape[0]
        if "target_texts" in result:
            target_texts = list(getattr(episode, "target_texts", ()))
            if target_texts and len(target_texts) != t:
                raise ValueError(
                    "target_texts must be empty or match the episode Target count"
                )
            result["target_texts"][bi][:len(target_texts)] = target_texts
        if "context_query_ids" in result:
            context_query_ids = list(getattr(episode, "context_query_ids", ()))
            target_query_ids = list(getattr(episode, "target_query_ids", ()))
            if context_query_ids and (
                len(context_query_ids) != m
                or any(len(values) != k for values in context_query_ids)
            ):
                raise ValueError("context_query_ids must match the episode M x K shape")
            if target_query_ids and len(target_query_ids) != t:
                raise ValueError("target_query_ids must match the episode Target count")
            for model_index, values in enumerate(context_query_ids):
                result["context_query_ids"][bi][model_index][:len(values)] = values
            result["target_query_ids"][bi][:len(target_query_ids)] = target_query_ids
        result["context_query"][bi, :m, :k] = episode.context_query
        result["context_features"][bi, :m, :k] = episode.context_features
        result["context_mask"][bi, :m, :k] = episode.context_mask
        result["candidate_mask"][bi, :m] = True
        result["target_query"][bi, :t] = episode.target_query
        result["target_distribution"][bi, :t, :m] = episode.target_distribution
        result["target_score"][bi, :t, :m] = episode.target_score
        result["target_routing_score"][bi, :t, :m] = episode.target_routing_score
        result["target_cost_log"][bi, :t, :m] = episode.target_cost_log
        result["target_cost_mask"][bi, :t, :m] = episode.target_cost_mask
        result["target_uncertainty"][bi, :t, :m] = episode.target_uncertainty
        result["target_mask"][bi, :t, :m] = episode.target_mask
        result["global_model_score"][bi, :m] = episode.global_model_score
    valid_target = result["target_mask"].any(dim=-1)
    routing_score = result["target_routing_score"]
    target_winner = routing_score.masked_fill(~result["target_mask"], -1e4).argmax(dim=-1)
    top_two = routing_score.masked_fill(~result["target_mask"], -1e4).topk(
        min(2, routing_score.shape[-1]), dim=-1
    ).values
    target_winner_margin = (
        top_two[..., 0] - top_two[..., 1]
        if top_two.shape[-1] == 2 else torch.zeros_like(top_two[..., 0])
    )
    global_winner = result["global_model_score"].masked_fill(
        ~result["candidate_mask"], -1e4
    ).argmax(dim=-1)
    result["target_winner"] = target_winner
    result["target_winner_margin"] = target_winner_margin
    result["target_opportunity_mask"] = valid_target & (target_winner != global_winner[:, None])
    result["distinct_target_winners"] = torch.tensor([
        torch.unique(target_winner[index].masked_select(valid_target[index])).numel()
        for index in range(batch)
    ], dtype=torch.long)
    return result
