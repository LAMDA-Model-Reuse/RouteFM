from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def quality_distribution_statistics(
    output: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return quality moments without assigning spread an unsupported meaning.

    With repeated-outcome supervision, predictive spread can estimate outcome
    stability. With point-label one-hot targets it cannot. It is consequently
    exposed only as ``predictive_spread``. Estimation uncertainty (for example
    ensemble disagreement) is a separate deployment-time input.
    """
    logits = output["quality_logits"]
    probability = logits.softmax(dim=-1)
    bins = torch.linspace(
        0.0, 1.0, logits.shape[-1], device=logits.device, dtype=logits.dtype
    )
    distribution_mean = (probability * bins).sum(dim=-1)
    mean = output.get("quality_mean", distribution_mean)
    variance = (
        probability * (bins - distribution_mean.unsqueeze(-1)).square()
    ).sum(dim=-1)
    return {
        "probability": probability,
        "distribution_mean": distribution_mean,
        "mean": mean,
        "predictive_variance": variance.clamp_min(0.0),
        "predictive_spread": variance.clamp_min(0.0).sqrt(),
    }


def deployment_agnostic_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    distribution_weight: float = 1.0,
    mean_weight: float = 0.25,
    pairwise_weight: float = 0.50,
    regret_weight: float = 0.50,
    gap_weight: float = 0.25,
    mean_huber_beta: float = 0.05,
    gap_huber_beta: float = 0.05,
    pairwise_target_temperature: float = 0.05,
    pairwise_prediction_temperature: float = 0.10,
    pairwise_reliability_prior: float = 1.0,
    routing_temperature: float = 0.10,
    pair_tie_tolerance: float = 1e-6,
    distribution_only_sources: list[str] | tuple[str, ...] = (),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Quality-only routing objective whose labels contain no deployment policy.

    It combines distribution calibration, mean calibration, soft pairwise
    ranking, differentiable quality regret, and true-top-two gap calibration.
    Cost, price, parameter count, stability preferences and estimation
    uncertainty never enter a routing label in this loss.
    """
    target_mask = batch["target_mask"].bool()
    target_score = batch["target_score"]
    mask = target_mask.to(target_score.dtype)
    excluded = set(distribution_only_sources)
    sources = batch.get("sources", ["" for _ in range(target_mask.shape[0])])
    calibrated_source = torch.tensor(
        [source not in excluded for source in sources],
        dtype=target_score.dtype,
        device=target_score.device,
    )
    auxiliary_mask = mask * calibrated_source[:, None, None]

    statistics = quality_distribution_statistics(output)
    probability = statistics["probability"]
    predicted_mean = statistics["mean"]
    log_probability = output["quality_logits"].log_softmax(dim=-1)

    distribution_nll = -(
        batch["target_distribution"].to(log_probability.dtype) * log_probability
    ).sum(dim=-1)
    distribution_loss = _masked_mean(distribution_nll, mask)

    mean_error = F.smooth_l1_loss(
        predicted_mean,
        target_score.to(predicted_mean.dtype),
        reduction="none",
        beta=max(float(mean_huber_beta), 1e-8),
    )
    mean_loss = _masked_mean(mean_error, auxiliary_mask)

    models = target_score.shape[-1]
    upper_triangle = torch.triu(
        torch.ones(models, models, dtype=torch.bool, device=target_score.device),
        diagonal=1,
    )
    pair_mask = (
        target_mask.unsqueeze(-1)
        & target_mask.unsqueeze(-2)
        & upper_triangle
        & calibrated_source.bool()[:, None, None, None]
    )
    true_difference = target_score.unsqueeze(-1) - target_score.unsqueeze(-2)
    predicted_difference = predicted_mean.unsqueeze(-1) - predicted_mean.unsqueeze(-2)
    score_gap_pair_target = torch.sigmoid(
        true_difference / max(float(pairwise_target_temperature), 1e-4)
    )
    # A repeated outcome distribution gives a better preference target than its
    # mean alone: P(Q_i > Q_j) + 0.5 P(Q_i == Q_j). Do not infer repeated
    # supervision merely from a one-hot vector; samplers must mark it explicitly.
    target_distribution = batch["target_distribution"].to(predicted_difference.dtype)
    target_distribution = target_distribution / target_distribution.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    strictly_lower_mass = target_distribution.cumsum(dim=-1) - target_distribution
    distribution_pair_target = torch.einsum(
        "btik,btjk->btij", target_distribution, strictly_lower_mass
    ) + 0.5 * torch.einsum(
        "btik,btjk->btij", target_distribution, target_distribution
    )
    repeated_mask = batch.get("target_distribution_mask")
    if repeated_mask is None and "target_stability_mask" in batch:
        # Older batches expose only a broad stability flag. A point estimate
        # with num_observations>1 may still be encoded as a one-hot histogram,
        # so use it as a pair probability only when the histogram is actually
        # non-degenerate. A dedicated target_distribution_mask is preferred.
        non_degenerate = (target_distribution > 1e-8).sum(dim=-1) > 1
        repeated_mask = batch["target_stability_mask"].bool() & non_degenerate
    if repeated_mask is None:
        repeated_pair = torch.zeros_like(pair_mask)
    else:
        repeated_mask = repeated_mask.bool() & target_mask
        repeated_pair = repeated_mask.unsqueeze(-1) & repeated_mask.unsqueeze(-2)
    soft_pair_target = torch.where(
        repeated_pair, distribution_pair_target, score_gap_pair_target
    )
    pairwise_element = F.binary_cross_entropy_with_logits(
        predicted_difference / max(float(pairwise_prediction_temperature), 1e-4),
        soft_pair_target.to(predicted_difference.dtype),
        reduction="none",
    )
    pair_reliability = torch.ones_like(pairwise_element)
    if "target_score_reliability" in batch:
        reliability = batch["target_score_reliability"].to(pair_reliability.dtype)
        reliability = reliability.clamp(0.0, 1.0)
        pair_reliability = (
            reliability.unsqueeze(-1) * reliability.unsqueeze(-2)
        ).clamp_min(0.0).sqrt()
    elif "target_num_observations" in batch:
        observations = batch["target_num_observations"].to(pair_reliability.dtype)
        reliability = observations / (
            observations + max(float(pairwise_reliability_prior), 1e-8)
        )
        pair_reliability = (
            reliability.unsqueeze(-1) * reliability.unsqueeze(-2)
        ).clamp_min(0.0).sqrt()
    pairwise_loss = _masked_mean(pairwise_element, pair_mask * pair_reliability)

    valid_target = target_mask.any(dim=-1)
    auxiliary_target = valid_target & calibrated_source.bool()[:, None]
    policy_logits = predicted_mean.masked_fill(~target_mask, -1e4)
    policy = (policy_logits / max(float(routing_temperature), 1e-4)).softmax(dim=-1)
    expected_quality = (policy * target_score * mask).sum(dim=-1)
    oracle_quality = target_score.masked_fill(~target_mask, -1e4).max(dim=-1).values
    quality_regret = (oracle_quality - expected_quality).clamp_min(0.0)
    regret_loss = _masked_mean(quality_regret, auxiliary_target)

    if models >= 2:
        true_top_two = target_score.masked_fill(~target_mask, -1e4).topk(2, dim=-1)
        enough_candidates = target_mask.sum(dim=-1) >= 2
        predicted_at_true_top_two = predicted_mean.gather(-1, true_top_two.indices)
        predicted_gap = predicted_at_true_top_two[..., 0] - predicted_at_true_top_two[..., 1]
        target_gap = true_top_two.values[..., 0] - true_top_two.values[..., 1]
        gap_element = F.smooth_l1_loss(
            predicted_gap,
            target_gap.to(predicted_gap.dtype),
            reduction="none",
            beta=max(float(gap_huber_beta), 1e-8),
        )
        gap_mask = enough_candidates & calibrated_source.bool()[:, None]
        gap_loss = _masked_mean(gap_element, gap_mask)
    else:
        predicted_gap = predicted_mean.new_zeros(predicted_mean.shape[:-1])
        target_gap = predicted_gap
        gap_mask = torch.zeros_like(valid_target)
        gap_loss = predicted_mean.sum() * 0.0

    total = (
        float(distribution_weight) * distribution_loss
        + float(mean_weight) * mean_loss
        + float(pairwise_weight) * pairwise_loss
        + float(regret_weight) * regret_loss
        + float(gap_weight) * gap_loss
    )

    non_tied_pair = pair_mask & (true_difference.abs() > float(pair_tie_tolerance))
    pairwise_correct = (predicted_difference * true_difference > 0).to(target_score.dtype)
    chosen = predicted_mean.masked_fill(~target_mask, -1e4).argmax(dim=-1)
    oracle = target_score.masked_fill(~target_mask, -1e4).argmax(dim=-1)
    route_accuracy = _masked_mean((chosen == oracle).to(target_score.dtype), auxiliary_target)

    target_stability_mask = batch.get("target_stability_mask")
    stability_mae = predicted_mean.new_zeros(())
    supervised_stability_fraction = predicted_mean.new_zeros(())
    if target_stability_mask is not None and "target_stability" in batch:
        stability_mask = target_stability_mask.bool() & target_mask
        stability_mae = _masked_mean(
            (statistics["predictive_spread"] - batch["target_stability"]).abs(),
            stability_mask,
        )
        supervised_stability_fraction = (
            stability_mask.float().sum() / target_mask.float().sum().clamp_min(1.0)
        )

    metrics = {
        "loss": total.detach(),
        "distribution_loss": distribution_loss.detach(),
        "mean_huber_loss": mean_loss.detach(),
        "soft_pairwise_loss": pairwise_loss.detach(),
        "soft_quality_regret_loss": regret_loss.detach(),
        "top2_gap_loss": gap_loss.detach(),
        "mean_absolute_error": _masked_mean(
            (predicted_mean.detach() - target_score).abs(), auxiliary_mask
        ).detach(),
        "pairwise_accuracy": _masked_mean(pairwise_correct, non_tied_pair).detach(),
        "routing_accuracy": route_accuracy.detach(),
        "soft_routed_quality": _masked_mean(expected_quality, auxiliary_target).detach(),
        "mean_target_top2_gap": _masked_mean(target_gap, gap_mask).detach(),
        "mean_predicted_top2_gap": _masked_mean(predicted_gap, gap_mask).detach(),
        "stability_mae": stability_mae.detach(),
        "supervised_stability_fraction": supervised_stability_fraction.detach(),
        "distribution_pair_fraction": _masked_mean(
            repeated_pair.to(target_score.dtype), pair_mask
        ).detach(),
        "mean_pair_reliability": _masked_mean(pair_reliability, pair_mask).detach(),
    }
    return total, metrics


def multi_output_point_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    score_weight: float = 1.0,
    score_pairwise_weight: float = 0.5,
    cost_weight: float = 0.25,
    cost_pairwise_weight: float = 0.1,
    score_huber_beta: float = 0.05,
    cost_huber_beta: float = 0.05,
    pairwise_target_temperature: float = 0.05,
    pairwise_prediction_temperature: float = 0.10,
    ensemble_member_weight: float = 0.0,
    ensemble_bootstrap_keep: float = 1.0,
    ensemble_member_score_only: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise exactly two point estimates: score and episode-relative cost."""
    mask = batch["target_mask"].bool()
    score_target = batch["target_score"].to(output["score_mean"].dtype)
    cost_target = batch["target_cost_normalized"].to(output["cost_mean"].dtype)
    score_prediction = output["score_mean"]
    cost_prediction = output["cost_mean"]

    score_error = F.smooth_l1_loss(
        score_prediction, score_target, reduction="none",
        beta=max(float(score_huber_beta), 1e-8),
    )
    cost_error = F.smooth_l1_loss(
        cost_prediction, cost_target, reduction="none",
        beta=max(float(cost_huber_beta), 1e-8),
    )
    score_loss = _masked_mean(score_error, mask)
    cost_loss = _masked_mean(cost_error, mask)

    models = score_target.shape[-1]
    upper = torch.triu(
        torch.ones(models, models, dtype=torch.bool, device=mask.device), diagonal=1
    )
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2) & upper

    def pairwise_loss(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_difference = target.unsqueeze(-1) - target.unsqueeze(-2)
        prediction_difference = prediction.unsqueeze(-1) - prediction.unsqueeze(-2)
        soft_target = torch.sigmoid(
            target_difference / max(float(pairwise_target_temperature), 1e-4)
        )
        element = F.binary_cross_entropy_with_logits(
            prediction_difference / max(float(pairwise_prediction_temperature), 1e-4),
            soft_target,
            reduction="none",
        )
        loss = _masked_mean(element, pair_mask)
        non_tied = pair_mask & (target_difference.abs() > 1e-6)
        correct = (prediction_difference * target_difference > 0).to(prediction.dtype)
        return loss, _masked_mean(correct, non_tied)

    score_pairwise, score_pairwise_accuracy = pairwise_loss(
        score_prediction, score_target
    )
    cost_pairwise, cost_pairwise_accuracy = pairwise_loss(
        cost_prediction, cost_target
    )
    total = (
        float(score_weight) * score_loss
        + float(score_pairwise_weight) * score_pairwise
        + float(cost_weight) * cost_loss
        + float(cost_pairwise_weight) * cost_pairwise
    )
    member_loss = total.new_zeros(())
    member_disagreement = total.new_zeros(())
    member_score_std = total.new_zeros(())
    score_members = output.get("score_members")
    cost_members = output.get("cost_members")
    if score_members is not None or cost_members is not None:
        if score_members is None or cost_members is None:
            raise ValueError("score_members and cost_members must be provided together")
        if score_members.shape != cost_members.shape:
            raise ValueError("score_members and cost_members must have identical shapes")
        if score_members.shape[:-1] != score_prediction.shape:
            raise ValueError("ensemble member tensors must append one member axis")
        if not 0.0 < float(ensemble_bootstrap_keep) <= 1.0:
            raise ValueError("ensemble_bootstrap_keep must be in (0, 1]")
        losses = []
        valid_target = mask.any(dim=-1)
        for member_index in range(score_members.shape[-1]):
            member_batch = batch
            if float(ensemble_bootstrap_keep) < 1.0:
                keep = (
                    torch.rand(valid_target.shape, device=mask.device)
                    < float(ensemble_bootstrap_keep)
                ) & valid_target
                # Keep at least one valid target per episode so every branch
                # receives a supervised update even for an unlucky bootstrap.
                missing = valid_target.any(dim=-1) & ~keep.any(dim=-1)
                if missing.any():
                    first = valid_target.float().argmax(dim=-1)
                    keep = keep.clone()
                    keep[missing, first[missing]] = True
                member_batch = dict(batch)
                member_batch["target_mask"] = mask & keep.unsqueeze(-1)
            branch_loss, _ = multi_output_point_loss(
                {
                    "score_mean": score_members[..., member_index],
                    "cost_mean": cost_members[..., member_index],
                },
                member_batch,
                score_weight=score_weight,
                score_pairwise_weight=score_pairwise_weight,
                cost_weight=0.0 if ensemble_member_score_only else cost_weight,
                cost_pairwise_weight=(
                    0.0 if ensemble_member_score_only else cost_pairwise_weight
                ),
                score_huber_beta=score_huber_beta,
                cost_huber_beta=cost_huber_beta,
                pairwise_target_temperature=pairwise_target_temperature,
                pairwise_prediction_temperature=pairwise_prediction_temperature,
                ensemble_member_weight=0.0,
                ensemble_bootstrap_keep=1.0,
                ensemble_member_score_only=ensemble_member_score_only,
            )
            losses.append(branch_loss)
        member_loss = torch.stack(losses).mean()
        total = total + float(ensemble_member_weight) * member_loss
        member_choice = score_members.masked_fill(
            ~mask.unsqueeze(-1), -1e4
        ).argmax(dim=2)
        modal_choice = torch.mode(member_choice, dim=-1).values
        member_disagreement = _masked_mean(
            (member_choice != modal_choice.unsqueeze(-1)).to(score_prediction.dtype),
            valid_target.unsqueeze(-1).expand_as(member_choice),
        )
        member_score_std = _masked_mean(
            score_members.std(dim=-1, correction=0), mask
        )
    predicted_choice = score_prediction.masked_fill(~mask, -1e4).argmax(dim=-1)
    target_choice = score_target.masked_fill(~mask, -1e4).argmax(dim=-1)
    valid_target = mask.any(dim=-1)
    metrics = {
        "loss": total.detach(),
        "score_huber_loss": score_loss.detach(),
        "score_pairwise_loss": score_pairwise.detach(),
        "cost_huber_loss": cost_loss.detach(),
        "cost_pairwise_loss": cost_pairwise.detach(),
        "score_mae": _masked_mean((score_prediction - score_target).abs(), mask).detach(),
        "cost_mae": _masked_mean((cost_prediction - cost_target).abs(), mask).detach(),
        "score_pairwise_accuracy": score_pairwise_accuracy.detach(),
        "cost_pairwise_accuracy": cost_pairwise_accuracy.detach(),
        "routing_accuracy": _masked_mean(
            (predicted_choice == target_choice).to(score_prediction.dtype), valid_target
        ).detach(),
    }
    if score_members is not None:
        metrics.update({
            "ensemble_member_loss": member_loss.detach(),
            "ensemble_member_choice_disagreement": member_disagreement.detach(),
            "ensemble_member_score_std": member_score_std.detach(),
        })
    return total, metrics


def deployment_decision(
    output: dict[str, torch.Tensor],
    candidate_mask: torch.Tensor,
    *,
    cost: torch.Tensor | None = None,
    cost_log: torch.Tensor | None = None,
    cost_mask: torch.Tensor | None = None,
    cost_weight: float = 0.0,
    cost_reference: float = 1.0,
    missing_cost: float | None = None,
    stability_risk: torch.Tensor | None = None,
    stability_weight: float = 0.0,
    estimation_uncertainty: torch.Tensor | None = None,
    uncertainty_weight: float = 0.0,
    missing_cost_policy: Literal["reference", "fixed", "error"] = "reference",
) -> dict[str, torch.Tensor]:
    """Apply deployment preferences to a pure quality prediction.

    Costs are absolute and divided by one fixed external reference; there is no
    candidate-pool min/max normalization. Unknown cost defaults to the reference
    cost (a penalty of one), so it is never silently interpreted as free.
    ``stability_risk`` and ``estimation_uncertainty`` are separate quantities
    with independent weights.
    """
    if cost is not None and cost_log is not None:
        raise ValueError("provide either absolute cost or log1p cost, not both")
    if float(cost_reference) <= 0.0:
        raise ValueError("cost_reference must be positive")
    if missing_cost_policy not in {"reference", "fixed", "error"}:
        raise ValueError(f"unknown missing_cost_policy: {missing_cost_policy}")
    if missing_cost_policy == "fixed" and missing_cost is None:
        raise ValueError("missing_cost is required for missing_cost_policy='fixed'")

    statistics = quality_distribution_statistics(output)
    quality = statistics["mean"]
    mask = candidate_mask.bool()
    while mask.ndim < quality.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(quality)

    supplied_cost = cost if cost is not None else cost_log
    if supplied_cost is None:
        supplied_cost = torch.zeros_like(quality)
        available_cost = torch.zeros_like(mask)
    else:
        supplied_cost = supplied_cost.to(device=quality.device, dtype=quality.dtype)
        supplied_cost = torch.broadcast_to(supplied_cost, quality.shape)
        if cost_log is not None:
            supplied_cost = torch.expm1(supplied_cost)
        available_cost = torch.isfinite(supplied_cost) & (supplied_cost >= 0.0) & mask
        if cost_mask is not None:
            available_cost = available_cost & torch.broadcast_to(
                cost_mask.bool().to(quality.device), quality.shape
            )

    missing = mask & ~available_cost
    if missing_cost_policy == "error" and bool(missing.any()):
        raise ValueError("candidate cost is missing under missing_cost_policy='error'")
    fallback = (
        float(cost_reference)
        if missing_cost_policy == "reference"
        else float(missing_cost if missing_cost is not None else cost_reference)
    )
    if fallback < 0.0:
        raise ValueError("missing_cost must be non-negative")
    effective_cost = torch.where(
        available_cost, supplied_cost, quality.new_full((), fallback)
    )
    normalized_cost = effective_cost / float(cost_reference)

    if stability_risk is None:
        # Predictive spread is not automatically a stability estimate because
        # point-label one-hot supervision cannot identify outcome variance.
        stability = torch.zeros_like(quality)
    else:
        stability = torch.broadcast_to(
            stability_risk.to(device=quality.device, dtype=quality.dtype), quality.shape
        ).clamp_min(0.0)
    if estimation_uncertainty is None:
        estimation = torch.zeros_like(quality)
    else:
        estimation = torch.broadcast_to(
            estimation_uncertainty.to(device=quality.device, dtype=quality.dtype),
            quality.shape,
        ).clamp_min(0.0)

    decision_score = (
        quality
        - float(cost_weight) * normalized_cost
        - float(stability_weight) * stability
        - float(uncertainty_weight) * estimation
    ).masked_fill(~mask, float("-inf"))
    return {
        "quality_mean": quality,
        "normalized_absolute_cost": normalized_cost,
        "cost_available": available_cost,
        "predictive_spread": statistics["predictive_spread"],
        "stability_risk": stability,
        "estimation_uncertainty": estimation,
        "decision_score": decision_score,
        "routing_probability": decision_score.softmax(dim=-1),
        "routing_choice": decision_score.argmax(dim=-1),
    }
