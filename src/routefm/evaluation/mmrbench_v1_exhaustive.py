"""Strict exhaustive evaluation on the original 10,370-row MMR-Bench.

Every released query is a target exactly once for each context-size setting.
Within each benchmark, deterministic folds are formed and a target fold may
draw Context only from its complement.  RouteFM remains frozen throughout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from routefm.models import RouteFM, RouteFMConfig


METHODS = (
    "routefm_quality_policy",
    "routefm_cost_utility_policy",
    "context_mean_quality_policy",
    "context_mean_cost_utility_policy",
    "one_nn_quality_policy",
    "one_nn_cost_utility_policy",
    "oracle_quality",
    "oracle_cost_utility",
)


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "::".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def _folds(dataset: np.ndarray, folds: int, seed: int) -> np.ndarray:
    assignment = np.full(len(dataset), -1, dtype=np.int64)
    for name in sorted(set(dataset.tolist())):
        indices = np.flatnonzero(dataset == name)
        generator = np.random.default_rng(_stable_seed(seed, name, "folds"))
        shuffled = generator.permutation(indices)
        assignment[shuffled] = np.arange(len(indices)) % folds
    if (assignment < 0).any():
        raise RuntimeError("fold assignment is incomplete")
    return assignment


def _posterior_uncertainty(score: torch.Tensor) -> torch.Tensor:
    observations = 1.0
    alpha = 1.0 + score * observations
    beta = 1.0 + (1.0 - score) * observations
    total = alpha + beta
    return (alpha * beta / (total.square() * (total + 1.0))).clamp_min(0).sqrt()


def _normalized_context_log_cost(cost: torch.Tensor) -> torch.Tensor:
    """Match the final protocol's single min/max over every observed episode context cell."""
    value = torch.log1p(cost.clamp_min(0.0))
    low = value.amin()
    high = value.amax()
    return (value - low) / (high - low).clamp_min(1e-8)


def _build_batch(
    embeddings: torch.Tensor,
    score: torch.Tensor,
    cost: torch.Tensor,
    context_indices: np.ndarray,
    target_indices: np.ndarray,
    model_order: np.ndarray,
    device: torch.device,
    include_context_cost: bool = True,
    observation_features: int = 7,
) -> dict[str, torch.Tensor]:
    context_index = torch.as_tensor(context_indices, dtype=torch.long)
    target_index = torch.as_tensor(target_indices, dtype=torch.long)
    order = torch.as_tensor(model_order, dtype=torch.long)
    context_embedding = embeddings[context_index].to(device)
    target_embedding = embeddings[target_index].to(device)
    context_score = score[context_index][:, order].T.to(device)
    context_cost = cost[context_index][:, order].T.to(device)
    target_score = score[target_index][:, order].to(device)
    target_cost = cost[target_index][:, order].to(device)
    models, context = context_score.shape
    targets = target_score.shape[0]

    context_query = context_embedding[None, None].expand(1, models, -1, -1).clone()
    context_mask = torch.ones(1, models, context, dtype=torch.bool, device=device)
    if observation_features not in {2, 7}:
        raise ValueError("MMR-Bench supports 2-D score/cost or extended 7-D observations")
    context_features = torch.zeros(
        1, models, context, observation_features, device=device
    )
    context_features[0, ..., 0] = context_score
    use_cost = include_context_cost or observation_features == 2
    context_log_cost = torch.log1p(context_cost.clamp_min(0.0))
    cost_low = context_log_cost.amin()
    cost_high = context_log_cost.amax()
    cost_span = (cost_high - cost_low).clamp_min(1e-8)
    if use_cost:
        context_features[0, ..., 1] = (context_log_cost - cost_low) / cost_span
    if observation_features == 7:
        context_features[0, ..., 2] = _posterior_uncertainty(context_score)
        context_features[0, ..., 3] = math.log(2.0)
        context_features[0, ..., 4] = float(include_context_cost)
        context_features[0, ..., 5] = 0.0
        context_features[0, ..., 6] = 1.0
    target_log_cost = torch.log1p(target_cost.clamp_min(0.0))
    target_cost_normalized = ((target_log_cost - cost_low) / cost_span).clamp(0.0, 1.0)
    return {
        "context_query": context_query,
        "context_features": context_features,
        "context_mask": context_mask,
        "candidate_mask": torch.ones(1, models, dtype=torch.bool, device=device),
        "target_query": target_embedding[None],
        "target_score": target_score[None],
        "target_cost_log": torch.log1p(target_cost.clamp_min(0.0))[None],
        "target_cost_normalized": target_cost_normalized[None],
        "target_cost_mask": torch.ones(1, targets, models, dtype=torch.bool, device=device),
        "target_uncertainty": _posterior_uncertainty(target_score)[None],
        "target_mask": torch.ones(1, targets, models, dtype=torch.bool, device=device),
    }


def _chosen(value: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
    return value.gather(-1, choice.unsqueeze(-1)).squeeze(-1)


def _bootstrap(values: list[float], seed: int, samples: int = 10_000) -> dict:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = array[generator.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "unit": "dataset_fold_mean",
        "clusters": len(array),
        "mean_delta": float(array.mean()),
        "ci95": {"low": float(low), "high": float(high)},
    }


def _summarize(values: dict[str, dict[str, list[np.ndarray]]]) -> dict:
    result = {}
    for method, metrics in values.items():
        merged = {name: np.concatenate(rows) for name, rows in metrics.items()}
        result[method] = {
            "targets": int(len(merged["quality"])),
            **{name: float(value.mean()) for name, value in merged.items()},
        }
    return result


def _point_metric_state(calibration_bins: int = 10) -> dict:
    return {
        "count": 0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "signed_error": 0.0,
        "prediction_sum": 0.0,
        "target_sum": 0.0,
        "prediction_square_sum": 0.0,
        "target_square_sum": 0.0,
        "cross_sum": 0.0,
        "pairwise_correct": 0,
        "pairs": 0,
        "calibration_count": np.zeros(calibration_bins, dtype=np.int64),
        "calibration_prediction_sum": np.zeros(calibration_bins, dtype=np.float64),
        "calibration_target_sum": np.zeros(calibration_bins, dtype=np.float64),
    }


def _update_point_metric(
    state: dict, prediction: torch.Tensor, target: torch.Tensor
) -> None:
    prediction = prediction.float()
    target = target.float()
    error = prediction - target
    count = int(target.numel())
    state["count"] += count
    state["absolute_error"] += float(error.abs().sum().item())
    state["squared_error"] += float(error.square().sum().item())
    state["signed_error"] += float(error.sum().item())
    state["prediction_sum"] += float(prediction.sum().item())
    state["target_sum"] += float(target.sum().item())
    state["prediction_square_sum"] += float(prediction.square().sum().item())
    state["target_square_sum"] += float(target.square().sum().item())
    state["cross_sum"] += float((prediction * target).sum().item())

    truth_difference = target.unsqueeze(-1) - target.unsqueeze(-2)
    prediction_difference = prediction.unsqueeze(-1) - prediction.unsqueeze(-2)
    models = target.shape[-1]
    upper = torch.triu(
        torch.ones(models, models, dtype=torch.bool, device=target.device), diagonal=1
    )
    comparable = upper & (truth_difference.abs() > 1e-6)
    state["pairwise_correct"] += int(
        ((truth_difference * prediction_difference > 0) & comparable).sum().item()
    )
    state["pairs"] += int(comparable.sum().item())

    bins = len(state["calibration_count"])
    bin_index = (prediction.clamp(0.0, 1.0) * bins).long().clamp_max(bins - 1)
    for index in range(bins):
        selected = bin_index == index
        selected_count = int(selected.sum().item())
        if selected_count:
            state["calibration_count"][index] += selected_count
            state["calibration_prediction_sum"][index] += float(
                prediction.masked_select(selected).sum().item()
            )
            state["calibration_target_sum"][index] += float(
                target.masked_select(selected).sum().item()
            )


def _finish_point_metric(state: dict) -> dict:
    count = max(1, int(state["count"]))
    covariance = state["cross_sum"] - (
        state["prediction_sum"] * state["target_sum"] / count
    )
    prediction_variance = state["prediction_square_sum"] - (
        state["prediction_sum"] ** 2 / count
    )
    target_variance = state["target_square_sum"] - state["target_sum"] ** 2 / count
    correlation = covariance / math.sqrt(
        max(prediction_variance * target_variance, 1e-12)
    )
    calibration = []
    ece = 0.0
    for index, bin_count in enumerate(state["calibration_count"].tolist()):
        if not bin_count:
            continue
        prediction_mean = state["calibration_prediction_sum"][index] / bin_count
        target_mean = state["calibration_target_sum"][index] / bin_count
        ece += (bin_count / count) * abs(prediction_mean - target_mean)
        calibration.append({
            "bin": index,
            "count": int(bin_count),
            "prediction_mean": prediction_mean,
            "target_mean": target_mean,
        })
    pairs = int(state["pairs"])
    return {
        "target_model_predictions": int(state["count"]),
        "mae": state["absolute_error"] / count,
        "rmse": math.sqrt(state["squared_error"] / count),
        "bias": state["signed_error"] / count,
        "pearson_correlation": correlation,
        "non_tied_pairs": pairs,
        "pairwise_accuracy": state["pairwise_correct"] / pairs if pairs else 0.0,
        "calibration_ece_10bin": ece,
        "calibration": calibration,
    }


@torch.inference_mode()
def evaluate_mmrbench_v1_exhaustive(
    checkpoint_path: str,
    data_root: str,
    context_sizes: list[int],
    folds: int = 5,
    target_batch_size: int = 256,
    seed: int = 31_001,
    device_name: str = "auto",
    output_path: str | None = None,
    context_cost_mode: str = "auto",
) -> dict:
    if folds < 2 or target_batch_size < 1:
        raise ValueError("folds must be >=2 and target_batch_size must be positive")
    context_sizes = sorted(set(context_sizes))
    if not context_sizes or min(context_sizes) < 1:
        raise ValueError("context_sizes must be positive")
    if context_cost_mode not in {"auto", "available", "masked"}:
        raise ValueError("context_cost_mode must be auto, available, or masked")
    root = Path(data_root)
    import pandas as pd

    frame = pd.read_pickle(root / "test.pkl")
    embeddings_np = np.load(root / "test_embeddings.npy", mmap_mode="r")
    models = json.loads((root / "models.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_query_dim = int(saved["model_config"]["query_dim"])
    if (
        len(frame) != 10_370
        or embeddings_np.shape != (10_370, checkpoint_query_dim)
        or len(models) != 9
    ):
        raise ValueError(
            "expected the strict 10,370-row, 9-model V1 artifact with "
            f"checkpoint-compatible {checkpoint_query_dim}-D embeddings"
        )
    performance_columns = [f"model_{index}_performance" for index in range(len(models))]
    cost_columns = [f"model_{index}_cost" for index in range(len(models))]
    score_np = frame[performance_columns].to_numpy(np.float32)
    cost_np = frame[cost_columns].to_numpy(np.float32)
    if not np.isfinite(score_np).all() or not np.isfinite(cost_np).all():
        raise ValueError("strict primary artifact must not contain missing model cells")

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = RouteFM(RouteFMConfig(**saved["model_config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    if model.config.query_dim != embeddings_np.shape[1]:
        raise ValueError("checkpoint query dimension does not match V1 embeddings")

    # Copy the read-only memmap once so PyTorch never receives a non-writable
    # ndarray (the evaluator itself remains immutable with respect to the artifact).
    embeddings = torch.from_numpy(np.array(embeddings_np, copy=True))
    score = torch.from_numpy(score_np)
    cost = torch.from_numpy(cost_np)
    dataset = frame["eval_name"].astype(str).to_numpy()
    fold = _folds(dataset, folds, seed)
    curve = {}
    cost_weight = float(model.config.utility_cost_weight)
    uncertainty_weight = float(model.config.utility_uncertainty_weight)
    loss_mode = str(saved.get("config", {}).get("training", {}).get("loss_mode", ""))
    include_context_cost = (
        loss_mode != "deployment_agnostic"
        if context_cost_mode == "auto"
        else context_cost_mode == "available"
    )
    max_context = max(context_sizes)

    for context_size in context_sizes:
        aggregate = {
            method: defaultdict(list) for method in METHODS
        }
        by_dataset = {
            name: {method: defaultdict(list) for method in METHODS}
            for name in sorted(set(dataset.tolist()))
        }
        target_counts = np.zeros(len(frame), dtype=np.int8)
        cluster_delta = defaultdict(list)
        cluster_metrics = defaultdict(list)
        cluster_ids = []
        score_point_metric = _point_metric_state()
        cost_point_metrics = {
            "routefm": _point_metric_state(),
            "context_mean": _point_metric_state(),
            "one_nn": _point_metric_state(),
        }
        for dataset_name in sorted(set(dataset.tolist())):
            dataset_indices = np.flatnonzero(dataset == dataset_name)
            for fold_id in range(folds):
                target_all = dataset_indices[fold[dataset_indices] == fold_id]
                complement = dataset_indices[fold[dataset_indices] != fold_id]
                if len(complement) < max_context:
                    raise ValueError(f"{dataset_name} fold {fold_id} has only {len(complement)} context candidates")
                generator = np.random.default_rng(_stable_seed(seed, dataset_name, fold_id, "context"))
                context_max = generator.choice(complement, size=max_context, replace=False)
                context_index = context_max[:context_size]
                model_order = np.random.default_rng(
                    _stable_seed(seed, dataset_name, fold_id, "models")
                ).permutation(len(models))
                fold_values = defaultdict(list)
                fold_method_metrics = defaultdict(list)
                for start in range(0, len(target_all), target_batch_size):
                    target_index = target_all[start : start + target_batch_size]
                    target_counts[target_index] += 1
                    batch = _build_batch(
                        embeddings, score, cost, context_index, target_index,
                        model_order, device, include_context_cost,
                        observation_features=model.config.observation_features,
                    )
                    output = model.predict(batch)
                    quality_prediction = output["quality_mean"][0]
                    target_score = batch["target_score"][0]
                    _update_point_metric(
                        score_point_metric, quality_prediction, target_score
                    )
                    if "cost_mean" in output:
                        predicted_cost = output["cost_mean"][0]
                        target_relative_cost = batch["target_cost_normalized"][0]
                        _update_point_metric(
                            cost_point_metrics["routefm"],
                            predicted_cost,
                            target_relative_cost,
                        )
                    route_quality = quality_prediction.argmax(dim=-1)
                    utility_prediction = output.get("routing_utility")
                    route_utility = (
                        utility_prediction[0] if utility_prediction is not None
                        else quality_prediction
                    ).argmax(dim=-1)
                    normalized_cost = model._normalized_target_cost(batch)[0]
                    realized_utility = (
                        target_score - cost_weight * normalized_cost
                        - uncertainty_weight * batch["target_uncertainty"][0]
                    )
                    context_mean = batch["context_features"][0, ..., 0].mean(dim=-1)
                    cm_quality = context_mean.argmax().expand(len(target_index))
                    cm_utility_prediction = context_mean[None] - cost_weight * normalized_cost
                    cm_utility = cm_utility_prediction.argmax(dim=-1)
                    context_query = F.normalize(batch["context_query"][0, 0], dim=-1)
                    target_query = F.normalize(batch["target_query"][0], dim=-1)
                    nearest = (target_query @ context_query.T).argmax(dim=-1)
                    nearest_score = batch["context_features"][0, :, nearest, 0].T
                    if "cost_mean" in output:
                        context_mean_cost = batch["context_features"][0, ..., 1].mean(
                            dim=-1
                        )[None].expand_as(target_relative_cost)
                        nearest_cost = batch["context_features"][0, :, nearest, 1].T
                        _update_point_metric(
                            cost_point_metrics["context_mean"],
                            context_mean_cost,
                            target_relative_cost,
                        )
                        _update_point_metric(
                            cost_point_metrics["one_nn"],
                            nearest_cost,
                            target_relative_cost,
                        )
                    nn_quality = nearest_score.argmax(dim=-1)
                    nn_utility = (nearest_score - cost_weight * normalized_cost).argmax(dim=-1)
                    oracle_quality = target_score.argmax(dim=-1)
                    oracle_utility = realized_utility.argmax(dim=-1)
                    choices = {
                        "routefm_quality_policy": route_quality,
                        "routefm_cost_utility_policy": route_utility,
                        "context_mean_quality_policy": cm_quality,
                        "context_mean_cost_utility_policy": cm_utility,
                        "one_nn_quality_policy": nn_quality,
                        "one_nn_cost_utility_policy": nn_utility,
                        "oracle_quality": oracle_quality,
                        "oracle_cost_utility": oracle_utility,
                    }
                    oracle_quality_value = target_score.max(dim=-1).values
                    oracle_utility_value = realized_utility.max(dim=-1).values
                    batch_metrics = {}
                    for method, choice in choices.items():
                        quality = _chosen(target_score, choice)
                        utility = _chosen(realized_utility, choice)
                        metrics = {
                            "quality": quality,
                            "utility": utility,
                            "quality_regret": oracle_quality_value - quality,
                            "utility_regret": oracle_utility_value - utility,
                            "oracle_quality_hit": (quality >= oracle_quality_value - 1e-7).float(),
                        }
                        batch_metrics[method] = metrics
                        for metric, value in metrics.items():
                            array = value.float().cpu().numpy()
                            aggregate[method][metric].append(array)
                            by_dataset[dataset_name][method][metric].append(array)
                            fold_method_metrics[f"{method}::{metric}"].append(
                                value.float().cpu()
                            )
                    for metric in ("quality", "utility"):
                        rp = batch_metrics["routefm_quality_policy"][metric]
                        fold_values[f"routefm_minus_context_mean_{metric}"].append(
                            (rp - batch_metrics["context_mean_quality_policy"][metric]).cpu()
                        )
                        fold_values[f"routefm_minus_one_nn_{metric}"].append(
                            (rp - batch_metrics["one_nn_quality_policy"][metric]).cpu()
                        )
                for name, parts in fold_values.items():
                    cluster_delta[name].append(float(torch.cat(parts).mean().item()))
                cluster_ids.append(f"{dataset_name}::fold{fold_id}")
                for name, parts in fold_method_metrics.items():
                    cluster_metrics[name].append(float(torch.cat(parts).mean().item()))
        if not np.all(target_counts == 1):
            raise RuntimeError(
                f"exhaustive target invariant failed: min={target_counts.min()} max={target_counts.max()}"
            )
        curve[str(context_size)] = {
            "metrics": _summarize(aggregate),
            "by_dataset": {
                name: _summarize(methods) for name, methods in by_dataset.items()
            },
            "paired_cluster_bootstrap": {
                name: _bootstrap(values, seed + context_size + offset * 10_000)
                for offset, (name, values) in enumerate(sorted(cluster_delta.items()))
            },
            "raw_cluster_metrics": {
                "cluster_ids": cluster_ids,
                "values": dict(sorted(cluster_metrics.items())),
            },
            "audit": {
                "targets": int(target_counts.sum()),
                "unique_targets": int((target_counts > 0).sum()),
                "minimum_target_uses": int(target_counts.min()),
                "maximum_target_uses": int(target_counts.max()),
                "context_target_overlap": 0,
                "context_per_model": context_size,
                "models": len(models),
                "fold_clusters": int(len(set(dataset.tolist())) * folds),
            },
            "score_point_prediction": _finish_point_metric(score_point_metric),
        }
        if cost_point_metrics["routefm"]["count"]:
            cost_metrics = {
                name: _finish_point_metric(state)
                for name, state in cost_point_metrics.items()
            }
            route_cost = cost_metrics["routefm"]
            curve[str(context_size)]["cost_prediction"] = {
                # Preserve the original compact fields for existing consumers.
                "target_model_predictions": route_cost["target_model_predictions"],
                "normalized_cost_mae": route_cost["mae"],
                "non_tied_pairs": route_cost["non_tied_pairs"],
                "pairwise_accuracy": route_cost["pairwise_accuracy"],
                "predictors": cost_metrics,
            }
        print(json.dumps({
            "context_size": context_size,
            "routefm_quality": curve[str(context_size)]["metrics"]["routefm_quality_policy"]["quality"],
            "context_mean_quality": curve[str(context_size)]["metrics"]["context_mean_quality_policy"]["quality"],
            "one_nn_quality": curve[str(context_size)]["metrics"]["one_nn_quality_policy"]["quality"],
        }), flush=True)

    result = {
        "benchmark": "MMR-Bench V1 exhaustive 9-full-coverage-model text-image transfer",
        "checkpoint": checkpoint_path,
        "checkpoint_step": int(saved.get("step", -1)),
        "data_root": data_root,
        "strict_holdout": {
            "checkpoint_updates": 0,
            "mmrbench_used_for_training": False,
            "mmrbench_used_for_checkpoint_selection": False,
            "model_identity_features_used": False,
        },
        "protocol": {
            "rows": len(frame),
            "datasets": sorted(set(dataset.tolist())),
            "models": models,
            "folds_per_dataset": folds,
            "target_rule": "every query exactly once; deterministic fold",
            "context_rule": "uniform without replacement from target-fold complement",
            "nested_context_sizes": context_sizes,
            "model_order": "deterministically anonymized independently per dataset-fold",
            "primary_policy": "quality-only",
            "cost_utility": {
                "cost_weight": cost_weight,
                "uncertainty_weight": uncertainty_weight,
                "cost_normalization": "per-query candidate min-max of log1p released cost",
            },
            "point_cost_target_normalization": (
                "episode Context-wide min-max of log1p released cost; the same "
                "scale is applied to Target costs and clipped to [0,1]"
            ),
            "embedding_input": metadata["embedding_input"],
            "excluded_from_embedding": metadata["excluded_from_embedding"],
            "embedding_adapter": metadata.get("embedding_adapter"),
            "checkpoint_training_loss_mode": loss_mode or None,
            "context_cost_mode": context_cost_mode,
            "context_cost_features": (
                "normalized released cost as feature 2; no availability bit"
                if model.config.observation_features == 2 else
                "normalized released cost + available bit"
                if include_context_cost else
                "masked to zero by the checkpoint's evaluation contract"
            ),
        },
        "seed": seed,
        "curve": curve,
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": output_path, "checkpoint_step": result["checkpoint_step"]}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustive frozen RouteFM evaluation on MMR-Bench V1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=31_001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument(
        "--context-cost-mode", choices=("auto", "available", "masked"),
        default="auto",
    )
    args = parser.parse_args()
    evaluate_mmrbench_v1_exhaustive(
        args.checkpoint, args.data_root, args.context_sizes, args.folds,
        args.target_batch_size, args.seed, args.device, args.output,
        args.context_cost_mode,
    )


if __name__ == "__main__":
    main()
