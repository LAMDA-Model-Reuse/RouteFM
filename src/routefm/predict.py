"""Route a custom model pool using observed Context and frozen RouteFM weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from routefm.checkpoints import (
    ResolvedCheckpoint,
    load_safetensors_config,
    resolve_checkpoint,
)
from routefm.models import RouteFM, RouteFMConfig


def load_router(
    checkpoint: str | Path | ResolvedCheckpoint,
    device: str = "cpu",
    expected_encoder: str | None = None,
) -> RouteFM:
    resolved = checkpoint if isinstance(checkpoint, ResolvedCheckpoint) else resolve_checkpoint(
        expected_encoder or "qwen", checkpoint
    )
    if resolved.weights.suffix == ".safetensors":
        from safetensors.torch import load_file

        saved = load_safetensors_config(resolved)
        if saved.get("format") != "routefm_safetensors":
            raise ValueError("unsupported RouteFM safetensors configuration")
        state = load_file(resolved.weights, device="cpu")
    else:
        saved = torch.load(resolved.weights, map_location="cpu", weights_only=True)
        if saved.get("format") != "routefm_unified_scratch":
            raise ValueError("unsupported RouteFM checkpoint")
        state = saved["model"]
    if expected_encoder is not None and saved.get("encoder") != expected_encoder:
        raise ValueError("checkpoint encoder does not match --encoder")
    model = RouteFM(RouteFMConfig(**saved["model_config"]))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def prepare_episode(input_path: str | Path, dimension: int, device: str) -> dict:
    with np.load(input_path, allow_pickle=False) as data:
        required = {"context_embeddings", "context_scores", "context_costs", "target_embeddings"}
        if not required.issubset(data.files):
            raise ValueError(f"missing arrays: {sorted(required - set(data.files))}")
        context = np.asarray(data["context_embeddings"], dtype=np.float32)
        scores = np.asarray(data["context_scores"], dtype=np.float32)
        costs = np.asarray(data["context_costs"], dtype=np.float32)
        targets = np.asarray(data["target_embeddings"], dtype=np.float32)
        mask = (np.asarray(data["context_mask"], dtype=bool)
                if "context_mask" in data.files else np.ones(scores.shape, dtype=bool))
    if context.ndim != 3 or context.shape[-1] != dimension:
        raise ValueError(f"context_embeddings must be [M,K,{dimension}]")
    models, width, _ = context.shape
    if models < 2 or width < 1 or scores.shape != (models, width):
        raise ValueError("context_scores must be [M,K] with M>=2 and K>=1")
    if costs.shape != scores.shape or mask.shape != scores.shape:
        raise ValueError("context_costs and context_mask must match [M,K]")
    if targets.ndim != 2 or targets.shape[1] != dimension or len(targets) < 1:
        raise ValueError(f"target_embeddings must be [T,{dimension}] with T>=1")
    if not np.all(mask.any(axis=1)):
        raise ValueError("every candidate needs at least one valid Context observation")
    if not np.isfinite(context[mask]).all() or not np.isfinite(targets).all():
        raise ValueError("embeddings contain non-finite values")
    if not np.isfinite(scores[mask]).all() or ((scores[mask] < 0) | (scores[mask] > 1)).any():
        raise ValueError("valid Context scores must be finite and in [0,1]")
    if not np.isfinite(costs[mask]).all() or (costs[mask] < 0).any():
        raise ValueError("valid Context costs must be finite and nonnegative")
    log_cost = np.log1p(np.where(mask, costs, 0.0))
    low, high = float(log_cost[mask].min()), float(log_cost[mask].max())
    normalized = (log_cost - low) / max(high - low, 1e-8)
    features = np.stack((np.where(mask, scores, 0.0),
                         np.where(mask, normalized, 0.0)), axis=-1).astype(np.float32)
    context = np.where(mask[..., None], context, 0.0)
    return {
        "context_query": torch.from_numpy(context[None]).to(device),
        "context_features": torch.from_numpy(features[None]).to(device),
        "context_mask": torch.from_numpy(mask[None]).to(device),
        "candidate_mask": torch.ones((1, models), dtype=torch.bool, device=device),
        "target_query": torch.from_numpy(targets[None]).to(device),
        "target_mask": torch.ones((1, len(targets), models), dtype=torch.bool, device=device),
    }


@torch.inference_mode()
def predict(checkpoint: str | Path | ResolvedCheckpoint, input_path: str | Path,
            device: str = "cpu", target_batch_size: int = 128,
            model_names: list[str] | None = None,
            expected_encoder: str | None = None) -> dict:
    model = load_router(checkpoint, device, expected_encoder)
    episode = prepare_episode(input_path, model.config.query_dim, device)
    candidates = episode["candidate_mask"].shape[1]
    if model_names is not None and (len(model_names) != candidates or len(set(model_names)) != candidates):
        raise ValueError("model_names must list each candidate exactly once in Context row order")
    if target_batch_size < 1:
        raise ValueError("target-batch-size must be positive")
    score_parts, cost_parts = [], []
    targets = episode["target_query"].shape[1]
    for offset in range(0, targets, target_batch_size):
        local = dict(episode)
        for key in ("target_query", "target_mask"):
            local[key] = episode[key][:, offset:offset + target_batch_size]
        output = model(local)
        score_parts.append(output["score_mean"][0].float().cpu())
        cost_parts.append(output["cost_mean"][0].float().cpu())
    score = torch.cat(score_parts)
    cost = torch.cat(cost_parts)
    choice = score.argmax(-1).tolist()
    result = {
        "chosen_model_index": choice,
        "predicted_score": score.tolist(),
        "predicted_relative_cost": cost.tolist(),
        "decision_rule": "highest predicted score; cost is not used for selection",
    }
    if model_names is not None:
        result["candidate_model_names"] = model_names
        result["chosen_model_name"] = [model_names[index] for index in choice]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("qwen", "bge"), required=True)
    parser.add_argument("--checkpoint", help="Defaults to the released weight for --encoder")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-names", help="Optional JSON array, in Context row order")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-batch-size", type=int, default=128)
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(args.encoder, args.checkpoint)
    names = json.loads(Path(args.model_names).read_text()) if args.model_names else None
    result = predict(
        checkpoint, args.input, args.device, args.target_batch_size, names, args.encoder
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
