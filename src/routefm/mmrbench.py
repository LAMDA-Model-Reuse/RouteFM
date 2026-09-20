"""Matched MMR-Bench V1 evaluation for Qwen and BGE RouteFM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import pandas as pd
import torch

from routefm.predict import load_router
from routefm.splits import DATASETS, SEEDS, load_canonical, load_split
from routefm.evaluation.mmrbench_v1_exhaustive import _build_batch, _stable_seed


RELEASE = Path(__file__).resolve().parents[2]
TARGET_KEYS = (
    "target_query", "target_score", "target_cost_log", "target_cost_normalized",
    "target_cost_mask", "target_uncertainty", "target_mask",
)


def load_data(root: Path, dimension: int, canonical_path: Path) -> tuple:
    frame = pd.read_pickle(root / "test.pkl")
    embeddings = np.load(root / "test_embeddings.npy", mmap_mode="r", allow_pickle=False)
    model_names = json.loads((root / "models.json").read_text())
    if embeddings.shape != (10370, dimension) or len(model_names) != 9:
        raise ValueError(f"expected 10,370 aligned {dimension}-D embeddings and nine models")
    order = load_canonical(canonical_path, frame, model_names)
    frame = frame.iloc[order].reset_index(drop=True)
    embeddings = np.asarray(embeddings[order], dtype=np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("non-finite query embedding")
    score = frame[[f"model_{i}_performance" for i in range(9)]].to_numpy(np.float32)
    cost = frame[[f"model_{i}_cost" for i in range(9)]].to_numpy(np.float32)
    if not np.isfinite(score).all() or not np.isfinite(cost).all():
        raise ValueError("MMR-Bench requires complete finite score/cost cells")
    datasets = frame["eval_name"].astype(str).to_numpy()
    if set(datasets.tolist()) != set(DATASETS):
        raise ValueError("MMR-Bench V1 dataset scope differs")
    return (frame, datasets, torch.from_numpy(embeddings),
            torch.from_numpy(score.copy()), torch.from_numpy(cost.copy()), model_names)


@torch.inference_mode()
def evaluate_episode(model, data: tuple, context: np.ndarray,
                     targets: np.ndarray, order: np.ndarray,
                     device: str, target_batch_size: int) -> tuple[float, float]:
    _, _, embeddings, scores, costs, _ = data
    if np.intersect1d(context, targets).size:
        raise ValueError("Context/Target overlap")
    batch = _build_batch(embeddings, scores, costs, context, targets, order,
                         torch.device(device), observation_features=2)
    mean_score = batch["context_features"][..., 0].mean(-1)
    selected, baseline = [], []
    for offset in range(0, len(targets), target_batch_size):
        local = dict(batch)
        for key in TARGET_KEYS:
            local[key] = batch[key][:, offset:offset + target_batch_size]
        predicted = model(local)["score_mean"]
        choice = predicted.argmax(-1)
        mean_choice = mean_score[:, None, :].expand_as(predicted).argmax(-1)
        selected.extend(local["target_score"].gather(-1, choice[..., None]).flatten().cpu().tolist())
        baseline.extend(local["target_score"].gather(-1, mean_choice[..., None]).flatten().cpu().tolist())
    return float(np.mean(selected)), float(np.mean(baseline))


def _summarize(records: list[dict], setting: str, expected_targets: int) -> dict:
    rows = [row for row in records if row["setting"] == setting]
    total = sum(row["targets"] for row in rows)
    if total != expected_targets:
        raise ValueError(f"{setting}: expected {expected_targets} Targets; found {total}")
    return {
        "targets": total,
        "quality": sum(row["quality"] * row["targets"] for row in rows) / total,
        "context_mean_quality": sum(
            row["context_mean_quality"] * row["targets"] for row in rows
        ) / total,
    }


def evaluate_small_seed(model, data: tuple, seed: int, sizes: tuple[int, ...],
                        split_path: Path, canonical: dict,
                        device: str, batch_size: int) -> dict:
    frame, _, _, _, _, model_names = data
    split = json.loads(split_path.read_text())
    if int(split["seed"]) != seed:
        raise ValueError(f"split file does not belong to seed {seed}")
    groups = load_split(split_path, canonical, frame)
    records = []
    for dataset in DATASETS:
        context_order = groups[dataset]["train"]
        targets = groups[dataset]["test"]
        model_order = np.random.default_rng(
            _stable_seed(seed, dataset, "40_60_models")
        ).permutation(len(model_names))
        for size in sizes:
            if len(context_order) < size:
                raise ValueError(f"{dataset}: fewer than {size} Context rows")
            quality, baseline = evaluate_episode(
                model, data, context_order[:size], targets, model_order,
                device, batch_size,
            )
            records.append({"dataset": dataset, "setting": f"K{size}",
                            "context": size, "targets": len(targets),
                            "quality": quality, "context_mean_quality": baseline})
    summary = {f"K{size}": _summarize(records, f"K{size}", 6223) for size in sizes}
    return {"seed": seed, "summary": summary, "records": records}


def evaluate_large_seed(model, data: tuple, seed: int, split_path: Path,
                        canonical: dict, device: str, batch_size: int) -> dict:
    frame, _, _, _, _, model_names = data
    split = json.loads(split_path.read_text())
    if int(split["seed"]) != seed:
        raise ValueError(f"split file does not belong to seed {seed}")
    groups = load_split(split_path, canonical, frame)
    records = []
    for dataset in DATASETS:
        context = groups[dataset]["train"]
        targets = groups[dataset]["test"]
        order = np.random.default_rng(
            _stable_seed(seed, dataset, "40_60_models")
        ).permutation(len(model_names))
        quality, baseline = evaluate_episode(model, data, context, targets,
                                             order, device, batch_size)
        records.append({"dataset": dataset, "setting": "split40_60",
                        "context": len(context), "targets": len(targets),
                        "quality": quality, "context_mean_quality": baseline})
    return {"seed": seed, "summary": {
        "split40_60": _summarize(records, "split40_60", 6223),
    }, "records": records}


def _cli(mode: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("qwen", "bge"), required=True)
    parser.add_argument("--checkpoint", help="Defaults to the released weight for --encoder")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits-dir", default=str(RELEASE / "splits"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--context-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--target-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = args.checkpoint or str(RELEASE / f"weights/routefm_{args.encoder}.pt")
    model = load_router(checkpoint, args.device)
    if model.config.query_dim != {"qwen": 4096, "bge": 768}[args.encoder]:
        raise ValueError("checkpoint dimension does not match --encoder")
    splits_dir = Path(args.splits_dir)
    canonical_path = splits_dir / "mmrbench_v1_canonical_ids.json"
    canonical = json.loads(canonical_path.read_text())
    data = load_data(Path(args.data_root), model.config.query_dim, canonical_path)
    if mode == "small":
        sizes = tuple(sorted(set(args.context_sizes)))
        if not sizes or min(sizes) < 1:
            raise ValueError("Context sizes must be positive")
        per_seed = [evaluate_small_seed(
                        model, data, seed, sizes,
                        splits_dir / f"mmrbench_v1_seed{seed}_40_60_ids.json",
                        canonical, args.device, args.target_batch_size)
                    for seed in args.seeds]
        settings = tuple(f"K{size}" for size in sizes)
    else:
        per_seed = [evaluate_large_seed(
            model, data, seed,
            splits_dir / f"mmrbench_v1_seed{seed}_40_60_ids.json",
            canonical, args.device, args.target_batch_size,
        ) for seed in args.seeds]
        settings = ("split40_60",)
    summary = {
        setting: {
            "quality": statistics.mean(row["summary"][setting]["quality"] for row in per_seed),
            "context_mean_quality": statistics.mean(
                row["summary"][setting]["context_mean_quality"] for row in per_seed
            ),
            "targets_per_seed": per_seed[0]["summary"][setting]["targets"],
        }
        for setting in settings
    }
    result = {
        "name": "RouteFM", "encoder": args.encoder,
        "scope": "MMR-Bench V1",
        "protocol": ("nested fixed-K prefixes of published 40:60 Context" if mode == "small"
                     else "published within-dataset 40:60 train/test IDs"),
        "rows": 10370, "models": 9, "seeds": args.seeds,
        "summary": summary, "per_seed": per_seed,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "encoder": args.encoder,
                      "summary": summary}, indent=2), flush=True)


def main_small() -> None:
    _cli("small")


def main_large() -> None:
    _cli("large")


if __name__ == "__main__":
    main_small()
