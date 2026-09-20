"""Publish and verify deterministic MMR-Bench V1 train/test query IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from routefm.evaluation.mmrbench_v1_exhaustive import _stable_seed


DATASETS = ("MMStar", "MathVerse", "MathVision", "MathVista", "OCRBench",
            "RealWorldQA", "SEEDBench2_Plus")
SEEDS = (31010,)


def _id_hash(ids: list[str]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_document(frame: pd.DataFrame, model_names: list[str]) -> dict:
    ids = frame["sample_id"].astype(str).tolist()
    if len(ids) != 10370 or len(set(ids)) != len(ids):
        raise ValueError("expected 10,370 unique sample IDs")
    datasets = frame["eval_name"].astype(str)
    if set(datasets) != set(DATASETS):
        raise ValueError("MMR-Bench V1 dataset scope differs")
    if len(model_names) != 9 or len(set(model_names)) != 9:
        raise ValueError("expected nine unique candidate models")
    return {
        "format": "routefm_mmrbench_v1_canonical_order_v1",
        "rows": len(ids), "datasets": list(DATASETS),
        "model_names": model_names, "ordered_ids_sha256": _id_hash(ids),
        "ordered_query_ids": ids,
    }


def split_document(frame: pd.DataFrame, canonical: dict, seed: int) -> dict:
    ids = frame["sample_id"].astype(str).to_numpy()
    datasets = frame["eval_name"].astype(str).to_numpy()
    groups = {}
    for dataset in DATASETS:
        local = ids[datasets == dataset]
        order = np.random.default_rng(
            _stable_seed(seed, dataset, "40_60_split")
        ).permutation(len(local))
        count = math.floor(len(local) * 0.4)
        train_ids = local[order[:count]].tolist()
        test_ids = local[order[count:]].tolist()
        groups[dataset] = {
            "train_ids": train_ids,
            "test_ids": test_ids,
            "train_count": len(train_ids),
            "test_count": len(test_ids),
        }
    return {
        "format": "routefm_mmrbench_v1_40_60_ids_v1", "seed": seed,
        "train_means": "observed Context; no gradient updates to RouteFM",
        "test_means": "held-out Target queries",
        "canonical_order_sha256": canonical["ordered_ids_sha256"],
        "datasets": groups,
        "train_count": sum(len(row["train_ids"]) for row in groups.values()),
        "test_count": sum(len(row["test_ids"]) for row in groups.values()),
    }


def load_canonical(path: Path, frame: pd.DataFrame, model_names: list[str]) -> np.ndarray:
    value = json.loads(path.read_text())
    if value.get("format") != "routefm_mmrbench_v1_canonical_order_v1":
        raise ValueError("unsupported canonical-ID document")
    if model_names != value["model_names"]:
        raise ValueError("candidate-model order differs from the published artifact")
    current = frame["sample_id"].astype(str).tolist()
    if len(current) != len(set(current)) or set(current) != set(value["ordered_query_ids"]):
        raise ValueError("query-ID set differs from the published artifact")
    index = {sample_id: position for position, sample_id in enumerate(current)}
    order = np.asarray([index[sample_id] for sample_id in value["ordered_query_ids"]], dtype=np.int64)
    if _id_hash(value["ordered_query_ids"]) != value["ordered_ids_sha256"]:
        raise ValueError("canonical query-ID hash mismatch")
    return order


def load_split(path: Path, canonical: dict, frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    value = json.loads(path.read_text())
    if value.get("format") != "routefm_mmrbench_v1_40_60_ids_v1":
        raise ValueError("unsupported split document")
    if value.get("canonical_order_sha256") != canonical["ordered_ids_sha256"]:
        raise ValueError("split does not match canonical query order")
    ids = frame["sample_id"].astype(str).tolist()
    index = {sample_id: position for position, sample_id in enumerate(ids)}
    result = {}
    for dataset in DATASETS:
        row = value["datasets"][dataset]
        train, test = row["train_ids"], row["test_ids"]
        if set(train) & set(test):
            raise ValueError(f"{dataset}: Context/Target overlap")
        expected = set(frame.loc[frame["eval_name"].astype(str) == dataset,
                                 "sample_id"].astype(str))
        if set(train) | set(test) != expected:
            raise ValueError(f"{dataset}: split is not exhaustive")
        result[dataset] = {
            "train": np.asarray([index[item] for item in train], dtype=np.int64),
            "test": np.asarray([index[item] for item in test], dtype=np.int64),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    root, output = Path(args.data_root), Path(args.output_dir)
    frame = pd.read_pickle(root / "test.pkl")
    model_names = json.loads((root / "models.json").read_text())
    canonical = canonical_document(frame, model_names)
    output.mkdir(parents=True, exist_ok=True)
    (output / "mmrbench_v1_canonical_ids.json").write_text(
        json.dumps(canonical, indent=2) + "\n"
    )
    for seed in args.seeds:
        split = split_document(frame, canonical, seed)
        (output / f"mmrbench_v1_seed{seed}_40_60_ids.json").write_text(
            json.dumps(split, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
