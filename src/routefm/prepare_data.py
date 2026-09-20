"""Validate and package a dense query-by-model score matrix for pretraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="CSV with one query per row")
    parser.add_argument("--embeddings", required=True, help="Aligned float embedding .npy")
    parser.add_argument("--models", required=True, help="JSON array of candidate model names")
    parser.add_argument("--split", choices=("train", "test", "val"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    destination = Path(args.output_dir)
    split_path = destination / f"{args.split}.pkl"
    if split_path.exists():
        raise FileExistsError(f"split already exists: {split_path}")
    frame = pd.read_csv(args.matrix)
    models = json.loads(Path(args.models).read_text())
    embeddings = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if not isinstance(models, list) or len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("models.json must contain at least two unique model names")
    if embeddings.ndim != 2 or len(embeddings) != len(frame) or not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be a finite [rows,dimension] matrix")
    if "sample_id" not in frame or "eval_name" not in frame:
        raise ValueError("matrix requires sample_id and eval_name columns")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique within a split")
    for index in range(len(models)):
        score = f"model_{index}_performance"
        cost = f"model_{index}_cost"
        if score not in frame:
            raise ValueError(f"missing column {score}")
        valid = pd.to_numeric(frame[score], errors="coerce").dropna()
        if ((valid < 0) | (valid > 1)).any():
            raise ValueError(f"{score} must be in [0,1] where present")
        if cost not in frame:
            frame[cost] = np.nan
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "models.json"
    if model_path.exists() and json.loads(model_path.read_text()) != models:
        raise ValueError("existing models.json has a different candidate order")
    model_path.write_text(json.dumps(models, indent=2) + "\n")
    frame.to_pickle(split_path)
    shutil.copyfile(args.embeddings, destination / f"{args.split}_embeddings.npy")


if __name__ == "__main__":
    main()
