"""Request normalized text/image embeddings from a compatible Qwen service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True,
                        help="One JSON object per row with an `input` field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    key = os.environ.get("ROUTEFM_EMBEDDING_API_KEY")
    if not key:
        raise RuntimeError("set ROUTEFM_EMBEDDING_API_KEY in the environment")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    inputs = [json.loads(line)["input"] for line in
              Path(args.input_jsonl).read_text().splitlines() if line.strip()]
    if not inputs:
        raise ValueError("input file is empty")
    vectors = []
    for offset in range(0, len(inputs), args.batch_size):
        batch = inputs[offset:offset + args.batch_size]
        body = json.dumps({"model": args.model, "input": batch}).encode()
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/embeddings", data=body,
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.load(response)
        rows = sorted(payload["data"], key=lambda row: row["index"])
        if len(rows) != len(batch):
            raise ValueError("embedding response row count does not match request")
        vectors.extend(row["embedding"] for row in rows)
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape != (len(inputs), 4096) or not np.isfinite(values).all():
        raise ValueError("embedding service must return finite 4096-D vectors")
    values /= np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, values)


if __name__ == "__main__":
    main()
