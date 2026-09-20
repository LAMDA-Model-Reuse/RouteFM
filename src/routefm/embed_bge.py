"""Encode text with BGE-base-en-v1.5 using normalized CLS pooling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True,
                        help="One JSON object per row with a `text` field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install the optional bge dependency") from error
    texts = [str(json.loads(line)["text"]) for line in
             Path(args.input_jsonl).read_text().splitlines() if line.strip()]
    if not texts:
        raise ValueError("input file is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    vectors = []
    with torch.inference_mode():
        for offset in range(0, len(texts), args.batch_size):
            tokens = tokenizer(texts[offset:offset + args.batch_size], padding=True,
                               truncation=True, max_length=512, return_tensors="pt")
            tokens = {key: value.to(args.device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(hidden.float(), dim=-1).cpu())
    values = torch.cat(vectors).numpy().astype(np.float32)
    if values.shape != (len(texts), 768) or not np.isfinite(values).all():
        raise ValueError("BGE encoder must produce finite 768-D vectors")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, values)


if __name__ == "__main__":
    main()
