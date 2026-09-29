"""Prefetch and verify the published RouteFM checkpoints."""
from __future__ import annotations

import argparse
import json

from routefm.checkpoints import HF_REPO_ID, HF_REVISION, resolve_checkpoint, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("qwen", "bge", "all"), default="all")
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory")
    args = parser.parse_args()
    encoders = ("qwen", "bge") if args.encoder == "all" else (args.encoder,)
    artifacts = {}
    for encoder in encoders:
        resolved = resolve_checkpoint(encoder, cache_dir=args.cache_dir)
        artifacts[encoder] = {
            "weights": str(resolved.weights),
            "weights_sha256": sha256_file(resolved.weights),
            "config": str(resolved.config),
            "config_sha256": sha256_file(resolved.config),
        }
    print(json.dumps({
        "repo_id": HF_REPO_ID,
        "revision": HF_REVISION,
        "artifacts": artifacts,
    }, indent=2))


if __name__ == "__main__":
    main()
