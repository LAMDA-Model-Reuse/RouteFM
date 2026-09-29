#!/usr/bin/env python3
"""Build the reproducible Hugging Face artifact directory for RouteFM 1.0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file


ENCODERS = ("qwen", "bge")
HF_REPO_ID = "AIGNLAI/RouteFM"
HF_REVISION = "b364300234794b0e6394453d224ddab7c94e41d0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def resolve_legacy_checkpoint(source: Path, encoder: str) -> Path:
    local = source / "weights" / f"routefm_{encoder}.pt"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"legacy/routefm_{encoder}.pt",
        revision=HF_REVISION,
        library_name="routefm-router",
    ))


def build(source: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("MODEL_CARD.md", "README.md"),
        ("LICENSE", "LICENSE"),
        ("NOTICE", "NOTICE"),
        ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ):
        shutil.copyfile(source / source_name, output / target_name)
    manifest = {
        "name": "RouteFM",
        "model_version": "1.0.0",
        "license": "Apache-2.0",
        "code_repository": "https://github.com/LAMDA-Model-Reuse/RouteFM",
        "source_code_revision": "v1.0.0",
        "artifacts": {},
    }
    for encoder in ENCODERS:
        legacy = resolve_legacy_checkpoint(source, encoder)
        saved = torch.load(legacy, map_location="cpu", weights_only=True)
        if saved.get("format") != "routefm_unified_scratch":
            raise ValueError(f"unsupported checkpoint format: {legacy}")
        if saved.get("encoder") != encoder:
            raise ValueError(f"encoder mismatch: {legacy}")
        state = {key: value.detach().contiguous().cpu() for key, value in saved["model"].items()}
        target = output / encoder
        target.mkdir(parents=True, exist_ok=True)
        weights = target / "model.safetensors"
        save_file(state, weights)
        restored = load_file(weights, device="cpu")
        if restored.keys() != state.keys() or any(
            not torch.equal(restored[key], value) for key, value in state.items()
        ):
            raise RuntimeError(f"safetensors conversion changed tensors for {encoder}")
        config = {
            "format": "routefm_safetensors",
            "model_version": "1.0.0",
            "release_version": saved.get("release_version"),
            "encoder": encoder,
            "step": int(saved["step"]),
            "model_config": saved["model_config"],
            "parameter_count": sum(tensor.numel() for tensor in state.values()),
            "source_checkpoint": f"legacy/routefm_{encoder}.pt",
            "source_checkpoint_sha256": sha256_file(legacy),
        }
        config_path = target / "config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        legacy_target = output / "legacy" / legacy.name
        legacy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy, legacy_target)
        manifest["artifacts"][encoder] = {
            "weights": str(weights.relative_to(output)),
            "weights_sha256": sha256_file(weights),
            "weights_size_bytes": weights.stat().st_size,
            "config": str(config_path.relative_to(output)),
            "config_sha256": sha256_file(config_path),
            "legacy_checkpoint": str(legacy_target.relative_to(output)),
            "legacy_checkpoint_sha256": sha256_file(legacy_target),
            "legacy_checkpoint_size_bytes": legacy_target.stat().st_size,
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source.resolve(), args.output.resolve())
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
