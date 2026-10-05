"""Resolve and validate published RouteFM checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


HF_REPO_ID = "AIGNLAI/RouteFM"
# Immutable Hub commit containing the RouteFM 1.0 artifacts and repository index.
HF_REVISION = "c9992e6bcfd11f0cb3e14033808d0e852fdc2c74"
REPOSITORY_CONFIG = {
    "path": "config.json",
    "sha256": "fcceac6c349d0c1126310ea2e6b1fc6f5c581d3aa5c5c9fbb2a9d0a1c9e2458a",
}

ARTIFACTS = {
    "qwen": {
        "weights": "qwen/model.safetensors",
        "weights_sha256": "881b648ead20565da38a7f6ee9516dd0a45d3747add41c44d300ac24433b9cb8",
        "config": "qwen/config.json",
        "config_sha256": "0a0f63b954fa3e4472fdd8c51abff33bbc6a12bf0ea267eab0a4604ebcaf6231",
    },
    "bge": {
        "weights": "bge/model.safetensors",
        "weights_sha256": "3245698f2f6bd4cb607cc3af8875d63ba04172f33abd25b286d1b2968c679662",
        "config": "bge/config.json",
        "config_sha256": "6cdc9e9d5ccad4ed9e2543e26ea5adf7def5830189a25e3349ae183c5e7535ea",
    },
}


@dataclass(frozen=True)
class ResolvedCheckpoint:
    weights: Path
    config: Path | None = None
    source: str = "local"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"RouteFM artifact checksum mismatch for {path}: "
            f"expected {expected}, found {actual}"
        )


def resolve_checkpoint(
    encoder: str,
    checkpoint: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> ResolvedCheckpoint:
    """Resolve a local override or the immutable published Hub artifact."""
    if encoder not in ARTIFACTS:
        raise ValueError(f"unknown RouteFM encoder: {encoder}")
    if checkpoint is not None:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        config = path.with_name("config.json") if path.suffix == ".safetensors" else None
        if config is not None and not config.is_file():
            raise FileNotFoundError(
                f"safetensors checkpoint requires a sibling config.json: {config}"
            )
        return ResolvedCheckpoint(path, config)

    if HF_REVISION.startswith("__ROUTEFM_"):
        raise RuntimeError("RouteFM Hub release revision has not been finalized")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "The default RouteFM checkpoint is hosted on Hugging Face. "
            "Install the package dependencies or pass --checkpoint explicitly."
        ) from error

    artifact = ARTIFACTS[encoder]
    common = {
        "repo_id": HF_REPO_ID,
        "revision": HF_REVISION,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "library_name": "routefm-router",
    }
    try:
        repository_config = Path(
            hf_hub_download(filename=REPOSITORY_CONFIG["path"], **common)
        )
        weights = Path(hf_hub_download(filename=artifact["weights"], **common))
        config = Path(hf_hub_download(filename=artifact["config"], **common))
    except Exception as error:
        raise RuntimeError(
            f"could not resolve RouteFM {encoder} artifacts and repository index from "
            f"{HF_REPO_ID}@{HF_REVISION}; pass --checkpoint for a local file"
        ) from error
    _validate(repository_config, REPOSITORY_CONFIG["sha256"])
    _validate(weights, artifact["weights_sha256"])
    _validate(config, artifact["config_sha256"])
    return ResolvedCheckpoint(weights, config, f"hf://{HF_REPO_ID}@{HF_REVISION}")


def load_safetensors_config(checkpoint: ResolvedCheckpoint) -> dict:
    if checkpoint.config is None:
        raise ValueError("safetensors checkpoint has no configuration file")
    value = json.loads(checkpoint.config.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("checkpoint configuration root must be an object")
    return value
