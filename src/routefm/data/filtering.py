from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).lower().strip())


def query_hash(text: str) -> str:
    return hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()


def underlying_dataset(name: str) -> str:
    """Map source-specific task names onto leakage-control benchmark families."""
    value = str(name).lower().split("::", 1)[0].replace("_router_dataset", "")
    value = value.replace("harness_", "").replace("truthfulqa_mc_0", "truthfulqa")
    value = value.replace("-", "_")
    aliases = (
        (("mmlu_pro",), "mmlu_pro"),
        (("mmlu_", "mmlu"), "mmlu"),
        (("grade_school_math", "gsm8k"), "gsm8k"),
        (("math_500", "math500"), "math"),
        (("mtbench_math",), "mtbench_math"),
        (("arc_challenge", "arc_easy", "arc"), "arc"),
        (("hellaswag",), "hellaswag"),
        (("winogrande",), "winogrande"),
        (("gpqa",), "gpqa"),
        (("truthfulqa",), "truthfulqa"),
        (("ifeval",), "ifeval"),
        (("musr",), "musr"),
        (("bbh",), "bbh"),
        (("drop_800", "drop"), "drop"),
        (("math",), "math"),
    )
    for prefixes, canonical in aliases:
        if any(value == prefix or value.startswith(prefix + "_") for prefix in prefixes):
            return canonical
    return value


@lru_cache(maxsize=32)
def load_hashes(path: str) -> frozenset[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("hashes", payload.get("query_hashes", []))
    if not isinstance(payload, list):
        raise ValueError(f"query hash file must contain a list: {path}")
    return frozenset(str(value) for value in payload)


def stable_model_partition(model_id: str, fraction: float, seed: int) -> str:
    if not 0.0 < fraction < 1.0:
        raise ValueError("model_holdout_fraction must be between 0 and 1")
    # Strip common checkpoint suffixes so a lineage cannot straddle the split.
    lineage = model_id.lower().split("::checkpoint::", 1)[0]
    lineage = re.sub(r"(?:checkpoint|ckpt|step|epoch)[-_ ]?\d+.*$", "", lineage)
    key = f"{seed}\0{lineage}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return "holdout" if bucket < fraction else "fit"


def entry_allows_query(entry: dict, eval_name: str, query_text: str) -> bool:
    family = underlying_dataset(eval_name)
    included = {underlying_dataset(x) for x in entry.get("include_dataset_families", [])}
    excluded = {underlying_dataset(x) for x in entry.get("exclude_dataset_families", [])}
    if included and family not in included:
        return False
    if family in excluded:
        return False
    digest = query_hash(query_text)
    include_path = entry.get("include_query_hashes_path")
    exclude_path = entry.get("exclude_query_hashes_path")
    if include_path and digest not in load_hashes(str(include_path)):
        return False
    if exclude_path and digest in load_hashes(str(exclude_path)):
        return False
    return True


def entry_allows_model(entry: dict, model_id: str) -> bool:
    partition = entry.get("model_partition", "all")
    if partition == "all":
        return True
    if partition not in {"fit", "holdout"}:
        raise ValueError("model_partition must be one of: all, fit, holdout")
    actual = stable_model_partition(
        model_id,
        float(entry.get("model_holdout_fraction", 0.4)),
        int(entry.get("model_partition_seed", 29_003)),
    )
    return actual == partition
