"""Data selection helpers shared by pretraining and validation."""
from __future__ import annotations

import copy
from pathlib import Path


def _apply_data_split(config: dict, data_split: str) -> None:
    """Apply source-specific split and query-partition overrides in-place."""
    entries = (
        config["data"]["entries"]
        if config["data"]["type"] == "dense_artifacts"
        else config["data"].get("dense_entries", [])
    )
    for entry in entries:
        entry["split"] = entry.get(f"{data_split}_split", data_split)
        entry["query_partition"] = entry.get(f"{data_split}_query_partition", "all")
    if config["data"]["type"] == "mixed":
        for entry in config["data"].get("jsonl_entries", []):
            entry["split"] = entry.get(f"{data_split}_split", data_split)
        for entry in config["data"].get("synthetic_entries", []):
            entry["enabled"] = False


def _apply_evaluation_suite(config: dict, suite_name: str) -> None:
    """Apply the named validation source and episode settings in-place."""
    suites = config["data"].get("evaluation_suites", {})
    if suite_name not in suites:
        raise ValueError(f"unknown evaluation suite {suite_name!r}; available={sorted(suites)}")
    suite = suites[suite_name]
    enabled_sources = set(suite.get("enabled_sources", []))
    entry_overrides = suite.get("entries", {})
    entries = (
        config["data"].get("entries", [])
        if config["data"]["type"] == "dense_artifacts"
        else config["data"].get("dense_entries", [])
    )
    entries = list(entries) + list(config["data"].get("jsonl_entries", []))
    for entry in entries:
        source = str(entry.get("source", Path(entry.get("path", "unknown")).stem))
        if enabled_sources:
            entry["enabled"] = source in enabled_sources
        for key, value in entry_overrides.get(source, {}).items():
            if value is None:
                entry.pop(key, None)
            else:
                entry[key] = copy.deepcopy(value)
    if "source_weights" in suite:
        config["data"]["source_weights"] = copy.deepcopy(suite["source_weights"])
    for key, value in suite.get("episodes", {}).items():
        config["episodes"][key] = copy.deepcopy(value)
