"""Regenerate the static Space data with the released RouteFM-BGE model."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from routefm import RouteFMRouter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "static_demo" / "data.json"
TARGETS = {
    "Mixed assistant": [
        "Prove that there are infinitely many prime numbers.",
        "Write a Python function that detects a cycle in a linked list.",
        "Summarize why central banks raise interest rates.",
    ],
    "Reasoning lab": [
        "A fair coin is tossed until two consecutive heads appear. What is the expected number of tosses?",
        "Prove that the square root of 3 is irrational.",
        "A bag has 3 red and 2 blue balls. Draw two without replacement: what is the probability both are red?",
    ],
    "Software team": [
        "Design an idempotent webhook consumer with retries and exactly-once effects.",
        "Diagnose a race condition in a shared in-memory cache.",
        "Write a SQL query that returns the second-highest salary in each department.",
    ],
}


def load_environments() -> dict:
    """Read the literal preset definition without importing the Gradio app."""
    tree = ast.parse((ROOT / "demo" / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "ENVIRONMENTS":
            return ast.literal_eval(node.value)
    raise RuntimeError("ENVIRONMENTS was not found in demo/app.py")


def main() -> None:
    router = RouteFMRouter.from_pretrained(
        encoder="bge", device="cpu", embedding_max_length=256
    )
    output = {
        "metadata": {
            "model": "AIGNLAI/RouteFM",
            "variant": "RouteFM-BGE",
            "generated": "2026-10-08",
            "decision_rule": "highest predicted quality",
        },
        "environments": {},
    }
    for environment, item in load_environments().items():
        generated = {
            "labels": item["labels"],
            "targets": TARGETS[environment],
            "observations": [
                {"query": query, "scores": scores, "costs": costs}
                for query, scores, costs in item["observations"]
            ],
            "results": {},
        }
        for budget in (1, 2, 4, 8):
            context = {
                f"Candidate {chr(65 + index)}": [
                    {"query": query, "score": scores[index], "cost": costs[index]}
                    for query, scores, costs in item["observations"][:budget]
                ]
                for index in range(len(item["labels"]))
            }
            router.set_context(context)
            generated["results"][str(budget)] = {}
            for query in TARGETS[environment]:
                decision = router.route(query)
                generated["results"][str(budget)][query] = {
                    "scores": [round(value, 6) for value in decision.predicted_scores.values()],
                    "costs": [round(value, 6) for value in decision.predicted_relative_costs.values()],
                }
        output["environments"][environment] = generated
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
