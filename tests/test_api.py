from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from routefm import ContextObservation, RouteFMRouter
from routefm.predict import prepare_episode_arrays


class _PredictableModel(torch.nn.Module):
    def __init__(self, dimension: int = 3) -> None:
        super().__init__()
        self.config = SimpleNamespace(query_dim=dimension)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        targets = batch["target_query"].shape[1]
        candidates = batch["candidate_mask"].shape[1]
        scores = torch.arange(
            candidates, dtype=torch.float32, device=batch["target_query"].device
        ).reshape(1, 1, candidates).expand(1, targets, candidates)
        costs = torch.flip(scores, dims=(-1,))
        return {"score_mean": scores, "cost_mean": costs}


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray([
            [len(text), sum(map(ord, text)) % 17, index + 1]
            for index, text in enumerate(texts)
        ], dtype=np.float32)


class RouteFMRouterTest(unittest.TestCase):
    def test_text_context_and_single_route(self) -> None:
        embedder = _RecordingEmbedder()
        router = RouteFMRouter(
            _PredictableModel(), encoder="bge", embedder=embedder
        ).set_context({
            "small": [
                ContextObservation("easy", 1.0, 0.1),
                {"query": "shared", "score": 0.4, "cost": 0.1},
            ],
            "large": [
                {"query": "hard", "score": 0.9, "cost": 1.0},
                {"query": "shared", "score": 1.0, "cost": 1.0},
            ],
        })

        decision = router.route("new query")

        self.assertEqual(router.candidate_names, ("small", "large"))
        self.assertEqual(embedder.calls[0], ["easy", "shared", "hard"])
        self.assertEqual(embedder.calls[1], ["new query"])
        self.assertEqual(decision.model_name, "large")
        self.assertEqual(decision.model_index, 1)
        self.assertEqual(decision.predicted_scores, {"small": 0.0, "large": 1.0})
        self.assertEqual(decision.to_dict()["model_name"], "large")

    def test_route_batch_requires_context(self) -> None:
        router = RouteFMRouter(
            _PredictableModel(), encoder="bge", embedder=_RecordingEmbedder()
        )
        with self.assertRaisesRegex(RuntimeError, "set_context"):
            router.route_batch(["query"])

    def test_context_rejects_incomplete_observation(self) -> None:
        router = RouteFMRouter(
            _PredictableModel(), encoder="bge", embedder=_RecordingEmbedder()
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            router.set_context({
                "a": [{"query": "x", "score": 1.0}],
                "b": [{"query": "x", "score": 0.0, "cost": 1.0}],
            })

    def test_array_api_supports_ragged_context_mask(self) -> None:
        router = RouteFMRouter(_PredictableModel(), encoder="qwen")
        result = router.predict_arrays(
            context_embeddings=np.ones((2, 2, 3), dtype=np.float32),
            context_scores=np.asarray([[1.0, 0.0], [0.5, np.nan]], dtype=np.float32),
            context_costs=np.asarray([[0.1, 0.2], [1.0, np.nan]], dtype=np.float32),
            context_mask=np.asarray([[True, True], [True, False]]),
            target_embeddings=np.ones((2, 3), dtype=np.float32),
            model_names=["a", "b"],
        )
        self.assertEqual(result["chosen_model_name"], ["b", "b"])


class PrepareEpisodeArraysTest(unittest.TestCase):
    def test_rejects_candidate_without_observations(self) -> None:
        with self.assertRaisesRegex(ValueError, "every candidate"):
            prepare_episode_arrays(
                context_embeddings=np.ones((2, 1, 3), dtype=np.float32),
                context_scores=np.ones((2, 1), dtype=np.float32),
                context_costs=np.ones((2, 1), dtype=np.float32),
                target_embeddings=np.ones((1, 3), dtype=np.float32),
                context_mask=np.asarray([[True], [False]]),
                dimension=3,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
