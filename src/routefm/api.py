"""User-facing in-memory and text APIs for frozen RouteFM inference."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from routefm.checkpoints import ResolvedCheckpoint, resolve_checkpoint
from routefm.models import RouteFM
from routefm.predict import load_router, predict_arrays as _predict_arrays


TextEmbedder = Callable[[Sequence[str]], np.ndarray]
DEFAULT_BGE_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


@dataclass(frozen=True)
class ContextObservation:
    """One observed outcome used to characterize a candidate model."""

    query: str
    score: float
    cost: float


@dataclass(frozen=True)
class RoutingDecision:
    """RouteFM's decision and per-candidate predictions for one query."""

    model_name: str
    model_index: int
    predicted_scores: Mapping[str, float]
    predicted_relative_costs: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "model_name": self.model_name,
            "model_index": self.model_index,
            "predicted_scores": dict(self.predicted_scores),
            "predicted_relative_costs": dict(self.predicted_relative_costs),
        }


class _BGETextEncoder:
    def __init__(
        self,
        model_name: str,
        revision: str | None,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Text routing requires the BGE extra: pip install 'routefm-router[bge]'"
            ) from error
        if batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if max_length < 1:
            raise ValueError("embedding_max_length must be positive")
        options = {"revision": revision} if revision is not None else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **options)
        self.model = AutoModel.from_pretrained(model_name, **options).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    @torch.inference_mode()
    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        parts = []
        for offset in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(
                list(texts[offset:offset + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            hidden = self.model(**tokens).last_hidden_state[:, 0]
            parts.append(torch.nn.functional.normalize(hidden.float(), dim=-1).cpu())
        return torch.cat(parts).numpy().astype(np.float32)


class RouteFMRouter:
    """Reusable frozen router with a convenient text and array interface.

    Candidate names are used only to label outputs. RouteFM's predictions use
    the observed query, quality, and cost values supplied through Context.
    """

    def __init__(
        self,
        model: RouteFM,
        *,
        encoder: str,
        device: str = "cpu",
        embedder: TextEmbedder | None = None,
        embedding_model: str = DEFAULT_BGE_MODEL,
        embedding_revision: str | None = DEFAULT_BGE_REVISION,
        embedding_batch_size: int = 32,
        embedding_max_length: int = 512,
    ) -> None:
        if encoder not in {"bge", "qwen"}:
            raise ValueError("encoder must be 'bge' or 'qwen'")
        self.model = model.to(device).eval()
        self.encoder = encoder
        self.device = device
        self._embedder = embedder
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.embedding_batch_size = embedding_batch_size
        self.embedding_max_length = embedding_max_length
        self._candidate_names: list[str] | None = None
        self._context_embeddings: np.ndarray | None = None
        self._context_scores: np.ndarray | None = None
        self._context_costs: np.ndarray | None = None
        self._context_mask: np.ndarray | None = None

    @classmethod
    def from_pretrained(
        cls,
        encoder: str = "bge",
        *,
        checkpoint: str | Path | ResolvedCheckpoint | None = None,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        embedding_model: str = DEFAULT_BGE_MODEL,
        embedding_revision: str | None = DEFAULT_BGE_REVISION,
        embedding_batch_size: int = 32,
        embedding_max_length: int = 512,
    ) -> "RouteFMRouter":
        """Load released weights from Hugging Face or a local checkpoint."""
        resolved = (
            checkpoint
            if isinstance(checkpoint, ResolvedCheckpoint)
            else resolve_checkpoint(encoder, checkpoint, cache_dir=cache_dir)
        )
        model = load_router(resolved, device=device, expected_encoder=encoder)
        return cls(
            model,
            encoder=encoder,
            device=device,
            embedder=embedder,
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
            embedding_batch_size=embedding_batch_size,
            embedding_max_length=embedding_max_length,
        )

    @property
    def candidate_names(self) -> tuple[str, ...]:
        """Candidate order currently stored in Context."""
        return tuple(self._candidate_names or ())

    def _text_embedder(self) -> TextEmbedder:
        if self._embedder is None:
            if self.encoder != "bge":
                raise RuntimeError(
                    "Built-in text routing is available for encoder='bge'. "
                    "For Qwen multimodal routing, pass a compatible 4096-D embedder "
                    "or call predict_arrays()."
                )
            self._embedder = _BGETextEncoder(
                self.embedding_model,
                self.embedding_revision,
                self.device,
                self.embedding_batch_size,
                self.embedding_max_length,
            )
        return self._embedder

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, (str, bytes)):
            raise TypeError("pass one query to route() or a sequence to route_batch()")
        values = list(texts)
        if not values:
            raise ValueError("at least one query is required")
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("queries must be non-empty strings")
        vectors = np.asarray(self._text_embedder()(values), dtype=np.float32)
        expected = (len(values), self.model.config.query_dim)
        if vectors.shape != expected:
            raise ValueError(f"embedder must return shape {expected}, got {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("embedder returned non-finite vectors")
        return vectors

    @staticmethod
    def _observation(value: ContextObservation | Mapping[str, Any]) -> ContextObservation:
        if isinstance(value, ContextObservation):
            observation = ContextObservation(
                query=value.query,
                score=float(value.score),
                cost=float(value.cost),
            )
        elif isinstance(value, Mapping):
            missing = {"query", "score", "cost"} - set(value)
            if missing:
                raise ValueError(f"Context observation is missing: {sorted(missing)}")
            observation = ContextObservation(
                query=value["query"],
                score=float(value["score"]),
                cost=float(value["cost"]),
            )
        else:
            raise TypeError("Context observations must be mappings or ContextObservation values")
        if not isinstance(observation.query, str) or not observation.query.strip():
            raise ValueError("Context query must be a non-empty string")
        if not np.isfinite(observation.score) or not 0.0 <= observation.score <= 1.0:
            raise ValueError("Context score must be finite and in [0,1]")
        if not np.isfinite(observation.cost) or observation.cost < 0.0:
            raise ValueError("Context cost must be finite and nonnegative")
        return observation

    def set_context(
        self,
        candidates: Mapping[
            str, Sequence[ContextObservation | Mapping[str, Any]]
        ],
    ) -> "RouteFMRouter":
        """Encode and store behavioral observations for a candidate pool."""
        if len(candidates) < 2:
            raise ValueError("Context requires at least two candidate models")
        names = list(candidates)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("candidate names must be non-empty strings")
        rows: list[list[ContextObservation]] = []
        for name in names:
            values = candidates[name]
            if isinstance(values, (str, bytes)) or not values:
                raise ValueError(f"candidate {name!r} needs at least one Context observation")
            rows.append([self._observation(value) for value in values])

        unique_queries = list(dict.fromkeys(
            observation.query for row in rows for observation in row
        ))
        encoded = self._encode(unique_queries)
        vectors = dict(zip(unique_queries, encoded, strict=True))
        width = max(len(row) for row in rows)
        dimension = self.model.config.query_dim
        context = np.zeros((len(rows), width, dimension), dtype=np.float32)
        scores = np.zeros((len(rows), width), dtype=np.float32)
        costs = np.zeros((len(rows), width), dtype=np.float32)
        mask = np.zeros((len(rows), width), dtype=bool)
        for candidate_index, row in enumerate(rows):
            for context_index, observation in enumerate(row):
                context[candidate_index, context_index] = vectors[observation.query]
                scores[candidate_index, context_index] = observation.score
                costs[candidate_index, context_index] = observation.cost
                mask[candidate_index, context_index] = True

        self._candidate_names = names
        self._context_embeddings = context
        self._context_scores = scores
        self._context_costs = costs
        self._context_mask = mask
        return self

    def predict_arrays(
        self,
        context_embeddings: np.ndarray,
        context_scores: np.ndarray,
        context_costs: np.ndarray,
        target_embeddings: np.ndarray,
        *,
        context_mask: np.ndarray | None = None,
        model_names: list[str] | None = None,
        target_batch_size: int = 128,
    ) -> dict:
        """Route precomputed embeddings without using the text helper."""
        return _predict_arrays(
            self.model,
            context_embeddings,
            context_scores,
            context_costs,
            target_embeddings,
            context_mask=context_mask,
            device=self.device,
            target_batch_size=target_batch_size,
            model_names=model_names,
        )

    def route(self, query: str) -> RoutingDecision:
        """Route one text query using the currently stored Context."""
        return self.route_batch([query])[0]

    def route_batch(self, queries: Sequence[str]) -> list[RoutingDecision]:
        """Route multiple text queries while encoding them in one batch."""
        if self._candidate_names is None:
            raise RuntimeError("call set_context() before route()")
        target_embeddings = self._encode(queries)
        result = self.predict_arrays(
            self._context_embeddings,
            self._context_scores,
            self._context_costs,
            target_embeddings,
            context_mask=self._context_mask,
            model_names=self._candidate_names,
        )
        decisions = []
        for index, chosen in enumerate(result["chosen_model_index"]):
            decisions.append(RoutingDecision(
                model_name=self._candidate_names[chosen],
                model_index=chosen,
                predicted_scores=dict(zip(
                    self._candidate_names, result["predicted_score"][index], strict=True
                )),
                predicted_relative_costs=dict(zip(
                    self._candidate_names,
                    result["predicted_relative_cost"][index],
                    strict=True,
                )),
            ))
        return decisions
