from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch
from torch import nn
from .ensemble_gate import EnsembleGate, confidence_features

from .blocks import (
    BidirectionalAxialBlock,
    MaskedReadoutBlock,
    ProfileBlock,
    ReadoutBlock,
    TargetProfileFeedbackBlock,
)


@dataclass
class RouteFMConfig:
    query_dim: int = 1024
    hidden_dim: int = 384
    projection_dim: int = 768
    ffn_dim: int = 1536
    heads: int = 8
    profile_layers: int = 4
    readout_layers: int = 2
    pool_layers: int = 4
    capability_tokens: int = 8
    quality_bins: int = 21
    observation_features: int = 7
    dropout: float = 0.1
    target_query_scale: float = 1.0
    outcome_residual_init: float = 3.0
    use_pool_transformer: bool = True
    architecture: str = "hierarchical"
    conditional_residual_init: float = 0.25
    conditional_residual_max: float = 0.5
    similarity_temperature: float = 0.10
    similarity_adapter_hidden: int = 32
    grid_layers: int = 2
    grid_residual_init: float = 0.10
    similarity_logit_scale: float = 0.75
    similarity_sigma: float = 0.12
    bilinear_scale: float = 1.0
    kernel_scale: float = 4.0
    kernel_temperature: float = 0.10
    ridge_feature_dim: int = 16
    ridge_lambda: float = 1.0
    utility_cost_weight: float = 0.0
    utility_uncertainty_weight: float = 0.0
    observation_schema: str = "extended7"
    local_context_layers: int = 1
    local_context_topk: int = 0
    local_context_residual_init: float = 0.1
    local_context_mode: str = "parallel_static"
    local_context_gate_hidden: int = 64
    local_context_gate_init: float = 0.25
    local_context_semantic_bias: float = 0.0
    target_profile_feedback_rounds: int = 0
    target_profile_feedback_residual_init: float = 0.1
    ensemble_members: int = 1
    ensemble_adapter_dim: int = 64
    ensemble_residual_init: float = 1e-3
    ensemble_branch_mode: str = "late_residual"
    ensemble_context_policies: tuple[str, ...] = ()
    ensemble_semantic_topk: int = 16
    ensemble_member_weights: tuple[float, ...] = ()
    ensemble_gate_mode: str = "none"
    ensemble_gate_hidden: int = 32
    context_refinement_rounds: int = 0
    context_refinement_shared: bool = False
    context_scale_conditioning: bool = False
    context_scale_max: int = 1024
    context_scale_residual_init: float = 0.10
    auxiliary_distribution_head: bool = False
    auxiliary_extended_observation_encoder: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class QueryProjection(nn.Module):
    def __init__(self, config: RouteFMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(config.query_dim),
            nn.Linear(config.query_dim, config.projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_dim, config.hidden_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ObservationEncoder(nn.Module):
    def __init__(self, config: RouteFMConfig):
        super().__init__()
        self.concatenate_features = config.architecture == "concatenated_features"
        self.score_cost = config.observation_schema == "score_cost"
        if self.concatenate_features:
            self.net = nn.Sequential(
                nn.Linear(config.hidden_dim + config.observation_features, config.ffn_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ffn_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
            )
            return
        if self.score_cost:
            if config.observation_features != 2:
                raise ValueError("score_cost observation schema requires exactly 2 features")
            self.score = nn.Sequential(
                nn.Linear(1, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.cost = nn.Sequential(
                nn.Linear(1, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.interaction = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
            self.net = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.ffn_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.ffn_dim, config.hidden_dim),
            )
            self.output_norm = nn.LayerNorm(config.hidden_dim)
            return
        self.outcome = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.auxiliary = nn.Linear(config.observation_features - 1, config.hidden_dim)
        self.interaction = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.outcome_residual = nn.Parameter(torch.tensor(float(config.outcome_residual_init)))

    def forward(self, query: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if self.concatenate_features:
            return self.net(torch.cat((query, features), dim=-1))
        if self.score_cost:
            score = self.score(features[..., :1])
            cost = self.cost(features[..., 1:2])
            interaction = self.interaction(
                torch.cat((query * score, query * cost), dim=-1)
            )
            base = query + score + cost + interaction
            return self.output_norm(base + self.net(base))
        outcome = self.outcome(features[..., :1])
        auxiliary = self.auxiliary(features[..., 1:])
        base = query + outcome + self.interaction(query * outcome) + auxiliary
        return self.output_norm(base + self.net(base) + self.outcome_residual * outcome)


class RouteFM(nn.Module):
    """RouteFM's hierarchical in-context routing architecture."""

    def __init__(self, config: RouteFMConfig):
        super().__init__()
        if config.hidden_dim % config.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.config = config
        self.query_projection = QueryProjection(config)
        # mixed-task routing Context consists only of model-specific winning-query embeddings.
        # Score/cost/variance construct labels and are deliberately not encoded.
        self.observation_encoder: ObservationEncoder | None = (
            None if config.architecture in {"context_only", "context_only_cost", "context_role", "context_role_masked"}
            else ObservationEncoder(config)
        )
        self.extended_observation_encoder: ObservationEncoder | None = None
        if config.auxiliary_extended_observation_encoder:
            if config.architecture not in {"profile_context", "profile_feedback"}:
                raise ValueError("extended-feature encoder requires profile_context/profile_feedback")
            extended_config = replace(
                config, observation_features=7, observation_schema="extended7"
            )
            self.extended_observation_encoder = ObservationEncoder(extended_config)
        self.context_role_embedding: nn.Embedding | None = None
        if config.architecture in {"context_role", "context_role_masked"}:
            self.context_role_embedding = nn.Embedding(2, config.hidden_dim)
            nn.init.normal_(self.context_role_embedding.weight, std=0.02)
        # metadata-aware routing treats deploy-time model metadata as a separate axis from the
        # query embedding: relative cost, context uncertainty, cost availability.
        self.target_known_encoder: nn.Module | None = None
        if config.architecture in {"metadata_utility", "metadata_quality", "profile_context_metadata"}:
            self.target_known_encoder = nn.Linear(3, config.hidden_dim)
        self.context_grid: nn.ModuleList | None = None
        if config.architecture in {"context_grid", "metric_alignment", "profile_adapter", "masked_matrix", "masked_matrix_kernel"}:
            self.context_grid = nn.ModuleList(
                [
                    BidirectionalAxialBlock(
                        config.hidden_dim,
                        config.heads,
                        config.ffn_dim,
                        config.dropout,
                        config.grid_residual_init,
                    )
                    for _ in range(config.grid_layers)
                ]
            )
        self.target_cell_token: nn.Parameter | None = None
        if config.architecture in {"masked_matrix", "masked_matrix_kernel"}:
            self.target_cell_token = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.kernel_projection: nn.Module | None = None
        if config.architecture == "masked_matrix_kernel":
            self.kernel_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            nn.init.eye_(self.kernel_projection.weight)
        self.ridge_skill_projection: nn.Module | None = None
        self.ridge_local_projection: nn.Module | None = None
        if config.architecture == "neural_ridge":
            self.ridge_skill_projection = nn.Linear(
                config.query_dim, config.ridge_feature_dim, bias=False
            )
            self.ridge_local_projection = nn.Linear(
                config.query_dim, config.ridge_feature_dim, bias=False
            )
        self.capability_tokens = nn.Parameter(torch.randn(config.capability_tokens, config.hidden_dim) * 0.02)
        self.context_scale_encoder: nn.Module | None = None
        self.context_scale_residual: nn.Parameter | None = None
        if config.context_scale_conditioning:
            self.context_scale_encoder = nn.Sequential(
                nn.Linear(4, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
            )
            self.context_scale_residual = nn.Parameter(
                torch.tensor(float(config.context_scale_residual_init))
            )
        self.profile_blocks = nn.ModuleList(
            [ProfileBlock(config.hidden_dim, config.heads, config.ffn_dim, config.dropout) for _ in range(config.profile_layers)]
        )
        self.readout_blocks = nn.ModuleList(
            [ReadoutBlock(config.hidden_dim, config.heads, config.ffn_dim, config.dropout) for _ in range(config.readout_layers)]
        )
        self.local_context_blocks: nn.ModuleList | None = None
        self.local_context_residual: nn.Parameter | None = None
        if config.architecture in {"profile_context_metadata", "profile_context", "profile_feedback"}:
            if config.local_context_layers < 1:
                raise ValueError("profile_context_metadata requires local_context_layers >= 1")
            if config.local_context_topk < 0:
                raise ValueError("local_context_topk must be non-negative")
            self.local_context_blocks = nn.ModuleList(
                [
                    MaskedReadoutBlock(
                        config.hidden_dim, config.heads, config.ffn_dim, config.dropout
                    )
                    for _ in range(config.local_context_layers)
                ]
            )
            self.local_context_residual = nn.Parameter(
                torch.tensor(float(config.local_context_residual_init))
            )
            if config.local_context_mode not in {
                "disabled", "parallel_static", "profile_static", "profile_dynamic",
            }:
                raise ValueError(f"unknown local_context_mode: {config.local_context_mode}")
            if config.local_context_gate_hidden < 1:
                raise ValueError("local_context_gate_hidden must be positive")
            if not 0.0 < config.local_context_gate_init < 1.0:
                raise ValueError("local_context_gate_init must be strictly between 0 and 1")
            if config.local_context_semantic_bias < 0.0:
                raise ValueError("local_context_semantic_bias must be non-negative")
        self.target_profile_feedback: nn.ModuleList | None = None
        # Optional post-readout depth. Defaults add no tensors and preserve all
        # the default parameter initialization. Each Target reads the same immutable memory
        # independently; parameter sharing is across rounds, never Targets.
        if config.context_refinement_rounds < 0:
            raise ValueError("context_refinement_rounds must be nonnegative")
        self.context_refinement_blocks: nn.ModuleList | None = None
        self.context_refinement_gates: nn.Parameter | None = None
        if config.context_refinement_rounds:
            if self.local_context_blocks is None:
                raise ValueError("Context refinement requires local Context readout")
            count = 1 if config.context_refinement_shared else config.context_refinement_rounds
            self.context_refinement_blocks = nn.ModuleList([
                MaskedReadoutBlock(config.hidden_dim, config.heads, config.ffn_dim, config.dropout)
                for _ in range(count)
            ])
            # Exactly identity at initialization, including in shared 3-round mode.
            self.context_refinement_gates = nn.Parameter(torch.zeros(count))
        if config.architecture == "profile_feedback":
            if config.target_profile_feedback_rounds < 1:
                raise ValueError("profile_feedback requires target_profile_feedback_rounds >= 1")
            self.target_profile_feedback = nn.ModuleList(
                [
                    TargetProfileFeedbackBlock(
                        config.hidden_dim,
                        config.heads,
                        config.ffn_dim,
                        config.dropout,
                        config.target_profile_feedback_residual_init,
                    )
                    for _ in range(config.target_profile_feedback_rounds)
                ]
            )
        if config.use_pool_transformer:
            pool_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.heads,
                dim_feedforward=config.ffn_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.pool: nn.Module | None = nn.TransformerEncoder(
                pool_layer, config.pool_layers, norm=nn.LayerNorm(config.hidden_dim)
            )
        else:
            self.pool = None
        self.quality_head: nn.Module | None = (
            None if (
                config.architecture in {"context_only", "context_only_cost", "context_role", "context_role_masked"}
                or (
                    config.architecture in {"profile_context", "profile_feedback"}
                    and not config.auxiliary_distribution_head
                )
            )
            else nn.Linear(config.hidden_dim, config.quality_bins)
        )
        self.score_head: nn.Module | None = (
            nn.Linear(config.hidden_dim, 1)
            if config.architecture in {"profile_context", "profile_feedback"} else None
        )
        self.routing_head: nn.Module | None = (
            nn.Linear(config.hidden_dim, 1)
            if config.architecture in {"context_only", "context_only_cost", "context_role", "context_role_masked"} else None
        )
        self.similarity_adapter: nn.Module | None = None
        self.similarity_scale: nn.Parameter | None = None
        self.profile_adapter: nn.Module | None = None
        if config.architecture == "metric_alignment":
            self.similarity_scale = nn.Parameter(
                torch.tensor(float(config.similarity_logit_scale))
            )
        if config.architecture == "profile_adapter":
            self.profile_adapter = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        if config.architecture == "similarity_adapter":
            self.similarity_adapter = nn.Sequential(
                nn.Linear(8, config.similarity_adapter_hidden),
                nn.GELU(),
                nn.Linear(config.similarity_adapter_hidden, config.quality_bins),
            )
            nn.init.zeros_(self.similarity_adapter[-1].weight)
            nn.init.zeros_(self.similarity_adapter[-1].bias)
        self.similarity_residual_head: nn.Module | None = None
        if config.architecture in {"similarity_residual", "similarity_calibrated"}:
            self.baseline_score_head: nn.Module | None = nn.Linear(config.hidden_dim, 1)
            self.quality_log_scale_head: nn.Module | None = nn.Linear(config.hidden_dim, 1)
            if config.architecture == "similarity_residual":
                self.residual_score_head: nn.Module | None = nn.Linear(config.hidden_dim, 1)
                nn.init.zeros_(self.residual_score_head.weight)
                nn.init.zeros_(self.residual_score_head.bias)
            else:
                self.residual_score_head = None
                # Similarity calibration calibrates two explicit similarity features together with
                # the frozen the profile readout query-to-profile readout. The trainable path is
                # still tiny and cannot memorize high-dimensional pooled states.
                self.similarity_residual_head = nn.Linear(4, 1)
                nn.init.zeros_(self.similarity_residual_head.weight)
                nn.init.zeros_(self.similarity_residual_head.bias)
            self.baseline_quality_head = None
            self.conditional_residual_scale = None
        elif config.architecture == "conditional":
            self.baseline_quality_head: nn.Module | None = nn.Linear(config.hidden_dim, config.quality_bins)
            self.conditional_residual_scale: nn.Parameter | None = nn.Parameter(
                torch.tensor(float(config.conditional_residual_init))
            )
        else:
            self.baseline_quality_head = None
            self.conditional_residual_scale = None
            self.baseline_score_head = None
            self.residual_score_head = None
            self.quality_log_scale_head = None
        if config.architecture not in {"similarity_residual", "similarity_calibrated"}:
            self.baseline_score_head = None
            self.residual_score_head = None
            self.quality_log_scale_head = None
        self.cost_head: nn.Module | None = (
            None if config.architecture in {"context_only", "context_only_cost", "context_role", "context_role_masked"}
            else nn.Linear(config.hidden_dim, 1)
        )
        self.ensemble_residual_heads: nn.ModuleList | None = None
        self.ensemble_branch_adapters: nn.ModuleList | None = None
        self.ensemble_score_heads: nn.ModuleList | None = None
        self.ensemble_gate = None
        if config.ensemble_gate_mode != "none":
            if config.ensemble_members < 2 or config.ensemble_branch_mode != "context_policy":
                raise ValueError("learned gates require a context_policy ensemble")
            self.ensemble_gate = EnsembleGate(
                config.ensemble_members, config.ensemble_gate_mode,
                config.ensemble_gate_hidden,
            )
        if config.ensemble_members < 1:
            raise ValueError("ensemble_members must be positive")
        if config.ensemble_members > 1:
            if config.architecture not in {"profile_context", "profile_feedback"}:
                raise ValueError("internal ensemble is currently supported only by profile_context/profile_feedback")
            if config.ensemble_adapter_dim < 1:
                raise ValueError("ensemble_adapter_dim must be positive")
            if config.ensemble_branch_mode not in {"late_residual", "context_policy"}:
                raise ValueError("unknown ensemble_branch_mode")
            if (
                config.ensemble_branch_mode == "context_policy"
                and len(config.ensemble_context_policies) != config.ensemble_members
            ):
                raise ValueError(
                    "context_policy ensemble requires one policy per member"
                )
            if (
                config.ensemble_member_weights
                and len(config.ensemble_member_weights) != config.ensemble_members
            ):
                raise ValueError("ensemble_member_weights must match ensemble_members")
            if config.ensemble_member_weights and not all(
                weight >= 0 for weight in config.ensemble_member_weights
            ):
                raise ValueError("ensemble_member_weights must be non-negative")
            if config.ensemble_member_weights and not any(
                weight > 0 for weight in config.ensemble_member_weights
            ):
                raise ValueError("at least one ensemble member weight must be positive")
            if config.ensemble_branch_mode == "late_residual":
                self.ensemble_residual_heads = nn.ModuleList()
                for _ in range(config.ensemble_members - 1):
                    branch = nn.Sequential(
                        nn.LayerNorm(config.hidden_dim),
                        nn.Linear(config.hidden_dim, config.ensemble_adapter_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.ensemble_adapter_dim, 2),
                    )
                    nn.init.normal_(
                        branch[-1].weight,
                        std=float(config.ensemble_residual_init),
                    )
                    nn.init.zeros_(branch[-1].bias)
                    self.ensemble_residual_heads.append(branch)
            else:
                self.ensemble_branch_adapters = nn.ModuleList()
                self.ensemble_score_heads = nn.ModuleList()
                for _ in range(config.ensemble_members):
                    adapter = nn.Sequential(
                        nn.LayerNorm(config.hidden_dim),
                        nn.Linear(config.hidden_dim, config.ensemble_adapter_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.ensemble_adapter_dim, config.hidden_dim),
                    )
                    nn.init.normal_(
                        adapter[-1].weight,
                        std=float(config.ensemble_residual_init),
                    )
                    nn.init.zeros_(adapter[-1].bias)
                    self.ensemble_branch_adapters.append(adapter)
                    self.ensemble_score_heads.append(nn.Linear(config.hidden_dim, 1))
        if config.architecture == "concatenated_features":
            self.rank_head: nn.Module | None = nn.Linear(config.hidden_dim, 1)
        else:
            self.rank_head = None
        # Initialize the optional gate last to keep the base parameter initialization stable.
        self.local_context_gate: nn.Module | None = None
        if (
            self.local_context_blocks is not None
            and config.local_context_mode == "profile_dynamic"
        ):
            gate_input = 2 * config.hidden_dim + 3
            self.local_context_gate = nn.Sequential(
                nn.LayerNorm(gate_input),
                nn.Linear(gate_input, config.local_context_gate_hidden),
                nn.GELU(),
                nn.Linear(config.local_context_gate_hidden, 1),
            )
            probability = float(config.local_context_gate_init)
            nn.init.normal_(self.local_context_gate[-1].weight, std=0.01)
            nn.init.constant_(
                self.local_context_gate[-1].bias,
                math.log(probability / (1.0 - probability)),
            )

    def encode_context_grid(
        self,
        context_query: torch.Tensor,
        context_features: torch.Tensor,
        context_mask: torch.Tensor,
        axis_mode: str = "both",
        projected_query: torch.Tensor | None = None,
        observation_schema: str = "score_cost",
    ) -> torch.Tensor:
        """Encode the model × context table before per-model profile pooling."""
        projected = (
            self.query_projection(context_query)
            if projected_query is None else projected_query
        )
        if self.observation_encoder is None:
            raise RuntimeError("mixed-task routing encodes Context queries directly")
        encoder = self.observation_encoder
        if observation_schema == "extended7":
            encoder = self.extended_observation_encoder
            if encoder is None:
                raise RuntimeError("extended7 Context requested without auxiliary encoder")
        observations = encoder(projected, context_features)
        if self.context_grid is not None:
            for block in self.context_grid:
                observations = block(observations, context_mask, axis_mode=axis_mode)
        return observations.masked_fill(~context_mask.unsqueeze(-1), 0.0)

    def encode_profiles(
        self,
        context_query: torch.Tensor,
        context_features: torch.Tensor,
        context_mask: torch.Tensor,
        grid_axis_mode: str = "both",
    ) -> torch.Tensor:
        batch, models, context, _ = context_query.shape
        observations = self.encode_context_grid(
            context_query, context_features, context_mask, axis_mode=grid_axis_mode
        )
        return self._encode_profiles_from_observations(observations, context_mask)

    def _encode_profiles_from_observations(
        self,
        observations: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, models, context, _ = observations.shape
        observations = observations.reshape(batch * models, context, -1)
        mask = context_mask.reshape(batch * models, context).clone()
        # MultiheadAttention cannot consume an entirely masked row (padded candidates).
        inactive = ~mask.any(dim=-1)
        if inactive.any():
            mask[inactive, 0] = True
            observations[inactive, 0] = 0.0
        latent = self.capability_tokens.unsqueeze(0).expand(batch * models, -1, -1)
        for block in self.profile_blocks:
            latent = block(latent, observations, mask)
        return latent.reshape(batch, models, self.config.capability_tokens, self.config.hidden_dim)

    def _context_scale_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Encode Context size and per-candidate coverage for one unified model.

        coverage-aware routing computed closely related coverage statistics outside its frozen
        experts.  Here they are ordinary differentiable features learned jointly
        with the single RouteFM from random initialization.
        """
        if self.context_scale_encoder is None:
            return None
        mask = batch["context_mask"]
        count = mask.sum(dim=-1).to(dtype=batch["context_query"].dtype)
        batch_size = mask.shape[0]
        context_sizes = batch.get("context_sizes")
        if context_sizes is None:
            context_sizes = count.new_full((batch_size,), mask.shape[-1])
        else:
            context_sizes = context_sizes.to(device=count.device, dtype=count.dtype)
        context_sizes = context_sizes.clamp_min(1.0)[:, None].expand_as(count)
        normalizer = math.log1p(max(1, int(self.config.context_scale_max)))
        features = torch.stack(
            (
                torch.log1p(count) / normalizer,
                torch.log1p(context_sizes) / normalizer,
                count / context_sizes,
                torch.rsqrt(count + 1.0),
            ),
            dim=-1,
        )
        return self.context_scale_encoder(features)

    def _local_topk_mask(
        self,
        target_query: torch.Tensor,
        context_query: torch.Tensor,
        context_mask: torch.Tensor,
        topk_override: int | None = None,
    ) -> torch.Tensor | None:
        """Select semantic Context neighbours independently for every target/model."""
        topk = int(
            self.config.local_context_topk
            if topk_override is None else topk_override
        )
        if topk <= 0 or topk >= context_query.shape[2]:
            return None
        # The shared learned query projection is both substantially cheaper
        # than a 4096-D raw cosine and aligned with the attention feature space.
        target = torch.nn.functional.normalize(target_query.float(), dim=-1)
        context = torch.nn.functional.normalize(context_query.float(), dim=-1)
        similarity = torch.einsum("btd,bmkd->bmtk", target, context)
        similarity = similarity.masked_fill(~context_mask[:, :, None, :], -torch.inf)
        indices = similarity.topk(min(topk, context_query.shape[2]), dim=-1).indices
        selected = torch.zeros_like(similarity, dtype=torch.bool)
        selected.scatter_(-1, indices, True)
        return selected & context_mask[:, :, None, :]

    @staticmethod
    def _local_similarity(
        target_query: torch.Tensor,
        context_query: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Projected cosine similarity for soft local retrieval and gating."""
        target = torch.nn.functional.normalize(target_query.float(), dim=-1)
        context = torch.nn.functional.normalize(context_query.float(), dim=-1)
        similarity = torch.einsum("btd,bmkd->bmtk", target, context)
        return similarity.masked_fill(~context_mask[:, :, None, :], -1.0)

    @staticmethod
    def _nonempty_context_mask(
        selected: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        empty = original.any(dim=-1) & ~selected.any(dim=-1)
        if empty.any():
            selected = selected.clone()
            selected[empty] = original[empty]
        return selected

    def _context_policy_mask(
        self,
        batch: dict[str, torch.Tensor],
        policy: str,
    ) -> torch.Tensor:
        original = batch["context_mask"].bool()
        context = original.shape[-1]
        if policy == "full" or policy.startswith("semantic"):
            return original
        position = torch.arange(context, device=original.device)
        if policy == "even":
            selected = original & (position.remainder(2) == 0)
        elif policy == "odd":
            selected = original & (position.remainder(2) == 1)
        elif policy.startswith("uniform"):
            suffix = policy.removeprefix("uniform")
            percentage = int(suffix) if suffix else 50
            stride = max(1, round(100 / max(1, percentage)))
            selected = original & (position.remainder(stride) == 0)
        elif policy.startswith("highscore"):
            suffix = policy.removeprefix("highscore")
            percentage = int(suffix) if suffix else 50
            keep = max(1, round(context * percentage / 100))
            score = batch["context_features"][..., 0].masked_fill(
                ~original, -torch.inf
            )
            indices = score.topk(min(keep, context), dim=-1).indices
            selected = torch.zeros_like(original)
            selected.scatter_(-1, indices, True)
            selected &= original
        elif policy == "boundary":
            score = batch["context_features"][..., 0]
            count = original.sum(dim=-1).clamp_min(1)
            model_mean = (score * original).sum(dim=-1) / count
            global_winner = model_mean.masked_fill(
                ~batch["candidate_mask"], -torch.inf
            ).argmax(dim=-1)
            query_score = score.transpose(1, 2).masked_fill(
                ~original.transpose(1, 2), -torch.inf
            )
            query_winner = query_score.argmax(dim=-1)
            enough_models = original.transpose(1, 2).sum(dim=-1) >= 2
            switching = enough_models & (query_winner != global_winner[:, None])
            selected = original & switching[:, None, :]
        else:
            raise ValueError(f"unknown ensemble Context policy: {policy}")
        return self._nonempty_context_mask(selected, original)

    def _forward_context_policy_ensemble(
        self,
        batch: dict[str, torch.Tensor],
        grid_axis_mode: str,
    ) -> dict[str, torch.Tensor]:
        branch_outputs = []
        for member_index, policy in enumerate(self.config.ensemble_context_policies):
            branch_batch = dict(batch)
            branch_batch["context_mask"] = self._context_policy_mask(batch, policy)
            branch_batch["_ensemble_branch_index"] = member_index
            if policy.startswith("semantic"):
                suffix = policy.removeprefix("semantic")
                branch_batch["_ensemble_local_topk"] = (
                    int(suffix) if suffix else int(self.config.ensemble_semantic_topk)
                )
            branch_outputs.append(
                self.forward(branch_batch, grid_axis_mode=grid_axis_mode)
            )
        score_members = torch.stack(
            [output["score_mean"] for output in branch_outputs], dim=-1
        )
        cost_members = torch.stack(
            [output["cost_mean"] for output in branch_outputs], dim=-1
        )
        if self.config.ensemble_member_weights:
            weights = score_members.new_tensor(self.config.ensemble_member_weights)
            weights = weights / weights.sum()
            score_mean = (score_members * weights).sum(dim=-1)
            cost_mean = (cost_members * weights).sum(dim=-1)
        else:
            score_mean = score_members.mean(dim=-1)
            cost_mean = cost_members.mean(dim=-1)
        output = dict(branch_outputs[0])
        if self.ensemble_gate is not None:
            mask = batch["candidate_mask"][:, None].expand_as(score_members[..., 0])
            if "target_mask" in batch:
                mask = mask & batch["target_mask"]
            weights = self.ensemble_gate(confidence_features(score_members, mask))
            score_mean = (score_members * weights.unsqueeze(-2)).sum(-1)
            # Quality gating must not learn to prefer a branch because of a
            # cost prediction scale difference. Cost retains its fixed blend.
            output["ensemble_gate_weights"] = weights
        output.update({
            "score_mean": score_mean,
            "cost_mean": cost_mean,
            "score_members": score_members,
            "cost_members": cost_members,
        })
        return output

    @torch.no_grad()
    def initialize_ensemble_from_base(self) -> None:
        """Warm-start independent score heads from a loaded score/cost routing head."""
        if self.ensemble_score_heads is None or self.score_head is None:
            return
        for head in self.ensemble_score_heads:
            head.weight.copy_(self.score_head.weight)
            head.bias.copy_(self.score_head.bias)

    def forward(
        self, batch: dict[str, torch.Tensor], grid_axis_mode: str = "both"
    ) -> dict[str, torch.Tensor]:
        if self.config.architecture in {"context_only", "context_only_cost", "context_role", "context_role_masked"}:
            return self._forward_context_only(batch)
        if self.config.architecture == "neural_ridge":
            return self._forward_neural_ridge(batch)
        if self.config.architecture in {"masked_matrix", "masked_matrix_kernel"}:
            return self._forward_masked_matrix(batch, grid_axis_mode=grid_axis_mode)
        if (
            self.config.architecture == "profile_context"
            and self.config.ensemble_members > 1
            and self.config.ensemble_branch_mode == "context_policy"
            and "_ensemble_branch_index" not in batch
            and not batch.get("_ensemble_base_only", False)
        ):
            return self._forward_context_policy_ensemble(batch, grid_axis_mode)
        context_query = batch["context_query"]
        batch_size, models = context_query.shape[:2]
        targets = batch["target_query"].shape[1]
        projected_context = self.query_projection(context_query)
        context_hidden_residual = batch.get("_context_hidden_residual")
        if context_hidden_residual is not None:
            if context_hidden_residual.shape != projected_context.shape:
                raise ValueError(
                    "_context_hidden_residual must match projected Context shape"
                )
            projected_context = projected_context + context_hidden_residual.to(
                dtype=projected_context.dtype
            )
        observations = self.encode_context_grid(
            context_query, batch["context_features"], batch["context_mask"],
            axis_mode=grid_axis_mode,
            projected_query=projected_context,
            observation_schema=str(batch.get("_context_schema", "score_cost")),
        )
        profiles = self._encode_profiles_from_observations(
            observations, batch["context_mask"]
        )
        scale_embedding = self._context_scale_embedding(batch)
        if scale_embedding is not None:
            profiles = profiles + self.context_scale_residual * scale_embedding.unsqueeze(2)
        projected_target = self.query_projection(batch["target_query"])
        target_hidden_residual = batch.get("_target_hidden_residual")
        if target_hidden_residual is not None:
            if target_hidden_residual.shape != projected_target.shape:
                raise ValueError(
                    "_target_hidden_residual must match projected Target shape"
                )
            projected_target = projected_target + target_hidden_residual.to(
                dtype=projected_target.dtype
            )
        query = self.config.target_query_scale * projected_target
        query = query[:, None].expand(-1, models, -1, -1)
        if scale_embedding is not None:
            query = query + self.context_scale_residual * scale_embedding.unsqueeze(2)
        if self.target_known_encoder is not None:
            context_mask = batch["context_mask"].to(query.dtype)
            context_uncertainty = (
                (batch["context_features"][..., 2] * context_mask).sum(dim=-1)
                / context_mask.sum(dim=-1).clamp_min(1.0)
            )
            context_uncertainty = context_uncertainty[:, :, None].expand(-1, -1, targets)
            target_cost = self._normalized_target_cost(batch).transpose(1, 2)
            target_cost_available = batch["target_cost_mask"].transpose(1, 2).to(query.dtype)
            if self.config.architecture in {"metadata_quality", "profile_context_metadata"}:
                # Keep the quality predictor pure. Known target cost is applied
                # exactly once after quality/uncertainty prediction in utility.
                target_cost = torch.zeros_like(target_cost)
                target_cost_available = torch.zeros_like(target_cost_available)
            known = torch.stack(
                (
                    target_cost,
                    context_uncertainty,
                    target_cost_available,
                ),
                dim=-1,
            )
            query = query + self.target_known_encoder(known)
        query = query.reshape(batch_size * models, targets, -1)
        base_query = query
        profile = profiles.reshape(batch_size * models, self.config.capability_tokens, -1)
        for block in self.readout_blocks:
            query = block(query, profile)
        local_gate_value = None
        local_enabled = (
            self.local_context_blocks is not None
            and self.config.local_context_mode != "disabled"
            and not batch.get("_disable_local_context", False)
        )
        if local_enabled:
            context = observations.reshape(batch_size * models, context_query.shape[2], -1)
            context_mask = batch["context_mask"].reshape(
                batch_size * models, context_query.shape[2]
            ).clone()
            inactive_context = ~context_mask.any(dim=-1)
            if inactive_context.any():
                context_mask[inactive_context, 0] = True
                context = context.clone()
                context[inactive_context, 0] = 0.0
            target_context_mask = self._local_topk_mask(
                projected_target, projected_context, batch["context_mask"],
                topk_override=batch.get("_ensemble_local_topk"),
            )
            if target_context_mask is not None:
                target_context_mask = target_context_mask.reshape(
                    batch_size * models, targets, context_query.shape[2]
                )
                if inactive_context.any():
                    target_context_mask = target_context_mask.clone()
                    target_context_mask[inactive_context, :, 0] = True
            needs_similarity = (
                self.config.local_context_mode == "profile_dynamic"
                or self.config.local_context_semantic_bias > 0.0
            )
            local_similarity = None
            if needs_similarity:
                local_similarity = self._local_similarity(
                    projected_target, projected_context, batch["context_mask"]
                ).reshape(batch_size * models, targets, context_query.shape[2])
                if inactive_context.any():
                    local_similarity = local_similarity.clone()
                    local_similarity[inactive_context, :, 0] = 0.0
            attention_bias = None
            if self.config.local_context_semantic_bias > 0.0:
                attention_bias = (
                    float(self.config.local_context_semantic_bias) * local_similarity
                )
            profile_conditioned = self.config.local_context_mode in {
                "profile_static", "profile_dynamic",
            }
            global_query = query
            local_query = global_query if profile_conditioned else base_query
            for block in self.local_context_blocks:
                local_query = block(
                    local_query, context, context_mask, target_context_mask,
                    attention_bias,
                )
            if self.config.local_context_mode == "profile_dynamic":
                count = batch["context_mask"].sum(dim=-1).to(global_query.dtype)
                count = count.reshape(batch_size * models, 1).expand(-1, targets)
                width = max(1, context_query.shape[2])
                normalizer = math.log1p(max(1, int(self.config.context_scale_max)))
                maximum_similarity = local_similarity.max(dim=-1).values.to(
                    global_query.dtype
                )
                statistics = torch.stack(
                    (
                        torch.log1p(count) / normalizer,
                        count / float(width),
                        maximum_similarity,
                    ),
                    dim=-1,
                )
                delta = local_query - global_query
                gate_input = torch.cat((global_query, delta, statistics), dim=-1)
                local_gate_value = torch.sigmoid(self.local_context_gate(gate_input))
                query = global_query + local_gate_value * delta
            elif profile_conditioned:
                query = global_query + self.local_context_residual * (
                    local_query - global_query
                )
            else:
                query = global_query + self.local_context_residual * (
                    local_query - base_query
                )
            if self.context_refinement_blocks is not None:
                for round_index in range(self.config.context_refinement_rounds):
                    index = 0 if self.config.context_refinement_shared else round_index
                    proposed = self.context_refinement_blocks[index](
                        query, context, context_mask, target_context_mask
                    )
                    query = query + self.context_refinement_gates[index].tanh() * (proposed - query)
        if self.target_profile_feedback is not None:
            # Clone profiles per target by placing targets in the batch axis.
            # Consequently no target can attend to, update, or otherwise leak
            # information into another target in the same episode.
            batch_models = batch_size * models
            target_state = query.reshape(batch_models * targets, 1, -1)
            conditioned_profile = profile[:, None].expand(
                -1, targets, -1, -1
            ).reshape(
                batch_models * targets, self.config.capability_tokens, -1
            )
            for block in self.target_profile_feedback:
                target_state, conditioned_profile = block(
                    target_state, conditioned_profile
                )
            query = target_state.reshape(batch_models, targets, -1)
        query = query.reshape(batch_size, models, targets, -1).transpose(1, 2)

        pooled = query.reshape(batch_size * targets, models, -1)
        candidate_mask = batch["candidate_mask"][:, None].expand(-1, targets, -1)
        # A model that was not evaluated on this target is not a candidate for
        # the target-specific decision and must not influence the other model
        # slots through cross-model attention.  ``candidate_mask`` alone is
        # episode-wide and therefore insufficient for sparse target tables.
        if "target_mask" in batch:
            candidate_mask = candidate_mask & batch["target_mask"]
        candidate_mask = candidate_mask.reshape(batch_size * targets, models)
        # Collation pads shorter target blocks with rows whose target_mask is
        # entirely false. Transformer attention cannot consume an all-masked
        # row, so install an inert fallback token; losses and decisions still
        # discard the padded target via target_mask.
        inactive_targets = ~candidate_mask.any(dim=-1)
        if inactive_targets.any():
            candidate_mask = candidate_mask.clone()
            pooled = pooled.clone()
            candidate_mask[inactive_targets, 0] = True
            pooled[inactive_targets, 0] = 0.0
        if self.pool is not None:
            pooled = self.pool(pooled, src_key_padding_mask=~candidate_mask)
        pooled = pooled.reshape(batch_size, targets, models, -1)
        conditional_logits = (
            self.quality_head(pooled) if self.quality_head is not None else None
        )
        output = {"profiles": profiles}
        if local_gate_value is not None:
            output["local_gate_mean"] = local_gate_value.mean()
            output["local_gate_min"] = local_gate_value.min()
            output["local_gate_max"] = local_gate_value.max()
        if self.config.architecture in {"profile_context", "profile_feedback"}:
            score_logit = self.score_head(pooled).squeeze(-1)
            cost_logit = self.cost_head(pooled).squeeze(-1)
            branch_index = batch.get("_ensemble_branch_index")
            if branch_index is not None:
                if self.config.ensemble_branch_mode == "context_policy":
                    branch_pooled = pooled + self.ensemble_branch_adapters[
                        branch_index
                    ](pooled)
                    score_logit = self.ensemble_score_heads[
                        branch_index
                    ](branch_pooled).squeeze(-1)
                elif branch_index:
                    residual = self.ensemble_residual_heads[branch_index - 1](pooled)
                    score_logit = score_logit + residual[..., 0]
                    cost_logit = cost_logit + residual[..., 1]
                output.update({
                    "score_mean": score_logit.sigmoid(),
                    "cost_mean": cost_logit.sigmoid(),
                })
            elif self.ensemble_residual_heads is None:
                output.update({
                    "score_mean": score_logit.sigmoid(),
                    "cost_mean": cost_logit.sigmoid(),
                })
            else:
                score_members = [score_logit.sigmoid()]
                cost_members = [cost_logit.sigmoid()]
                for branch in self.ensemble_residual_heads:
                    residual = branch(pooled)
                    score_members.append((score_logit + residual[..., 0]).sigmoid())
                    cost_members.append((cost_logit + residual[..., 1]).sigmoid())
                score_members = torch.stack(score_members, dim=-1)
                cost_members = torch.stack(cost_members, dim=-1)
                if self.config.ensemble_member_weights:
                    weights = score_members.new_tensor(
                        self.config.ensemble_member_weights
                    )
                    weights = weights / weights.sum()
                    score_mean = (score_members * weights).sum(dim=-1)
                    cost_mean = (cost_members * weights).sum(dim=-1)
                else:
                    score_mean = score_members.mean(dim=-1)
                    cost_mean = cost_members.mean(dim=-1)
                output.update({
                    "score_mean": score_mean,
                    "cost_mean": cost_mean,
                    "score_members": score_members,
                    "cost_members": cost_members,
                })
        else:
            output["cost_log"] = self.cost_head(pooled).squeeze(-1)
        if self.similarity_scale is not None:
            # Learn the metric jointly with the router. Raw encoder cosine is a
            # useful baseline but need not align with model-specific capability.
            target_embedding = torch.nn.functional.normalize(
                self.query_projection(batch["target_query"]), dim=-1
            )
            context_embedding = torch.nn.functional.normalize(
                self.query_projection(batch["context_query"]), dim=-1
            )
            cosine = torch.einsum("btd,bmkd->btmk", target_embedding, context_embedding)
            evidence_mask = batch["context_mask"][:, None]
            masked_cosine = cosine.masked_fill(~evidence_mask, -1e4)
            attention = (
                masked_cosine / max(float(self.config.similarity_temperature), 1e-4)
            ).softmax(dim=-1) * evidence_mask
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            context_score = batch["context_features"][..., 0]
            similarity_score = (attention * context_score[:, None]).sum(dim=-1)
            bins = torch.linspace(
                0.0, 1.0, self.config.quality_bins, device=conditional_logits.device
            )
            sigma = max(float(self.config.similarity_sigma), 1e-3)
            evidence_logits = -0.5 * (
                (bins - similarity_score.unsqueeze(-1)) / sigma
            ).square()
            evidence_logits = evidence_logits - evidence_logits.mean(dim=-1, keepdim=True)
            scale = self.similarity_scale.clamp(0.0, 4.0)
            conditional_logits = conditional_logits + scale * evidence_logits
            output.update(
                {
                    "similarity_score": similarity_score,
                    "similarity_attention": attention,
                    "similarity_scale": scale,
                }
            )
        if self.profile_adapter is not None:
            # Explicit query × inferred-capability interaction. Subtracting the
            # projected zero vector makes the diagnostic intervention exact:
            # zero-query contributes no bilinear routing residual.
            zero_query = torch.zeros_like(batch["target_query"])
            target_direct = self.query_projection(batch["target_query"]) - self.query_projection(zero_query)
            target_direct = torch.nn.functional.normalize(target_direct, dim=-1)
            profile_summary = profiles.mean(dim=2)
            capability = torch.nn.functional.normalize(
                self.profile_adapter(profile_summary), dim=-1
            )
            bilinear_score = torch.einsum("bth,bmh->btm", target_direct, capability)
            bins = torch.linspace(
                -1.0, 1.0, self.config.quality_bins, device=conditional_logits.device
            )
            conditional_logits = conditional_logits + (
                float(self.config.bilinear_scale)
                * bilinear_score.unsqueeze(-1)
                * bins
            )
            output["bilinear_score"] = bilinear_score
        if self.similarity_adapter is not None:
            target_embedding = torch.nn.functional.normalize(batch["target_query"], dim=-1)
            context_embedding = torch.nn.functional.normalize(batch["context_query"], dim=-1)
            cosine = torch.einsum("btd,bmkd->btmk", target_embedding, context_embedding)
            evidence_mask = batch["context_mask"][:, None]
            masked_cosine = cosine.masked_fill(~evidence_mask, -1e4)
            attention = (
                masked_cosine / max(float(self.config.similarity_temperature), 1e-4)
            ).softmax(dim=-1) * evidence_mask
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            context_score = batch["context_features"][..., 0]
            context_count = batch["context_mask"].sum(dim=-1).clamp_min(1)
            context_mean = (
                (context_score * batch["context_mask"]).sum(dim=-1) / context_count
            )
            context_variance = (
                ((context_score - context_mean[..., None]).square() * batch["context_mask"]).sum(dim=-1)
                / context_count
            )
            similarity_score = (attention * context_score[:, None]).sum(dim=-1)
            nearest_index = masked_cosine.argmax(dim=-1)
            nearest_score = context_score[:, None].expand(-1, targets, -1, -1).gather(
                -1, nearest_index.unsqueeze(-1)
            ).squeeze(-1)
            maximum_similarity = masked_cosine.max(dim=-1).values
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
            maximum_entropy = context_count.float().log()[:, None]
            confidence = torch.where(
                maximum_entropy > 0,
                1.0 - entropy / maximum_entropy.clamp_min(1e-8),
                torch.ones_like(entropy),
            ).clamp(0.0, 1.0)
            similarity_evidence = torch.stack(
                (
                    similarity_score,
                    nearest_score,
                    context_mean[:, None].expand_as(similarity_score),
                    similarity_score - context_mean[:, None],
                    nearest_score - context_mean[:, None],
                    maximum_similarity,
                    confidence,
                    context_variance.sqrt()[:, None].expand_as(similarity_score),
                ),
                dim=-1,
            )
            similarity_adapter_logits = self.similarity_adapter(similarity_evidence)
            conditional_logits = conditional_logits + similarity_adapter_logits
            output.update(
                {
                    "similarity_score": similarity_score,
                    "nearest_context_score": nearest_score,
                    "similarity_confidence": confidence,
                    "similarity_adapter_logits": similarity_adapter_logits,
                }
            )
        if self.baseline_score_head is not None:
            profile_summary = profiles.mean(dim=2)
            baseline_score = torch.sigmoid(self.baseline_score_head(profile_summary).squeeze(-1))
            extra_output = {}
            if self.similarity_residual_head is not None:
                target_embedding = torch.nn.functional.normalize(batch["target_query"], dim=-1)
                context_embedding = torch.nn.functional.normalize(batch["context_query"], dim=-1)
                similarity = torch.einsum("btd,bmkd->btmk", target_embedding, context_embedding)
                similarity = similarity / max(float(self.config.similarity_temperature), 1e-4)
                evidence_mask = batch["context_mask"][:, None]
                similarity = similarity.masked_fill(~evidence_mask, -1e4)
                attention = similarity.softmax(dim=-1) * evidence_mask
                attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                context_score = batch["context_features"][..., 0]
                similarity_score = (attention * context_score[:, None]).sum(dim=-1)
                context_count = batch["context_mask"].sum(dim=-1).clamp_min(1)
                context_mean = (
                    (context_score * batch["context_mask"]).sum(dim=-1) / context_count
                )
                entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
                maximum_entropy = context_count.float().log()[:, None]
                confidence = torch.where(
                    maximum_entropy > 0,
                    1.0 - entropy / maximum_entropy.clamp_min(1e-8),
                    torch.ones_like(entropy),
                ).clamp(0.0, 1.0)
                evidence_delta = similarity_score - context_mean[:, None]
                evidence_bins = torch.linspace(
                    0.0, 1.0, self.config.quality_bins, device=conditional_logits.device
                )
                neural_score = (conditional_logits.softmax(dim=-1) * evidence_bins).sum(dim=-1)
                neural_delta = neural_score - baseline_score[:, None]
                evidence = torch.stack(
                    (
                        evidence_delta,
                        evidence_delta * confidence,
                        neural_delta,
                        evidence_delta * neural_delta,
                    ),
                    dim=-1,
                )
                conditional_residual = self.config.conditional_residual_max * torch.tanh(
                    self.similarity_residual_head(evidence).squeeze(-1)
                )
                extra_output = {
                    "similarity_score": similarity_score,
                    "similarity_confidence": confidence,
                    "similarity_evidence_delta": evidence_delta,
                    "neural_evidence_delta": neural_delta,
                }
            else:
                conditional_residual = self.config.conditional_residual_max * torch.tanh(
                    self.residual_score_head(pooled).squeeze(-1)
                )
            final_score = (baseline_score[:, None] + conditional_residual).clamp(0.0, 1.0)
            scale = 0.02 + 0.25 * torch.sigmoid(self.quality_log_scale_head(pooled).squeeze(-1))
            bins = torch.linspace(0.0, 1.0, self.config.quality_bins, device=pooled.device)
            quality_logits = -0.5 * ((bins - final_score.unsqueeze(-1)) / scale.unsqueeze(-1)).square()
            output.update(
                {
                    "baseline_score": baseline_score,
                    "conditional_residual": conditional_residual,
                    "quality_mean": final_score,
                    "quality_logits": quality_logits,
                    **extra_output,
                }
            )
        elif self.baseline_quality_head is not None:
            profile_summary = profiles.mean(dim=2)
            baseline_logits = self.baseline_quality_head(profile_summary)
            # Only relative bin preference is a residual; removing the constant
            # logit direction makes the decomposition identifiable.
            conditional_logits = conditional_logits - conditional_logits.mean(dim=-1, keepdim=True)
            scaled_residual = self.conditional_residual_scale * conditional_logits
            output.update(
                {
                    "baseline_quality_logits": baseline_logits,
                    "conditional_quality_logits": scaled_residual,
                    "quality_logits": baseline_logits[:, None] + scaled_residual,
                }
            )
        elif conditional_logits is not None:
            output["quality_logits"] = conditional_logits
        if self.rank_head is not None:
            output["rank_score"] = self.rank_head(pooled).squeeze(-1)
        return output

    def _forward_context_only(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Route directly from target queries and per-model Context query sets."""
        if self.routing_head is None:
            raise RuntimeError("mixed-task routing routing head was not initialized")
        context_query = batch["context_query"]
        batch_size, models, context, _ = context_query.shape
        targets = batch["target_query"].shape[1]

        observations = self.query_projection(context_query)
        if self.context_role_embedding is not None:
            observations = observations + self.context_role_embedding(
                batch["context_role"]
            )
        observations = observations.masked_fill(
            ~batch["context_mask"].unsqueeze(-1), 0.0
        )
        observations = observations.reshape(batch_size * models, context, -1)
        context_mask = batch["context_mask"].reshape(
            batch_size * models, context
        ).clone()
        inactive_context = ~context_mask.any(dim=-1)
        if inactive_context.any():
            context_mask[inactive_context, 0] = True
            observations[inactive_context, 0] = 0.0

        profiles = self.capability_tokens.unsqueeze(0).expand(
            batch_size * models, -1, -1
        )
        for block in self.profile_blocks:
            profiles = block(profiles, observations, context_mask)

        target = self.config.target_query_scale * self.query_projection(
            batch["target_query"]
        )
        target = target[:, None].expand(-1, models, -1, -1)
        target = target.reshape(batch_size * models, targets, -1)
        for block in self.readout_blocks:
            target = block(target, profiles)
        target = target.reshape(batch_size, models, targets, -1).transpose(1, 2)

        pooled = target.reshape(batch_size * targets, models, -1)
        if self.config.architecture == "context_role_masked":
            target_candidate_mask = batch.get(
                "target_candidate_mask", batch["target_mask"]
            )
            candidate_mask = target_candidate_mask.reshape(
                batch_size * targets, models
            )
        else:
            candidate_mask = batch["candidate_mask"][:, None].expand(
                -1, targets, -1
            ).reshape(batch_size * targets, models)
        if self.pool is not None:
            safe_candidate_mask = candidate_mask.clone()
            inactive_target = ~safe_candidate_mask.any(dim=-1)
            if inactive_target.any():
                safe_candidate_mask[inactive_target, 0] = True
                pooled = pooled.clone()
                pooled[inactive_target, 0] = 0.0
            pooled = self.pool(pooled, src_key_padding_mask=~safe_candidate_mask)
        pooled = pooled.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        pooled = pooled.reshape(batch_size, targets, models, -1)
        profiles = profiles.reshape(
            batch_size, models, self.config.capability_tokens, self.config.hidden_dim
        )
        return {
            "routing_logits": self.routing_head(pooled).squeeze(-1),
            "profiles": profiles,
        }

    @staticmethod
    def _normalized_target_cost(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize log-cost within each candidate pool, preserving missingness."""
        cost = batch["target_cost_log"]
        available = batch.get("target_cost_mask", batch["target_mask"]) & batch["target_mask"]
        low = cost.masked_fill(~available, float("inf")).amin(dim=-1, keepdim=True)
        high = cost.masked_fill(~available, float("-inf")).amax(dim=-1, keepdim=True)
        valid_row = available.any(dim=-1, keepdim=True)
        low = torch.where(valid_row, low, torch.zeros_like(low))
        high = torch.where(valid_row, high, torch.zeros_like(high))
        normalized = (cost - low) / (high - low).clamp_min(1e-8)
        return normalized.masked_fill(~available, 0.0)

    def _ridge_features(self, query: torch.Tensor) -> torch.Tensor:
        if self.ridge_skill_projection is None or self.ridge_local_projection is None:
            raise RuntimeError("neural-ridge modules were not initialized")
        return torch.cat(
            (
                torch.tanh(self.ridge_skill_projection(query)),
                torch.sin(self.ridge_local_projection(query)),
            ),
            dim=-1,
        )

    def _forward_neural_ridge(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fit one differentiable query→quality regressor per candidate model."""
        context_feature = self._ridge_features(batch["context_query"])
        target_feature = self._ridge_features(batch["target_query"])
        mask = batch["context_mask"].to(context_feature.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        feature_mean = (context_feature * mask.unsqueeze(-1)).sum(dim=2) / count
        context_score = batch["context_features"][..., 0]
        score_mean = (context_score * mask).sum(dim=-1) / count.squeeze(-1)
        centered_feature = (context_feature - feature_mean.unsqueeze(2)) * mask.unsqueeze(-1)
        centered_score = (context_score - score_mean.unsqueeze(-1)) * mask
        covariance = torch.einsum("bmkf,bmkg->bmfg", centered_feature, centered_feature)
        cross_covariance = torch.einsum("bmkf,bmk->bmf", centered_feature, centered_score)
        feature_dim = covariance.shape[-1]
        identity = torch.eye(feature_dim, device=covariance.device, dtype=covariance.dtype)
        ridge = covariance + float(self.config.ridge_lambda) * identity
        # CUDA's batched LU does not support BF16, and this small solve is also
        # the numerically sensitive part of the model. Keep it in FP32 under AMP.
        capability = torch.linalg.solve(
            ridge.float(), cross_covariance.float().unsqueeze(-1)
        ).squeeze(-1).to(context_feature.dtype)
        target_centered = target_feature[:, :, None] - feature_mean[:, None]
        delta = torch.einsum("btmf,bmf->btm", target_centered, capability)
        score = (score_mean[:, None] + delta).clamp(1e-4, 1.0 - 1e-4)

        # The calibrated synthetic corpus represents repeated binary outcomes
        # at bins 0 and 1. Parameterize that distribution directly while still
        # exposing its expectation for routing and auxiliary regression losses.
        bins = self.config.quality_bins
        quality_logits = score.new_full((*score.shape, bins), -20.0)
        quality_logits[..., 0] = (1.0 - score).log()
        quality_logits[..., -1] = score.log()
        cost_mean = (
            batch["context_features"][..., 1] * mask
        ).sum(dim=-1) / count.squeeze(-1)
        return {
            "quality_logits": quality_logits,
            "quality_mean": score,
            "cost_log": cost_mean[:, None].expand_as(score),
            "profiles": capability.unsqueeze(2),
            "ridge_capability": capability,
            "ridge_query_delta": delta,
        }

    def _forward_masked_matrix(
        self, batch: dict[str, torch.Tensor], grid_axis_mode: str = "both"
    ) -> dict[str, torch.Tensor]:
        """Predict masked target cells in a model × query performance matrix.

        Unlike the profile bottleneck used by earlier versions, target queries
        participate in every axial block. A target cell carries its query but no
        outcome, and can recover model-specific evidence only by attending along
        its model's query axis to observed context cells.
        """
        if self.context_grid is None or self.target_cell_token is None:
            raise RuntimeError("masked_matrix masked-matrix modules were not initialized")
        context_query = batch["context_query"]
        batch_size, models, context, _ = context_query.shape
        targets = batch["target_query"].shape[1]
        context_projected = self.query_projection(context_query)
        target_projected = self.query_projection(batch["target_query"])
        context_cells = self.observation_encoder(context_projected, batch["context_features"])
        target_cells = target_projected
        target_cells = target_cells[:, None].expand(-1, models, -1, -1)
        target_cells = target_cells + self.target_cell_token.view(1, 1, 1, -1)
        target_mask = batch["target_mask"].transpose(1, 2)
        cell_mask = torch.cat((batch["context_mask"], target_mask), dim=2)
        cells = torch.cat((context_cells, target_cells), dim=2)
        cells = cells.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
        for block in self.context_grid:
            cells = block(cells, cell_mask, axis_mode=grid_axis_mode)

        encoded_context = cells[:, :, :context]
        context_weight = batch["context_mask"].unsqueeze(-1)
        profile_summary = (encoded_context * context_weight).sum(dim=2)
        profile_summary = profile_summary / context_weight.sum(dim=2).clamp_min(1)
        profiles = profile_summary.unsqueeze(2)

        target_state = cells[:, :, context:].transpose(1, 2)
        pooled = target_state.reshape(batch_size * targets, models, -1)
        pool_mask = batch["target_mask"].reshape(batch_size * targets, models)
        if self.pool is not None:
            safe_pool_mask = pool_mask.clone()
            inactive = ~safe_pool_mask.any(dim=-1)
            if inactive.any():
                safe_pool_mask[inactive, 0] = True
                pooled = pooled.clone()
                pooled[inactive, 0] = 0.0
            pooled = self.pool(pooled, src_key_padding_mask=~safe_pool_mask)
        pooled = pooled.masked_fill(~pool_mask.unsqueeze(-1), 0.0)
        pooled = pooled.reshape(batch_size, targets, models, -1)
        quality_logits = self.quality_head(pooled)
        output = {
            "quality_logits": quality_logits,
            "cost_log": self.cost_head(pooled).squeeze(-1),
            "profiles": profiles,
            "masked_matrix_state": target_state,
        }
        if self.kernel_projection is not None:
            target_metric = torch.nn.functional.normalize(
                self.kernel_projection(target_projected), dim=-1
            )
            context_metric = torch.nn.functional.normalize(
                self.kernel_projection(context_projected), dim=-1
            )
            similarity = torch.einsum("bth,bmkh->btmk", target_metric, context_metric)
            similarity = similarity.masked_fill(~batch["context_mask"][:, None], -1e4)
            attention = (
                similarity / max(float(self.config.kernel_temperature), 1e-4)
            ).softmax(dim=-1)
            context_score = batch["context_features"][..., 0]
            local_score = (attention * context_score[:, None]).sum(dim=-1)
            context_count = batch["context_mask"].sum(dim=-1).clamp_min(1)
            context_mean = (
                (context_score * batch["context_mask"]).sum(dim=-1) / context_count
            )
            local_delta = local_score - context_mean[:, None]
            bins = torch.linspace(-1.0, 1.0, self.config.quality_bins, device=pooled.device)
            output["quality_logits"] = quality_logits + (
                float(self.config.kernel_scale) * local_delta.unsqueeze(-1) * bins
            )
            output.update({
                "kernel_score": local_score,
                "kernel_delta": local_delta,
                "kernel_attention": attention,
            })
        return output

    @torch.no_grad()
    def predict(
        self, batch: dict[str, torch.Tensor], grid_axis_mode: str = "both"
    ) -> dict[str, torch.Tensor]:
        output = self(batch, grid_axis_mode=grid_axis_mode)
        if self.config.architecture in {"context_only", "context_only_cost", "context_role", "context_role_masked"}:
            if self.config.architecture == "context_role_masked":
                decision_mask = batch.get(
                    "target_candidate_mask", batch["target_mask"]
                )
            else:
                decision_mask = batch["candidate_mask"][:, None]
            logits = output["routing_logits"].masked_fill(
                ~decision_mask, -1e4
            )
            output.update(
                {
                    "routing_probability": logits.softmax(dim=-1),
                    "routing_choice": logits.argmax(dim=-1),
                }
            )
            return output
        if self.config.architecture in {"profile_context", "profile_feedback"}:
            output["quality_mean"] = output["score_mean"]
            return output
        probability = output["quality_logits"].softmax(dim=-1)
        bins = torch.linspace(0.0, 1.0, self.config.quality_bins, device=probability.device)
        distribution_mean = (probability * bins).sum(dim=-1)
        mean = output.get("quality_mean", distribution_mean)
        variance = (probability * (bins - distribution_mean.unsqueeze(-1)).square()).sum(dim=-1)
        output.update(
            {
                "quality_probability": probability,
                "quality_mean": mean,
                "distribution_mean": distribution_mean,
                "quality_variance": variance,
            }
        )
        if self.config.architecture in {"metadata_utility", "metadata_quality", "profile_context_metadata"}:
            output["routing_utility"] = (
                mean
                - float(self.config.utility_cost_weight) * self._normalized_target_cost(batch)
                - float(self.config.utility_uncertainty_weight) * variance.clamp_min(0.0).sqrt()
            )
        return output
