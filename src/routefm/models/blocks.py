from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ProfileBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        update, _ = self.cross(
            self.cross_norm(latent),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        latent = latent + update
        normalized = self.self_norm(latent)
        update, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        latent = latent + update
        return latent + self.ffn(self.ffn_norm(latent))


class ReadoutBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.profile_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

    def forward(self, query: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        update, _ = self.attn(
            self.query_norm(query), self.profile_norm(profile), self.profile_norm(profile), need_weights=False
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class MaskedReadoutBlock(nn.Module):
    """Cross-attend target queries to the uncompressed Context observations."""

    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        target_context_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a target-conditioned Context readout.

        ``context_mask`` is ``[N, K]``. An optional target-specific mask is
        ``[N, T, K]`` and marks the entries that may participate in attention.
        ``MultiheadAttention`` expects the inverse convention and one copy per
        head, hence the conversion below.
        """
        attention_mask = None
        key_padding_mask = ~context_mask
        if attention_bias is not None:
            expected = (query.shape[0], query.shape[1], context.shape[1])
            if attention_bias.shape != expected:
                raise ValueError(
                    f"attention_bias must have shape {expected}, got "
                    f"{tuple(attention_bias.shape)}"
                )
            allowed = context_mask[:, None, :].expand_as(attention_bias)
            if target_context_mask is not None:
                allowed = allowed & target_context_mask
            attention_mask = attention_bias.to(dtype=query.dtype).masked_fill(
                ~allowed, -torch.inf
            )
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
            # The additive mask includes both padding and any target-specific
            # retrieval restriction, avoiding mixed bool/float MHA masks.
            key_padding_mask = None
        elif target_context_mask is not None:
            allowed = target_context_mask & context_mask[:, None, :]
            attention_mask = (~allowed).repeat_interleave(self.heads, dim=0)
            # The target-specific mask already includes padding.
            key_padding_mask = None
        update, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class TargetProfileFeedbackBlock(nn.Module):
    """One target-isolated profile-feedback round.

    Each row represents exactly one ``(episode, model, target)`` triple.  The
    model profile first reads the target state, after which that same target
    re-reads the conditioned profile.  Keeping targets in the batch dimension
    prevents any target-to-target communication.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        residual_init: float,
    ):
        super().__init__()
        self.profile_norm = nn.LayerNorm(hidden_dim)
        self.target_for_profile_norm = nn.LayerNorm(hidden_dim)
        self.profile_reads_target = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.profile_ffn_norm = nn.LayerNorm(hidden_dim)
        self.profile_ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.target_norm = nn.LayerNorm(hidden_dim)
        self.conditioned_profile_norm = nn.LayerNorm(hidden_dim)
        self.target_reads_profile = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.target_ffn_norm = nn.LayerNorm(hidden_dim)
        self.target_ffn = FeedForward(hidden_dim, ffn_dim, dropout)
        self.profile_gate = nn.Parameter(torch.tensor(float(residual_init)))
        self.target_gate = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self, target: torch.Tensor, profile: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        profile_update, _ = self.profile_reads_target(
            self.profile_norm(profile),
            self.target_for_profile_norm(target),
            self.target_for_profile_norm(target),
            need_weights=False,
        )
        proposed_profile = profile + profile_update
        proposed_profile = proposed_profile + self.profile_ffn(
            self.profile_ffn_norm(proposed_profile)
        )
        profile = profile + torch.tanh(self.profile_gate) * (
            proposed_profile - profile
        )

        target_update, _ = self.target_reads_profile(
            self.target_norm(target),
            self.conditioned_profile_norm(profile),
            self.conditioned_profile_norm(profile),
            need_weights=False,
        )
        proposed_target = target + target_update
        proposed_target = proposed_target + self.target_ffn(
            self.target_ffn_norm(proposed_target)
        )
        target = target + torch.tanh(self.target_gate) * (
            proposed_target - target
        )
        return target, profile


class AxisTransformerBlock(nn.Module):
    """A permutation-equivariant Transformer block shared by both table axes."""

    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # MultiheadAttention returns NaNs for a completely padded sequence.
        # Insert one zero sentinel for computation and erase inactive rows after
        # every residual branch so padding cannot leak into later axis passes.
        safe_mask = mask.clone()
        inactive = ~safe_mask.any(dim=-1)
        if inactive.any():
            safe_mask[inactive, 0] = True
            value = value.clone()
            value[inactive, 0] = 0.0
        normalized = self.attention_norm(value)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        value = value + update
        value = value + self.ffn(self.ffn_norm(value))
        return value.masked_fill(~mask.unsqueeze(-1), 0.0)


class BidirectionalAxialBlock(nn.Module):
    """Symmetric row/column attention for a ``[B, M, K, H]`` table.

    Context-axis and model-axis attention run from the same input with shared
    Transformer parameters. Averaging the two views makes the block
    transpose-equivariant while requiring only two axis-attention calls:
    ``block(x.transpose(1, 2)).transpose(1, 2) == block(x)`` in eval mode.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        residual_init: float,
    ):
        super().__init__()
        self.axis = AxisTransformerBlock(hidden_dim, heads, ffn_dim, dropout)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_init)))

    @staticmethod
    def _context_axis(
        axis: AxisTransformerBlock, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, models, context, hidden = value.shape
        encoded = axis(value.reshape(batch * models, context, hidden), mask.reshape(batch * models, context))
        return encoded.reshape(batch, models, context, hidden)

    @staticmethod
    def _model_axis(
        axis: AxisTransformerBlock, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, models, context, hidden = value.shape
        transposed = value.transpose(1, 2)
        transposed_mask = mask.transpose(1, 2)
        encoded = axis(
            transposed.reshape(batch * context, models, hidden),
            transposed_mask.reshape(batch * context, models),
        )
        return encoded.reshape(batch, context, models, hidden).transpose(1, 2)

    def forward(
        self, value: torch.Tensor, mask: torch.Tensor, axis_mode: str = "both"
    ) -> torch.Tensor:
        if axis_mode == "context":
            transformed = self._context_axis(self.axis, value, mask)
        elif axis_mode == "model":
            transformed = self._model_axis(self.axis, value, mask)
        elif axis_mode == "both":
            context_view = self._context_axis(self.axis, value, mask)
            model_view = self._model_axis(self.axis, value, mask)
            transformed = 0.5 * (context_view + model_view)
        else:
            raise ValueError(f"unknown axis_mode: {axis_mode}")
        gate = torch.tanh(self.residual_gate)
        output = value + gate * (transformed - value)
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)
