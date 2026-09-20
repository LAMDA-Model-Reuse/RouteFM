"""Small gates over frozen routing branches; no target labels are inputs."""
from __future__ import annotations

import torch
from torch import nn


def confidence_features(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B,T,M,H] -> [B,T,H,8], invariant to anonymous candidate ordering.

    Each target is summarized separately. Padded candidates are excluded, and
    empty padded targets have a finite inert representation.
    """
    scores = scores.float()
    mask = mask.bool().clone()
    empty = ~mask.any(dim=-1)
    scores = scores.masked_fill(empty[..., None, None], 0.0)
    mask[..., 0] |= empty
    valid = mask.unsqueeze(-1)
    count = valid.sum(dim=-2).clamp_min(1)
    mean = (scores * valid).sum(dim=-2) / count
    variance = ((scores - mean.unsqueeze(-2)).square() * valid).sum(-2) / count
    masked = scores.masked_fill(~valid, -1e4)
    top = masked.topk(min(2, scores.shape[-2]), dim=-2).values
    gap = top[..., 0, :] - top[..., -1, :]
    gap = torch.where(count > 1, gap, torch.zeros_like(gap))
    probs = torch.softmax(masked / 0.1, dim=-2)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-2)
    entropy = entropy / count.float().log().clamp_min(1.0)
    deviation = ((scores - scores.mean(-1, keepdim=True)).abs() * valid).sum(-2) / count
    # Symmetric agreement: probability mass that the ensemble assigns to this
    # branch's preferred candidate. No model IDs or labels enter the gate.
    choices = masked.argmax(dim=-2)
    ensemble_probs = torch.softmax(masked.mean(-1) / 0.1, dim=-1)
    agreement = ensemble_probs.gather(-1, choices)
    return torch.stack((
        mean, variance.clamp_min(0).sqrt(), top[..., 0, :], gap,
        entropy, deviation, agreement,
        count.float().log1p().expand_as(mean) / 4.0,
    ), dim=-1)


class EnsembleGate(nn.Module):
    def __init__(self, members: int, mode: str, hidden: int = 32):
        super().__init__()
        if mode not in {"global", "dynamic"}:
            raise ValueError(f"unknown ensemble gate mode {mode!r}")
        self.members = members
        self.mode = mode
        self.logits = nn.Parameter(torch.zeros(members))
        self.network = None
        if mode == "dynamic":
            self.network = nn.Sequential(
                nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.logits.expand(*features.shape[:-2], self.members)
        if self.network is not None:
            logits = logits + self.network(features).squeeze(-1)
        return logits.softmax(dim=-1)
