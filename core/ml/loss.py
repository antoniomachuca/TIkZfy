"""Teacher-forced cross-entropy objective, AdamW optimizer, and cosine scheduler.

References:
    Goodfellow et al., Deep Learning — teacher forcing (§10.2.1) and the
        negative log-likelihood objective for softmax outputs (§6.2.2).
    Loshchilov & Hutter, Decoupled Weight Decay Regularization — AdamW.
    Vaswani et al., Attention Is All You Need — linear warmup scheduling.
"""

import math
from functools import partial
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from core.exceptions import TensorTopologyError
from core.models import PAD_INDEX, TokenVocabulary


def build_teacher_forcing_pair(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a padded token batch into decoder inputs and shifted targets.

    Args:
        tokens (torch.Tensor): Rank-2 index tensor with shape ``(B, L)`` whose
            rows follow the ``[BOS, ..., EOS, PAD, ...]`` layout.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(tokens[:, :-1], tokens[:, 1:])``,
        each with shape ``(B, L - 1)``.

    Temporal complexity: O(1) — pure strided views, no data movement.
    """
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
        raise TensorTopologyError("Tokens must be a rank-2 torch.long tensor with shape (B, L).")
    if tokens.shape[1] < 2:
        raise TensorTopologyError("Token sequences must contain at least two positions.")
    return tokens[:, :-1], tokens[:, 1:]


class TeacherForcingCrossEntropy(nn.Module):
    """Mean token-level cross-entropy over causally shifted targets.

    Padding positions are excluded from the average via ``ignore_index``.
    """

    def __init__(self, ignore_index: int = PAD_INDEX) -> None:
        super().__init__()
        self.ignore_index: int = ignore_index

    def forward(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Return the mean cross-entropy over non-ignored target positions.

        Args:
            logits (torch.Tensor): Unnormalized scores with shape ``(B, L, V)``.
            target_tokens (torch.Tensor): Shifted targets with shape ``(B, L)``.

        Returns:
            torch.Tensor: Scalar loss tensor.

        Temporal complexity: O(B * L * V).
        """
        if logits.ndim != 3:
            raise TensorTopologyError("Logits must be a rank-3 tensor with shape (B, L, V).")
        if target_tokens.ndim != 2 or target_tokens.dtype != torch.long:
            raise TensorTopologyError("Target tokens must be a rank-2 torch.long tensor.")
        if tuple(logits.shape[:2]) != tuple(target_tokens.shape):
            raise TensorTopologyError("Logit and target batch/sequence dimensions must match.")
        loss: torch.Tensor = F.cross_entropy(
            logits.transpose(1, 2), target_tokens, ignore_index=self.ignore_index
        )
        return loss


class SpatialAwareHybridLoss(nn.Module):
    """Hybrid objective combining token cross-entropy with continuous coordinate Huber loss.

    Penalizes syntax and structure errors via Cross-Entropy while computing continuous
    Smooth L1 spatial distance over numerical coordinate predictions.
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        spatial_weight: float = 0.5,
        ignore_index: int = PAD_INDEX,
    ) -> None:
        super().__init__()
        self.spatial_weight: float = spatial_weight
        self.ignore_index: int = ignore_index

        vocab_size: int = len(vocabulary.token_to_index)
        is_coord: list[bool] = [False] * vocab_size
        coord_values: list[float] = [0.0] * vocab_size

        for token, idx in vocabulary.token_to_index.items():
            try:
                val: float = float(token)
                is_coord[idx] = True
                coord_values[idx] = val
            except ValueError:
                pass

        self.register_buffer("is_coord_mask", torch.tensor(is_coord, dtype=torch.bool))
        self.register_buffer("coord_values", torch.tensor(coord_values, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Return scalar hybrid loss = CrossEntropy + lambda * SmoothL1(coords)."""
        if logits.ndim != 3:
            raise TensorTopologyError("Logits must be a rank-3 tensor with shape (B, L, V).")
        if target_tokens.ndim != 2 or target_tokens.dtype != torch.long:
            raise TensorTopologyError("Target tokens must be a rank-2 torch.long tensor.")
        if tuple(logits.shape[:2]) != tuple(target_tokens.shape):
            raise TensorTopologyError("Logit and target batch/sequence dimensions must match.")

        ce_loss: torch.Tensor = F.cross_entropy(
            logits.transpose(1, 2), target_tokens, ignore_index=self.ignore_index
        )

        if self.spatial_weight <= 0.0:
            return ce_loss

        is_coord_mask: torch.Tensor = cast(torch.Tensor, self.is_coord_mask)
        coord_values: torch.Tensor = cast(torch.Tensor, self.coord_values)

        is_target_coord: torch.Tensor = is_coord_mask[target_tokens]
        if not bool(is_target_coord.any().item()):
            return ce_loss

        probs: torch.Tensor = F.softmax(logits, dim=-1)
        coord_probs: torch.Tensor = probs * is_coord_mask.unsqueeze(0).unsqueeze(0)
        coord_sum: torch.Tensor = coord_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        normalized_coord_probs: torch.Tensor = coord_probs / coord_sum

        pred_coords: torch.Tensor = (
            normalized_coord_probs * coord_values.unsqueeze(0).unsqueeze(0)
        ).sum(dim=-1)
        gt_coords: torch.Tensor = coord_values[target_tokens]

        coord_loss: torch.Tensor = F.smooth_l1_loss(
            pred_coords[is_target_coord], gt_coords[is_target_coord], beta=0.1
        )

        return ce_loss + self.spatial_weight * coord_loss


def build_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    epsilon: float = 1e-8,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer with decoupled weight-decay parameter groups.

    Parameters with ``ndim >= 2`` (weight matrices, embeddings) decay; 1-D
    parameters (biases, normalization gains) are excluded from decay.

    Temporal complexity: O(P) where P is the number of model parameters.
    """
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive. Got {learning_rate}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative. Got {weight_decay}.")
    if not 0.0 < betas[0] < 1.0 or not 0.0 < betas[1] < 1.0:
        raise ValueError(f"betas must lie in the open interval (0, 1). Got {betas}.")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive. Got {epsilon}.")

    decay_parameters: list[nn.Parameter] = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    no_decay_parameters: list[nn.Parameter] = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim < 2
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    active_groups: list[dict[str, Any]] = [group for group in parameter_groups if group["params"]]
    return torch.optim.AdamW(active_groups, lr=learning_rate, betas=betas, eps=epsilon)


def warmup_cosine_ratio(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    """Return the learning-rate multiplier for a 0-indexed ``step``.

    Linear ramp from ``1 / warmup_steps`` to ``1.0`` during warmup, then cosine
    decay reaching ``min_lr_ratio`` at ``total_steps``.

    Temporal complexity: O(1).
    """
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_span: int = max(1, total_steps - warmup_steps)
    progress: float = min(1.0, float(step - warmup_steps) / float(decay_span))
    cosine: float = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a LambdaLR scheduler with linear warmup and cosine decay.

    The scheduler must be stepped once per optimizer step.
    """
    if warmup_steps < 1:
        raise ValueError(f"warmup_steps must be at least 1. Got {warmup_steps}.")
    if total_steps <= warmup_steps:
        raise ValueError(
            f"total_steps must exceed warmup_steps. Got {total_steps} <= {warmup_steps}."
        )
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(
            f"min_lr_ratio must lie in the closed interval [0, 1]. Got {min_lr_ratio}."
        )

    lr_lambda = partial(
        warmup_cosine_ratio,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
