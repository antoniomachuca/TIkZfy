"""Teacher-forced cross-entropy objective, AdamW optimizer, and cosine scheduler.

References:
    Goodfellow et al., Deep Learning — teacher forcing (§10.2.1) and the
        negative log-likelihood objective for softmax outputs (§6.2.2).
    Loshchilov & Hutter, Decoupled Weight Decay Regularization — AdamW.
    Vaswani et al., Attention Is All You Need — linear warmup scheduling.
"""

import math
from dataclasses import dataclass
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


def apply_word_dropout(
    decoder_input: torch.Tensor,
    unk_index: int = 3,
    dropout_probability: float = 0.35,
    bos_index: int = 1,
) -> torch.Tensor:
    """Randomly replace decoder input tokens with UNK to mitigate posterior collapse.

    Forces the autoregressive decoder to attend to visual encoder features rather
    than relying exclusively on autoregressive language priors.
    """
    if decoder_input.ndim != 2:
        raise TensorTopologyError("decoder_input must be a rank-2 tensor (B, L).")
    if decoder_input.dtype != torch.long:
        raise TensorTopologyError("decoder_input must have torch.long dtype.")
    if not (0.0 <= dropout_probability <= 1.0):
        raise TensorTopologyError("dropout_probability must be in the range [0.0, 1.0].")
    if dropout_probability == 0.0:
        return decoder_input
    # Shape: (B, L)
    mask: torch.Tensor = (
        torch.rand_like(decoder_input, dtype=torch.float32) < dropout_probability
    ) & (decoder_input != bos_index)
    masked_input: torch.Tensor = torch.where(
        mask, torch.full_like(decoder_input, unk_index), decoder_input
    )
    return masked_input


def build_token_loss_weights(
    vocabulary: TokenVocabulary,
    coordinate_weight: float = 6.0,
    geometric_weight: float = 2.5,
    boilerplate_weight: float = 0.3,
    default_weight: float = 1.0,
) -> torch.Tensor:
    """Compute per-class cross-entropy loss weights to counteract token distribution imbalance.

    Heavily penalizes errors on continuous spatial coordinate bins and geometric operators,
    preventing the model from settling into trivial local minima dictated by boilerplate.

    Args:
        vocabulary (TokenVocabulary): Discrete token vocabulary instance.
        coordinate_weight (float): Multiplier for numerical coordinate bins (default: 6.0).
        geometric_weight (float): Multiplier for geometric operators (default: 2.5).
        boilerplate_weight (float): Multiplier for boilerplate delimiters (default: 0.3).
        default_weight (float): Baseline multiplier for intermediate tokens (default: 1.0).

    Returns:
        torch.Tensor: Shape ``(V,)`` tensor of per-class positive scalar loss weights.

    Temporal complexity: O(|V|).
    Spatial complexity: O(|V|).
    """
    vocab_size: int = len(vocabulary.token_to_index)
    weights: torch.Tensor = torch.full((vocab_size,), default_weight, dtype=torch.float32)

    geometric_keywords: set[str] = {
        "--",
        "->",
        "<-",
        "<->",
        "|-",
        "-|",
        "circle",
        "arc",
        "plot",
        "grid",
        "node",
        "fill",
        "rectangle",
        "draw",
        "dashed",
        "thick",
        "thin",
        "very",
        "ultra",
        "smooth",
        "red",
        "blue",
        "cyan",
        "magenta",
        "orange",
        "green",
        "black",
        "gray",
        "brown",
        "domain",
        "step",
        "scale",
        "rotate",
        "at",
        "cos",
        "sin",
        "exp",
        "tan",
        "\\x",
    }
    boilerplate_keywords: set[str] = {
        r"\begin{tikzpicture}",
        r"\end{tikzpicture}",
        ";",
        r"\draw",
    }

    for token, idx in vocabulary.token_to_index.items():
        try:
            _ = float(token)
            weights[idx] = coordinate_weight
        except ValueError:
            if token in geometric_keywords:
                weights[idx] = geometric_weight
            elif token in boilerplate_keywords:
                weights[idx] = boilerplate_weight

    return weights


class TeacherForcingCrossEntropy(nn.Module):
    """Mean token-level cross-entropy over causally shifted targets with optional class weights.

    Padding positions are excluded from the average via ``ignore_index``.
    """

    def __init__(
        self,
        ignore_index: int = PAD_INDEX,
        token_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ignore_index: int = ignore_index
        self.label_smoothing: float = label_smoothing
        if token_weights is not None:
            if token_weights.ndim != 1:
                raise TensorTopologyError("token_weights must be a rank-1 tensor (V,).")
            self.register_buffer("token_weights", token_weights)
        else:
            self.token_weights: torch.Tensor | None = None

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
            logits.transpose(1, 2),
            target_tokens,
            weight=self.token_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss


class SpatialAwareHybridLoss(nn.Module):
    """Hybrid objective combining token cross-entropy with continuous Huber coordinate loss.

    Penalizes syntax and structure errors via Weighted Cross-Entropy while computing continuous
    Smooth L1 spatial distance over numerical coordinate predictions.
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        spatial_weight: float = 1.5,
        token_weights: torch.Tensor | None = None,
        use_automatic_reweighting: bool = True,
        ignore_index: int = PAD_INDEX,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial_weight: float = spatial_weight
        self.ignore_index: int = ignore_index
        self.label_smoothing: float = label_smoothing

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

        resolved_weights: torch.Tensor | None = token_weights
        if resolved_weights is None and use_automatic_reweighting:
            resolved_weights = build_token_loss_weights(vocabulary)

        if resolved_weights is not None:
            self.register_buffer("token_weights", resolved_weights)
        else:
            self.token_weights = None

    def forward(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Return scalar hybrid loss = WeightedCrossEntropy + lambda * SmoothL1(coords)."""
        if logits.ndim != 3:
            raise TensorTopologyError("Logits must be a rank-3 tensor with shape (B, L, V).")
        if target_tokens.ndim != 2 or target_tokens.dtype != torch.long:
            raise TensorTopologyError("Target tokens must be a rank-2 torch.long tensor.")
        if tuple(logits.shape[:2]) != tuple(target_tokens.shape):
            raise TensorTopologyError("Logit and target batch/sequence dimensions must match.")

        token_weights: torch.Tensor | None = (
            cast(torch.Tensor, self.token_weights) if self.token_weights is not None else None
        )
        ce_loss: torch.Tensor = F.cross_entropy(
            logits.transpose(1, 2),
            target_tokens,
            weight=token_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
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


@dataclass(frozen=True)
class LossComponents:
    """Decomposed scalar values of the multi-task composite loss."""

    total_loss: torch.Tensor
    syntax_loss: torch.Tensor
    gaussian_ord_loss: torch.Tensor
    family_loss: torch.Tensor
    huber_loss: torch.Tensor


class GaussianOrdinalCoordinateLoss(nn.Module):
    """Gaussian Ordinal smoothing loss over numerical coordinate token distributions.

    References:
        Golub & Van Loan, Matrix Computations — Gram matrix construction.
        Goodfellow et al., Deep Learning — Softmax cross-entropy (§6.2.2).

    Constructs a normalized Gaussian distribution around continuous target coordinates:
        q_j = exp(-(c_j - c*)^2 / (2 * sigma^2)) / sum_m exp(-(c_m - c*)^2 / (2 * sigma^2))
    Penalizes deviations via cross-entropy:
        L_GaussianOrd = -sum_{j in C} q_j * log(p_j)
    Executed in O(1) logical GPU parallel algebra through precomputed transition matrices.
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        sigma: float = 0.2,
        ignore_index: int = PAD_INDEX,
    ) -> None:
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive. Got {sigma}.")

        self.sigma: float = sigma
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

        # Build Gaussian transition matrix Q: Shape (V, V)
        q_matrix: torch.Tensor = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
        coord_indices: list[int] = [idx for idx, c in enumerate(is_coord) if c]
        if coord_indices:
            coords_tensor: torch.Tensor = torch.tensor(
                [coord_values[i] for i in coord_indices], dtype=torch.float32
            )
            # Difference matrix: Shape (C, C)
            diff: torch.Tensor = coords_tensor.unsqueeze(1) - coords_tensor.unsqueeze(0)
            weights: torch.Tensor = torch.exp(-0.5 * (diff / sigma) ** 2)
            row_sums: torch.Tensor = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            normalized_weights: torch.Tensor = weights / row_sums

            coord_idx_tensor: torch.Tensor = torch.tensor(coord_indices, dtype=torch.long)
            grid_i, grid_j = torch.meshgrid(coord_idx_tensor, coord_idx_tensor, indexing="ij")
            q_matrix[grid_i, grid_j] = normalized_weights

        self.register_buffer("ordinal_smoothing_matrix", q_matrix)

    def forward(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Compute mean Gaussian ordinal cross-entropy over coordinate positions."""
        if logits.ndim != 3:
            raise TensorTopologyError("Logits must be a rank-3 tensor with shape (B, L, V).")
        if target_tokens.ndim != 2 or target_tokens.dtype != torch.long:
            raise TensorTopologyError("Target tokens must be a rank-2 torch.long tensor.")
        if tuple(logits.shape[:2]) != tuple(target_tokens.shape):
            raise TensorTopologyError("Logit and target batch/sequence dimensions must match.")

        is_coord_mask: torch.Tensor = cast(torch.Tensor, self.is_coord_mask)
        is_target_coord: torch.Tensor = is_coord_mask[target_tokens] & (
            target_tokens != self.ignore_index
        )
        if not bool(is_target_coord.any().item()):
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        ordinal_matrix: torch.Tensor = cast(torch.Tensor, self.ordinal_smoothing_matrix)
        # Shape: (B, L, V)
        target_dist: torch.Tensor = ordinal_matrix[target_tokens]
        # Shape: (B, L, V)
        log_probs: torch.Tensor = F.log_softmax(logits, dim=-1)
        # Token-level cross-entropy: Shape (B, L)
        per_token_loss: torch.Tensor = -(target_dist * log_probs).sum(dim=-1)

        return per_token_loss[is_target_coord].mean()


class CompositeMultiTaskLoss(nn.Module):
    """Composite multi-task objective with Gaussian Ordinal, Huber, and Family classification.

    Decouples four complementary objectives:
        L_total = L_syntax + lambda_c * L_GaussianOrd + lambda_s * L_Huber + lambda_f * L_family

    1. L_syntax: Cross-entropy with label smoothing (eps = 0.05) on non-coordinate tokens.
    2. L_GaussianOrd: Ordinal cross-entropy with Gaussian smoothing over coordinate bins.
    3. L_family: Supervised cross-entropy over visual encoder GAP summary representation.
    4. L_Huber: Smooth L1 penalty between predicted expected coordinate and ground truth.
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        lambda_coord: float = 1.0,
        lambda_family: float = 1.5,
        lambda_spatial: float = 2.0,
        label_smoothing: float = 0.05,
        sigma: float = 0.2,
        huber_beta: float = 0.1,
        ignore_index: int = PAD_INDEX,
        token_weights: torch.Tensor | None = None,
        use_automatic_reweighting: bool = False,
    ) -> None:
        super().__init__()
        if lambda_coord < 0.0 or lambda_family < 0.0 or lambda_spatial < 0.0:
            raise ValueError("Loss balance multipliers must be non-negative.")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive. Got {sigma}.")
        if huber_beta <= 0.0:
            raise ValueError(f"huber_beta must be strictly positive. Got {huber_beta}.")
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(f"label_smoothing must be in [0.0, 1.0]. Got {label_smoothing}.")

        self.lambda_coord: float = lambda_coord
        self.lambda_family: float = lambda_family
        self.lambda_spatial: float = lambda_spatial
        self.label_smoothing: float = label_smoothing
        self.huber_beta: float = huber_beta
        self.ignore_index: int = ignore_index

        self.gaussian_ordinal_loss: GaussianOrdinalCoordinateLoss = GaussianOrdinalCoordinateLoss(
            vocabulary=vocabulary,
            sigma=sigma,
            ignore_index=ignore_index,
        )

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

        resolved_weights: torch.Tensor | None = token_weights
        if resolved_weights is None and use_automatic_reweighting:
            resolved_weights = build_token_loss_weights(vocabulary)

        if resolved_weights is not None:
            self.register_buffer("token_weights", resolved_weights)
        else:
            self.token_weights = None

    def forward(
        self,
        token_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        family_logits: torch.Tensor | None = None,
        family_targets: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | LossComponents:
        """Compute composite multi-task loss across syntax, coordinates, and family."""
        if token_logits.ndim != 3:
            raise TensorTopologyError("token_logits must be a rank-3 tensor with shape (B, L, V).")
        if target_tokens.ndim != 2 or target_tokens.dtype != torch.long:
            raise TensorTopologyError("target_tokens must be a rank-2 torch.long tensor.")
        if tuple(token_logits.shape[:2]) != tuple(target_tokens.shape):
            raise TensorTopologyError(
                "token_logits and target_tokens batch/sequence shapes must match."
            )

        device: torch.device = token_logits.device
        dtype: torch.dtype = token_logits.dtype

        is_coord_mask: torch.Tensor = cast(torch.Tensor, self.is_coord_mask)
        coord_values: torch.Tensor = cast(torch.Tensor, self.coord_values)
        token_weights: torch.Tensor | None = (
            cast(torch.Tensor, self.token_weights) if self.token_weights is not None else None
        )

        is_target_coord: torch.Tensor = is_coord_mask[target_tokens] & (
            target_tokens != self.ignore_index
        )
        is_target_syntax: torch.Tensor = (~is_coord_mask[target_tokens]) & (
            target_tokens != self.ignore_index
        )

        # 1. Syntax Cross-Entropy Loss
        if bool(is_target_syntax.any().item()):
            syntax_loss: torch.Tensor = F.cross_entropy(
                token_logits[is_target_syntax],
                target_tokens[is_target_syntax],
                weight=token_weights,
                label_smoothing=self.label_smoothing,
            )
        else:
            syntax_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # 2. Gaussian Ordinal Coordinate Loss
        if self.lambda_coord > 0.0 and bool(is_target_coord.any().item()):
            gaussian_ord_loss: torch.Tensor = self.gaussian_ordinal_loss(
                token_logits, target_tokens
            )
        else:
            gaussian_ord_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # 3. Continuous Spatial Huber Loss
        if self.lambda_spatial > 0.0 and bool(is_target_coord.any().item()):
            probs: torch.Tensor = F.softmax(token_logits, dim=-1)
            coord_probs: torch.Tensor = probs * is_coord_mask.unsqueeze(0).unsqueeze(0)
            coord_sum: torch.Tensor = coord_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            normalized_coord_probs: torch.Tensor = coord_probs / coord_sum

            # Expected continuous coordinate: Shape (B, L)
            pred_coords: torch.Tensor = (
                normalized_coord_probs * coord_values.unsqueeze(0).unsqueeze(0)
            ).sum(dim=-1)
            gt_coords: torch.Tensor = coord_values[target_tokens]

            huber_loss: torch.Tensor = F.smooth_l1_loss(
                pred_coords[is_target_coord], gt_coords[is_target_coord], beta=self.huber_beta
            )
        else:
            huber_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # 4. Multi-Task Geometric Family Loss
        if self.lambda_family > 0.0 and family_logits is not None and family_targets is not None:
            family_loss: torch.Tensor = F.cross_entropy(family_logits, family_targets)
        else:
            family_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # Total Composite Objective
        total_loss: torch.Tensor = (
            syntax_loss
            + self.lambda_coord * gaussian_ord_loss
            + self.lambda_spatial * huber_loss
            + self.lambda_family * family_loss
        )

        if return_components:
            return LossComponents(
                total_loss=total_loss,
                syntax_loss=syntax_loss,
                gaussian_ord_loss=gaussian_ord_loss,
                family_loss=family_loss,
                huber_loss=huber_loss,
            )
        return total_loss


# Canonical V4 architectural alias
CompositeMultiTaskLossV4 = CompositeMultiTaskLoss


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
