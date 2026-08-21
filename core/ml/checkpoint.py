"""Model and optimizer checkpoint snapshot/restore helpers.

References:
    Goodfellow et al., Deep Learning — SGD (§8.3.1) and Adam (§8.5.3): the
        optimizer state dict captures the first/second-moment estimates that
        must be resumed for a training run to continue exactly.
"""

import torch
from torch import nn

from core.models import TrainingCheckpoint


def snapshot_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> TrainingCheckpoint:
    """Capture an immutable checkpoint of model and optimizer state.

    Args:
        model (nn.Module): Trainable module whose parameters are captured.
        optimizer (torch.optim.Optimizer): Optimizer whose per-parameter
            moments and hyperparameter groups are captured.
        epoch (int): 0-indexed epoch counter persisted alongside the state.

    Returns:
        TrainingCheckpoint: Validated snapshot of the current training state.

    Temporal complexity: O(P) where P is the number of model parameters.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module instance.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer instance.")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"epoch must be a non-negative integer. Got {epoch}.")

    return TrainingCheckpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        epoch=epoch,
    )


def restore_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: TrainingCheckpoint,
) -> None:
    """Load model and optimizer state from a validated checkpoint in place.

    Args:
        model (nn.Module): Target module whose parameters are overwritten.
        optimizer (torch.optim.Optimizer): Target optimizer whose moments and
            parameter groups are overwritten.
        checkpoint (TrainingCheckpoint): Snapshot produced by ``snapshot_checkpoint``.

    Temporal complexity: O(P) where P is the number of model parameters.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module instance.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer instance.")
    if not isinstance(checkpoint, TrainingCheckpoint):
        raise TypeError("checkpoint must be a TrainingCheckpoint instance.")

    model.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
