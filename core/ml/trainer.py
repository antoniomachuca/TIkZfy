"""Mini-batch teacher-forced training loop (SGD).

References:
    Goodfellow et al., Deep Learning — mini-batch SGD (§8.1.3), teacher forcing (§10.2.1).
"""

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from core.exceptions import TensorTopologyError
from core.ml.loss import build_teacher_forcing_pair


@dataclass(frozen=True)
class TrainingMetrics:
    """Scalar loss trace: mean per-epoch and per-step cross-entropy."""

    epoch_losses: tuple[float, ...]
    step_losses: tuple[float, ...]


def iter_batch_bounds(dataset_size: int, batch_size: int) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` bounds covering ``[0, dataset_size)``.

    Trailing batch clamps to ``dataset_size``. O(N / B).
    """
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive. Got {dataset_size}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive. Got {batch_size}.")

    starts: range = range(0, dataset_size, batch_size)
    return [(start, min(start + batch_size, dataset_size)) for start in starts]


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: nn.Module,
    images: torch.Tensor,
    tokens: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[float]:
    """One teacher-forced epoch over mini-batches; returns detached per-step losses.

    Images ``(B, C, H, W)``, tokens ``(B, L)``, logits ``(B, L, V)``. O(N * L * V).
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise TensorTopologyError("Images must be a rank-4 tensor with shape (B, C, H, W).")
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
        raise TensorTopologyError("Tokens must be a rank-2 torch.long tensor with shape (B, L).")
    if images.shape[0] != tokens.shape[0]:
        raise TensorTopologyError("Image and token batch dimensions must match.")
    if tokens.shape[1] < 2:
        raise TensorTopologyError("Token sequences must contain at least two positions.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive. Got {batch_size}.")

    dataset_size: int = images.shape[0]
    if shuffle:
        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        order: torch.Tensor = torch.randperm(dataset_size, generator=generator).to(images.device)
    else:
        order = torch.arange(dataset_size, device=images.device)

    images = images[order]
    tokens = tokens[order]

    model_device: torch.device = next(model.parameters()).device
    if images.device != model_device:
        images = images.to(model_device)
    if tokens.device != model_device:
        tokens = tokens.to(model_device)

    model.train()
    step_losses: list[float] = []
    for start, end in iter_batch_bounds(dataset_size, batch_size):
        decoder_input, targets = build_teacher_forcing_pair(tokens[start:end])
        optimizer.zero_grad()
        logits: torch.Tensor = cast(torch.Tensor, model(images[start:end], decoder_input))
        loss: torch.Tensor = cast(torch.Tensor, criterion(logits, targets))
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step_losses.append(loss.detach().item())

    return step_losses


def fit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: nn.Module,
    images: torch.Tensor,
    tokens: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> TrainingMetrics:
    """``num_epochs`` teacher-forced epochs; returns the loss trace.

    O(num_epochs * N * L * V).
    """
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive. Got {num_epochs}.")

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    for _ in range(num_epochs):
        epoch_steps: list[float] = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            images=images,
            tokens=tokens,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        epoch_losses.append(sum(epoch_steps) / len(epoch_steps))
        step_losses.extend(epoch_steps)

    return TrainingMetrics(epoch_losses=tuple(epoch_losses), step_losses=tuple(step_losses))
