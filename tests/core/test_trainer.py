import pytest
import torch

from core.exceptions import TensorTopologyError
from core.math.tokenization import batch_encode, build_vocabulary
from core.ml.loss import TeacherForcingCrossEntropy, build_adamw_optimizer
from core.ml.model import VisionAutoregressiveModel
from core.ml.trainer import (
    TrainingMetrics,
    fit,
    iter_batch_bounds,
    train_one_epoch,
)
from core.models import TikzTokens

SAMPLE_MARKUP: str = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"
DATASET_SIZE: int = 8


def _tiny_model() -> VisionAutoregressiveModel:
    torch.manual_seed(11)
    vocabulary = build_vocabulary([TikzTokens(markup=SAMPLE_MARKUP)])
    return VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=32,
        max_length=64,
        num_layers=1,
        num_heads=4,
    )


def _dataset(model: VisionAutoregressiveModel) -> tuple[torch.Tensor, torch.Tensor]:
    corpus = [TikzTokens(markup=SAMPLE_MARKUP)] * DATASET_SIZE
    tokens = batch_encode(corpus, model.vocabulary, max_length=32)
    images = torch.randn(DATASET_SIZE, 3, 32, 32)
    return images, tokens


def test_iter_batch_bounds_partitions_dataset() -> None:
    assert iter_batch_bounds(10, 3) == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert iter_batch_bounds(6, 6) == [(0, 6)]
    assert iter_batch_bounds(7, 4) == [(0, 4), (4, 7)]


def test_iter_batch_bounds_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        iter_batch_bounds(0, 4)
    with pytest.raises(ValueError):
        iter_batch_bounds(4, 0)


def test_train_one_epoch_returns_loss_per_batch() -> None:
    model = _tiny_model()
    images, tokens = _dataset(model)
    optimizer = build_adamw_optimizer(model, learning_rate=1e-3)
    criterion = TeacherForcingCrossEntropy()

    losses = train_one_epoch(model, optimizer, None, criterion, images, tokens, batch_size=4)

    assert len(losses) == DATASET_SIZE // 4
    assert all(isinstance(loss, float) for loss in losses)


def test_train_one_epoch_handles_partial_final_batch() -> None:
    model = _tiny_model()
    images, tokens = _dataset(model)
    optimizer = build_adamw_optimizer(model, learning_rate=1e-3)
    criterion = TeacherForcingCrossEntropy()

    losses = train_one_epoch(model, optimizer, None, criterion, images, tokens, batch_size=3)

    assert len(losses) == 3


def test_fit_records_epoch_and_step_losses() -> None:
    model = _tiny_model()
    images, tokens = _dataset(model)
    optimizer = build_adamw_optimizer(model, learning_rate=1e-3)
    criterion = TeacherForcingCrossEntropy()

    metrics = fit(model, optimizer, None, criterion, images, tokens, num_epochs=3, batch_size=4)

    assert len(metrics.epoch_losses) == 3
    assert len(metrics.step_losses) == 3 * (DATASET_SIZE // 4)
    assert all(isinstance(loss, float) for loss in metrics.step_losses)


def test_fit_reduces_teacher_forced_loss() -> None:
    torch.manual_seed(3)
    model = _tiny_model()
    images, tokens = _dataset(model)
    optimizer = build_adamw_optimizer(model, learning_rate=1e-2)
    criterion = TeacherForcingCrossEntropy()

    metrics = fit(
        model,
        optimizer,
        None,
        criterion,
        images,
        tokens,
        num_epochs=6,
        batch_size=4,
        seed=0,
    )

    assert metrics.epoch_losses[-1] < metrics.epoch_losses[0]


def test_fit_is_deterministic_given_seed() -> None:
    def run() -> TrainingMetrics:
        torch.manual_seed(7)
        model = _tiny_model()
        images, tokens = _dataset(model)
        optimizer = build_adamw_optimizer(model, learning_rate=1e-3)
        criterion = TeacherForcingCrossEntropy()
        return fit(
            model,
            optimizer,
            None,
            criterion,
            images,
            tokens,
            num_epochs=2,
            batch_size=4,
            seed=42,
        )

    first = run()
    second = run()

    assert len(first.step_losses) == len(second.step_losses)
    for expected, actual in zip(first.step_losses, second.step_losses, strict=True):
        assert actual == pytest.approx(expected)


def test_train_one_epoch_rejects_invalid_inputs() -> None:
    model = _tiny_model()
    images, tokens = _dataset(model)
    optimizer = build_adamw_optimizer(model)
    criterion = TeacherForcingCrossEntropy()

    with pytest.raises(TensorTopologyError):
        train_one_epoch(model, optimizer, None, criterion, images.unsqueeze(0), tokens, 4)
    with pytest.raises(TensorTopologyError):
        train_one_epoch(model, optimizer, None, criterion, images, tokens.float(), 4)
    with pytest.raises(TensorTopologyError):
        train_one_epoch(model, optimizer, None, criterion, images[:4], tokens, 4)
    with pytest.raises(ValueError):
        train_one_epoch(model, optimizer, None, criterion, images, tokens, 0)


def test_fit_rejects_invalid_epoch_count() -> None:
    model = _tiny_model()
    images, tokens = _dataset(model)
    optimizer = build_adamw_optimizer(model)
    criterion = TeacherForcingCrossEntropy()

    with pytest.raises(ValueError):
        fit(model, optimizer, None, criterion, images, tokens, num_epochs=0, batch_size=4)
