import pytest
import torch

from core.math.tokenization import build_vocabulary
from core.ml.checkpoint import restore_checkpoint, snapshot_checkpoint
from core.ml.loss import build_adamw_optimizer
from core.ml.model import VisionAutoregressiveModel
from core.models import TikzTokens, TrainingCheckpoint

SAMPLE_MARKUP: str = r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}"


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


def test_snapshot_checkpoint_captures_state_and_epoch() -> None:
    model = _tiny_model()
    optimizer = build_adamw_optimizer(model, learning_rate=1e-3)

    checkpoint = snapshot_checkpoint(model, optimizer, epoch=4)

    assert isinstance(checkpoint, TrainingCheckpoint)
    assert checkpoint.epoch == 4
    assert set(checkpoint.model_state.keys()) == set(model.state_dict().keys())
    assert "param_groups" in checkpoint.optimizer_state


def test_snapshot_checkpoint_rejects_invalid_arguments() -> None:
    model = _tiny_model()
    optimizer = build_adamw_optimizer(model)

    with pytest.raises(TypeError):
        snapshot_checkpoint("not-a-module", optimizer, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        snapshot_checkpoint(model, "not-an-optimizer", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        snapshot_checkpoint(model, optimizer, -1)


def test_restore_checkpoint_requires_checkpoint_instance() -> None:
    model = _tiny_model()
    optimizer = build_adamw_optimizer(model)

    with pytest.raises(TypeError):
        restore_checkpoint(model, optimizer, "not-a-checkpoint")  # type: ignore[arg-type]
