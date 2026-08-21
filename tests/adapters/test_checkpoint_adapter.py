from pathlib import Path

import pytest
import torch

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from core.exceptions import DomainError
from core.math.tokenization import build_vocabulary
from core.ml.checkpoint import restore_checkpoint, snapshot_checkpoint
from core.ml.loss import build_adamw_optimizer
from core.ml.model import VisionAutoregressiveModel
from core.models import TikzTokens, TrainingCheckpoint
from ports.outbound import CheckpointPersistencePort

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


def test_adapter_implements_checkpoint_port() -> None:
    assert isinstance(AtomicCheckpointAdapter(), CheckpointPersistencePort)


def test_save_load_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = build_adamw_optimizer(model, learning_rate=1e-3)
    reference_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    checkpoint = snapshot_checkpoint(model, optimizer, epoch=2)
    adapter = AtomicCheckpointAdapter()
    file_path = str(tmp_path / "checkpoint.pt")

    adapter.save_checkpoint(checkpoint, file_path)
    loaded = adapter.load_checkpoint(file_path)

    assert isinstance(loaded, TrainingCheckpoint)
    assert loaded.epoch == 2
    for name, tensor in reference_state.items():
        assert torch.equal(loaded.model_state[name], tensor)


def test_restore_checkpoint_resumes_model_state(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = build_adamw_optimizer(model, learning_rate=1e-3)
    adapter = AtomicCheckpointAdapter()
    file_path = str(tmp_path / "checkpoint.pt")

    adapter.save_checkpoint(snapshot_checkpoint(model, optimizer, epoch=1), file_path)

    restored_model = _tiny_model()
    restored_optimizer = build_adamw_optimizer(restored_model, learning_rate=1e-3)
    restore_checkpoint(
        restored_model, restored_optimizer, adapter.load_checkpoint(file_path)
    )

    for name, tensor in model.state_dict().items():
        assert torch.equal(restored_model.state_dict()[name], tensor)


def test_save_is_atomic_without_temporary_artifacts(tmp_path: Path) -> None:
    model = _tiny_model()
    optimizer = build_adamw_optimizer(model)
    adapter = AtomicCheckpointAdapter()
    file_path = tmp_path / "nested" / "checkpoint.pt"

    adapter.save_checkpoint(snapshot_checkpoint(model, optimizer, epoch=0), str(file_path))

    assert file_path.exists()
    assert not file_path.with_name(file_path.name + ".tmp").exists()


def test_load_nonexistent_checkpoint_raises_domain_error(tmp_path: Path) -> None:
    adapter = AtomicCheckpointAdapter()

    with pytest.raises(DomainError):
        adapter.load_checkpoint(str(tmp_path / "missing.pt"))


def test_load_corrupt_checkpoint_raises_domain_error(tmp_path: Path) -> None:
    file_path = tmp_path / "corrupt.pt"
    file_path.write_bytes(b"not a torch payload")
    adapter = AtomicCheckpointAdapter()

    with pytest.raises(DomainError):
        adapter.load_checkpoint(str(file_path))


def test_save_rejects_non_checkpoint_input(tmp_path: Path) -> None:
    adapter = AtomicCheckpointAdapter()

    with pytest.raises(DomainError):
        adapter.save_checkpoint("not-a-checkpoint", str(tmp_path / "x.pt"))  # type: ignore[arg-type]
