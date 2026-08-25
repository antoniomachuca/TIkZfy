"""Atomic checkpoint persistence adapter (model + optimizer state).

References:
    R. C. Martin, Clean Architecture — I/O lives at the boundary; the
        persistence adapter isolates disk writes from the domain.
    Atomic commit — the payload is serialized to a sibling temporary file and
        renamed onto the destination via ``os.replace`` (atomic on POSIX), so
        an interrupted write never corrupts an existing checkpoint.
"""

import os
import pickle
from pathlib import Path

import torch

from core.exceptions import DomainError, TensorTopologyError
from core.models import TrainingCheckpoint
from ports.outbound import CheckpointPersistencePort


class AtomicCheckpointAdapter(CheckpointPersistencePort):
    """PyTorch checkpoint adapter with atomic, corruption-safe persistence."""

    def save_checkpoint(self, checkpoint: TrainingCheckpoint, destination_path: str) -> None:
        """Serialize a checkpoint atomically to ``destination_path``.

        The payload is written to a temporary sibling file and renamed onto the
        destination, guaranteeing that a crash mid-write leaves any prior
        checkpoint intact.

        Temporal complexity: O(P) where P is the number of model parameters.
        """
        if not isinstance(checkpoint, TrainingCheckpoint):
            raise DomainError("Input must be a TrainingCheckpoint instance.")

        path: Path = Path(destination_path)
        temporary_path: Path = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, object] = {
                "model_state": checkpoint.model_state,
                "optimizer_state": checkpoint.optimizer_state,
                "epoch": checkpoint.epoch,
            }
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        except (OSError, TypeError, RuntimeError) as e:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise DomainError(f"Failed to save checkpoint to '{destination_path}': {str(e)}") from e

    def load_checkpoint(self, source_path: str) -> TrainingCheckpoint:
        """Load and validate a checkpoint from ``source_path``.

        Temporal complexity: O(P) where P is the number of model parameters.
        """
        try:
            path: Path = Path(source_path)
            if not path.exists():
                raise DomainError(f"Source path does not exist: '{source_path}'")

            payload: object = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise DomainError("Checkpoint payload must be a mapping.")

            return TrainingCheckpoint(
                model_state=payload["model_state"],
                optimizer_state=payload["optimizer_state"],
                epoch=payload["epoch"],
            )
        except (OSError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as e:
            raise DomainError(f"Failed to load checkpoint from '{source_path}': {str(e)}") from e
        except TensorTopologyError as e:
            raise DomainError(f"Loaded checkpoint violates invariants: {str(e)}") from e
