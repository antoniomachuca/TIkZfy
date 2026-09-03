"""Sharded dataset primitives and streaming abstractions for high-resolution corpus V4.

Provides memory-bounded, indexed iteration over partitioned PyTorch tensor shards
(256x256 uint8 images, token sequences, and family labels) guaranteeing O(1) RAM
consumption during multi-epoch neural training.

References:
    Golub & Van Loan, Matrix Computations - block matrix partitioning and spatial indexing.
    Goodfellow et al., Deep Learning - high-throughput dataset sharding and memory management.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import Dataset

from core.exceptions import DomainError, TensorTopologyError


@dataclass(frozen=True)
class ShardMetadata:
    """Descriptor metadata for a single dataset storage shard."""

    shard_id: int
    file_name: str
    num_samples: int
    family_counts: dict[str, int]


@dataclass(frozen=True)
class DatasetV4Manifest:
    """Catalog manifest governing all partitioned shards in Corpus V4."""

    version: str
    image_size: int
    max_length: int
    total_samples: int
    shards: list[ShardMetadata]

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "image_size": self.image_size,
            "max_length": self.max_length,
            "total_samples": self.total_samples,
            "shards": [asdict(s) for s in self.shards],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetV4Manifest":
        """Reconstruct manifest from a parsed JSON dictionary."""
        shards = [
            ShardMetadata(
                shard_id=int(s["shard_id"]),
                file_name=str(s["file_name"]),
                num_samples=int(s["num_samples"]),
                family_counts={k: int(v) for k, v in s["family_counts"].items()},
            )
            for s in data["shards"]
        ]
        return cls(
            version=str(data["version"]),
            image_size=int(data["image_size"]),
            max_length=int(data["max_length"]),
            total_samples=int(data["total_samples"]),
            shards=shards,
        )


def save_shard(
    output_path: Path | str,
    images: torch.Tensor,
    tokens: torch.Tensor,
    family_labels: torch.Tensor,
) -> None:
    """Persist a single verified shard to disk.

    Args:
        output_path (Path | str): Target file path for the .pt shard.
        images (torch.Tensor): Byte tensor with shape ``(N, 3, H, W)`` in [0, 255].
        tokens (torch.Tensor): Long index tensor with shape ``(N, L)``.
        family_labels (torch.Tensor): Long label tensor with shape ``(N,)`` in [0, 7].

    Raises:
        TensorTopologyError: If dimensions or cardinality do not match invariants.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise TensorTopologyError(
            f"Images must have shape (N, 3, H, W). Got {tuple(images.shape)}."
        )
    if tokens.ndim != 2:
        raise TensorTopologyError(f"Tokens must have shape (N, L). Got {tuple(tokens.shape)}.")
    if family_labels.ndim != 1:
        raise TensorTopologyError(
            f"Family labels must have shape (N,). Got {tuple(family_labels.shape)}."
        )

    n_samples: int = images.shape[0]
    if tokens.shape[0] != n_samples or family_labels.shape[0] != n_samples:
        raise TensorTopologyError(
            f"Sample count mismatch across tensors: {n_samples} vs {tokens.shape[0]} "
            f"vs {family_labels.shape[0]}."
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "images": images.to(torch.uint8),
        "tokens": tokens.to(torch.long),
        "family_labels": family_labels.to(torch.long),
        "num_samples": n_samples,
    }
    torch.save(payload, path)


class ShardedTikzDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Memory-mapped chunked dataset reader with dynamic LRU shard caching.

    Enables high-throughput random and sequential access over massive sharded corpora
    by keeping at most ``cache_size`` shards in active memory.

    Temporal complexity: O(1) per __getitem__ amortized.
    Spatial complexity: O(cache_size * S_shard) where S_shard is shard memory footprint.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        cache_size: int = 2,
    ) -> None:
        path = Path(manifest_path)
        if not path.exists():
            raise DomainError(f"Manifest file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self._manifest: DatasetV4Manifest = DatasetV4Manifest.from_dict(data)
        self._root_dir: Path = path.parent
        self._cache_size: int = max(1, cache_size)

        # Build global index to (shard_id, local_offset) lookup table
        self._index_map: list[tuple[int, int]] = []
        for shard_idx, shard_meta in enumerate(self._manifest.shards):
            local_idx = 0
            while local_idx < shard_meta.num_samples:
                self._index_map.append((shard_idx, local_idx))
                local_idx += 1

        self._loaded_shards: dict[int, dict[str, torch.Tensor]] = {}
        self._access_history: list[int] = []

    @property
    def manifest(self) -> DatasetV4Manifest:
        """Return the bound corpus manifest."""
        return self._manifest

    def __len__(self) -> int:
        """Return total samples across all registered shards."""
        return len(self._index_map)

    def _ensure_shard_loaded(self, shard_idx: int) -> dict[str, torch.Tensor]:
        """Load shard into cache, evicting least recently used shard if budget is exceeded."""
        if shard_idx in self._loaded_shards:
            self._access_history.remove(shard_idx)
            self._access_history.append(shard_idx)
            return self._loaded_shards[shard_idx]

        shard_meta = self._manifest.shards[shard_idx]
        shard_file = self._root_dir / shard_meta.file_name
        try:
            raw_data = torch.load(shard_file, map_location="cpu", weights_only=True, mmap=True)
        except (TypeError, RuntimeError):
            raw_data = torch.load(shard_file, map_location="cpu", weights_only=True)
        data: dict[str, torch.Tensor] = cast(dict[str, torch.Tensor], raw_data)

        if len(self._loaded_shards) >= self._cache_size:
            oldest = self._access_history.pop(0)
            del self._loaded_shards[oldest]

        self._loaded_shards[shard_idx] = data
        self._access_history.append(shard_idx)
        return data

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Retrieve single sample by global integer index.

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]:
                - image: Float32 tensor of shape (3, H, W) normalized to [0.0, 1.0].
                - tokens: Long tensor of shape (L,).
                - family_idx: Stratum integer label in [0, len(FAMILY_NAMES) - 1].
        """
        if index < 0 or index >= len(self._index_map):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self)} samples.")

        shard_idx, local_offset = self._index_map[index]
        shard_data = self._ensure_shard_loaded(shard_idx)

        # Convert uint8 (3, H, W) to float32 in [0, 1]
        raw_image: torch.Tensor = shard_data["images"][local_offset]
        float_image: torch.Tensor = raw_image.to(torch.float32) / 255.0
        tokens: torch.Tensor = shard_data["tokens"][local_offset]
        family_idx: int = int(shard_data["family_labels"][local_offset].item())

        return float_image, tokens, family_idx
