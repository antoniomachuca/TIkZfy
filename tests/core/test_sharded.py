"""Unit tests for sharded dataset storage and streaming primitives."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from core.dataset.sharded import (
    DatasetV4Manifest,
    ShardedTikzDataset,
    ShardMetadata,
    save_shard,
)
from core.exceptions import TensorTopologyError


def test_save_shard_validates_tensor_topology() -> None:
    with TemporaryDirectory() as temp_dir:
        shard_path = Path(temp_dir) / "shard_000.pt"

        # Valid inputs
        images = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        tokens = torch.zeros((4, 16), dtype=torch.long)
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        save_shard(shard_path, images, tokens, labels)
        assert shard_path.exists()

        # Invalid shape: 3D images
        with pytest.raises(TensorTopologyError):
            save_shard(shard_path, images[0], tokens, labels)

        # Mismatched sample counts
        with pytest.raises(TensorTopologyError):
            save_shard(shard_path, images, tokens[:2], labels)


def test_manifest_serialization_roundtrip() -> None:
    shards = [
        ShardMetadata(
            shard_id=0,
            file_name="shard_000.pt",
            num_samples=100,
            family_counts={"line_segment": 50, "circle_arc": 50},
        ),
        ShardMetadata(
            shard_id=1,
            file_name="shard_001.pt",
            num_samples=100,
            family_counts={"grid_axes": 50, "node_arrow": 50},
        ),
    ]
    manifest = DatasetV4Manifest(
        version="v4",
        image_size=256,
        max_length=512,
        total_samples=200,
        shards=shards,
    )

    data_dict = manifest.to_dict()
    reconstructed = DatasetV4Manifest.from_dict(data_dict)

    assert reconstructed.version == "v4"
    assert reconstructed.image_size == 256
    assert reconstructed.total_samples == 200
    assert len(reconstructed.shards) == 2
    assert reconstructed.shards[0].file_name == "shard_000.pt"
    assert reconstructed.shards[1].family_counts["node_arrow"] == 50


def test_sharded_tikz_dataset_streaming_and_caching() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        # Create 3 shards with 10 samples each
        shards_meta: list[ShardMetadata] = []
        for s_id in range(3):
            file_name = f"shard_{s_id:03d}.pt"
            shard_file = root / file_name
            images = torch.full((10, 3, 32, 32), fill_value=s_id * 50, dtype=torch.uint8)
            tokens = torch.full((10, 8), fill_value=s_id, dtype=torch.long)
            labels = torch.full((10,), fill_value=s_id, dtype=torch.long)
            save_shard(shard_file, images, tokens, labels)

            shards_meta.append(
                ShardMetadata(
                    shard_id=s_id,
                    file_name=file_name,
                    num_samples=10,
                    family_counts={"test": 10},
                )
            )

        manifest = DatasetV4Manifest(
            version="v4",
            image_size=32,
            max_length=8,
            total_samples=30,
            shards=shards_meta,
        )
        manifest_path = root / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f)

        # Instantiate dataset with cache_size = 2
        dataset = ShardedTikzDataset(manifest_path, cache_size=2)

        assert len(dataset) == 30

        # Sample 0 (from shard 0)
        img0, tok0, fam0 = dataset[0]
        assert img0.shape == (3, 32, 32)
        assert img0.dtype == torch.float32
        assert torch.allclose(img0, torch.zeros_like(img0))
        assert tok0.tolist() == [0] * 8
        assert fam0 == 0

        # Sample 15 (from shard 1)
        img15, tok15, fam15 = dataset[15]
        assert tok15.tolist() == [1] * 8
        assert fam15 == 1

        # Sample 25 (from shard 2) - should evict shard 0
        img25, tok25, fam25 = dataset[25]
        assert tok25.tolist() == [2] * 8
        assert fam25 == 2

        # Verify cache size bound
        assert len(dataset._loaded_shards) == 2
        assert 0 not in dataset._loaded_shards  # Evicted
        assert 1 in dataset._loaded_shards
        assert 2 in dataset._loaded_shards

        # Out of bounds
        with pytest.raises(IndexError):
            _ = dataset[30]
        with pytest.raises(IndexError):
            _ = dataset[-1]
