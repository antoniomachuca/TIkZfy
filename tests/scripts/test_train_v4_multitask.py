"""Smoke integration test for V4 curriculum training script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from core.dataset.sharded import DatasetV4Manifest, ShardMetadata, save_shard
from core.math.tokenization import build_vocabulary
from core.models import TikzTokens
from scripts.train_v4_multitask import execute_v4_curriculum_training


def test_train_v4_multitask_smoke_execution() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        data_dir = root / "data"
        results_dir = root / "results"
        data_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)

        vocab = build_vocabulary(
            [TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}")]
        )
        vocab_path = data_dir / "vocabulary_v4.json"
        with vocab_path.open("w", encoding="utf-8") as f:
            json.dump(vocab.token_to_index, f)

        # Create train and val shards
        for split_name in ("train", "val"):
            split_dir = data_dir / split_name
            split_dir.mkdir(parents=True)
            shard_path = split_dir / f"{split_name}_shard_0000.pt"
            # 4 samples, 256x256, 16 tokens
            images = torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8)
            tokens = torch.zeros(4, 16, dtype=torch.long)
            labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            save_shard(shard_path, images, tokens, labels)

            shard_meta = ShardMetadata(
                shard_id=0,
                file_name=f"{split_name}/{split_name}_shard_0000.pt",
                num_samples=4,
                family_counts={"line_segment": 1, "polyline": 1, "polygon": 1, "circle_arc": 1},
            )
            manifest = DatasetV4Manifest(
                version="v4",
                image_size=256,
                max_length=16,
                total_samples=4,
                shards=[shard_meta],
            )
            manifest_path = data_dir / f"manifest_{split_name}.json"
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f)

        args = argparse.Namespace(
            data_dir=str(data_dir),
            vocab_path=str(vocab_path),
            results_dir=str(results_dir),
            batch_size=2,
            grad_accum_steps=1,
            model_dim=32,
            max_length=16,
            num_layers=1,
            num_heads=2,
            dim_ff=64,
            num_encoder_blocks=1,
            dropout=0.0,
            weight_decay=0.0,
            lambda_coord=1.0,
            lambda_family=1.5,
            lambda_spatial=2.0,
            label_smoothing=0.05,
            sigma=0.20,
            huber_beta=0.10,
            cache_size=2,
            num_workers=0,
            max_samples_per_epoch=4,
            start_stage=3,  # Run stage 3 only (1 epoch via monkeypatch)
            resume=None,
            seed=42,
            device="cpu",
            auto_shutdown=False,
        )

        import scripts.train_v4_multitask
        from scripts.train_v4_multitask import CurriculumStage

        # Override canonical stage 3 to run 1 epoch for smoke test
        test_stage = CurriculumStage(
            stage_idx=3,
            name="Smoke Test Stage",
            num_epochs=1,
            learning_rate=1e-3,
            family_weights=(0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0),
            enable_augmentation=False,
            description="Quick test",
        )
        original_stages = scripts.train_v4_multitask.CANONICAL_STAGES
        scripts.train_v4_multitask.CANONICAL_STAGES = (test_stage,)

        try:
            execute_v4_curriculum_training(args)
        finally:
            scripts.train_v4_multitask.CANONICAL_STAGES = original_stages

        # Verify artifacts
        latest_cp_path = results_dir / "checkpoints" / "curriculum_v4_latest.pt"
        telemetry_path = results_dir / "telemetry.json"
        log_path = results_dir / "train_v4.log"

        assert latest_cp_path.exists()
        assert telemetry_path.exists()
        assert log_path.exists()

        with telemetry_path.open("r", encoding="utf-8") as f:
            telemetry = json.load(f)
        assert len(telemetry) == 1
        assert telemetry[0]["stage_idx"] == 3
        assert telemetry[0]["epoch"] == 1
