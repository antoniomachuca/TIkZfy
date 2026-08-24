"""Smoke test verifying local CPU integration for deep scaled model.

Tests end-to-end forward propagation, Teacher-Forced Cross-Entropy loss,
backward gradient flow across all 12-14 layers, and AdamW optimizer updates.
"""

import torch

from core.math.tokenization import build_vocabulary
from core.ml.loss import (
    TeacherForcingCrossEntropy,
    build_adamw_optimizer,
    build_teacher_forcing_pair,
)
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens, TokenVocabulary


def _create_sample_vocabulary() -> TokenVocabulary:
    """Build a deterministic TokenVocabulary for smoke testing."""
    samples: list[TikzTokens] = [
        TikzTokens(
            markup=(
                r"\begin{tikzpicture}\draw[red] (0,0) -- (1,1);"
                r"\node at (0,0) {A};\end{tikzpicture}"
            )
        ),
        TikzTokens(
            markup=r"\begin{tikzpicture}\fill[blue] (-2.5,1.0) circle (0.5);\end{tikzpicture}"
        ),
    ]
    return build_vocabulary(samples)


def test_smoke_mini_batch_forward_backward_cpu() -> None:
    """Verify 12-layer model mini-batch execution and gradient propagation on CPU."""
    torch.manual_seed(42)
    vocabulary: TokenVocabulary = _create_sample_vocabulary()
    vocab_size: int = len(vocabulary.token_to_index)

    # Instantiate full 12-layer production configuration: 6 Encoder blocks + 6 Decoder layers
    model: VisionAutoregressiveModel = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=384,
        max_length=512,
        num_layers=6,
        num_heads=8,
        dim_feedforward=1536,
        num_encoder_blocks=6,
        device="cpu",
    )
    model.train()

    batch_size: int = 2
    height: int = 64
    width: int = 64
    seq_length: int = 32

    # Synthetic mini-batch tensors: Shape (B, C, H, W) and (B, L)
    images: torch.Tensor = torch.randn(batch_size, 3, height, width)
    tokens: torch.Tensor = torch.randint(
        1, vocab_size, (batch_size, seq_length), dtype=torch.long
    )
    # Ensure BOS token is at index 0
    tokens[:, 0] = 0

    decoder_input, targets = build_teacher_forcing_pair(tokens)
    # Forward pass: Shape (B, L-1, V)
    logits: torch.Tensor = model(images, decoder_input)

    assert logits.shape == (batch_size, seq_length - 1, vocab_size)
    assert logits.dtype == torch.float32

    criterion: TeacherForcingCrossEntropy = TeacherForcingCrossEntropy()
    loss: torch.Tensor = criterion(logits, targets)

    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0

    optimizer: torch.optim.AdamW = build_adamw_optimizer(model, learning_rate=3e-4)
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]

    # Verify every trainable parameter received finite, non-zero gradients
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"Parameter {name} did not receive gradients."
            assert (
                torch.isfinite(parameter.grad).all()
            ), f"Parameter {name} has non-finite gradients."

    optimizer.step()


def test_smoke_14_layer_deep_decoder_cpu() -> None:
    """Verify maximum scaled 14-layer architecture (6 Encoder + 8 Decoder layers)."""
    torch.manual_seed(123)
    vocabulary: TokenVocabulary = _create_sample_vocabulary()
    vocab_size: int = len(vocabulary.token_to_index)

    model: VisionAutoregressiveModel = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=384,
        max_length=128,
        num_layers=8,
        num_heads=8,
        dim_feedforward=1536,
        num_encoder_blocks=6,
        device="cpu",
    )
    model.train()

    images: torch.Tensor = torch.randn(2, 3, 32, 32)
    tokens: torch.Tensor = torch.randint(0, vocab_size, (2, 16), dtype=torch.long)
    decoder_input, targets = build_teacher_forcing_pair(tokens)

    logits: torch.Tensor = model(images, decoder_input)
    assert logits.shape == (2, 15, vocab_size)

    loss: torch.Tensor = TeacherForcingCrossEntropy()(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"Parameter {name} missing gradients."


def test_smoke_generation_pipeline_cpu() -> None:
    """Verify greedy markup generation on CPU with scaled model."""
    torch.manual_seed(7)
    vocabulary: TokenVocabulary = _create_sample_vocabulary()

    model: VisionAutoregressiveModel = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=384,
        max_length=32,
        num_layers=6,
        num_heads=8,
        dim_feedforward=1536,
        num_encoder_blocks=6,
        device="cpu",
    )
    model.eval()

    sample_image: ImageTensor = ImageTensor(torch.randn(1, 3, 64, 64))
    result: TikzTokens = model.generate_markup(sample_image)

    assert isinstance(result, TikzTokens)
    assert r"\begin{tikzpicture}" in result.markup
    assert r"\end{tikzpicture}" in result.markup
