import torch

from core.math.tokenization import build_vocabulary
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens, TokenVocabulary


def _vocabulary() -> TokenVocabulary:
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0);\end{tikzpicture}")
    return build_vocabulary([sample])


def _model(max_length: int = 16) -> VisionAutoregressiveModel:
    torch.manual_seed(7)
    return VisionAutoregressiveModel(
        vocabulary=_vocabulary(),
        input_channels=3,
        model_dimension=32,
        max_length=max_length,
        num_layers=1,
        num_heads=4,
    )


def test_model_preserves_teacher_forced_shapes() -> None:
    model = _model()
    images = torch.randn(2, 3, 32, 32)
    target_tokens = torch.ones(2, 16, dtype=torch.long)

    logits = model(images, target_tokens)

    assert logits.shape == (2, 16, len(model.vocabulary.token_to_index))
    assert logits.dtype == torch.float32


def test_decoder_is_causally_masked() -> None:
    model = _model(max_length=8)
    model.eval()
    images = torch.randn(1, 3, 32, 32)
    baseline_targets = torch.arange(8, dtype=torch.long).remainder(
        len(model.vocabulary.token_to_index)
    ).unsqueeze(0)
    modified_targets = baseline_targets.clone()
    modified_targets[:, 5:] = 0

    baseline_logits = model(images, baseline_targets)
    modified_logits = model(images, modified_targets)

    assert torch.allclose(baseline_logits[:, :5, :], modified_logits[:, :5, :])


def test_model_generation_returns_valid_tikz_tokens() -> None:
    model = _model(max_length=4)

    generated = model.generate_markup(ImageTensor(torch.randn(1, 3, 32, 32)))

    assert isinstance(generated, TikzTokens)
    assert r"\begin{tikzpicture}" in generated.markup
    assert r"\end{tikzpicture}" in generated.markup


def test_conv_residual_block_preserves_spatial_shape_and_flow() -> None:
    from core.ml.model import ConvResidualBlock

    block = ConvResidualBlock(channels=32)
    inputs = torch.randn(2, 32, 8, 8, requires_grad=True)
    outputs = block(inputs)

    assert outputs.shape == (2, 32, 8, 8)
    assert outputs.dtype == torch.float32

    loss = outputs.sum()
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_vision_encoder_six_blocks_default_topology() -> None:
    from core.ml.model import VisionEncoder

    encoder = VisionEncoder(input_channels=3, model_dimension=32)
    assert len(encoder.residual_blocks) == 6

    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    tokens = encoder(images)

    # 32 / 4 = 8 -> 8 * 8 = 64 visual tokens
    assert tokens.shape == (2, 64, 32)
    assert tokens.dtype == torch.float32

    loss = tokens.sum()
    loss.backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def test_vision_encoder_custom_block_count() -> None:
    from core.ml.model import VisionEncoder

    encoder = VisionEncoder(input_channels=3, model_dimension=16, num_blocks=4)
    assert len(encoder.residual_blocks) == 4

    images = torch.randn(1, 3, 16, 16)
    tokens = encoder(images)
    # 16 / 4 = 4 -> 4 * 4 = 16 tokens
    assert tokens.shape == (1, 16, 16)

