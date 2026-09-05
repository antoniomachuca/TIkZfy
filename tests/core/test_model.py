import pytest
import torch
import torch.nn.functional as F

from core.exceptions import TensorTopologyError
from core.math.tokenization import build_vocabulary
from core.ml.model import (
    AutoregressiveDecoder,
    ConvResidualBlock,
    VisionAutoregressiveModel,
    VisionAutoregressiveModelV4,
    VisionEncoder,
    VisionEncoderV4,
    build_2d_sinusoidal_positional_encoding,
    resolve_device,
)
from core.models import FAMILY_NAMES, ImageTensor, TikzTokens, TokenVocabulary


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
        num_downsampling_stages=3,
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
    baseline_targets = (
        torch.arange(8, dtype=torch.long)
        .remainder(len(model.vocabulary.token_to_index))
        .unsqueeze(0)
    )
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
    encoder = VisionEncoder(input_channels=3, model_dimension=32)
    assert len(encoder.residual_blocks) == 6

    # Canonical 256x256 input under 3 downsampling stages: 256 / 8 = 32 -> 32 * 32 = 1024 tokens
    images_256 = torch.randn(2, 3, 256, 256, requires_grad=True)
    tokens_256 = encoder(images_256)

    assert tokens_256.shape == (2, 1024, 32)
    assert tokens_256.dtype == torch.float32

    loss = tokens_256.sum()
    loss.backward()
    assert images_256.grad is not None
    assert torch.isfinite(images_256.grad).all()


def test_vision_encoder_custom_block_count() -> None:
    encoder = VisionEncoder(
        input_channels=3,
        model_dimension=16,
        num_blocks=4,
        num_downsampling_stages=2,
    )
    assert len(encoder.residual_blocks) == 4

    images = torch.randn(1, 3, 16, 16)
    tokens = encoder(images)
    # 16 / 4 = 4 -> 4 * 4 = 16 tokens
    assert tokens.shape == (1, 16, 16)


def test_autoregressive_decoder_scaling_and_feedforward() -> None:
    decoder_6 = AutoregressiveDecoder(
        vocabulary_size=100,
        model_dimension=384,
        max_length=64,
        num_layers=6,
        num_heads=8,
        dim_feedforward=1536,
    )
    assert decoder_6.model_dimension == 384
    assert decoder_6.num_layers == 6
    assert decoder_6.num_heads == 8
    assert decoder_6.dim_feedforward == 1536

    decoder_8 = AutoregressiveDecoder(
        vocabulary_size=100,
        model_dimension=384,
        max_length=64,
        num_layers=8,
        num_heads=8,
        dim_feedforward=1536,
    )
    assert decoder_8.num_layers == 8

    visual_tokens = torch.randn(2, 64, 384)
    target_tokens = torch.randint(0, 100, (2, 32))
    logits = decoder_6(visual_tokens, target_tokens)
    assert logits.shape == (2, 32, 100)


def test_resolve_device_and_cuda_assignment() -> None:
    auto_device = resolve_device(None)
    assert isinstance(auto_device, torch.device)
    assert auto_device.type in ("cuda", "cpu")

    cpu_device = resolve_device("cpu")
    assert cpu_device == torch.device("cpu")


def test_vision_autoregressive_model_scaled_architecture() -> None:
    vocab = _vocabulary()
    model = VisionAutoregressiveModel(
        vocabulary=vocab,
        input_channels=3,
        model_dimension=384,
        max_length=64,
        num_layers=6,
        num_heads=8,
        dim_feedforward=1536,
        num_encoder_blocks=6,
        device="cpu",
    )

    images = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, len(vocab.token_to_index), (2, 16))
    logits = model(images, targets)

    assert logits.shape == (2, 16, len(vocab.token_to_index))
    assert model.target_device == torch.device("cpu")


def test_build_2d_sinusoidal_positional_encoding() -> None:
    pe = build_2d_sinusoidal_positional_encoding(
        height=8, width=8, dimension=64, device=torch.device("cpu")
    )
    assert pe.shape == (1, 64, 64)
    assert not torch.isnan(pe).any()
    assert not torch.allclose(pe[0, 0, :], pe[0, 1, :])
    assert not torch.allclose(pe[0, 0, :], pe[0, 8, :])


def test_vision_encoder_v4_spatial_resolution_256_to_1024_tokens() -> None:
    """Validate 256x256 image downsampling through 3 stride-2 stages yields exactly 1024 tokens."""
    encoder = VisionEncoderV4(
        input_channels=3,
        model_dimension=64,
        num_blocks=6,
        num_downsampling_stages=3,
    )
    images = torch.randn(4, 3, 256, 256, dtype=torch.float32)

    # Forward pass through encoder
    tokens = encoder(images)

    # Verification: 256 / 2^3 = 32 -> 32 * 32 = 1024 tokens
    assert tokens.shape == (4, 1024, 64)
    assert tokens.dtype == torch.float32
    assert not torch.isnan(tokens).any()


def test_vision_encoder_auxiliary_family_head_shapes_and_gradients() -> None:
    """Validate auxiliary family classification head output shapes and full gradient flow."""
    model_dim: int = 64
    num_families: int = len(FAMILY_NAMES)
    encoder = VisionEncoder(
        input_channels=3,
        model_dimension=model_dim,
        num_blocks=2,
        num_downsampling_stages=3,
        num_families=num_families,
    )

    assert encoder.family_head is not None
    assert encoder.family_head.weight.shape == (num_families, model_dim)
    assert encoder.family_head.bias.shape == (num_families,)

    images = torch.randn(2, 3, 256, 256, requires_grad=True)
    family_logits = encoder.classify_family(images)

    assert family_logits.shape == (2, num_families)
    assert family_logits.dtype == torch.float32

    # Supervised classification loss on auxiliary head
    target_families = torch.tensor([0, 6], dtype=torch.long)
    loss = F.cross_entropy(family_logits, target_families)
    loss.backward()  # type: ignore[no-untyped-call]

    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert encoder.family_head.weight.grad is not None
    assert torch.isfinite(encoder.family_head.weight.grad).all()


def test_vision_encoder_summary_token_computation_and_equivalence() -> None:
    """Validate GAP produces identical summary tokens from 4D map or 3D sequence."""
    features_4d = torch.randn(2, 32, 16, 16)
    features_3d = features_4d.reshape(2, 32, 256).transpose(1, 2)

    summary_from_4d = VisionEncoder.compute_summary_token(features_4d)
    summary_from_3d = VisionEncoder.compute_summary_token(features_3d)

    assert summary_from_4d.shape == (2, 32)
    assert summary_from_3d.shape == (2, 32)
    assert torch.allclose(summary_from_4d, summary_from_3d, atol=1e-6)

    # Invalid topology rejection
    with pytest.raises(TensorTopologyError):
        VisionEncoder.compute_summary_token(torch.randn(2, 32))


def test_vision_encoder_multi_task_forward_modes() -> None:
    """Validate all four return permutations of VisionEncoder.forward."""
    encoder = VisionEncoder(
        input_channels=3,
        model_dimension=32,
        num_blocks=2,
        num_downsampling_stages=3,
        num_families=8,
    )
    images = torch.randn(2, 3, 256, 256)

    # Mode 1: Default tokens only
    tokens = encoder(images)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (2, 1024, 32)

    # Mode 2: Summary token requested
    tokens_sub, summary = encoder(images, return_summary=True)
    assert tokens_sub.shape == (2, 1024, 32)
    assert summary.shape == (2, 32)

    # Mode 3: Family logits requested
    tokens_fam, fam_logits = encoder(images, return_family_logits=True)
    assert tokens_fam.shape == (2, 1024, 32)
    assert fam_logits.shape == (2, 8)

    # Mode 4: Both summary and family logits requested
    tokens_all, summary_all, fam_all = encoder(
        images, return_summary=True, return_family_logits=True
    )
    assert tokens_all.shape == (2, 1024, 32)
    assert summary_all.shape == (2, 32)
    assert fam_all.shape == (2, 8)


def test_vision_encoder_family_head_error_when_unconfigured() -> None:
    """Verify TensorTopologyError when requesting family logits from unconfigured encoder."""
    encoder = VisionEncoder(
        input_channels=3,
        model_dimension=32,
        num_blocks=2,
        num_downsampling_stages=3,
        num_families=None,
    )
    images = torch.randn(1, 3, 64, 64)

    assert encoder.family_head is None
    with pytest.raises(TensorTopologyError):
        encoder.classify_family(images)
    with pytest.raises(TensorTopologyError):
        encoder(images, return_family_logits=True)


def test_vision_autoregressive_model_v4_multitask_shapes() -> None:
    """Validate VisionAutoregressiveModelV4 multi-task forward shapes and prediction API."""
    vocab = _vocabulary()
    num_families = len(FAMILY_NAMES)
    model = VisionAutoregressiveModelV4(
        vocabulary=vocab,
        input_channels=3,
        model_dimension=32,
        max_length=16,
        num_layers=1,
        num_heads=4,
        num_downsampling_stages=3,
        num_families=num_families,
    )

    images = torch.randn(2, 3, 256, 256)
    targets = torch.ones(2, 16, dtype=torch.long)

    # Single-task forward returns token logits: Shape (B, L, V)
    token_logits = model(images, targets)
    assert token_logits.shape == (2, 16, len(vocab.token_to_index))

    # Multi-task forward returns (token_logits, family_logits)
    token_logits_mt, family_logits = model(images, targets, return_family_logits=True)
    assert token_logits_mt.shape == (2, 16, len(vocab.token_to_index))
    assert family_logits.shape == (2, num_families)

    # Direct predict_family API
    family_preds = model.predict_family(images)
    assert family_preds.shape == (2, num_families)
    assert model.family_head is not None


def test_vision_autoregressive_model_v4_multitask_loss_backward() -> None:
    """Validate joint backpropagation of syntax cross-entropy and family classification loss."""
    vocab = _vocabulary()
    num_families = len(FAMILY_NAMES)
    model = VisionAutoregressiveModel(
        vocabulary=vocab,
        input_channels=3,
        model_dimension=32,
        max_length=16,
        num_layers=1,
        num_heads=4,
        num_downsampling_stages=3,
        num_families=num_families,
    )
    model.train()

    images = torch.randn(2, 3, 256, 256, requires_grad=True)
    targets = torch.zeros(2, 8, dtype=torch.long)
    family_targets = torch.tensor([1, 7], dtype=torch.long)

    token_logits, family_logits = model(images, targets, return_family_logits=True)

    syntax_loss = F.cross_entropy(token_logits.transpose(1, 2), targets)
    family_loss = F.cross_entropy(family_logits, family_targets)
    composite_loss = syntax_loss + 1.5 * family_loss
    composite_loss.backward()  # type: ignore[no-untyped-call]

    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert model.family_head is not None
    assert model.family_head.weight.grad is not None
    assert torch.isfinite(model.family_head.weight.grad).all()
    assert model.decoder.output_projection.weight.grad is not None
    assert torch.isfinite(model.decoder.output_projection.weight.grad).all()


def test_vision_autoregressive_model_causality_with_1024_tokens() -> None:
    """Validate causal masking strictly prevents information leakage with 1024 tokens."""
    model = _model(max_length=12)
    model.eval()

    # Canonical 256x256 image -> 1024 visual tokens
    images = torch.randn(1, 3, 256, 256)
    baseline_targets = torch.randint(0, len(model.vocabulary.token_to_index), (1, 12))
    modified_targets = baseline_targets.clone()

    # Perturb targets from index 7 onward
    modified_targets[:, 7:] = (modified_targets[:, 7:] + 1).remainder(
        len(model.vocabulary.token_to_index)
    )

    baseline_logits = model(images, baseline_targets)
    modified_logits = model(images, modified_targets)

    # Causality invariant: past token logits (indices 0..6) must be identical
    assert torch.allclose(baseline_logits[:, :7, :], modified_logits[:, :7, :], atol=1e-6)


def test_vision_autoregressive_model_generate_markup_with_family_prefix() -> None:
    """Validate generate_markup with family prefix injection enabled."""
    model = _model(max_length=6)
    image = ImageTensor(torch.randn(1, 3, 256, 256))

    generated = model.generate_markup(image, inject_family_prefix=True)

    assert isinstance(generated, TikzTokens)
    assert r"\begin{tikzpicture}" in generated.markup
    assert r"\end{tikzpicture}" in generated.markup


def test_vision_autoregressive_model_backward_compat_checkpoint_loading() -> None:
    """Validate state_dict loading succeeds when pre-V4 checkpoint lacks family_head."""
    model_without_head = VisionAutoregressiveModel(
        vocabulary=_vocabulary(),
        input_channels=3,
        model_dimension=32,
        max_length=16,
        num_layers=1,
        num_heads=4,
        num_families=None,
    )
    legacy_state = model_without_head.state_dict()
    assert not any("family_head" in k for k in legacy_state.keys())

    # Load legacy state into standard V4 model that has family_head configured
    model_v4 = VisionAutoregressiveModel(
        vocabulary=_vocabulary(),
        input_channels=3,
        model_dimension=32,
        max_length=16,
        num_layers=1,
        num_heads=4,
        num_families=len(FAMILY_NAMES),
    )
    incompatible = model_v4.load_state_dict(legacy_state)
    assert len(incompatible.unexpected_keys) == 0

    # Verify that a genuinely missing core weight (e.g. decoder projection) still raises
    corrupted_state = {k: v for k, v in legacy_state.items() if "output_projection" not in k}
    with pytest.raises(RuntimeError):
        model_v4.load_state_dict(corrupted_state)
