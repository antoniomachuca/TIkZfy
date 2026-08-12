import torch

from core.math.tokenization import build_vocabulary
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens


def _vocabulary():
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
