import torch

from adapters.model_inference import TorchModelInferenceAdapter
from core.math.tokenization import build_vocabulary
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens
from ports.outbound import ModelInferencePort


def test_model_adapter_implements_inference_port() -> None:
    torch.manual_seed(11)
    sample = TikzTokens(markup=r"\begin{tikzpicture}\draw (0,0);\end{tikzpicture}")
    vocabulary = build_vocabulary([sample])
    model = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        model_dimension=32,
        max_length=4,
        num_layers=1,
        num_heads=4,
    )
    adapter = TorchModelInferenceAdapter(model)

    generated = adapter.infer_markup(ImageTensor(torch.randn(1, 3, 32, 32)))

    assert isinstance(adapter, ModelInferencePort)
    assert isinstance(generated, TikzTokens)
    assert r"\begin{tikzpicture}" in generated.markup
