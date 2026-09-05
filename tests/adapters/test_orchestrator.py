"""Unit tests for the inbound ImageToTikzOrchestrator adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.orchestrator import ImageToTikzOrchestrator
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.exceptions import TensorTopologyError
from core.ml.model import VisionAutoregressiveModel
from core.models import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    UNK_INDEX,
    ImageTensor,
    TikzTokens,
    TokenVocabulary,
    TrainingCheckpoint,
)


def _build_test_vocabulary() -> TokenVocabulary:
    tokens: tuple[str, ...] = (
        "\\begin{tikzpicture}",
        "\\draw",
        "(0,0)",
        "--",
        "(1,1)",
        ";",
        "\\end{tikzpicture}",
    )
    special: dict[int, str] = {
        PAD_INDEX: "<PAD>",
        BOS_INDEX: "<BOS>",
        EOS_INDEX: "<EOS>",
        UNK_INDEX: "<UNK>",
    }
    index_to_token: dict[int, str] = {
        **special,
        **{idx + 4: tok for idx, tok in enumerate(tokens)},
    }
    token_to_index: dict[str, int] = {tok: idx for idx, tok in index_to_token.items()}
    return TokenVocabulary(
        token_to_index=token_to_index,
        index_to_token=index_to_token,
    )


def _build_test_model(vocabulary: TokenVocabulary) -> VisionAutoregressiveModel:
    return VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=32,
        max_length=64,
        num_layers=2,
        num_heads=2,
    )


def test_orchestrator_initialization_and_properties() -> None:
    vocabulary: TokenVocabulary = _build_test_vocabulary()
    model: VisionAutoregressiveModel = _build_test_model(vocabulary)

    orchestrator = ImageToTikzOrchestrator(
        model=model,
        vocabulary=vocabulary,
        max_length=64,
        search_strategy="greedy",
    )

    assert orchestrator.model is model
    assert orchestrator.vocabulary is vocabulary
    assert orchestrator.max_length == 64


def test_orchestrator_initialization_validation() -> None:
    vocabulary: TokenVocabulary = _build_test_vocabulary()
    model: VisionAutoregressiveModel = _build_test_model(vocabulary)

    with pytest.raises(TypeError):
        ImageToTikzOrchestrator(model="invalid", vocabulary=vocabulary)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ImageToTikzOrchestrator(model=model, vocabulary="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ImageToTikzOrchestrator(model=model, vocabulary=vocabulary, max_length=0)
    with pytest.raises(ValueError):
        ImageToTikzOrchestrator(model=model, vocabulary=vocabulary, search_strategy="random")
    with pytest.raises(ValueError):
        ImageToTikzOrchestrator(model=model, vocabulary=vocabulary, beam_width=0)


def test_orchestrator_execute_greedy() -> None:
    vocabulary: TokenVocabulary = _build_test_vocabulary()
    model: VisionAutoregressiveModel = _build_test_model(vocabulary)
    orchestrator = ImageToTikzOrchestrator(
        model=model, vocabulary=vocabulary, max_length=32, search_strategy="greedy"
    )

    image = ImageTensor(raw_tensor=torch.randn(1, 3, 32, 32))
    tokens: TikzTokens = orchestrator.execute(image)

    assert isinstance(tokens, TikzTokens)
    assert "\\begin{" in tokens.markup
    assert "\\end{" in tokens.markup


def test_orchestrator_execute_beam() -> None:
    vocabulary: TokenVocabulary = _build_test_vocabulary()
    model: VisionAutoregressiveModel = _build_test_model(vocabulary)
    orchestrator = ImageToTikzOrchestrator(
        model=model,
        vocabulary=vocabulary,
        max_length=32,
        search_strategy="beam",
        beam_width=2,
    )

    image = ImageTensor(raw_tensor=torch.randn(1, 3, 32, 32))
    tokens: TikzTokens = orchestrator.execute(image)

    assert isinstance(tokens, TikzTokens)
    assert "\\begin{" in tokens.markup
    assert "\\end{" in tokens.markup


def test_orchestrator_execute_rejects_invalid_batch_or_type() -> None:
    vocabulary: TokenVocabulary = _build_test_vocabulary()
    model: VisionAutoregressiveModel = _build_test_model(vocabulary)
    orchestrator = ImageToTikzOrchestrator(model=model, vocabulary=vocabulary)

    with pytest.raises(TypeError):
        orchestrator.execute("not_an_image")  # type: ignore[arg-type]
    with pytest.raises(TensorTopologyError):
        orchestrator.execute(ImageTensor(raw_tensor=torch.randn(2, 3, 32, 32)))


def test_orchestrator_from_checkpoint() -> None:
    vocabulary: TokenVocabulary = _build_test_vocabulary()
    model: VisionAutoregressiveModel = _build_test_model(vocabulary)
    checkpoint = TrainingCheckpoint(
        model_state=model.state_dict(),
        optimizer_state={"param_groups": []},
        epoch=1,
    )

    with TemporaryDirectory() as temp_dir:
        vocab_path = Path(temp_dir) / "vocabulary.json"
        ckpt_path = Path(temp_dir) / "checkpoint.pt"

        JsonVocabularyAdapter().save_vocabulary(vocabulary, str(vocab_path))
        AtomicCheckpointAdapter().save_checkpoint(checkpoint, str(ckpt_path))

        config = {
            "input_channels": 3,
            "model_dimension": 32,
            "max_length": 64,
            "num_layers": 2,
            "num_heads": 2,
        }

        orchestrator = ImageToTikzOrchestrator.from_checkpoint(
            checkpoint_path=ckpt_path,
            vocabulary_path=vocab_path,
            config=config,
            device=torch.device("cpu"),
        )

        assert isinstance(orchestrator, ImageToTikzOrchestrator)
        assert orchestrator.max_length == 64
        image = ImageTensor(raw_tensor=torch.randn(1, 3, 32, 32))
        tokens = orchestrator.execute(image)
        assert isinstance(tokens, TikzTokens)


def test_orchestrator_execute_grammar_greedy_and_best_of_n() -> None:
    vocabulary = _build_test_vocabulary()
    model = _build_test_model(vocabulary)
    image = ImageTensor(raw_tensor=torch.randn(1, 3, 32, 32))

    for strat in ("grammar_greedy", "grammar_beam", "sample", "best_of_n"):
        orchestrator = ImageToTikzOrchestrator(
            model=model,
            vocabulary=vocabulary,
            max_length=32,
            search_strategy=strat,
            beam_width=2,
        )
        tokens = orchestrator.execute(image)
        assert isinstance(tokens, TikzTokens)
        assert "\\begin{" in tokens.markup
        assert "\\end{" in tokens.markup


@pytest.mark.anyio
async def test_orchestrator_execute_reranked() -> None:
    from core.models import CompilationResult
    from ports.outbound import ImageRasterizerPort, TexCompilerPort

    class DummyCompiler(TexCompilerPort):
        async def compile_tikz(self, tokens: TikzTokens) -> CompilationResult:
            return CompilationResult(is_successful=False, pdf_data=b"")

    class DummyRasterizer(ImageRasterizerPort):
        async def rasterize_pdf(self, pdf_data: bytes, dpi: int = 150) -> bytes:
            return b""

    vocabulary = _build_test_vocabulary()
    model = _build_test_model(vocabulary)
    orchestrator = ImageToTikzOrchestrator(
        model=model,
        vocabulary=vocabulary,
        max_length=32,
    )
    image = ImageTensor(raw_tensor=torch.randn(1, 3, 32, 32))
    best_tokens, ssim_score = await orchestrator.execute_reranked(
        image, DummyCompiler(), DummyRasterizer(), n_hypotheses=2
    )

    assert isinstance(best_tokens, TikzTokens)
    assert "\\begin{" in best_tokens.markup
    assert "\\end{" in best_tokens.markup
    assert isinstance(ssim_score, float)

