import pytest

from ports.inbound import ImageToTikzUseCase
from ports.outbound import ModelInferencePort, TexCompilerPort


def test_cannot_instantiate_usecase() -> None:
    """Verify ImageToTikzUseCase is purely abstract."""
    with pytest.raises(TypeError):
        ImageToTikzUseCase() # type: ignore

def test_cannot_instantiate_model_port() -> None:
    """Verify ModelInferencePort is purely abstract."""
    with pytest.raises(TypeError):
        ModelInferencePort() # type: ignore

def test_cannot_instantiate_compiler_port() -> None:
    """Verify TexCompilerPort is purely abstract."""
    with pytest.raises(TypeError):
        TexCompilerPort() # type: ignore
