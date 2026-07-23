import pytest
from ports.inbound import ImageToTikzUseCase
from ports.outbound import ModelInferencePort, TexCompilerPort

def test_cannot_instantiate_usecase():
    """Verify ImageToTikzUseCase is purely abstract."""
    with pytest.raises(TypeError):
        ImageToTikzUseCase() # type: ignore

def test_cannot_instantiate_model_port():
    """Verify ModelInferencePort is purely abstract."""
    with pytest.raises(TypeError):
        ModelInferencePort() # type: ignore

def test_cannot_instantiate_compiler_port():
    """Verify TexCompilerPort is purely abstract."""
    with pytest.raises(TypeError):
        TexCompilerPort() # type: ignore
