class DomainError(Exception):
    """Base exception for all logical invariant violations in the pure domain."""
    pass

class TensorTopologyError(DomainError):
    """Raised when spatial dimensionality or tensor shapes are invalid."""
    pass

class SyntaxTopologicalError(DomainError):
    """Raised when the generative markup syntax violates formatting constraints."""
    pass

class VocabularyInvariantError(DomainError):
    """Raised when token vocabulary constraints are violated."""
    pass

