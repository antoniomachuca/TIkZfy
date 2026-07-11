class DomainError(Exception):
    """Base exception for all logical invariant violations in the pure domain."""
    pass

class TensorTopologyError(DomainError):
    """Raised when spatial dimensionality or tensor structural invariants fail in O(1)."""
    pass

class SyntaxTopologicalError(DomainError):
    """Raised when the generative markup syntax violates structural existence constraints."""
    pass
