class LayerViolationError(RuntimeError):
    """Raised when a BMS layer crosses a forbidden dependency boundary."""
