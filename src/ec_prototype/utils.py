"""General utility functions."""

from torch.backends import mps


def get_default_device() -> str | None:
    """Returns the name of a GPU device if available."""
    # Check if a macOS Metal device is available
    if mps.is_available() and mps.is_built():
        return "mps"
    # PyTorch will already get an available cuda GPU device if None is passed in.
    return None
