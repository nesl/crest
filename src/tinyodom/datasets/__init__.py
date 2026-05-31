"""Concrete dataset implementations for the modular TinyODOM pipeline."""

from __future__ import annotations

from typing import Any

__all__ = ["OxIODDataset", "UrbanSound8KMelDataset"]


def __getattr__(name: str) -> Any:
    """Load concrete dataset classes only when callers request them.

    Importing dataset utility modules such as ``urbansound8k_common`` should not
    import OxIOD and its optional scientific stack. Keeping these exports lazy
    avoids pulling unrelated dataset dependencies into standalone prep scripts.

    Parameters
    ----------
    name : str
        Component name to resolve from the registry.

    Returns
    -------
    Any
        Lazily imported dataset class exposed by this package.

    Raises
    ------
    AttributeError
        If existing validation or execution checks fail.
    """
    if name == "OxIODDataset":
        from .oxiod import OxIODDataset

        return OxIODDataset
    if name == "UrbanSound8KMelDataset":
        from .urbansound8k_mel import UrbanSound8KMelDataset

        return UrbanSound8KMelDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
