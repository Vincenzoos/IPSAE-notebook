"""Backward-compatible AF3 pairing re-exports; prefer model_pairing."""

from __future__ import annotations

from pathlib import Path

from model_pairing import (  # noqa: F401
    ModelType,
    PairInfo as Af3PairInfo,
    parse_af3_pae_pairing,
    parse_af3_structure_pairing,
)
from model_pairing import validate_structure_pae_pairing as _validate_typed


def validate_structure_pae_pairing(structure_file: str | Path, pae_file: str | Path) -> None:
    """AF3-only validation retained for older callers without an explicit type."""
    _validate_typed(structure_file, pae_file, ModelType.AF3)
