"""Pure helpers for Single Model UI state (type-specific paths and filters)."""

from __future__ import annotations

from pathlib import Path

from model_pairing import (
    MODEL_TYPE_CHOICES,
    ModelType,
    expected_boltz_summary_path,
    parse_af2_pae_pairing,
    parse_af2_structure_pairing,
    parse_af3_pae_pairing,
    parse_af3_structure_pairing,
    parse_boltz_pae_pairing,
    parse_boltz_structure_pairing,
    parse_boltz_summary_pairing,
    parse_model_type,
    type_hints_for,
)


def selected_model_type(value: str | None) -> ModelType | None:
    text = (value or "").strip()
    if not text:
        return None
    return parse_model_type(text)


def placeholders_for(model_type: ModelType | None) -> dict[str, str]:
    if model_type is None:
        return {
            "structure": "Select a model type first",
            "pae": "Select a model type first",
            "summary": "confidence_<complex>_model_0.json",
        }
    hints = type_hints_for(model_type)
    return {
        "structure": hints.structure_placeholder,
        "pae": hints.pae_placeholder,
        "summary": hints.summary_placeholder or "confidence_<complex>_model_0.json",
    }


def upload_extensions_for(model_type: ModelType | None) -> dict[str, frozenset[str]]:
    if model_type is None:
        return {
            "structure": frozenset({".pdb", ".cif"}),
            "pae": frozenset({".json", ".npz"}),
            "summary": frozenset({".json"}),
        }
    hints = type_hints_for(model_type)
    return {
        "structure": hints.structure_extensions,
        "pae": hints.pae_extensions,
        "summary": frozenset({".json"}),
    }


def hint_text_for(model_type: ModelType | None) -> str:
    if model_type is None:
        return "Select a model type to see the accepted filename convention and an example pair."
    return type_hints_for(model_type).hint_html


def path_compatible_with_type(path: str, model_type: ModelType, kind: str) -> bool:
    text = (path or "").strip()
    if not text:
        return True
    candidate = Path(text)
    ext = candidate.suffix
    allowed = upload_extensions_for(model_type)[kind]
    if ext not in allowed:
        return False
    parsers = {
        (ModelType.AF2, "structure"): parse_af2_structure_pairing,
        (ModelType.AF2, "pae"): parse_af2_pae_pairing,
        (ModelType.AF3, "structure"): parse_af3_structure_pairing,
        (ModelType.AF3, "pae"): parse_af3_pae_pairing,
        (ModelType.BOLTZ, "structure"): parse_boltz_structure_pairing,
        (ModelType.BOLTZ, "pae"): parse_boltz_pae_pairing,
        (ModelType.BOLTZ, "summary"): parse_boltz_summary_pairing,
    }
    parser = parsers.get((model_type, kind))
    return parser(candidate) is not None if parser else True


def clear_incompatible_paths(
    *,
    model_type: ModelType | None,
    structure: str,
    pae: str,
    summary: str,
) -> dict[str, str]:
    """Clear path values that are incompatible with the newly selected type."""
    if model_type is None:
        return {"structure": "", "pae": "", "summary": ""}
    out = {
        "structure": structure if path_compatible_with_type(structure, model_type, "structure") else "",
        "pae": pae if path_compatible_with_type(pae, model_type, "pae") else "",
        "summary": "",
    }
    if model_type is ModelType.BOLTZ:
        out["summary"] = summary if path_compatible_with_type(summary, model_type, "summary") else ""
    return out


def maybe_prefill_boltz_summary(pae_path: str, current_summary: str) -> str | None:
    """Return an existing sibling summary path to prefill, or None.

    Does not overwrite a non-empty user-entered summary value.
    """
    if (current_summary or "").strip():
        return None
    text = (pae_path or "").strip()
    if not text:
        return None
    try:
        from paths import resolve_repo_path

        expected = expected_boltz_summary_path(resolve_repo_path(text))
    except ValueError:
        return None
    if expected.is_file():
        return str(expected)
    return None


def single_inputs_locked(
    *,
    model_type: ModelType | None,
    running: bool = False,
) -> bool:
    """True when path/upload widgets should be disabled (no type or busy)."""
    return running or model_type is None


def single_run_ready(
    *,
    model_type: ModelType | None,
    structure: str,
    pae: str,
    running: bool = False,
) -> bool:
    if single_inputs_locked(model_type=model_type, running=running):
        return False
    return bool(structure.strip()) and bool(pae.strip())


__all__ = [
    "MODEL_TYPE_CHOICES",
    "selected_model_type",
    "placeholders_for",
    "upload_extensions_for",
    "hint_text_for",
    "path_compatible_with_type",
    "clear_incompatible_paths",
    "maybe_prefill_boltz_summary",
    "single_inputs_locked",
    "single_run_ready",
]
