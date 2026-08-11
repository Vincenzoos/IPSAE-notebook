"""Structure/PAE filename pairing validation and bulk discovery helpers.

Supports AlphaFold2 (single model), AlphaFold Server (AF3), and Boltz naming.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class ModelType(str, Enum):
    AF2 = "af2"
    AF3 = "af3"
    BOLTZ = "boltz"

    @property
    def label(self) -> str:
        return {
            ModelType.AF2: "AlphaFold2",
            ModelType.AF3: "AlphaFold3",
            ModelType.BOLTZ: "Boltz",
        }[self]


MODEL_TYPE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Select model type…", ""),
    ("AlphaFold2", ModelType.AF2.value),
    ("AlphaFold3", ModelType.AF3.value),
    ("Boltz", ModelType.BOLTZ.value),
)

BOLTZ_MISSING_SUMMARY_WARNING = (
    "Boltz summary file not found. ipSAE will still run, but Boltz ipTM values "
    "may be unavailable or zero."
)

AF2_STRUCTURE_RE = re.compile(
    r"^(?P<complex>.+)_unrelaxed_rank_(?P<rank>\d+)_(?P<details>.+)$"
)
AF2_PAE_RE = re.compile(
    r"^(?P<complex>.+)_scores_rank_(?P<rank>\d+)_(?P<details>.+)$"
)
AF3_STRUCTURE_RE = re.compile(r"^(?P<complex>.+)_model_(?P<index>\d+)$")
AF3_PAE_RE = re.compile(r"^(?P<complex>.+)_full_data_(?P<index>\d+)$")
AF3_SUMMARY_RE = re.compile(r"^(?P<complex>.+)_summary_confidences_(?P<index>\d+)$")
BOLTZ_STRUCTURE_RE = re.compile(r"^(?P<complex>.+)_model_(?P<index>\d+)$")
BOLTZ_PAE_RE = re.compile(r"^pae_(?P<complex>.+)_model_(?P<index>\d+)$")
BOLTZ_SUMMARY_RE = re.compile(r"^confidence_(?P<complex>.+)_model_(?P<index>\d+)$")

# Signatures used for bulk type detection (any model index).
AF3_PAE_ANY_RE = re.compile(r"^.+_full_data_\d+\.json$")
BOLTZ_PAE_ANY_RE = re.compile(r"^pae_.+_model_\d+\.npz$")
AF2_BULK_HINT_RE = re.compile(
    r"(^.+_unrelaxed_rank_\d+_.+\.pdb$)|(^.+_scores_rank_\d+_.+\.json$)",
)


@dataclass(frozen=True)
class PairInfo:
    complex_id: str
    model_index: int | None = None
    rank: str | None = None
    details: str | None = None


@dataclass(frozen=True)
class TypeHints:
    structure_placeholder: str
    pae_placeholder: str
    structure_extensions: frozenset[str]
    pae_extensions: frozenset[str]
    hint_html: str
    summary_placeholder: str | None = None


TYPE_HINTS: dict[ModelType, TypeHints] = {
    ModelType.AF2: TypeHints(
        structure_placeholder="<complex>_unrelaxed_rank_001_<details>.pdb",
        pae_placeholder="<complex>_scores_rank_001_<details>.json",
        structure_extensions=frozenset({".pdb"}),
        pae_extensions=frozenset({".json"}),
        hint_html=(
            "AlphaFold2 / ColabFold default naming: structure "
            "<code>*_unrelaxed_rank_&lt;rank&gt;_&lt;details&gt;.pdb</code> "
            "with matching PAE <code>*_scores_rank_&lt;rank&gt;_&lt;details&gt;.json</code>. "
            "Example: <code>RAF1_KSR1_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_003.pdb</code> "
            "+ <code>RAF1_KSR1_scores_rank_001_alphafold2_multimer_v3_model_4_seed_003.json</code>."
        ),
    ),
    ModelType.AF3: TypeHints(
        structure_placeholder="<complex>_model_0.cif",
        pae_placeholder="<complex>_full_data_0.json",
        structure_extensions=frozenset({".cif"}),
        pae_extensions=frozenset({".json"}),
        hint_html=(
            "AlphaFold Server naming: structure <code>*_model_N.cif</code> with matching "
            "PAE <code>*_full_data_N.json</code> (same complex and N). "
            "Example: <code>fold_aurka_tpx2_model_0.cif</code> + "
            "<code>fold_aurka_tpx2_full_data_0.json</code>."
        ),
    ),
    ModelType.BOLTZ: TypeHints(
        structure_placeholder="<complex>_model_0.pdb or .cif",
        pae_placeholder="pae_<complex>_model_0.npz",
        structure_extensions=frozenset({".pdb", ".cif"}),
        pae_extensions=frozenset({".npz"}),
        summary_placeholder="confidence_<complex>_model_0.json",
        hint_html=(
            "Boltz naming: structure <code>&lt;complex&gt;_model_N.pdb</code> or "
            "<code>.cif</code> with PAE <code>pae_&lt;complex&gt;_model_N.npz</code>. "
            "Optional summary <code>confidence_&lt;complex&gt;_model_N.json</code>. "
            "Example: <code>AURKA_TPX2_model_0.cif</code> + "
            "<code>pae_AURKA_TPX2_model_0.npz</code>."
        ),
    ),
}


def parse_model_type(value: str | ModelType | None) -> ModelType:
    if isinstance(value, ModelType):
        return value
    if value is None or str(value).strip() == "":
        raise ValueError("Model type is required. Select AlphaFold2, AlphaFold3, or Boltz.")
    key = str(value).strip().lower()
    aliases = {
        "af2": ModelType.AF2,
        "alphafold2": ModelType.AF2,
        "af3": ModelType.AF3,
        "alphafold3": ModelType.AF3,
        "alphafold3 server": ModelType.AF3,
        "alphafold server": ModelType.AF3,
        "boltz": ModelType.BOLTZ,
        "boltz1": ModelType.BOLTZ,
        "boltz2": ModelType.BOLTZ,
    }
    if key not in aliases:
        raise ValueError(f"Unknown model type: {value!r}. Expected af2, af3, or boltz.")
    return aliases[key]


def type_hints_for(model_type: str | ModelType) -> TypeHints:
    return TYPE_HINTS[parse_model_type(model_type)]


def structure_extensions_for(model_type: str | ModelType) -> frozenset[str]:
    return type_hints_for(model_type).structure_extensions


def pae_extensions_for(model_type: str | ModelType) -> frozenset[str]:
    return type_hints_for(model_type).pae_extensions


def boltz_companion_paths(pae_file: str | Path) -> tuple[Path, Path]:
    """Derive confidence JSON and pLDDT NPZ siblings from a Boltz PAE path.

    Only the filename is transformed so parent directories containing ``pae``
    are left unchanged.
    """
    pae_path = Path(pae_file)
    name = pae_path.name
    if pae_path.suffix != ".npz" or BOLTZ_PAE_RE.match(pae_path.stem) is None:
        raise ValueError(
            "Boltz PAE file must be named like 'pae_<complex>_model_N.npz'. "
            f"Got '{name}'."
        )
    base = name[len("pae_") : -len(".npz")]
    confidence = pae_path.with_name(f"confidence_{base}.json")
    plddt = pae_path.with_name(f"plddt_{base}.npz")
    return confidence, plddt


def expected_boltz_summary_path(pae_file: str | Path) -> Path:
    confidence, _plddt = boltz_companion_paths(pae_file)
    return confidence


def parse_af2_structure_pairing(structure_file: str | Path) -> PairInfo | None:
    path = Path(structure_file)
    if path.suffix != ".pdb":
        return None
    match = AF2_STRUCTURE_RE.match(path.stem)
    if not match:
        return None
    return PairInfo(
        complex_id=match.group("complex"),
        rank=match.group("rank"),
        details=match.group("details"),
    )


def parse_af2_pae_pairing(pae_file: str | Path) -> PairInfo | None:
    path = Path(pae_file)
    if path.suffix != ".json":
        return None
    match = AF2_PAE_RE.match(path.stem)
    if not match:
        return None
    return PairInfo(
        complex_id=match.group("complex"),
        rank=match.group("rank"),
        details=match.group("details"),
    )


def parse_af3_structure_pairing(structure_file: str | Path) -> PairInfo | None:
    path = Path(structure_file)
    if path.suffix != ".cif":
        return None
    match = AF3_STRUCTURE_RE.match(path.stem)
    if not match:
        return None
    return PairInfo(
        complex_id=match.group("complex"),
        model_index=int(match.group("index")),
    )


def parse_af3_pae_pairing(pae_file: str | Path) -> PairInfo | None:
    path = Path(pae_file)
    if path.suffix != ".json":
        return None
    match = AF3_PAE_RE.match(path.stem)
    if not match:
        return None
    return PairInfo(
        complex_id=match.group("complex"),
        model_index=int(match.group("index")),
    )


def parse_af3_summary_pairing(summary_file: str | Path) -> PairInfo | None:
    path = Path(summary_file)
    if path.suffix != ".json":
        return None
    match = AF3_SUMMARY_RE.match(path.stem)
    if not match:
        return None
    return PairInfo(
        complex_id=match.group("complex"),
        model_index=int(match.group("index")),
    )


def parse_boltz_structure_pairing(structure_file: str | Path) -> PairInfo | None:
    path = Path(structure_file)
    if path.suffix not in {".pdb", ".cif"}:
        return None
    match = BOLTZ_STRUCTURE_RE.match(path.stem)
    if not match:
        return None
    return PairInfo(
        complex_id=match.group("complex"),
        model_index=int(match.group("index")),
    )


def parse_boltz_pae_pairing(pae_file: str | Path) -> PairInfo | None:
    path = Path(pae_file)
    if path.suffix != ".npz":
        return None
    match = BOLTZ_PAE_RE.match(path.stem)
    if not match:
        return None
    return PairInfo(
        complex_id=match.group("complex"),
        model_index=int(match.group("index")),
    )


def parse_boltz_summary_pairing(summary_file: str | Path) -> PairInfo | None:
    path = Path(summary_file)
    if path.suffix != ".json":
        return None
    match = BOLTZ_SUMMARY_RE.match(path.stem)
    if not match:
        return None
    return PairInfo(
        complex_id=match.group("complex"),
        model_index=int(match.group("index")),
    )


def _require_regular_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")


def _require_same_folder(paths: Iterable[Path], labels: Iterable[str]) -> None:
    resolved = [p.resolve().parent for p in paths]
    if len({str(parent) for parent in resolved}) != 1:
        parts = ", ".join(f"{label}='{parent}'" for label, parent in zip(labels, resolved))
        raise ValueError(
            "All files for one job must share the same parent folder. "
            f"Got {parts}."
        )


def _validate_af2(structure_path: Path, pae_path: Path) -> PairInfo:
    if structure_path.suffix != ".pdb":
        raise ValueError(
            "AlphaFold2 structure must be a .pdb file matching "
            "'*_unrelaxed_rank_<rank>_<details>.pdb'. "
            f"Got '{structure_path.name}'."
        )
    if pae_path.suffix != ".json":
        raise ValueError(
            "AlphaFold2 PAE must be a .json file matching "
            "'*_scores_rank_<rank>_<details>.json'. "
            f"Got '{pae_path.name}'."
        )
    structure = parse_af2_structure_pairing(structure_path)
    pae = parse_af2_pae_pairing(pae_path)
    if not structure or not pae:
        raise ValueError(
            "AlphaFold2 structure/PAE mismatch: v1 accepts only the default ColabFold-style pair "
            "'<complex>_unrelaxed_rank_<rank>_<details>.pdb' with "
            "'<complex>_scores_rank_<rank>_<details>.json' (same complex, rank, and details). "
            f"Got '{structure_path.name}' and '{pae_path.name}'. "
            "Other AF2 naming schemes (including relaxed / ranked_N) are unsupported."
        )
    if (
        structure.complex_id != pae.complex_id
        or structure.rank != pae.rank
        or structure.details != pae.details
    ):
        raise ValueError(
            "AlphaFold2 structure/PAE mismatch: complex, rank, and details must match. "
            f"Got '{structure_path.name}' and '{pae_path.name}'."
        )
    _require_same_folder([structure_path, pae_path], ["structure", "PAE"])
    return structure


def _validate_af3(structure_path: Path, pae_path: Path) -> PairInfo:
    if structure_path.suffix != ".cif":
        raise ValueError(
            "AlphaFold3 Server structure must be a .cif file matching '*_model_N.cif'. "
            f"Got '{structure_path.name}'."
        )
    if pae_path.suffix != ".json":
        raise ValueError(
            "AlphaFold3 Server PAE must be a .json file matching '*_full_data_N.json'. "
            f"Got '{pae_path.name}'."
        )
    structure = parse_af3_structure_pairing(structure_path)
    pae = parse_af3_pae_pairing(pae_path)
    if not structure or not pae:
        raise ValueError(
            "AF3 structure/PAE mismatch: expected a structure like '*_model_N.cif' "
            f"paired with a PAE file like '*_full_data_N.json'. Got '{structure_path.name}' "
            f"and '{pae_path.name}'."
        )
    if structure.complex_id != pae.complex_id:
        raise ValueError(
            "AF3 structure/PAE mismatch: structure and PAE appear to come from "
            f"different complexes ('{structure_path.name}' vs '{pae_path.name}')."
        )
    if structure.model_index != pae.model_index:
        raise ValueError(
            "AF3 structure/PAE mismatch: structure and PAE refer to different model "
            f"indices for the same complex ('{structure_path.name}' vs '{pae_path.name}')."
        )
    _require_same_folder([structure_path, pae_path], ["structure", "PAE"])
    return structure


def _validate_af3_summary(
    summary_path: Path,
    structure: PairInfo,
    pae_path: Path,
) -> None:
    if summary_path.suffix != ".json":
        raise ValueError(
            "AlphaFold3 Server summary must be a .json file matching "
            "'*_summary_confidences_N.json'. "
            f"Got '{summary_path.name}'."
        )
    summary = parse_af3_summary_pairing(summary_path)
    expected = af3_summary_path_for(
        pae_path,
        structure.complex_id,
        int(structure.model_index),
    )
    if not summary or summary_path.name != expected.name:
        raise ValueError(
            "AlphaFold3 Server summary filename mismatch: expected sibling "
            f"'{expected.name}' for PAE '{pae_path.name}'. Got '{summary_path.name}'."
        )
    if summary.complex_id != structure.complex_id or summary.model_index != structure.model_index:
        raise ValueError(
            "AlphaFold3 Server summary mismatch: summary must match the structure/PAE "
            "complex and model index. "
            f"Got '{summary_path.name}'."
        )


def _validate_boltz_summary(
    summary_path: Path,
    structure: PairInfo,
    pae_path: Path,
) -> None:
    if summary_path.suffix != ".json":
        raise ValueError(
            "Boltz summary file must be a .json file matching "
            "'confidence_<complex>_model_N.json'. "
            f"Got '{summary_path.name}'."
        )
    summary = parse_boltz_summary_pairing(summary_path)
    expected = expected_boltz_summary_path(pae_path)
    if not summary:
        raise ValueError(
            "Boltz summary filename mismatch: expected "
            f"'{expected.name}'. Got '{summary_path.name}'."
        )
    if summary.complex_id != structure.complex_id or summary.model_index != structure.model_index:
        raise ValueError(
            "Boltz summary mismatch: summary must match the structure/PAE complex and model index. "
            f"Got '{summary_path.name}' for structure complex '{structure.complex_id}' "
            f"model {structure.model_index}."
        )
    if summary_path.name != expected.name:
        raise ValueError(
            "Boltz summary filename mismatch: expected sibling "
            f"'{expected.name}' for PAE '{pae_path.name}'. Got '{summary_path.name}'."
        )


def _validate_boltz(
    structure_path: Path,
    pae_path: Path,
    summary_file: str | Path | None = None,
) -> tuple[PairInfo, Path | None, str | None]:
    if structure_path.suffix not in {".pdb", ".cif"}:
        raise ValueError(
            "Boltz structure must be a .pdb or .cif file matching '<complex>_model_N'. "
            f"Got '{structure_path.name}'."
        )
    if pae_path.suffix != ".npz":
        raise ValueError(
            "Boltz PAE must be a .npz file matching 'pae_<complex>_model_N.npz'. "
            f"Got '{pae_path.name}'."
        )
    structure = parse_boltz_structure_pairing(structure_path)
    pae = parse_boltz_pae_pairing(pae_path)
    if not structure or not pae:
        raise ValueError(
            "Boltz structure/PAE mismatch: expected structure '<complex>_model_N.pdb|.cif' "
            f"with PAE 'pae_<complex>_model_N.npz'. Got '{structure_path.name}' and '{pae_path.name}'."
        )
    if structure.complex_id != pae.complex_id or structure.model_index != pae.model_index:
        raise ValueError(
            "Boltz structure/PAE mismatch: complex and model index must match. "
            f"Got '{structure_path.name}' and '{pae_path.name}'."
        )

    summary_path: Path | None = None
    warning: str | None = None
    if summary_file is not None and str(summary_file).strip():
        summary_path = Path(summary_file)
        _require_regular_file(summary_path, "Boltz summary file")
        _validate_boltz_summary(summary_path, structure, pae_path)
        _require_same_folder(
            [structure_path, pae_path, summary_path],
            ["structure", "PAE", "summary"],
        )
    else:
        _require_same_folder([structure_path, pae_path], ["structure", "PAE"])
        expected = expected_boltz_summary_path(pae_path)
        if expected.is_file():
            summary_path = expected
        else:
            warning = BOLTZ_MISSING_SUMMARY_WARNING
    return structure, summary_path, warning


def validate_structure_pae_pairing(
    structure_file: str | Path,
    pae_file: str | Path,
    model_type: str | ModelType,
    summary_file: str | Path | None = None,
    *,
    require_exists: bool = True,
) -> dict:
    """Validate structure/PAE pairing for an explicit model type.

    Returns a dict with pair info and optional Boltz summary path/warning.
    """
    model = parse_model_type(model_type)
    structure_path = Path(structure_file)
    pae_path = Path(pae_file)

    if require_exists:
        _require_regular_file(structure_path, "Structure file")
        _require_regular_file(pae_path, "PAE file")

    warning: str | None = None
    summary_path: Path | None = None
    if model is ModelType.AF2:
        if summary_file is not None and str(summary_file).strip():
            raise ValueError("Summary file is only used for Boltz jobs.")
        info = _validate_af2(structure_path, pae_path)
    elif model is ModelType.AF3:
        info = _validate_af3(structure_path, pae_path)
        if summary_file is not None and str(summary_file).strip():
            summary_path = Path(summary_file)
            _require_regular_file(summary_path, "AlphaFold3 Server summary file")
            _validate_af3_summary(summary_path, info, pae_path)
            _require_same_folder(
                [structure_path, pae_path, summary_path],
                ["structure", "PAE", "summary"],
            )
        else:
            expected = af3_summary_path_for(
                pae_path,
                info.complex_id,
                int(info.model_index),
            )
            if expected.is_file():
                summary_path = expected
    else:
        info, summary_path, warning = _validate_boltz(structure_path, pae_path, summary_file)

    return {
        "model_type": model,
        "pair": info,
        "summary_file": summary_path,
        "warning": warning,
    }


# Back-compat alias used by older call sites.
validate_af3_structure_pae_pairing = lambda structure_file, pae_file: validate_structure_pae_pairing(
    structure_file, pae_file, ModelType.AF3
)


def detect_bulk_model_type(folder: str | Path) -> ModelType:
    """Detect AF3 Server vs Boltz from PAE signatures under a folder."""
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    has_af3 = False
    has_boltz = False
    has_af2_hint = False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if AF3_PAE_ANY_RE.match(name):
            has_af3 = True
        elif BOLTZ_PAE_ANY_RE.match(name):
            has_boltz = True
        elif AF2_BULK_HINT_RE.match(name):
            has_af2_hint = True

    if has_af3 and has_boltz:
        raise ValueError(
            "Mixed model-type folder: found both AlphaFold Server "
            "(*_full_data_N.json) and Boltz (pae_*_model_N.npz) PAE signatures. "
            "Bulk evaluation requires a folder with only one supported type."
        )
    if has_af3:
        return ModelType.AF3
    if has_boltz:
        return ModelType.BOLTZ
    if has_af2_hint:
        raise ValueError(
            "Unsupported AF2-style bulk input. v1 bulk evaluation supports AlphaFold Server "
            "and Boltz folders only. Use Single Model for AlphaFold2 / ColabFold outputs."
        )
    raise ValueError(
        "Unrecognized folder for bulk evaluation. v1 supports AlphaFold Server folders "
        "with '*_full_data_N.json' or Boltz folders with 'pae_*_model_N.npz'. "
        "AF2 bulk discovery is not supported."
    )


def af3_summary_path_for(structure_or_pae: Path, complex_id: str, model_index: int) -> Path:
    return structure_or_pae.with_name(f"{complex_id}_summary_confidences_{model_index}.json")
