"""Bulk AF3 Server / Boltz discovery and ipSAE execution helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from model_pairing import (
    BOLTZ_MISSING_SUMMARY_WARNING,
    ModelType,
    af3_summary_path_for,
    detect_bulk_model_type,
    expected_boltz_summary_path,
    parse_af3_pae_pairing,
    parse_af3_structure_pairing,
    parse_boltz_pae_pairing,
    parse_boltz_structure_pairing,
)
from naming import model_name_from_structure
from paths import DEFAULT_EVALS_DIR, ROOT, rel_repo_path, resolve_repo_path
from single_model_eval import IpsaeJob, run_ipsae

PREVIEW_COLUMNS = [
    "Type",
    "Model",
    "Folder",
    "ModelIndex",
    "Structure",
    "PAE",
    "Summary",
    "SummaryStatus",
    "Ready",
    "Error",
]


def new_bulk_output_dir(base: str = "bulk_ipsae_evals") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = ROOT / f"{base}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = ROOT / f"{base}_{stamp}_{suffix}"
        suffix += 1
    return candidate


def bulk_log_path(output_dir: str | Path) -> Path:
    out_dir = resolve_repo_path(output_dir)
    return out_dir / f"{out_dir.name}_log.json"


def write_bulk_run_log(
    output_dir: str | Path,
    source_folder: str | Path,
    model_index: int,
    jobs: list[IpsaeJob],
    ok_count: int,
    error_count: int,
    detected_type: str | ModelType | None = None,
) -> Path:
    out_dir = resolve_repo_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve_repo_path(source_folder)
    if detected_type is None and jobs:
        detected_type = jobs[0].resolved_model_type().value
    elif isinstance(detected_type, ModelType):
        detected_type = detected_type.value
    log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_folder": rel_repo_path(source_path),
        "detected_type": detected_type,
        "model_index": int(model_index),
        "model_count": len(jobs),
        "successful_models": int(ok_count),
        "failed_models": int(error_count),
        "models": [
            {
                "model": job.label,
                "model_type": job.resolved_model_type().value,
                "structure_file": rel_repo_path(resolve_repo_path(job.structure_file)),
                "pae_file": rel_repo_path(resolve_repo_path(job.pae_file)),
                "summary_file": (
                    rel_repo_path(resolve_repo_path(job.summary_file))
                    if job.summary_file
                    else None
                ),
            }
            for job in jobs
        ],
    }
    log_path = bulk_log_path(out_dir)
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    return log_path


def _preview_row(
    *,
    model_type: ModelType,
    model_name: str,
    folder: Path,
    model_index: int,
    structure_name: str | None,
    pae_name: str | None,
    summary_name: str | None,
    summary_status: str,
    ready: bool,
    error: str = "",
) -> dict:
    return {
        "Type": model_type.value,
        "Model": model_name,
        "Folder": rel_repo_path(folder),
        "ModelIndex": int(model_index),
        "Structure": structure_name or "",
        "PAE": pae_name or "",
        "Summary": summary_name or "",
        "SummaryStatus": summary_status,
        "Ready": bool(ready),
        "Error": error,
    }


def discover_af3_models(
    folder: str | Path,
    model_index: int = 0,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> tuple[list[IpsaeJob], pd.DataFrame]:
    """Find AlphaFold Server model_N CIF files and matching full_data_N JSON files."""
    root = resolve_repo_path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    jobs: list[IpsaeJob] = []
    rows: list[dict] = []
    pae_by_key: dict[tuple[Path, str], Path] = {}
    structure_by_key: dict[tuple[Path, str], Path] = {}

    for pae in sorted(p for p in root.rglob(f"*_full_data_{model_index}.json") if p.is_file()):
        info = parse_af3_pae_pairing(pae)
        if info is not None and info.model_index == model_index:
            pae_by_key[(pae.parent.resolve(), info.complex_id)] = pae
    for structure in sorted(p for p in root.rglob(f"*_model_{model_index}.cif") if p.is_file()):
        info = parse_af3_structure_pairing(structure)
        if info is not None and info.model_index == model_index:
            structure_by_key[(structure.parent.resolve(), info.complex_id)] = structure

    for folder_key, complex_id in sorted(
        set(pae_by_key) | set(structure_by_key),
        key=lambda item: (str(item[0]), item[1]),
    ):
        key = (folder_key, complex_id)
        pae = pae_by_key.get(key)
        structure = structure_by_key.get(key)
        folder_path = (pae or structure).parent
        expected_pae_name = f"{complex_id}_full_data_{model_index}.json"
        expected_structure_name = f"{complex_id}_model_{model_index}.cif"
        summary = af3_summary_path_for(folder_path / expected_pae_name, complex_id, model_index)
        summary_ok = summary.is_file()
        summary_status = "present" if summary_ok else "optional/absent"
        model_name = model_name_from_structure(structure) if structure else complex_id

        error = ""
        if structure is None:
            error = f"Missing structure counterpart '{expected_structure_name}'."
        elif pae is None:
            error = f"Missing PAE counterpart '{expected_pae_name}'."

        rows.append(
            _preview_row(
                model_type=ModelType.AF3,
                model_name=model_name,
                folder=folder_path,
                model_index=model_index,
                structure_name=structure.name if structure else expected_structure_name,
                pae_name=pae.name if pae else expected_pae_name,
                summary_name=summary.name if summary_ok else "",
                summary_status=summary_status,
                ready=not error,
                error=error,
            )
        )
        if error:
            continue
        jobs.append(
            IpsaeJob(
                label=model_name,
                structure_file=structure,
                pae_file=pae,
                model_type=ModelType.AF3,
                pae_cutoff=float(pae_cutoff),
                dist_cutoff=float(dist_cutoff),
                summary_file=summary if summary_ok else None,
            )
        )

    preview = pd.DataFrame(rows, columns=PREVIEW_COLUMNS)
    if preview.empty:
        raise FileNotFoundError(
            f"No AlphaFold Server files found for model index {model_index}. "
            f"Expected files like fold_<name>_full_data_{model_index}.json with matching "
            f"fold_<name>_model_{model_index}.cif in the same folder."
        )
    return jobs, preview


def discover_boltz_models(
    folder: str | Path,
    model_index: int = 0,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> tuple[list[IpsaeJob], pd.DataFrame]:
    """Find Boltz pae_*_model_N.npz files and matching structure/summary siblings."""
    root = resolve_repo_path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    jobs: list[IpsaeJob] = []
    rows: list[dict] = []
    pae_by_key: dict[tuple[Path, str], Path] = {}
    structures_by_key: dict[tuple[Path, str], list[Path]] = {}

    for pae in sorted(p for p in root.rglob(f"pae_*_model_{model_index}.npz") if p.is_file()):
        info = parse_boltz_pae_pairing(pae)
        if info is not None and info.model_index == model_index:
            pae_by_key[(pae.parent.resolve(), info.complex_id)] = pae
    for suffix in ("cif", "pdb"):
        for structure in sorted(p for p in root.rglob(f"*_model_{model_index}.{suffix}") if p.is_file()):
            info = parse_boltz_structure_pairing(structure)
            if info is not None and info.model_index == model_index:
                key = (structure.parent.resolve(), info.complex_id)
                structures_by_key.setdefault(key, []).append(structure)

    for folder_key, complex_id in sorted(
        set(pae_by_key) | set(structures_by_key),
        key=lambda item: (str(item[0]), item[1]),
    ):
        key = (folder_key, complex_id)
        pae = pae_by_key.get(key)
        structures = sorted(structures_by_key.get(key, []))
        folder_path = (pae or structures[0]).parent
        expected_pae_name = f"pae_{complex_id}_model_{model_index}.npz"
        summary = expected_boltz_summary_path(folder_path / expected_pae_name)
        summary_ok = summary.is_file()
        summary_status = "present" if summary_ok else "missing (warning)"

        error = ""
        if len(structures) > 1:
            names = " | ".join(path.name for path in structures)
            error = f"Ambiguous Boltz structure: multiple counterparts exist: {names}."
        elif not structures:
            error = (
                "Missing structure counterpart "
                f"'{complex_id}_model_{model_index}.cif' or "
                f"'{complex_id}_model_{model_index}.pdb'."
            )
        elif pae is None:
            error = f"Missing PAE counterpart '{expected_pae_name}'."

        structure = structures[0] if len(structures) == 1 else None
        model_name = model_name_from_structure(structure) if structure else complex_id
        warning = "" if summary_ok else BOLTZ_MISSING_SUMMARY_WARNING
        rows.append(
            _preview_row(
                model_type=ModelType.BOLTZ,
                model_name=model_name,
                folder=folder_path,
                model_index=model_index,
                structure_name=(
                    structure.name
                    if structure
                    else " | ".join(path.name for path in structures)
                    or f"{complex_id}_model_{model_index}.cif|.pdb"
                ),
                pae_name=pae.name if pae else expected_pae_name,
                summary_name=summary.name if summary_ok else "",
                summary_status=summary_status,
                ready=not error,
                error=error or warning,
            )
        )
        if error:
            continue
        jobs.append(
            IpsaeJob(
                label=model_name,
                structure_file=structure,
                pae_file=pae,
                model_type=ModelType.BOLTZ,
                pae_cutoff=float(pae_cutoff),
                dist_cutoff=float(dist_cutoff),
                summary_file=summary if summary_ok else None,
            )
        )

    preview = pd.DataFrame(rows, columns=PREVIEW_COLUMNS)
    if preview.empty:
        raise FileNotFoundError(
            f"No Boltz files found for model index {model_index}. "
            f"Expected files like pae_<complex>_model_{model_index}.npz with matching "
            f"<complex>_model_{model_index}.pdb or .cif in the same folder."
        )
    return jobs, preview


def discover_bulk_models(
    folder: str | Path,
    model_index: int = 0,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> tuple[list[IpsaeJob], pd.DataFrame, ModelType]:
    """Auto-detect AF3 or Boltz under folder, then discover jobs for model_index."""
    root = resolve_repo_path(folder)
    model_type = detect_bulk_model_type(root)
    if model_type is ModelType.AF3:
        jobs, preview = discover_af3_models(
            root,
            model_index=model_index,
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
        )
    else:
        jobs, preview = discover_boltz_models(
            root,
            model_index=model_index,
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
        )
    return jobs, preview, model_type


discover_af3_model_jobs = discover_af3_models


def run_bulk_ipsae(
    jobs: list[IpsaeJob],
    overwrite: bool = False,
    collect_outputs: bool = False,
    output_dir: str | Path = DEFAULT_EVALS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    result_frames = []
    error_rows = []
    warnings: list[str] = []
    for job in jobs:
        try:
            scores, warning = run_ipsae(
                job,
                overwrite=overwrite,
                collect_outputs=collect_outputs,
                output_dir=output_dir,
            )
            result_frames.append(scores)
            if warning:
                warnings.append(f"{job.label}: {warning}")
        except Exception as exc:
            error_rows.append(
                {
                    "Model": job.label,
                    "Model_Type": job.resolved_model_type().value,
                    "Structure_File": rel_repo_path(resolve_repo_path(job.structure_file)),
                    "PAE_File": rel_repo_path(resolve_repo_path(job.pae_file)),
                    "Error": str(exc),
                }
            )

    scores = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    errors = pd.DataFrame(error_rows)
    return scores, errors, warnings
