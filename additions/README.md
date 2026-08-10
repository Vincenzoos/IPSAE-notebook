# Adds-ons to original IPSAE

This folder contains code added to the vendored
DunbrackLab/IPSAE repository.

Original upstream IPSAE: https://github.com/DunbrackLab/IPSAE

- `ipsae.py`
- `README.md`
- `LICENSE`

These added files live in `additions/`:

- `paths.py`: repository root constants, project-root resolution, and upload path helpers.
- `naming.py`: model/output filename helpers for ipSAE runs.
- `model_pairing.py`: AF2 / AF3 Server / Boltz filename pairing, validation, and bulk type detection.
- `af3_pairing.py`: thin AF3-only compatibility shim over `model_pairing.py`.
- `single_model_ui_state.py`: pure helpers for Single Model type-specific UI state.
- `ui_helpers.py`: shared ipywidgets styling and display helpers for notebook UIs.
- `folder_picker.py`: reusable folder dropdown + refresh controls for notebook UIs.
- `uploads.py`: upload normalization, file saving, and safe zip extraction under `upload/`.
- `upload_widgets.py`: reusable FileUpload / zip / uploaded-folder picker widgets.
- `ipsae_eval_ui.py`: ipywidgets UI wiring for single-model and bulk evaluation.
- `ipsae_comparison_ui.py`: ipywidgets UI for summarizing collected outputs into a comparison CSV.
- `single_model_eval.py`: single-model execution, output collection, and score parsing.
- `bulk_eval.py`: AF3 Server / Boltz folder discovery, batch execution, and run logging.
- `ipsae_evals_summary.py`: collected-output discovery and best-ipSAE summary CSV generation.
- `tests/`: focused unit tests for uploads, pairing/discovery, and score parsing.
- `README.md`: this note.

The UI calls the original upstream `../ipsae.py` script; it does not replace
DunbrackLab scoring logic (aside from a Boltz companion-path basename fix).

## v1 support matrix

| Model type | Single Model | Bulk Evaluation |
|---|---:|---:|
| AlphaFold2 | Yes | No |
| AlphaFold3 Server | Yes | Yes |
| Boltz | Yes | Yes |

- Single Model requires an explicit **Model type** (no Auto).
- Bulk auto-detects AF3 Server vs Boltz from PAE signatures and has no type selector.
- AF2 bulk is unsupported in v1.
- AF3 support is limited to AlphaFold Server naming (`*_model_N.cif` + `*_full_data_N.json`).

## Accepted filename contracts

Filename extensions must be lowercase because the `ipsae.py` format dispatch is
case-sensitive.

### AlphaFold2 — Single Model only

- Structure: `<complex>_unrelaxed_rank_<rank>_<details>.pdb`
- PAE: `<complex>_scores_rank_<rank>_<details>.json`
- Example:
  - `RAF1_KSR1_unrelaxed_rank_001_alphafold2_multimer_v3_model_4_seed_003.pdb`
  - `RAF1_KSR1_scores_rank_001_alphafold2_multimer_v3_model_4_seed_003.json`

### AlphaFold3 Server — Single and Bulk

- Structure: `<complex>_model_<N>.cif`
- PAE: `<complex>_full_data_<N>.json`
- Optional summary: `<complex>_summary_confidences_<N>.json`
- Example:
  - `fold_aurka_tpx2_model_0.cif`
  - `fold_aurka_tpx2_full_data_0.json`
  - `fold_aurka_tpx2_summary_confidences_0.json`

### Boltz — Single and Bulk

- Structure: `<complex>_model_<N>.pdb` or `.cif`
- PAE: `pae_<complex>_model_<N>.npz`
- Optional summary: `confidence_<complex>_model_<N>.json`
- Example:
  - `AURKA_TPX2_model_0.cif`
  - `pae_AURKA_TPX2_model_0.npz`
  - `confidence_AURKA_TPX2_model_0.json`
- A missing Boltz summary does not block the run; the UI shows a soft warning because Boltz ipTM values may be unavailable/zero.

## Single-model validation

`single_model_eval.py` validates using the selected model type before calling
`ipsae.py`. Extensions, same-folder placement, and name pairing are checked.
Changing the model type updates upload filters, placeholders, and clears
incompatible paths. Boltz shows an optional summary upload row.

## File and folder uploads

Uploads are stored under the project root:

```text
<PROJECT_ROOT>/upload/files/          # single structure / PAE / summary uploads
<PROJECT_ROOT>/upload/folders/<stem>/ # extracted bulk zip uploads
```

Single Model upload extensions depend on the selected type (AF2 `.pdb`+`.json`;
AF3 `.cif`+`.json`; Boltz `.pdb`/`.cif`+`.npz`, optional `.json` summary).

Bulk Evaluation: upload a zip via the JupyterLab file browser, paste the path into
**Zip path**, and click **Extract zip** (up to 2 GB). Extracted zips land in
`upload/folders/<zip-stem>/`. You can also type any server-side folder path into
**Folder**.

## Bulk evaluation

Point Bulk Evaluation at a parent folder of AlphaFold Server or Boltz outputs.
Click **Find models** to auto-detect the type and preview candidates for the
selected **Model index** (default `0`).

```text
AF3_outputs/
  fold_binder_001/
    fold_binder_001_model_0.cif
    fold_binder_001_full_data_0.json
    fold_binder_001_summary_confidences_0.json

Boltz_outputs/
  AURKA_TPX2/
    AURKA_TPX2_model_0.cif
    pae_AURKA_TPX2_model_0.npz
    confidence_AURKA_TPX2_model_0.json
```

Detection uses PAE signatures only (`*_full_data_N.json` vs `pae_*_model_N.npz`).
Mixed AF3/Boltz folders are rejected. Incomplete and ambiguous candidates appear
in the preview; only unambiguous pairs become runnable jobs.

Each bulk run creates `bulk_ipsae_evals_YYYYMMDD_HHMMSS/` with per-model output
subfolders and a `*_log.json` recording `source_folder`, detected type, model
index, and job paths.

Official AlphaFold Server output reference:
https://www.ebi.ac.uk/training/online/courses/alphafold/alphafold-3-and-alphafold-server/alphafold-server-your-gateway-to-alphafold-3/interpreting-results-from-alphafold-server/

## Summary comparison

Summary comparison is separate from running ipSAE. Use the **ipSAE Comparison CSV**
cell in `ipsae_eval.ipynb` to enter a folder path that already contains collected
ipSAE metric outputs, such as `ipsae_evals/` or a `bulk_ipsae_evals_YYYYMMDD_HHMMSS/`
folder. The summary ranks models by the best ipSAE row per model and can save the
comparison table to CSV.
