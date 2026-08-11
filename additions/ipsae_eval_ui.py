"""
ipSAE evaluation UI for single-model and bulk model scoring.

Launch from notebooks/ipsae_eval.ipynb:

    from ipsae_eval_ui import launch_ipsae_eval_ui
    launch_ipsae_eval_ui()

The notebook stays intentionally small; execution and output handling live here.
DunbrackLab/IPSAE itself remains in ../ipsae.py.
"""

from __future__ import annotations

import threading
from pathlib import Path

from bulk_eval import discover_bulk_models, new_bulk_output_dir, run_bulk_ipsae, write_bulk_run_log
from model_pairing import MODEL_TYPE_CHOICES, ModelType
from naming import model_name_from_structure, safe_name
from paths import DEFAULT_EVALS_DIR, IPSAE_SCRIPT, rel_repo_path, resolve_repo_path
from single_model_eval import IpsaeJob, run_ipsae
from single_model_ui_state import (
    clear_incompatible_paths,
    hint_text_for,
    maybe_prefill_boltz_summary,
    placeholders_for,
    selected_model_type,
    single_inputs_locked,
    single_run_ready,
    upload_extensions_for,
)
from ui_helpers import (
    ERR,
    INFO,
    OK,
    SOFT,
    banner,
    clear_output,
    display,
    enable_colab_iframe_resize,
    enable_colab_widget_manager,
    example,
    html,
    publish_cell,
    responsive_layout,
    scrollable_ui_layout,
    warning,
    wrapping_row,
    widgets,
    HTML,
)
from upload_widgets import (
    make_single_file_upload_row,
    make_zip_folder_upload_panel,
    path_for_textbox,
)
from uploads import ensure_upload_dirs

AF3_SERVER_OUTPUT_URL = "https://www.ebi.ac.uk/training/online/courses/alphafold/alphafold-3-and-alphafold-server/alphafold-server-your-gateway-to-alphafold-3/interpreting-results-from-alphafold-server/"

RESULT_MSG_ID = "ipsae-eval-result-msg"
RESULT_TABLE_ID = "ipsae-eval-result-table"
RESULT_EXTRA_ID = "ipsae-eval-result-extra"


def launch_ipsae_eval_ui() -> None:
    if widgets is None:
        raise RuntimeError("ipywidgets is required to launch this UI.")

    # Package installation earlier in a Colab session can reset the widget
    # frontend. Re-enable it immediately before publishing widget views.
    enable_colab_widget_manager()

    # Replace any previously rendered copy of this UI when the notebook cell is rerun.
    clear_output(wait=True)
    enable_colab_iframe_resize()
    ensure_upload_dirs()

    status = widgets.HTML(value=f'<span style="{SOFT}">Ready. Choose single or bulk evaluation, then run ipSAE.</span>')
    state = {
        "running": False,
        "bulk_jobs": [],
        "bulk_source": None,
        "bulk_type": None,
        "auto_summary": None,
    }
    cell_initialized: set[str] = set()

    pae_cutoff = widgets.FloatText(value=10.0, description="PAE cutoff", layout=widgets.Layout(width="220px"))
    dist_cutoff = widgets.FloatText(value=10.0, description="Dist cutoff", layout=widgets.Layout(width="220px"))
    collect_outputs = widgets.Checkbox(
        value=False,
        description="Copy outputs to separate folder",
        disabled=True,
    )
    output_dir = widgets.Text(
        value="ipsae_evals",
        description="Output dir",
        disabled=True,
        layout=responsive_layout("700px"),
    )

    model_type = widgets.Dropdown(
        options=list(MODEL_TYPE_CHOICES),
        value="",
        description="Model type",
        layout=widgets.Layout(width="280px"),
    )
    type_hint = widgets.HTML(
        value=f'<div style="{SOFT}">{hint_text_for(None)}</div>',
        layout=responsive_layout("660px"),
    )

    label = widgets.Text(
        value="",
        description="Label",
        placeholder="defaults to structure filename",
        disabled=True,
        layout=responsive_layout("520px"),
    )
    structure_path = widgets.Text(
        value="",
        description="Structure",
        placeholder=placeholders_for(None)["structure"],
        disabled=True,
        layout=responsive_layout(),
    )
    pae_path = widgets.Text(
        value="",
        description="PAE",
        placeholder=placeholders_for(None)["pae"],
        disabled=True,
        layout=responsive_layout(),
    )
    summary_path = widgets.Text(
        value="",
        description="Boltz summary",
        placeholder=placeholders_for(None)["summary"],
        disabled=True,
        layout=responsive_layout(),
    )
    run_single = widgets.Button(description="Run ipSAE", button_style="primary", icon="play", disabled=True)
    discover_bulk = widgets.Button(description="Find models", button_style="info", icon="search", disabled=True)
    run_bulk = widgets.Button(description="Run ipSAE", button_style="primary", icon="play", disabled=True)

    bulk_folder_path = widgets.Text(
        value="",
        description="Folder",
        placeholder="server path to AF3 Server or Boltz export folder (or extract a zip below first)",
        layout=responsive_layout(),
    )
    bulk_model_index = widgets.BoundedIntText(
        value=0,
        min=0,
        description="Model index",
        layout=widgets.Layout(width="180px"),
    )

    def on_structure_saved(path: Path) -> None:
        structure_path.value = path_for_textbox(path)
        sync_single_controls(state["running"])

    def on_pae_saved(path: Path) -> None:
        pae_path.value = path_for_textbox(path)
        maybe_auto_summary()
        sync_single_controls(state["running"])

    def on_summary_saved(path: Path) -> None:
        state["auto_summary"] = None
        summary_path.value = path_for_textbox(path)
        sync_single_controls(state["running"])

    structure_upload, set_structure_ext, set_structure_disabled = make_single_file_upload_row(
        description="Upload structure",
        allowed_extensions=upload_extensions_for(None)["structure"],
        on_saved=on_structure_saved,
        disabled=True,
    )
    pae_upload, set_pae_ext, set_pae_disabled = make_single_file_upload_row(
        description="Upload PAE",
        allowed_extensions=upload_extensions_for(None)["pae"],
        on_saved=on_pae_saved,
        disabled=True,
    )
    summary_upload, set_summary_ext, set_summary_disabled = make_single_file_upload_row(
        description="Upload Boltz summary",
        allowed_extensions=upload_extensions_for(None)["summary"],
        on_saved=on_summary_saved,
        disabled=True,
    )
    summary_box = widgets.VBox([summary_path, summary_upload])
    summary_box.layout.display = "none"

    def on_zip_extracted(path: Path) -> None:
        bulk_folder_path.value = path_for_textbox(path)
        on_bulk_input_changed()

    zip_upload_panel = make_zip_folder_upload_panel(on_extracted=on_zip_extracted)

    settings_panel = widgets.VBox([wrapping_row([pae_cutoff, dist_cutoff])])

    single_panel = widgets.VBox(
        [
            html(
                f'<span style="{SOFT}">Select a model type, then provide one matching structure and PAE file. '
                "Type a server path or upload files below. Pairing is validated for the selected type "
                "(not inferred from filenames).</span>"
            ),
            wrapping_row([model_type, type_hint]),
            label,
            structure_path,
            structure_upload,
            pae_path,
            pae_upload,
            summary_box,
            collect_outputs,
            output_dir,
            run_single,
        ]
    )

    bulk_panel = widgets.VBox(
        [
            html(
                f'<span style="{SOFT}">Select a model-output folder already on the server, '
                "or upload a zip via the JupyterLab file browser and Extract zip below. "
                "Bulk mode auto-detects AlphaFold Server or Boltz from PAE filenames, "
                "then discovers matching structure files for the selected model index.</span>"
            ),
            warning(
                "AF2 / ColabFold bulk discovery is not supported in v1. Use Single Model for AlphaFold2 outputs. "
                "Bulk supports AlphaFold Server (*_full_data_N.json) and Boltz (pae_*_model_N.npz) folders only."
            ),
            example(
                "Example AlphaFold Server layout\n"
                "AF3_outputs/\n"
                "  fold_binder_001/\n"
                "    fold_binder_001_model_0.cif\n"
                "    fold_binder_001_full_data_0.json\n"
                "    fold_binder_001_summary_confidences_0.json\n"
                "\n"
                "Example Boltz layout\n"
                "Boltz_outputs/\n"
                "  AURKA_TPX2/\n"
                "    AURKA_TPX2_model_0.cif\n"
                "    pae_AURKA_TPX2_model_0.npz\n"
                "    confidence_AURKA_TPX2_model_0.json"
            ),
            html(
                f'<span style="{SOFT}">Model index defaults to 0 (best-ranked / default model for both AF3 and Boltz). '
                f'<a href="{AF3_SERVER_OUTPUT_URL}" target="_blank">Official AlphaFold Server output reference</a>.</span>'
            ),
            bulk_folder_path,
            zip_upload_panel,
            bulk_model_index,
            discover_bulk,
            run_bulk,
        ]
    )

    tabs = widgets.Tab(children=[single_panel, bulk_panel])
    tabs.set_title(0, "Single Model")
    tabs.set_title(1, "Bulk Evaluation")

    def set_status(message: str, style: str = SOFT) -> None:
        status.value = f'<div style="{style}">{message}</div>'

    def show_cell_result(
        message: str,
        style: str = SOFT,
        table=None,
        extra_title: str | None = None,
        extra_table=None,
    ) -> None:
        publish_cell(
            RESULT_MSG_ID,
            HTML(f'<div style="{style};margin:8px 0;font-family:sans-serif">{message}</div>'),
            cell_initialized,
        )
        if table is not None:
            publish_cell(RESULT_TABLE_ID, table, cell_initialized)
        if extra_title is not None and extra_table is not None:
            publish_cell(RESULT_EXTRA_ID, HTML(f"<b>{extra_title}</b>"), cell_initialized)
            publish_cell(f"{RESULT_EXTRA_ID}-table", extra_table, cell_initialized)

    def current_model_type() -> ModelType | None:
        return selected_model_type(model_type.value)

    def maybe_auto_summary() -> None:
        if current_model_type() is not ModelType.BOLTZ:
            return
        current = summary_path.value.strip()
        previous_auto = state["auto_summary"]
        # Preserve a genuinely user-entered/uploaded value, but allow a prior
        # auto-fill to follow the PAE selection or clear when no sibling exists.
        if current and current != previous_auto:
            state["auto_summary"] = None
            return
        prefill = maybe_prefill_boltz_summary(pae_path.value, "")
        if prefill:
            display_path = path_for_textbox(Path(prefill))
            state["auto_summary"] = display_path
            summary_path.value = display_path
        elif previous_auto is not None:
            state["auto_summary"] = None
            summary_path.value = ""

    def apply_model_type_ui(_change=None) -> None:
        mt = current_model_type()
        placeholders = placeholders_for(mt)
        extensions = upload_extensions_for(mt)
        type_hint.value = f'<div style="{SOFT}">{hint_text_for(mt)}</div>'
        structure_path.placeholder = placeholders["structure"]
        pae_path.placeholder = placeholders["pae"]
        summary_path.placeholder = placeholders["summary"]
        set_structure_ext(extensions["structure"])
        set_pae_ext(extensions["pae"])
        set_summary_ext(extensions["summary"])

        cleared = clear_incompatible_paths(
            model_type=mt,
            structure=structure_path.value,
            pae=pae_path.value,
            summary=summary_path.value,
        )
        structure_path.value = cleared["structure"]
        pae_path.value = cleared["pae"]
        summary_path.value = cleared["summary"]

        if mt is ModelType.BOLTZ:
            summary_box.layout.display = None
            maybe_auto_summary()
        else:
            summary_box.layout.display = "none"
            state["auto_summary"] = None
            summary_path.value = ""
        sync_single_controls(state["running"])

    def current_job() -> IpsaeJob:
        mt = current_model_type()
        if mt is None:
            raise ValueError("Select a model type before running.")
        structure = structure_path.value.strip()
        pae = pae_path.value.strip()
        if not structure:
            raise ValueError("Structure path is empty.")
        if not pae:
            raise ValueError("PAE path is empty.")
        summary = summary_path.value.strip() or None
        if mt is not ModelType.BOLTZ:
            summary = None
        return IpsaeJob(
            label=label.value.strip() or model_name_from_structure(structure),
            pae_file=Path(pae),
            structure_file=Path(structure),
            model_type=mt,
            pae_cutoff=float(pae_cutoff.value),
            dist_cutoff=float(dist_cutoff.value),
            summary_file=Path(summary) if summary else None,
        )

    def selected_bulk_folder() -> str:
        return bulk_folder_path.value.strip()

    def bulk_input_key() -> tuple[str, int]:
        return (selected_bulk_folder(), int(bulk_model_index.value))

    def sync_single_controls(disabled: bool = False) -> None:
        mt = current_model_type()
        locked = single_inputs_locked(model_type=mt, running=disabled)
        label.disabled = locked
        structure_path.disabled = locked
        pae_path.disabled = locked
        summary_path.disabled = locked
        set_structure_disabled(locked)
        set_pae_disabled(locked)
        set_summary_disabled(locked)
        collect_outputs.disabled = locked
        output_dir.disabled = locked or not collect_outputs.value
        ready = single_run_ready(
            model_type=mt,
            structure=structure_path.value,
            pae=pae_path.value,
            running=disabled,
        )
        run_single.disabled = not ready

    def sync_bulk_controls(disabled: bool = False) -> None:
        has_folder = bool(selected_bulk_folder())
        discover_bulk.disabled = disabled or not has_folder
        run_bulk.disabled = disabled or not state["bulk_jobs"] or state["bulk_source"] != bulk_input_key()

    def set_buttons_disabled(disabled: bool) -> None:
        sync_single_controls(disabled)
        sync_bulk_controls(disabled)

    def on_bulk_input_changed(_change=None) -> None:
        state["bulk_jobs"] = []
        state["bulk_source"] = None
        state["bulk_type"] = None
        sync_bulk_controls(state["running"])

    def on_run_single(_button) -> None:
        if state["running"]:
            return
        try:
            job = current_job()
        except Exception as exc:
            set_status(f"FAILED: {exc}", ERR)
            show_cell_result(f"FAILED: {exc}", ERR)
            return

        collect = collect_outputs.value
        eval_output_dir = output_dir.value.strip() or DEFAULT_EVALS_DIR
        state["running"] = True
        set_buttons_disabled(True)
        set_status("Running ipSAE...", INFO)
        show_cell_result("Running ipSAE...", INFO)

        try:
            scores, warning_msg = run_ipsae(
                job,
                overwrite=True,
                collect_outputs=collect,
                output_dir=eval_output_dir,
            )
            structure = resolve_repo_path(job.structure_file)
            msg = f"SUCCESS: ipSAE completed. Outputs saved to {rel_repo_path(structure.parent)}"
            if collect:
                copied_dir = resolve_repo_path(eval_output_dir) / safe_name(job.label)
                msg += f"<br/>Copied outputs to {rel_repo_path(copied_dir)}"
            if warning_msg:
                msg += f'<br/><span style="color:#5c4400;font-weight:600">Warning: {warning_msg}</span>'
            style = INFO if warning_msg else OK
            set_status(msg, style)
            show_cell_result(msg, style, table=scores)
        except Exception as exc:
            set_status(f"FAILED: {exc}", ERR)
            show_cell_result(f"FAILED: {exc}", ERR)
        finally:
            state["running"] = False
            set_buttons_disabled(False)

    def on_discover_bulk(_button) -> None:
        if state["running"]:
            return
        try:
            jobs, preview, detected = discover_bulk_models(
                selected_bulk_folder(),
                model_index=int(bulk_model_index.value),
                pae_cutoff=float(pae_cutoff.value),
                dist_cutoff=float(dist_cutoff.value),
            )
            state["bulk_jobs"] = jobs
            state["bulk_source"] = bulk_input_key()
            state["bulk_type"] = detected
            sync_bulk_controls(False)
            ready_count = int(preview["Ready"].sum()) if "Ready" in preview.columns else len(jobs)
            warned = 0
            if "SummaryStatus" in preview.columns:
                warned = int(preview["SummaryStatus"].astype(str).str.contains("warning", case=False).sum())
            if not jobs:
                msg = (
                    f"Detected {detected.label}, but no ready models for index "
                    f"{int(bulk_model_index.value)}. Review incomplete or ambiguous rows below."
                )
                set_status(msg, ERR)
                show_cell_result(msg, ERR, table=preview)
                return
            msg = (
                f"Detected {detected.label}. Found {ready_count}/{len(preview)} ready model(s) "
                f"for index {int(bulk_model_index.value)}. Review the table, then run ipSAE."
            )
            if warned:
                msg += f" {warned} Boltz row(s) are missing a summary file (runnable with warning)."
            set_status(msg, OK if ready_count == len(preview) and not warned else INFO)
            show_cell_result(msg, OK if ready_count == len(preview) and not warned else INFO, table=preview)
        except Exception as exc:
            state["bulk_jobs"] = []
            state["bulk_source"] = None
            state["bulk_type"] = None
            sync_bulk_controls(False)
            set_status(f"FAILED: {exc}", ERR)
            show_cell_result(f"FAILED: {exc}", ERR)

    def on_run_bulk(_button) -> None:
        if state["running"]:
            return
        if not state["bulk_jobs"] or state["bulk_source"] != bulk_input_key():
            set_status("No models found for the current folder/model selection. Click Find models first.", ERR)
            sync_bulk_controls(False)
            return

        jobs = list(state["bulk_jobs"])
        source_folder = selected_bulk_folder()
        model_index = int(bulk_model_index.value)
        detected = state["bulk_type"]
        state["running"] = True
        set_buttons_disabled(True)
        set_status("Running bulk ipSAE...", INFO)
        show_cell_result("Running bulk ipSAE...", INFO)

        def work() -> None:
            try:
                bulk_out_dir = new_bulk_output_dir()
                scores, errors, warnings = run_bulk_ipsae(
                    jobs,
                    overwrite=True,
                    collect_outputs=True,
                    output_dir=bulk_out_dir,
                )
                ok_count = len(jobs) - len(errors)
                log_path = write_bulk_run_log(
                    bulk_out_dir,
                    source_folder,
                    model_index,
                    jobs,
                    ok_count,
                    len(errors),
                    detected_type=detected,
                )
                msg = f"SUCCESS: bulk ipSAE finished for {ok_count}/{len(jobs)} model(s)."
                msg += f"<br/>Logged folder/model selection to {rel_repo_path(log_path)}"
                if ok_count:
                    msg += f"<br/>Copied ipSAE outputs to {rel_repo_path(bulk_out_dir)}"
                    job = jobs[0]
                    pae_tag = str(int(job.pae_cutoff))
                    dist_tag = str(int(job.dist_cutoff))
                    suffix = f"{pae_tag}_{dist_tag}"
                    msg += (
                        f'<br/><span style="{SOFT}">Each model gets 3 score files from one run: '
                        f"<code>*_{suffix}.txt</code> (summary), "
                        f"<code>*_{suffix}_byres.txt</code> (per-residue), "
                        f"<code>*_{suffix}.pml</code> (PyMOL).</span>"
                    )
                if warnings:
                    msg += (
                        f'<br/><span style="color:#5c4400;font-weight:600">'
                        f"Warnings ({len(warnings)}): {warnings[0]}"
                        + (f" …and {len(warnings) - 1} more." if len(warnings) > 1 else "")
                        + "</span>"
                    )
                if not errors.empty:
                    msg += f"<br/>{len(errors)} model(s) failed; see errors below."
                style = OK if errors.empty and not warnings else INFO
                set_status(msg, style)
                show_cell_result(
                    msg,
                    style,
                    table=scores if not scores.empty else None,
                    extra_title="Errors" if not errors.empty else None,
                    extra_table=errors if not errors.empty else None,
                )
            except Exception as exc:
                failure = f"FAILED: {exc}"
                set_status(failure, ERR)
                show_cell_result(failure, ERR)
            finally:
                state["running"] = False
                set_buttons_disabled(False)

        threading.Thread(target=work, daemon=True).start()

    def on_pae_path_change(_change=None) -> None:
        maybe_auto_summary()
        sync_single_controls(state["running"])

    def on_summary_path_change(change=None) -> None:
        new_value = ((change or {}).get("new") or summary_path.value).strip()
        if new_value != state["auto_summary"]:
            state["auto_summary"] = None
        sync_single_controls(state["running"])

    collect_outputs.observe(lambda _change: sync_single_controls(state["running"]), names="value")
    model_type.observe(apply_model_type_ui, names="value")
    structure_path.observe(lambda change: sync_single_controls(state["running"]), names="value")
    pae_path.observe(on_pae_path_change, names="value")
    summary_path.observe(on_summary_path_change, names="value")
    bulk_folder_path.observe(on_bulk_input_changed, names="value")
    bulk_model_index.observe(on_bulk_input_changed, names="value")
    apply_model_type_ui()
    sync_bulk_controls(False)
    run_single.on_click(on_run_single)
    discover_bulk.on_click(on_discover_bulk)
    run_bulk.on_click(on_run_bulk)

    display(
        widgets.VBox(
            [
                banner("ipSAE Evaluation"),
                html(
                    f'<span style="{SOFT}">Using original DunbrackLab script: {rel_repo_path(IPSAE_SCRIPT)}. '
                    "Single Model supports AF2, AF3 Server, and Boltz. Bulk auto-detects AF3 Server or Boltz.</span>"
                ),
                settings_panel,
                tabs,
                status,
            ],
            layout=scrollable_ui_layout(),
        )
    )
    show_cell_result(
        "Ready. Run ipSAE to show results here in the notebook cell output.",
        SOFT,
    )


if __name__ == "__main__":
    print(f"IPSAE script: {IPSAE_SCRIPT}")
    print("Launch from notebooks/ipsae_eval.ipynb via launch_ipsae_eval_ui().")
