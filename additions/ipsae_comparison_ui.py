"""
ipSAE comparison CSV UI for summarizing collected metric outputs.

Launch from notebooks/ipsae_eval.ipynb:

    from ipsae_comparison_ui import launch_ipsae_comparison_ui
    launch_ipsae_comparison_ui()
"""

from __future__ import annotations

from ipsae_evals_summary import resolve_summary_csv_path, summarize_ipsae_folder
from paths import rel_repo_path
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
    html,
    panel,
    publish_cell,
    responsive_layout,
    scrollable_ui_layout,
    widgets,
    HTML,
)

RESULT_MSG_ID = "ipsae-comparison-result-msg"
RESULT_TABLE_ID = "ipsae-comparison-result-table"


def launch_ipsae_comparison_ui() -> None:
    if widgets is None:
        raise RuntimeError("ipywidgets is required to launch this UI.")

    enable_colab_widget_manager()
    clear_output(wait=True)
    enable_colab_iframe_resize()

    state = {"running": False}
    cell_initialized: set[str] = set()

    summary_folder = widgets.Text(
        value="ipsae_evals",
        description="Metrics folder",
        placeholder="server path to an ipSAE output folder (e.g. ipsae_evals or bulk_ipsae_evals_...)",
        layout=responsive_layout(),
    )
    summary_csv = widgets.Text(
        value="ipsae_evals_summary.csv",
        description="Summary CSV",
        layout=responsive_layout(),
    )
    run_summary = widgets.Button(description="Summarize metrics", button_style="success", icon="bar-chart")
    summary_status = widgets.HTML(value=f'<span style="{SOFT}">Ready. Enter a metrics folder path, then summarize.</span>')

    def set_summary_status(message: str, style: str = SOFT) -> None:
        summary_status.value = f'<div style="{style}">{message}</div>'

    def show_summary_result(message: str, style: str = SOFT, table=None, section_title: str | None = None) -> None:
        publish_cell(
            RESULT_MSG_ID,
            HTML(f'<div style="{style};margin:8px 0;font-family:sans-serif">{message}</div>'),
            cell_initialized,
        )
        if section_title is not None:
            publish_cell(RESULT_TABLE_ID, HTML(f"<b>{section_title}</b>"), cell_initialized)
        if table is not None:
            publish_cell(f"{RESULT_TABLE_ID}-data", table, cell_initialized)

    def on_run_summary(_button) -> None:
        if state["running"]:
            return
        folder = summary_folder.value.strip() or "ipsae_evals"
        csv_name = summary_csv.value.strip() or "ipsae_evals_summary.csv"
        state["running"] = True
        run_summary.disabled = True
        set_summary_status("Summarizing ipSAE metrics...", INFO)
        show_summary_result("Summarizing ipSAE metrics...", INFO)

        try:
            summary, all_scores = summarize_ipsae_folder(folder)
            out_path = resolve_summary_csv_path(folder, csv_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out_path, index=False)
            msg = (
                f"SUCCESS: summarized {len(all_scores)} score file(s) from "
                f"{summary['Model'].nunique() if 'Model' in summary else len(summary)} model(s). "
                f"Saved to {rel_repo_path(out_path)}"
            )
            set_summary_status(msg, OK)
            show_summary_result(msg, OK, table=summary, section_title="models evaluation ranked by ipSAE score")
        except Exception as exc:
            set_summary_status(f"FAILED: {exc}", ERR)
            show_summary_result(f"FAILED: {exc}", ERR)
        finally:
            state["running"] = False
            run_summary.disabled = False

    run_summary.on_click(on_run_summary)

    comparison_panel = panel(
        [
            html(
                f'<span style="{SOFT}">Enter an ipSAE output folder path to summarize the results.</span>'
            ),
            summary_folder,
            summary_csv,
            run_summary,
        ]
    )

    display(
        widgets.VBox(
            [
                banner("ipSAE eval summary"),
                comparison_panel,
                summary_status,
            ],
            layout=scrollable_ui_layout(),
        )
    )
    show_summary_result(
        "Ready. Enter an ipSAE output folder path, then summarize to show results here.",
        SOFT,
    )


if __name__ == "__main__":
    print("Launch from notebooks/ipsae_eval.ipynb via launch_ipsae_comparison_ui().")
