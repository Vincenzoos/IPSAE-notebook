"""Shared ipywidgets helpers for IPSAE notebook UIs."""

from __future__ import annotations

try:
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display, update_display
except ImportError:  # Allows import in non-notebook contexts for tests/parsing.
    widgets = None
    clear_output = None
    display = print
    update_display = None
    HTML = None

BANNER = (
    "background:#e8d5f5;padding:12px 16px;border-radius:6px;"
    "margin:8px 0;font-family:sans-serif;"
)
SOFT = "color:#57606a;"
OK = "color:#1a7f37;font-weight:600;"
ERR = "color:#cf222e;font-weight:600;"
INFO = "color:#0969da;"
WARNING_CARD = (
    "background:#fff8c5;border:1px solid #d4a72c;color:#5c4400;"
    "padding:12px 16px;border-radius:6px;margin:8px 0;font-family:sans-serif;"
)
EXAMPLE_CARD = (
    "background:#f6f8fa;border:1px solid #d0d7de;color:#24292f;"
    "padding:12px 16px;border-radius:6px;margin:8px 0;font-family:monospace;"
)

AF2_OUTPUT_URL = (
    "https://github.com/google-deepmind/alphafold/blob/main/README.md#alphafold-output"
)
AF3_SERVER_OUTPUT_URL = (
    "https://www.ebi.ac.uk/training/online/courses/alphafold/alphafold-3-and-alphafold-server/"
    "alphafold-server-your-gateway-to-alphafold-3/interpreting-results-from-alphafold-server/"
)
BOLTZ_OUTPUT_URL = (
    "https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md#output"
)


def doc_link(url: str, label: str) -> str:
    return f'<a href="{url}" target="_blank">{label}</a>'


def banner(text: str) -> "widgets.HTML":
    return widgets.HTML(f'<div style="{BANNER}"><b>{text}</b></div>')


def html(text: str) -> "widgets.HTML":
    return widgets.HTML(text)


def warning(text: str) -> "widgets.HTML":
    return widgets.HTML(f'<div style="{WARNING_CARD}"><b>Warning:</b> {text}</div>')


def example(text: str) -> "widgets.HTML":
    return widgets.HTML(f'<pre style="{EXAMPLE_CARD}">{text}</pre>')


def panel(children) -> "widgets.VBox":
    return widgets.VBox(
        children,
        layout=widgets.Layout(
            width="100%",
            border="1px solid #d0d7de",
            padding="12px 16px",
            margin="0 0 8px 0",
        ),
    )


def publish_cell(display_id: str, obj, initialized: set[str]) -> None:
    if display_id in initialized:
        update_display(obj, display_id=display_id)
    else:
        display(obj, display_id=display_id)
        initialized.add(display_id)
