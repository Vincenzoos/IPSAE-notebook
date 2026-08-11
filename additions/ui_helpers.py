"""Shared ipywidgets helpers for IPSAE notebook UIs."""

from __future__ import annotations

import sys

try:
    import ipywidgets as widgets
    from IPython.display import HTML, Javascript, clear_output, display, update_display
except ImportError:  # Allows import in non-notebook contexts for tests/parsing.
    widgets = None
    clear_output = None
    display = print
    update_display = None
    HTML = None
    Javascript = None

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


def enable_colab_iframe_resize(max_height: int = 10_000) -> None:
    """Let a Colab cell output grow with a tall, dynamically changing UI."""
    if "google.colab" not in sys.modules or Javascript is None:
        return
    display(
        Javascript(
            "google.colab.output.setIframeHeight(0, true, "
            f"{{maxHeight: {int(max_height)}}});"
        )
    )


def responsive_layout(max_width: str = "940px", **kwargs) -> "widgets.Layout":
    """Return a full-width layout capped at a comfortable desktop width."""
    return widgets.Layout(width="100%", max_width=max_width, **kwargs)


def scrollable_ui_layout(
    max_width: str = "940px",
    max_height: str = "800px",
) -> "widgets.Layout":
    """Keep a tall UI usable when its surrounding notebook output is clipped."""
    return responsive_layout(
        max_width,
        max_height=max_height,
        overflow="hidden auto",
        padding="0 8px 8px 0",
    )


def wrapping_row(children) -> "widgets.HBox":
    """Build a row that wraps instead of overflowing narrow notebook outputs."""
    return widgets.HBox(
        children,
        layout=widgets.Layout(width="100%", flex_flow="row wrap"),
    )


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
