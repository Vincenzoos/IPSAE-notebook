"""Reusable folder dropdown widgets for IPSAE notebook UIs.

Extracted from the zip-folder picker pattern: list immediate child folders
under the FreeBindCraft root, expose a Dropdown, and optionally a Refresh button.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from paths import ROOT
from ui_helpers import widgets

NO_FOLDERS = "(no folders)"


def list_repo_folders(base: str | Path | None = None) -> list[str]:
    """Return sorted immediate child directory names under ``base`` (default: repo ROOT)."""
    root = Path(base) if base is not None else ROOT
    root = root.expanduser().resolve()
    if not root.is_dir():
        return [NO_FOLDERS]
    folders = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    return folders or [NO_FOLDERS]


def refresh_folder_dropdown(
    dropdown: "widgets.Dropdown",
    *,
    base: str | Path | None = None,
    preferred: str | None = None,
) -> None:
    """Reload dropdown options, keeping the current or preferred selection when still valid."""
    options = list_repo_folders(base)
    current = preferred if preferred is not None else dropdown.value
    dropdown.options = options
    if current in options:
        dropdown.value = current
    else:
        dropdown.value = options[0]


def make_folder_dropdown(
    *,
    description: str = "Folder",
    width: str = "520px",
    base: str | Path | None = None,
    value: str | None = None,
    placeholder_when_empty: str | None = None,
) -> "widgets.Dropdown":
    """Create a Dropdown listing folders under ``base`` (default: repo ROOT)."""
    if widgets is None:
        raise RuntimeError("ipywidgets is required to create a folder dropdown.")

    options = list_repo_folders(base)
    if value and value in options:
        selected = value
    elif value and options == [NO_FOLDERS]:
        selected = NO_FOLDERS
    elif placeholder_when_empty and placeholder_when_empty in options:
        selected = placeholder_when_empty
    else:
        selected = options[0]

    return widgets.Dropdown(
        options=options,
        value=selected,
        description=description,
        layout=widgets.Layout(width=width),
    )


def make_folder_picker(
    *,
    description: str = "Folder",
    dropdown_width: str = "520px",
    refresh_width: str = "100px",
    base: str | Path | None = None,
    value: str | None = None,
    placeholder_when_empty: str | None = None,
    on_change: Callable | None = None,
) -> tuple["widgets.Dropdown", "widgets.Button", "widgets.HBox"]:
    """Build a folder Dropdown + Refresh button row.

    Returns ``(dropdown, refresh_button, row)``. The dropdown ``.value`` is the
    selected folder name relative to ``base`` (repo ROOT by default).
    """
    if widgets is None:
        raise RuntimeError("ipywidgets is required to create a folder picker.")

    dropdown = make_folder_dropdown(
        description=description,
        width=dropdown_width,
        base=base,
        value=value,
        placeholder_when_empty=placeholder_when_empty,
    )
    refresh_button = widgets.Button(
        description="Refresh",
        icon="refresh",
        layout=widgets.Layout(width=refresh_width),
    )

    def _refresh(_button=None) -> None:
        refresh_folder_dropdown(dropdown, base=base)

    refresh_button.on_click(_refresh)
    if on_change is not None:
        dropdown.observe(on_change, names="value")

    row = widgets.HBox([dropdown, refresh_button])
    return dropdown, refresh_button, row


def folder_value(dropdown: "widgets.Dropdown") -> str:
    """Return the selected folder name, or empty string when none are available."""
    value = str(dropdown.value or "").strip()
    if not value or value == NO_FOLDERS:
        return ""
    return value
