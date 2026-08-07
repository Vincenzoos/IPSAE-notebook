"""Reusable upload widgets for IPSAE notebook UIs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from folder_picker import make_folder_picker, refresh_folder_dropdown
from paths import UPLOAD_FOLDERS_DIR, rel_repo_path
from ui_helpers import ERR, INFO, OK, SOFT, widgets
from uploads import (
    STRUCTURE_EXTENSIONS,
    UploadError,
    describe_upload_path,
    ensure_upload_dirs,
    extract_uploaded_zip,
    save_uploaded_file,
    UPLOAD_FILES_DIR,
)


def make_single_file_upload_row(
    *,
    description: str,
    allowed_extensions: set[str] | frozenset[str],
    on_saved: Callable[[Path], None],
) -> "widgets.VBox":
    """Build a FileUpload row that saves one file and calls on_saved(path)."""
    if widgets is None:
        raise RuntimeError("ipywidgets is required to create upload widgets.")

    ensure_upload_dirs()
    accept = ",".join(sorted({ext if ext.startswith(".") else f".{ext}" for ext in allowed_extensions}))
    uploader = widgets.FileUpload(
        accept=accept,
        multiple=False,
        description=description,
        layout=widgets.Layout(width="320px"),
    )
    status = widgets.HTML(
        value=f'<span style="{SOFT}">No upload yet.</span>'
    )

    def set_status(message: str, style: str = SOFT) -> None:
        status.value = f'<div style="{style}">{message}</div>'

    def on_upload(change=None) -> None:
        if not uploader.value:
            return
        set_status("Saving upload...", INFO)
        try:
            paths = save_uploaded_file(uploader, UPLOAD_FILES_DIR, set(allowed_extensions))
            if len(paths) != 1:
                raise UploadError("Upload exactly one file.")
            path = paths[0]
            set_status(f"Saved {describe_upload_path(path)}", OK)
            on_saved(path)
        except Exception as exc:
            set_status(f"Upload failed: {exc}", ERR)

    uploader.observe(on_upload, names="value")
    return widgets.VBox([widgets.HBox([uploader]), status])


def make_zip_folder_upload_panel(
    *,
    on_extracted: Callable[[Path], None],
) -> "widgets.VBox":
    """Build a zip upload control and status line for bulk folder uploads."""
    if widgets is None:
        raise RuntimeError("ipywidgets is required to create upload widgets.")

    ensure_upload_dirs()
    uploader = widgets.FileUpload(
        accept=".zip",
        multiple=False,
        description="Upload AF3 zip",
        layout=widgets.Layout(width="320px"),
    )
    status = widgets.HTML(
        value=f'<span style="{SOFT}">Upload a zip of an AlphaFold Server output folder.</span>'
    )

    def set_status(message: str, style: str = SOFT) -> None:
        status.value = f'<div style="{style}">{message}</div>'

    def on_upload(change=None) -> None:
        if not uploader.value:
            return
        set_status("Extracting zip...", INFO)
        try:
            dest = extract_uploaded_zip(uploader)
            set_status(f"Extracted to {describe_upload_path(dest)}", OK)
            on_extracted(dest)
        except Exception as exc:
            set_status(f"Zip upload failed: {exc}", ERR)

    uploader.observe(on_upload, names="value")
    return widgets.VBox([uploader, status])


def make_uploaded_folder_picker(
    *,
    description: str = "Uploaded folder",
    on_change: Callable | None = None,
) -> tuple["widgets.Dropdown", "widgets.Button", "widgets.HBox"]:
    """Build a picker scoped to upload/folders."""
    ensure_upload_dirs()
    return make_folder_picker(
        description=description,
        dropdown_width="840px",
        base=UPLOAD_FOLDERS_DIR,
        on_change=on_change,
    )


def refresh_uploaded_folder_dropdown(
    dropdown: "widgets.Dropdown",
    *,
    preferred: str | None = None,
) -> None:
    """Reload the uploaded-folder dropdown options."""
    ensure_upload_dirs()
    refresh_folder_dropdown(dropdown, base=UPLOAD_FOLDERS_DIR, preferred=preferred)


def path_for_textbox(path: Path) -> str:
    """Prefer a repo-relative path string for path text boxes."""
    return rel_repo_path(Path(path).resolve())


# Keep STRUCTURE_EXTENSIONS import usable from UI callers without importing uploads.
__all__ = [
    "STRUCTURE_EXTENSIONS",
    "make_single_file_upload_row",
    "make_zip_folder_upload_panel",
    "make_uploaded_folder_picker",
    "refresh_uploaded_folder_dropdown",
    "path_for_textbox",
]
