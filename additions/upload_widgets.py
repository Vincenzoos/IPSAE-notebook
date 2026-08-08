"""Reusable upload widgets for IPSAE notebook UIs."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from folder_picker import make_folder_picker, refresh_folder_dropdown
from paths import UPLOAD_FOLDERS_DIR, rel_repo_path
from ui_helpers import ERR, INFO, OK, SOFT, warning, widgets
from uploads import (
    STRUCTURE_EXTENSIONS,
    MAX_WIDGET_ZIP_BYTES,
    UploadError,
    describe_upload_path,
    ensure_upload_dirs,
    extract_uploaded_zip,
    extract_zip_file,
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
    """Build zip extract controls for bulk folder uploads.

    Large AlphaFold Server zips should be uploaded via the JupyterLab file
    browser, then extracted with a server path. The FileUpload widget is kept
    only for small archives (websocket size limit).
    """
    if widgets is None:
        raise RuntimeError("ipywidgets is required to create upload widgets.")

    ensure_upload_dirs()
    limit_mb = MAX_WIDGET_ZIP_BYTES // (1024 * 1024)
    state = {"busy": False}

    zip_path = widgets.Text(
        value="",
        description="Zip path",
        placeholder="e.g. AF3_outputs.zip or upload/files/AF3_outputs.zip",
        layout=widgets.Layout(width="760px"),
    )
    extract_btn = widgets.Button(
        description="Extract zip",
        button_style="info",
        icon="folder-open",
        layout=widgets.Layout(width="160px"),
    )
    uploader = widgets.FileUpload(
        accept=".zip",
        multiple=False,
        description="Small zip only",
        layout=widgets.Layout(width="320px"),
    )
    status = widgets.HTML(
        value=(
            f'<span style="{SOFT}">Recommended on Binder: upload the zip in the left file browser, '
            "paste its path above, then Extract zip.</span>"
        )
    )

    def set_status(message: str, style: str = SOFT) -> None:
        status.value = f'<div style="{style}">{message}</div>'

    def set_busy(busy: bool) -> None:
        state["busy"] = busy
        extract_btn.disabled = busy
        uploader.disabled = busy

    def finish_ok(dest: Path) -> None:
        set_status(f"Extracted to {describe_upload_path(dest)}", OK)
        on_extracted(dest)

    def run_extract(work: Callable[[], Path], pending_message: str) -> None:
        if state["busy"]:
            return
        set_busy(True)
        set_status(pending_message, INFO)

        def _worker() -> None:
            try:
                dest = work()
                finish_ok(dest)
            except Exception as exc:
                set_status(f"Zip extract failed: {exc}", ERR)
            finally:
                set_busy(False)

        threading.Thread(target=_worker, daemon=True).start()

    def on_extract_clicked(_button=None) -> None:
        path = zip_path.value.strip()
        if not path:
            set_status("Zip path is empty. Upload via the file browser, then paste the path.", ERR)
            return
        run_extract(lambda: extract_zip_file(path), f"Extracting {path}...")

    def on_upload(change=None) -> None:
        if not uploader.value or state["busy"]:
            return
        run_extract(
            lambda: extract_uploaded_zip(uploader),
            "Receiving widget upload and extracting (small zips only)...",
        )

    extract_btn.on_click(on_extract_clicked)
    uploader.observe(on_upload, names="value")

    return widgets.VBox(
        [
            warning(
                "AF3 Server export zips are often larger than the Jupyter widget upload limit "
                f"(~{limit_mb} MB). If Upload shows (1) and nothing appears under upload/folders, "
                "the transfer is stuck — use the file browser + Extract zip path instead."
            ),
            widgets.HBox([zip_path, extract_btn]),
            widgets.HTML(
                f'<span style="{SOFT}">Optional: widget upload for tiny zips only '
                f"(≤{limit_mb} MB).</span>"
            ),
            uploader,
            status,
        ]
    )


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
