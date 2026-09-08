"""Reusable upload widgets for IPSAE notebook UIs."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from folder_picker import make_folder_picker, refresh_folder_dropdown
from paths import UPLOAD_FOLDERS_DIR, rel_repo_path
from ui_helpers import (
    AF3_SERVER_OUTPUT_URL,
    BOLTZ_OUTPUT_URL,
    doc_link,
    ERR,
    INFO,
    OK,
    SOFT,
    warning,
    widgets,
)
from uploads import (
    STRUCTURE_EXTENSIONS,
    MAX_ZIP_BYTES,
    UploadError,
    describe_upload_path,
    ensure_upload_dirs,
    extract_uploaded_zip,
    extract_zip_file,
    save_uploaded_file,
    UPLOAD_FILES_DIR,
)


def _accept_string(allowed_extensions: set[str] | frozenset[str]) -> str:
    return ",".join(
        sorted({ext if ext.startswith(".") else f".{ext}" for ext in allowed_extensions})
    )


def make_single_file_upload_row(
    *,
    description: str,
    allowed_extensions: set[str] | frozenset[str],
    on_saved: Callable[[Path], None],
    disabled: bool = False,
) -> tuple[
    "widgets.VBox",
    Callable[[set[str] | frozenset[str]], None],
    Callable[[bool], None],
]:
    """Build a FileUpload row that saves one file and calls on_saved(path).

    Returns ``(widget, set_allowed_extensions, set_disabled)`` so callers can
    update accept filters and lock the control when model type is unset.
    """
    if widgets is None:
        raise RuntimeError("ipywidgets is required to create upload widgets.")

    ensure_upload_dirs()
    state = {"allowed": set(allowed_extensions)}
    uploader = widgets.FileUpload(
        accept=_accept_string(state["allowed"]),
        multiple=False,
        description=description,
        disabled=disabled,
        layout=widgets.Layout(width="320px"),
    )
    status = widgets.HTML(
        value=f'<span style="{SOFT}">No upload yet.</span>'
    )

    def set_status(message: str, style: str = SOFT) -> None:
        status.value = f'<div style="{style}">{message}</div>'

    def set_allowed_extensions(extensions: set[str] | frozenset[str]) -> None:
        state["allowed"] = set(extensions)
        uploader.accept = _accept_string(state["allowed"])

    def set_disabled(locked: bool) -> None:
        uploader.disabled = locked

    def on_upload(change=None) -> None:
        if uploader.disabled or not uploader.value:
            return
        set_status("Saving upload...", INFO)
        try:
            paths = save_uploaded_file(uploader, UPLOAD_FILES_DIR, set(state["allowed"]))
            if len(paths) != 1:
                raise UploadError("Upload exactly one file.")
            path = paths[0]
            set_status(f"Saved {describe_upload_path(path)}", OK)
            on_saved(path)
        except Exception as exc:
            set_status(f"Upload failed: {exc}", ERR)

    uploader.observe(on_upload, names="value")
    return (
        widgets.VBox([widgets.HBox([uploader]), status]),
        set_allowed_extensions,
        set_disabled,
    )


def make_zip_folder_upload_panel(
    *,
    on_extracted: Callable[[Path], None],
) -> "widgets.VBox":
    """Build zip extract controls for bulk model-output folder uploads.

    Full export zips should be uploaded via the JupyterLab file browser (HTTP),
    then extracted with a server path. Archive size is capped at
    ``MAX_ZIP_BYTES`` (zip-slip / bomb checks still apply).
    """
    if widgets is None:
        raise RuntimeError("ipywidgets is required to create upload widgets.")

    ensure_upload_dirs()
    limit_gb = MAX_ZIP_BYTES // (1024 * 1024 * 1024)
    state = {"busy": False}

    zip_path = widgets.Text(
        value="",
        description="Zip path",
        placeholder="e.g. model_outputs.zip or upload/files/model_outputs.zip",
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
        description="Upload zip",
        layout=widgets.Layout(width="320px"),
    )
    status = widgets.HTML(
        value=(
            f'<span style="{SOFT}">If using file browser instead of the upload widget for uploading'
            f"{doc_link(AF3_SERVER_OUTPUT_URL, 'AF3 Server')} or "
            f"{doc_link(BOLTZ_OUTPUT_URL, 'Boltz 1 / Boltz 2')} zip file"
            f"(up to {limit_gb} GB), then paste its path above and click Extract zip.</span>"
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
            "Receiving zip upload and extracting...",
        )

    extract_btn.on_click(on_extract_clicked)
    uploader.observe(on_upload, names="value")

    return widgets.VBox(
        [
            warning(
                f"Upload limit for {doc_link(AF3_SERVER_OUTPUT_URL, 'AF3 Server')} or "
                f"{doc_link(BOLTZ_OUTPUT_URL, 'Boltz 1 / Boltz 2')} zip "
                f"is upto {limit_gb} GB. The Interface can hang on large transfers even though upload and evaluation jobs completed."
            ),
            widgets.HBox([zip_path, extract_btn]),
            widgets.HTML(
                f'<span style="{SOFT}">Use Jupyterlab/Github codespaces file browser instead of the upload widget for large imports.</span>'
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
