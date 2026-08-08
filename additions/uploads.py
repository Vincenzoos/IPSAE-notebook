"""Upload storage, validation, and safe zip extraction for IPSAE notebook UIs."""

from __future__ import annotations

import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path

from paths import UPLOAD_DIR, UPLOAD_FILES_DIR, UPLOAD_FOLDERS_DIR, rel_repo_path, resolve_repo_path

STRUCTURE_EXTENSIONS = frozenset({".pdb", ".cif"})
PAE_EXTENSIONS = frozenset({".json", ".npz"})
ZIP_EXTENSIONS = frozenset({".zip"})

MAX_ZIP_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXTRACTED_BYTES = 5 * 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 20000
MAX_SINGLE_FILE_BYTES = 1024 * 1024 * 1024
# ipywidgets FileUpload sends the whole file over the Jupyter websocket
# (default ~10 MB). Larger AF3 zips hang with "(1)" and never reach the kernel.
MAX_WIDGET_ZIP_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class UploadedItem:
    name: str
    content: bytes
    size: int


class UploadError(ValueError):
    """Raised when an upload is rejected for validation or safety reasons."""


def ensure_upload_dirs() -> None:
    """Create upload/files and upload/folders."""
    UPLOAD_FILES_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_FOLDERS_DIR.mkdir(parents=True, exist_ok=True)


def uploaded_items(upload_widget) -> list[UploadedItem]:
    """Normalize ipywidgets v7/v8 FileUpload values."""
    value = getattr(upload_widget, "value", None)
    if not value:
        return []

    items: list[UploadedItem] = []
    if isinstance(value, dict):
        # ipywidgets v7: {filename: {content, metadata, ...}}
        for name, entry in value.items():
            content = _as_bytes(entry.get("content", b""))
            meta = entry.get("metadata") or {}
            display_name = str(meta.get("name") or name or "upload")
            size = int(entry.get("size") or meta.get("size") or len(content))
            items.append(UploadedItem(name=display_name, content=content, size=size))
        return items

    if isinstance(value, (list, tuple)):
        # ipywidgets v8: tuple/list of {name, type, size, content}
        for entry in value:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "upload")
            content = _as_bytes(entry.get("content", b""))
            size = int(entry.get("size") or len(content))
            items.append(UploadedItem(name=name, content=content, size=size))
        return items

    raise UploadError(f"Unsupported FileUpload value type: {type(value).__name__}")


def save_uploaded_file(
    upload_widget,
    dest_dir: Path,
    allowed_extensions: set[str] | frozenset[str],
) -> list[Path]:
    """Save one or more uploaded files by basename, overwriting existing files."""
    ensure_upload_dirs()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    allowed = {_normalize_ext(ext) for ext in allowed_extensions}
    items = uploaded_items(upload_widget)
    if not items:
        raise UploadError("No file uploaded.")

    saved: list[Path] = []
    for item in items:
        basename = Path(item.name).name
        if not basename or basename in {".", ".."}:
            raise UploadError(f"Invalid upload filename: {item.name!r}")
        ext = _normalize_ext(Path(basename).suffix)
        if ext not in allowed:
            allowed_list = ", ".join(sorted(allowed))
            raise UploadError(
                f"Unsupported extension for {basename!r}. Allowed: {allowed_list}"
            )
        if item.size > MAX_SINGLE_FILE_BYTES or len(item.content) > MAX_SINGLE_FILE_BYTES:
            raise UploadError(
                f"File {basename!r} exceeds max size "
                f"({MAX_SINGLE_FILE_BYTES // (1024 * 1024)} MB)."
            )
        out = dest_dir / basename
        out.write_bytes(item.content)
        saved.append(out.resolve())
    return saved


def save_single_upload(upload_widget, kind: str) -> Path:
    """Save exactly one structure or PAE file to upload/files and return its path."""
    kind_key = str(kind).strip().lower()
    if kind_key == "structure":
        allowed = STRUCTURE_EXTENSIONS
    elif kind_key == "pae":
        allowed = PAE_EXTENSIONS
    else:
        raise UploadError(f"Unknown upload kind: {kind!r}. Use 'structure' or 'pae'.")

    paths = save_uploaded_file(upload_widget, UPLOAD_FILES_DIR, allowed)
    if len(paths) != 1:
        raise UploadError(f"Upload exactly one {kind_key} file.")
    return paths[0]


def extract_zip_file(zip_path: str | Path) -> Path:
    """Extract a server-side .zip into ``upload/folders/<zip-stem>``.

    Prefer this for AlphaFold Server exports: upload the zip with the
    JupyterLab file browser (HTTP), then pass the path here. That avoids the
    ipywidgets FileUpload websocket size limit that hangs large archives.
    """
    ensure_upload_dirs()
    path = resolve_repo_path(zip_path)
    if not path.is_file():
        raise UploadError(f"Zip not found: {path}")
    ext = _normalize_ext(path.suffix)
    if ext not in ZIP_EXTENSIONS:
        raise UploadError(f"Only .zip archives are accepted (got {path.name!r}).")
    if path.stat().st_size > MAX_ZIP_BYTES:
        raise UploadError(
            f"Zip {path.name!r} exceeds max size "
            f"({MAX_ZIP_BYTES // (1024 * 1024 * 1024)} GB)."
        )

    stem = _safe_folder_name(path.stem)
    dest = UPLOAD_FOLDERS_DIR / stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=False)
    try:
        safe_extract_zip(path, dest)
        _unwrap_matching_top_dir(dest, stem)
    except Exception:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        raise
    return dest.resolve()


def extract_uploaded_zip(upload_widget) -> Path:
    """Save one uploaded zip and extract it to upload/folders/<zip-stem>."""
    ensure_upload_dirs()
    items = uploaded_items(upload_widget)
    if not items:
        raise UploadError("No zip uploaded.")
    if len(items) != 1:
        raise UploadError("Upload exactly one zip file.")

    item = items[0]
    basename = Path(item.name).name
    ext = _normalize_ext(Path(basename).suffix)
    if ext not in ZIP_EXTENSIONS:
        raise UploadError(f"Only .zip uploads are accepted (got {basename!r}).")
    payload_size = max(int(item.size), len(item.content))
    if payload_size > MAX_WIDGET_ZIP_BYTES:
        limit_mb = MAX_WIDGET_ZIP_BYTES // (1024 * 1024)
        raise UploadError(
            f"Zip {basename!r} is too large for the widget uploader "
            f"({payload_size // (1024 * 1024)} MB > {limit_mb} MB). "
            "On Binder/JupyterLab: upload the zip via the left file browser, "
            "then use Extract zip with the server path."
        )
    if payload_size > MAX_ZIP_BYTES:
        raise UploadError(
            f"Zip {basename!r} exceeds max size ({MAX_ZIP_BYTES // (1024 * 1024 * 1024)} GB)."
        )

    zip_path = UPLOAD_FILES_DIR / basename
    zip_path.write_bytes(item.content)
    return extract_zip_file(zip_path)


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract with zip-slip, symlink, size, and file-count checks."""
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    if _normalize_ext(zip_path.suffix) not in ZIP_EXTENSIONS:
        raise UploadError(f"Only .zip archives are accepted (got {zip_path.name!r}).")
    if not zip_path.is_file():
        raise UploadError(f"Zip not found: {zip_path}")
    if zip_path.stat().st_size > MAX_ZIP_BYTES:
        raise UploadError(
            f"Zip exceeds max size ({MAX_ZIP_BYTES // (1024 * 1024 * 1024)} GB)."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()

    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as exc:
        raise UploadError(f"Invalid zip archive: {zip_path.name}") from exc

    with zf:
        members = zf.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise UploadError(
                f"Zip has too many entries ({len(members)} > {MAX_ARCHIVE_FILES})."
            )

        total_size = 0
        for info in members:
            _validate_zip_member(info, dest_root)
            if info.is_dir():
                continue
            if info.file_size > MAX_SINGLE_FILE_BYTES:
                raise UploadError(
                    f"Archive member {info.filename!r} exceeds max single-file size."
                )
            total_size += int(info.file_size)
            if total_size > MAX_EXTRACTED_BYTES:
                raise UploadError(
                    f"Extracted size would exceed limit "
                    f"({MAX_EXTRACTED_BYTES // (1024 * 1024 * 1024)} GB)."
                )

        for info in members:
            _validate_zip_member(info, dest_root)
            target = (dest_root / info.filename).resolve()
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def list_uploaded_folders() -> list[str]:
    """Return immediate child folders under upload/folders."""
    ensure_upload_dirs()
    folders = sorted(
        p.name
        for p in UPLOAD_FOLDERS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return folders


def uploaded_folder_path(name: str) -> Path:
    """Return upload/folders/<name> as an absolute path."""
    safe = _safe_folder_name(name)
    return (UPLOAD_FOLDERS_DIR / safe).resolve()


def describe_upload_path(path: Path) -> str:
    """Return a concise repo-relative path string for status messages."""
    return rel_repo_path(Path(path).resolve())


def _as_bytes(content) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, memoryview):
        return content.tobytes()
    if isinstance(content, bytearray):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    return bytes(content)


def _normalize_ext(ext: str) -> str:
    text = str(ext or "").strip().lower()
    if not text:
        return ""
    return text if text.startswith(".") else f".{text}"


def _safe_folder_name(name: str) -> str:
    text = Path(str(name).strip()).name
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise UploadError(f"Invalid folder name: {name!r}")
    return text


def _is_symlink_member(info: zipfile.ZipInfo) -> bool:
    # Unix symlink: high 4 bits of external_attr are mode; S_IFLNK == 0o120000
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and stat.S_ISLNK(mode):
        return True
    # Some archives mark create_system=3 (Unix) with symlink bit only.
    if info.external_attr & 0xA0000000 == 0xA0000000:
        return True
    return False


def _validate_zip_member(info: zipfile.ZipInfo, dest_root: Path) -> None:
    name = info.filename
    if not name or name.endswith("\x00"):
        raise UploadError("Zip contains an invalid empty or null entry name.")
    if name.startswith("/") or name.startswith("\\") or (len(name) > 1 and name[1] == ":"):
        raise UploadError(f"Zip contains absolute path: {name!r}")
    parts = Path(name).parts
    if any(part == ".." for part in parts):
        raise UploadError(f"Zip contains path traversal: {name!r}")
    if _is_symlink_member(info):
        raise UploadError(f"Zip contains symlink entry: {name!r}")

    target = (dest_root / name).resolve()
    try:
        target.relative_to(dest_root)
    except ValueError as exc:
        raise UploadError(f"Zip entry escapes destination: {name!r}") from exc


def _unwrap_matching_top_dir(dest: Path, expected_name: str) -> None:
    """Avoid upload/folders/<stem>/<stem>/... when the zip has a single matching top dir."""
    children = [p for p in dest.iterdir() if not p.name.startswith(".")]
    if len(children) != 1:
        return
    only = children[0]
    if not only.is_dir() or only.name != expected_name:
        return

    # Move inner contents up one level, then remove the empty wrapper.
    for child in list(only.iterdir()):
        target = dest / child.name
        if target.exists():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        child.rename(target)
    only.rmdir()


# Re-export upload roots for convenience.
__all__ = [
    "UPLOAD_DIR",
    "UPLOAD_FILES_DIR",
    "UPLOAD_FOLDERS_DIR",
    "STRUCTURE_EXTENSIONS",
    "PAE_EXTENSIONS",
    "MAX_ZIP_BYTES",
    "MAX_EXTRACTED_BYTES",
    "MAX_ARCHIVE_FILES",
    "MAX_SINGLE_FILE_BYTES",
    "UploadedItem",
    "UploadError",
    "ensure_upload_dirs",
    "uploaded_items",
    "save_uploaded_file",
    "save_single_upload",
    "extract_zip_file",
    "extract_uploaded_zip",
    "safe_extract_zip",
    "MAX_WIDGET_ZIP_BYTES",
    "list_uploaded_folders",
    "uploaded_folder_path",
    "describe_upload_path",
]
