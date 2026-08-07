"""Repository path constants and resolution helpers for IPSAE additions."""

from __future__ import annotations

from pathlib import Path

ADDITIONS_DIR = Path(__file__).resolve().parent
IPSAE_DIR = ADDITIONS_DIR.parent

# Parent-app markers when IPSAE is vendored under dependencies/ (or similar).
_APP_ROOT_MARKERS = (
    "bindcraft.py",
    "main_UI.py",
    "install_bindcraft.sh",
    ".gitmodules",
)
_VENDOR_DIR_NAMES = frozenset(
    {"dependencies", "vendor", "third_party", "submodules", "external"}
)


def resolve_project_root() -> Path:
    """Resolve standalone IPSAE root or parent application root.

    - Standalone: the IPSAE directory (contains ``ipsae.py``).
    - Embedded under ``dependencies/`` (etc.): the parent application root
      when known project markers are present.
    """
    ipsae = IPSAE_DIR
    parent = ipsae.parent
    if parent.name in _VENDOR_DIR_NAMES:
        app_root = parent.parent
        if any((app_root / marker).exists() for marker in _APP_ROOT_MARKERS):
            return app_root.resolve()
        # Embedded layout without markers: still prefer the parent of dependencies/.
        return app_root.resolve()
    if (ipsae / "ipsae.py").exists():
        return ipsae.resolve()
    return ipsae.resolve()


ROOT = resolve_project_root()
IPSAE_SCRIPT = IPSAE_DIR / "ipsae.py"
DEFAULT_EVALS_DIR = ROOT / "ipsae_evals"

UPLOAD_DIR = ROOT / "upload"
UPLOAD_FILES_DIR = UPLOAD_DIR / "files"
UPLOAD_FOLDERS_DIR = UPLOAD_DIR / "folders"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def rel_repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
