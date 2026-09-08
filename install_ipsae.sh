#!/usr/bin/env bash
# All-in-one local setup for the IPSAE notebook (conda + deps + editor extensions).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="ipsae-notebook"
PYTHON_VERSION="3.11"

EXTENSIONS=(
  "ms-python.python"
  "ms-toolsai.jupyter"
  "ms-toolsai.jupyter-renderers"
  "openai.chatgpt"
  "cweijan.vscode-office"
  "ArianJamasb.protein-viewer"
)

echo "==> IPSAE notebook local setup"
echo "    repo: ${ROOT}"
echo "    conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH."
  echo "Install Miniconda or Anaconda, then re-run this script."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Conda env '${ENV_NAME}' already exists — updating packages"
else
  echo "==> Creating conda env '${ENV_NAME}'"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

echo "==> Installing Python dependencies"
python -m pip install -U pip
python -m pip install -r "${ROOT}/requirements.txt" jupyterlab ipykernel notebook

echo "==> Registering Jupyter kernel '${ENV_NAME}'"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

if command -v zip >/dev/null 2>&1 && command -v unzip >/dev/null 2>&1; then
  echo "==> zip/unzip already available"
else
  echo "NOTE: zip/unzip not found. Bulk Extract zip / pack helpers need them."
  echo "      Debian/Ubuntu: sudo apt-get install -y zip unzip"
  echo "      macOS:        brew install zip unzip"
fi

install_extensions() {
  local cli="$1"
  echo "==> Installing VS Code / Cursor extensions via: ${cli}"
  local failed=0
  local ext
  for ext in "${EXTENSIONS[@]}"; do
    if "${cli}" --install-extension "${ext}" --force; then
      echo "    OK  ${ext}"
    else
      echo "    FAIL ${ext}"
      failed=1
    fi
  done
  return "${failed}"
}

EXT_CLI=""
if command -v cursor >/dev/null 2>&1; then
  EXT_CLI="cursor"
elif command -v code >/dev/null 2>&1; then
  EXT_CLI="code"
fi

if [[ -n "${EXT_CLI}" ]]; then
  if ! install_extensions "${EXT_CLI}"; then
    echo "WARNING: one or more extensions failed to install (marketplace / network)."
    echo "         You can install them manually from the Extensions view."
  fi
else
  echo "NOTE: neither 'cursor' nor 'code' CLI found — skipping extension install."
  echo "      Open this folder in Cursor/VS Code and accept recommended extensions"
  echo "      (see .vscode/extensions.json), or install:"
  for ext in "${EXTENSIONS[@]}"; do
    echo "        ${ext}"
  done
fi

# Jupyter large-upload limit (same as Binder / Codespaces).
mkdir -p "${HOME}/.jupyter"
if [[ -f "${ROOT}/binder/jupyter_server_config.py" ]]; then
  cp "${ROOT}/binder/jupyter_server_config.py" "${HOME}/.jupyter/jupyter_server_config.py"
  echo "==> Installed Jupyter server config (large upload limit)"
fi

echo
echo "Setup complete."
echo
echo "Activate and run JupyterLab:"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${ROOT}"
echo "  jupyter lab ipsae_eval.ipynb"
echo
echo "Or open this folder in Cursor/VS Code, select the '${ENV_NAME}' kernel, and run the notebook."
