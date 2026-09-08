#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ipsae-notebook"
VENV_DIR="${PWD}/${ENV_NAME}"
PYTHON_BIN="${VENV_DIR}/bin/python"

echo "==> Ensuring venv '${ENV_NAME}' at ${VENV_DIR}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

echo "==> Installing notebook dependencies into '${ENV_NAME}'"
"${PYTHON_BIN}" -m pip install -q -U pip
"${PYTHON_BIN}" -m pip install -q -r requirements.txt jupyterlab ipykernel notebook

# Kernel used by JupyterLab (Codespaces editor=jupyter) and the VS Code Jupyter extension.
echo "==> Registering Jupyter kernel '${ENV_NAME}'"
"${PYTHON_BIN}" -m ipykernel install --user \
  --name "${ENV_NAME}" \
  --display-name "Python (${ENV_NAME})"

# Same large-upload websocket limit as Binder.
mkdir -p "${HOME}/.jupyter"
cp "${PWD}/binder/jupyter_server_config.py" "${HOME}/.jupyter/jupyter_server_config.py"

echo "==> Ready: select kernel 'Python (${ENV_NAME})' for ipsae_eval.ipynb"
echo "    python: ${PYTHON_BIN}"
