#!/usr/bin/env bash
set -euo pipefail

# Re-apply requirements in case this branch changed them after the image was built.
python -m pip install -q -r requirements.txt

# Register a kernel for JupyterLab and the VS Code Jupyter extension.
python -m ipykernel install --user --name python3 --display-name "Python 3"

# Same large-upload websocket limit as Binder (bulk zip via file browser / widgets).
mkdir -p "${HOME}/.jupyter"
cp "${PWD}/binder/jupyter_server_config.py" "${HOME}/.jupyter/jupyter_server_config.py"
