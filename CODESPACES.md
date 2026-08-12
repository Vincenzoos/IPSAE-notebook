# Use this notebook in GitHub Codespaces

Use Codespaces if [Binder](https://mybinder.org/v2/gh/Vincenzoos/IPSAE-notebook/main?urlpath=lab/tree/ipsae_eval.ipynb) is slow or unavailable.

You only need a **free GitHub account**. Nothing is installed on your computer. The environment is already set up.

## 1. Open Codespaces

1. Sign in to [GitHub](https://github.com/login) (create a free account if you do not have one).
2. Click this button:

   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Vincenzoos/IPSAE-notebook?quickstart=1&editor=jupyter)

   Or open this link:

   ```text
   https://codespaces.new/Vincenzoos/IPSAE-notebook?quickstart=1&editor=jupyter
   ```

3. Wait until it finishes starting. The **first** launch can take several minutes.

JupyterLab should open in your browser (similar to Binder). In the file list on the left, click **`ipsae_eval.ipynb`**.

If Visual Studio Code opens instead: click **`ipsae_eval.ipynb`** in the left file list. It should open as a notebook.

## 2. Run the notebook

1. Select the first cell (**Package Installation**) and run it (play button, or **Shift+Enter**).
2. Run the **ipSAE Evaluation** cell to open the scoring UI.
3. Upload structure/PAE files in the UI.

For **bulk** AF3 Server or Boltz folders: upload the zip in the JupyterLab file browser (left sidebar), then use **Extract zip** in the UI (up to 2 GB).

Optional: run **ipSAE Comparison CSV** to rank collected outputs, and the zip helper cell to pack a folder for download.

## 3. Download results

In the JupyterLab file browser, right-click an output folder or zip → **Download**.

## 4. Stop when you are done

Codespaces uses a monthly free-hour allowance. Stop it when you finish:

1. Open [github.com/codespaces](https://github.com/codespaces).
2. Find this codespace → **Stop** (or **Delete** if you do not need it again).

You can reopen later from that same page; GitHub may offer to resume the last codespace.
