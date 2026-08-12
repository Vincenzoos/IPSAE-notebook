FROM mcr.microsoft.com/devcontainers/python:3.11

# zip/unzip: notebook zip helper + bulk Extract zip (same as Binder apt.txt)
RUN apt-get update \
    && apt-get install -y --no-install-recommends zip unzip \
    && rm -rf /var/lib/apt/lists/*

# Project deps + Jupyter so Codespaces / Dev Containers are ready on first open.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        jupyterlab \
        ipykernel \
        notebook \
    && rm /tmp/requirements.txt
