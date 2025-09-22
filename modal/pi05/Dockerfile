FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    linux-headers-generic \
    build-essential \
    clang \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python 3.11 and create virtual environment
ARG PYTHON_VERSION=3.11
RUN uv python install $PYTHON_VERSION
RUN uv python pin $PYTHON_VERSION

# Create virtual environment using the pinned Python version
RUN uv venv --python $PYTHON_VERSION .venv

# Activate the virtual environment by setting PATH and VIRTUAL_ENV
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip in the virtual environment
RUN uv pip install pip

# Verify we're using the correct Python version
RUN python --version