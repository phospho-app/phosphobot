# Reference: https://github.com/Physical-Intelligence/openpi/blob/b31173a8c39b8d959f4e88a36c792329801fcd13/scripts/docker/serve_policy.Dockerfile

# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
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

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN echo "v1"
#RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/Physical-Intelligence/openpi.git@5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d
RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/phospho-app/openpi/tree/fred-update@fred-update

# Install dependencies for serving the policy on Modal
RUN uv pip install \
    sentry-sdk \
    loguru>=0.7.3 \
    supabase \
    huggingface_hub[hf_transfer] \
    hf_xet \
    wandb \
    httpx>=0.28.1 \
    fastparquet>=2024.11.0 \
    opencv-python-headless>=4.0 \
    json-numpy>=2.1.0 \
    fastapi>=0.115.11 \
    uvicorn>=0.32.1 \
    pytest>=8.3.4

# Install pip in the uv environment for Modal compatibility
RUN uv pip install pip

# Add Python to PATH so Modal can find it
ENV PATH="/.venv/bin:$PATH"
