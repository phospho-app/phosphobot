FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace

RUN echo "v0"

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    tzdata netcat dnsutils libgl1-mesa-glx libvulkan-dev \
    git git-lfs curl ca-certificates ffmpeg

# Set the timezone to PST
RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Clean up apt cache to reduce image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN git clone https://github.com/huggingface/lerobot.git
RUN uv pip install './lerobot[smolvla]'

RUN uv pip install \
        loguru \
        supabase \
        sentry-sdk \
        "huggingface_hub[hf_transfer]" \
        hf_xet \
        wandb \
        accelerate \
        "httpx>=0.28.1" \
        "pydantic>=2.10.5" \
        "fastparquet>=2024.11.0" \
        "numpy<2" \
        "opencv-python-headless>=4.0" \
        "rich>=13.9.4" \
        "pandas>=2.2.2.240807" \
        "json-numpy>=2.1.0" \
        "fastapi>=0.115.11" \
        "torch>=2.2.1" \
        "torchvision>=0.21.0" \
        "pyarrow>=8.0.0" \
        uvicorn

# Install pip in the uv environment for Modal compatibility
RUN uv pip install pip

# Add Python to PATH so Modal can find it
ENV PATH="/.venv/bin:$PATH"
