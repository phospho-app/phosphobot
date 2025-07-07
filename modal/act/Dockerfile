FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        netcat \
        dnsutils \
        libgl1-mesa-glx \
        git \
        git-lfs \
        libvulkan-dev \
        zip \
        unzip \
        wget \
        curl \
        build-essential \
        cmake \
        vim \
        less \
        sudo \
        htop \
        ca-certificates \
        man \
        ffmpeg \
        tmux && \
    ln -fs /usr/share/zoneinfo/Europe/Paris /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update Conda to the latest version
RUN conda update -n base -c conda-forge conda -y

# Clone the lerobot repository
RUN echo "v0"
RUN git clone https://github.com/huggingface/lerobot.git
WORKDIR /lerobot

# Create and set up the lerobot Conda environment
RUN conda create -y -n lerobot python=3.10 && \
    /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate lerobot && \
    conda install -y -c conda-forge av && \
    pip install -e ."

# Ensure the lerobot environment is used by default
CMD ["/bin/bash"]