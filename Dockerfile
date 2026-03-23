# Base: CUDA 12.8 runtime on Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/.cache/huggingface

# System deps — python3-pip intentionally excluded; we bootstrap pip for 3.11 below
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip specifically for Python 3.11 (python3-pip installs for 3.10)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Canonical aliases so `python` and `pip` both target 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip

WORKDIR /app

# PyTorch with CUDA 12.8 first — other packages must link against this build
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Project requirements (torch excluded — already installed above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ctransformers with CUDA (override the cpu-only wheel from requirements.txt)
RUN pip install --no-cache-dir "ctransformers[cuda]"

# Project source (models/, outputs/ are bind-mounted at runtime)
COPY . .

EXPOSE 8000 8501

VOLUME ["/app/models", "/app/outputs"]

CMD ["python", "launch.py"]
