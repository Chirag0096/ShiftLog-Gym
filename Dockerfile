# Use NVIDIA CUDA base image for GPU training at runtime
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ─── System deps ──────────────────────────────────────────────────────────────
# We use the native Python 3.10 from Ubuntu 22.04 to avoid repo/PPA issues
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# ─── Install Python deps ───────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy project ─────────────────────────────────────────────────────────────
COPY . .
RUN pip install --no-cache-dir -e ".[observatory]"

# ─── Runtime env ──────────────────────────────────────────────────────────────
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV TOKENIZERS_PARALLELISM=false

# ─── Start server ─────────────────────────────────────────────────────────────
EXPOSE 7860
CMD ["python", "app.py"]
