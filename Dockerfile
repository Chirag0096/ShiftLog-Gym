# Use NVIDIA CUDA base image for GPU training at runtime
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ─── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ─── Install Python deps ───────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install training deps — these are heavy but needed for GPU training
# They are installed at BUILD time so the Space doesn't need to download them at runtime
RUN pip install --no-cache-dir \
    "trl==0.8.6" \
    "peft>=0.10.0" \
    "bitsandbytes>=0.43.0" \
    "accelerate>=0.28.0" \
    "transformers>=4.40.0" \
    "datasets>=2.18.0" \
    "wandb>=0.16.0" \
    "scipy" \
    "einops"

# ─── Copy project ─────────────────────────────────────────────────────────────
COPY . .
RUN pip install --no-cache-dir -e ".[observatory]"

# ─── Runtime env ──────────────────────────────────────────────────────────────
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV TOKENIZERS_PARALLELISM=false
# TRAIN_ENABLED is set as a Space Variable in Settings, not here

# ─── Start server ─────────────────────────────────────────────────────────────
EXPOSE 7860
CMD ["python", "app.py"]
