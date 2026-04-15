# =============================================================================
# Dockerfile for CLI Coding Agent Training
# =============================================================================
# Base: Unsloth Docker image with CUDA support
# Model: Google Gemma 4 E4B (8B)
# Training: Unsloth + LoRA
# =============================================================================

# Use Unsloth base image
FROM unsloth/base:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TRANSFORMERS_CACHE=/workspace/model_cache
ENV HF_HOME=/workspace/model_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
    unsloth \
    transformers \
    datasets \
    huggingface_hub \
    trl \
    peft \
    bitsandbytes \
    accelerate \
    scipy

# Create workspace
WORKDIR /workspace

# Copy scripts
COPY generate_training_data.py /workspace/
COPY finetune.py /workspace/
COPY export_to_gguf.py /workspace/

# Make scripts executable
RUN chmod +x /workspace/*.sh

# Default command
CMD ["/bin/bash"]
