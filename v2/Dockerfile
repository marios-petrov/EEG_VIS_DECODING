FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="7.0" \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (order matters!)
# First, install dependencies that mamba-ssm needs
RUN pip install --no-cache-dir \
    packaging \
    ninja \
    einops \
    transformers \
    sentencepiece \
    protobuf

# Install diffusers and related packages for SD 3.5
RUN pip install --no-cache-dir \
    diffusers \
    accelerate \
    safetensors \
    scipy \
    matplotlib \
    pillow \
    tqdm \
    opencv-python \
    scikit-learn \
    pandas

# Now install mamba-ssm and causal-conv1d (compile from source to match PyTorch 2.4.1)
# Use --no-build-isolation to use the existing PyTorch installation
RUN pip install --no-cache-dir --no-build-isolation causal-conv1d
RUN pip install --no-cache-dir --no-build-isolation mamba-ssm

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); from diffusers import SD3Transformer2DModel; print('✓ SD3 available')"
RUN python -c "import mamba_ssm; print('✓ mamba-ssm installed')"

# Set up volume mount points
VOLUME ["/workspace", "/data"]

# Default command
CMD ["/bin/bash"]
