#!/bin/bash
# Setup script for EEG2Video Docker environment

set -e

echo "🐳 EEG2Video Docker Setup"
echo "========================="
echo ""

# Check if Docker storage is configured
DOCKER_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $4}')
if [[ ! "$DOCKER_ROOT" == *"/local-scratch/marios-datasets"* ]]; then
    echo "⚠️  Docker is not configured to use /local-scratch/marios-datasets"
    echo "   Current Docker Root Dir: $DOCKER_ROOT"
    echo ""
    echo "   Run this first to configure storage:"
    echo "   sudo bash configure_docker_storage.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if nvidia-docker is available
echo "🔍 Checking NVIDIA Docker runtime..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✓ Docker and NVIDIA runtime are available"
else
    echo "❌ NVIDIA Docker runtime not available."
    echo "   Run: sudo ./install_nvidia_docker_fixed.sh"
    exit 1
fi
echo ""

# Build the Docker image
echo "📦 Building Docker image (this may take 10-15 minutes)..."
docker build -t eeg2video:latest .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "🚀 Usage:"
echo "========="
echo ""
echo "1. Start container with docker-compose:"
echo "   docker-compose up -d"
echo "   docker-compose exec eeg2video bash"
echo ""
echo "2. Or start container directly:"
echo "   docker run --gpus all -it --rm \\"
echo "     -v ~/EEG_Reconstruction:/workspace \\"
echo "     -v /local-scratch/marios-datasets:/data \\"
echo "     --shm-size=32g \\"
echo "     eeg2video:latest"
echo ""
echo "3. Run training inside container:"
echo "   cd /workspace"
echo "   python train_eegmamba_adapter.py \\"
echo "     --data_dir /data/NATVIEW \\"
echo "     --output_dir /data/outputs/eegmamba_sd35"
echo ""
echo "4. To use specific GPUs:"
echo "   docker run --gpus '\"device=0,1,2,3\"' ..."
echo ""
echo "💡 Tips:"
echo "- Your code is mounted at /workspace"
echo "- Your datasets are at /data"
echo "- Changes in /workspace persist on host"
echo "- All outputs should go to /data/outputs/"
echo ""
