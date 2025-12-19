#!/bin/bash
# Install nvidia-docker2 for GPU support in Docker

set -e

echo "🐳 Installing NVIDIA Docker Support"
echo "===================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    echo "❌ Cannot detect Linux distribution"
    exit 1
fi

echo "📋 Detected: $OS $VER"
echo ""

# Add NVIDIA Docker repository
echo "📦 Adding NVIDIA Docker repository..."

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo ""
echo "🔄 Updating package list..."
apt-get update

echo ""
echo "📥 Installing nvidia-docker2..."
apt-get install -y nvidia-docker2

echo ""
echo "🔄 Restarting Docker..."
systemctl restart docker

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Testing NVIDIA Docker..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "✅ NVIDIA Docker is working!"
echo ""
echo "You can now run: ./setup.sh"
