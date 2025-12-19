#!/bin/bash
# Install NVIDIA Container Toolkit (replaces nvidia-docker2 for newer Ubuntu)

set -e

echo "🐳 Installing NVIDIA Container Toolkit"
echo "======================================="
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

# Remove old sources list if it exists (and is broken)
if [ -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
    echo "🗑️  Removing old nvidia-container-toolkit.list..."
    rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
fi

echo "📦 Installing NVIDIA Container Toolkit using apt repository..."
echo ""

# Configure the production repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo ""
echo "🔄 Updating package list..."
apt-get update

echo ""
echo "📥 Installing nvidia-container-toolkit..."
apt-get install -y nvidia-container-toolkit

echo ""
echo "⚙️  Configuring Docker to use NVIDIA runtime..."
nvidia-ctk runtime configure --runtime=docker

echo ""
echo "🔄 Restarting Docker..."
systemctl restart docker

echo ""
echo "✅ Installation complete!"
echo ""
echo "🧪 Testing NVIDIA Container Toolkit..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "✅ NVIDIA Container Toolkit is working!"
echo ""
echo "You can now run: ./setup.sh"
