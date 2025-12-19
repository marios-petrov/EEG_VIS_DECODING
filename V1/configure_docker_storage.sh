#!/bin/bash
# Script to configure Docker to use /local-scratch/marios-datasets for storage

set -e

echo "🐳 Configuring Docker to use /local-scratch/marios-datasets"
echo "============================================================"
echo ""

# Target directory for Docker data
DOCKER_DATA_DIR="/local-scratch/marios-datasets/docker"

# Create directory if it doesn't exist
echo "📁 Creating Docker data directory..."
mkdir -p "$DOCKER_DATA_DIR"

# Backup existing daemon.json if it exists
if [ -f /etc/docker/daemon.json ]; then
    echo "📋 Backing up existing /etc/docker/daemon.json..."
    sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.backup.$(date +%Y%m%d_%H%M%S)
fi

# Create new daemon.json
echo "⚙️  Creating new Docker daemon configuration..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "data-root": "$DOCKER_DATA_DIR",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF

echo ""
echo "✅ Docker daemon.json created:"
cat /etc/docker/daemon.json
echo ""

# Stop Docker
echo "🛑 Stopping Docker service..."
sudo systemctl stop docker

# Optional: Move existing Docker data (uncomment if you want to migrate)
# echo "📦 Moving existing Docker data..."
# if [ -d /var/lib/docker ]; then
#     sudo rsync -aP /var/lib/docker/ "$DOCKER_DATA_DIR/"
# fi

# Start Docker
echo "🚀 Starting Docker service..."
sudo systemctl start docker

echo ""
echo "✅ Docker configured successfully!"
echo ""
echo "🔍 Verifying configuration..."
docker info | grep "Docker Root Dir"
echo ""

echo "📊 Storage usage:"
df -h /local-scratch/marios-datasets
echo ""

echo "✅ Setup complete! Docker will now store all data in:"
echo "   $DOCKER_DATA_DIR"
echo ""
echo "⚠️  Note: If you had existing Docker images/containers, you'll need to rebuild them."
echo ""
