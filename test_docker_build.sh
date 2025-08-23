#!/bin/bash
# Test script to verify Docker build works with the updated Dockerfile

set -e

echo "Testing Docker build with fixed Dockerfile.dev..."
echo "================================================"

# Clean any previous builds
echo "Cleaning previous Docker builds..."
docker rmi webxr-dev:test 2>/dev/null || true

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.dev -t webxr-dev:test --build-arg CUDA_ARCH=86 .

if [ $? -eq 0 ]; then
    echo "✅ Docker build completed successfully!"
    echo ""
    echo "The following changes fixed the issue:"
    echo "1. Removed redundant PTX compilation from Dockerfile (lines 88-100)"
    echo "2. Removed PTX file copy step from Dockerfile (lines 119-120)" 
    echo "3. Updated build.rs to use -dc flag and device linking"
    echo "4. Added cudadevrt library linking"
    echo ""
    echo "The build.rs now properly handles:"
    echo "- Compiling CUDA code to object files with -dc flag"
    echo "- Device linking with nvcc -dlink"
    echo "- Creating static library with both object files"
    echo "- Linking with cudart, cudadevrt, and stdc++"
else
    echo "❌ Docker build failed"
    exit 1
fi