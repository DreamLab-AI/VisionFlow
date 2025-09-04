#!/bin/bash

# Docker build script for WebXR with CUDA support
# This script builds the Docker image with proper CUDA and cudarc configuration

echo "Building WebXR Docker image with CUDA support..."

# Build arguments
CUDA_ARCH=${CUDA_ARCH:-86}  # Default to SM_86 for RTX A6000

# Development build
echo "Building development image..."
docker build \
  --build-arg CUDA_ARCH=$CUDA_ARCH \
  -f Dockerfile.dev \
  -t webxr:dev \
  .

# Production build
echo "Building production image..."
docker build \
  --build-arg CUDA_ARCH=$CUDA_ARCH \
  -f Dockerfile.production \
  -t webxr:production \
  .

echo "Build complete!"
echo ""
echo "To run development container:"
echo "  docker run --gpus all -p 3001:3001 -p 4000:4000 webxr:dev"
echo ""
echo "To run production container:"
echo "  docker run --gpus all -p 4000:4000 webxr:production"