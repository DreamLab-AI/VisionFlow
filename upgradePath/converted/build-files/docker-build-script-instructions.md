# Build Configuration Instructions - Docker Build Script

## **Task 2.1: Create Docker Build Script**
*   **Goal:** Add automated Docker build script for WebXR project with CUDA support
*   **Actions:**
    1. Create new file `docker-build.sh` in project root:
       - Set executable permissions with `chmod +x docker-build.sh`
       - Add shebang `#!/bin/bash`
    
    2. Configure build parameters:
       - Set default CUDA architecture: `CUDA_ARCH=${CUDA_ARCH:-86}` (RTX A6000 default)
       - Support environment variable override for different GPU architectures
    
    3. Implement development image build:
       - Use `docker build` command with build args
       - Pass `CUDA_ARCH` as build argument
       - Use `Dockerfile.dev` for development builds
       - Tag as `webxr:dev`
    
    4. Implement production image build:
       - Use `docker build` command with build args
       - Pass `CUDA_ARCH` as build argument  
       - Use `Dockerfile.production` for production builds
       - Tag as `webxr:production`
    
    5. Add user-friendly output:
       - Progress messages for each build step
       - Usage instructions after successful build
       - Development container run command: `docker run --gpus all -p 3001:3001 -p 4000:4000 webxr:dev`
       - Production container run command: `docker run --gpus all -p 4000:4000 webxr:production`

## **Script Contents:**
```bash
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
```

## **Implementation Notes:**
- Script provides unified interface for both development and production builds
- CUDA architecture is configurable for different GPU targets
- Includes GPU support flags in run commands
- Exposes appropriate ports for development vs production environments