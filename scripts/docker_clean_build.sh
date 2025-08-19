#!/bin/bash
# Clean Docker build script for VisionFlow
# Removes legacy artifacts and builds fresh

set -e

echo "=== VisionFlow Clean Docker Build ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Clean up legacy artifacts
echo -e "${YELLOW}Cleaning up legacy artifacts...${NC}"
rm -f Cargo.lock
rm -rf target/
echo -e "${GREEN}✓ Legacy artifacts removed${NC}"

# Prune Docker build cache (optional)
read -p "Do you want to clear Docker build cache? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Clearing Docker build cache...${NC}"
    docker builder prune -f
    echo -e "${GREEN}✓ Docker build cache cleared${NC}"
fi

# Build the Docker image
echo ""
echo -e "${YELLOW}Building VisionFlow Docker image...${NC}"
echo "Profile: ${1:-dev}"
echo ""

if [ "${1:-dev}" = "production" ]; then
    docker build -f Dockerfile.production -t visionflow:production .
else
    docker build -f Dockerfile.dev -t visionflow:dev --build-arg CUDA_ARCH=${CUDA_ARCH:-86} .
fi

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Docker build completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run: docker-compose -f docker-compose.dev.yml up"
    echo "2. Access the application at http://localhost:3001"
else
    echo ""
    echo -e "${RED}✗ Docker build failed${NC}"
    echo "Check the error messages above for details"
    exit 1
fi