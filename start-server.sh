#!/bin/bash
# Correct startup script for the WebXR server

echo "Starting WebXR Server..."
echo ""
echo "Architecture Overview:"
echo "  - Nginx (port 3001) → Rust backend (port 4000 internal)"
echo "  - Nginx (port 3001) → Vite frontend (port 5173 internal)"
echo "  - Client connects to port 3001 for everything"
echo ""

# Check Docker daemon
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running!"
    echo "Please start Docker first:"
    echo "  sudo service docker start"
    exit 1
fi

echo "Using docker-compose for development environment..."
echo ""

# The correct way to start the server
if [ -f docker-compose.dev.yml ]; then
    echo "Starting services with docker-compose..."
    docker-compose -f docker-compose.dev.yml up --build
else
    echo "ERROR: docker-compose.dev.yml not found!"
    echo "Please run this script from the /workspace/ext directory"
    exit 1
fi