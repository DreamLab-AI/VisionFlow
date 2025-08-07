#!/bin/bash

# Setup script for Agent Control Interface
# Ensures all dependencies are installed and environment is configured

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Agent Control Interface Setup"
echo "======================================"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env configuration file..."
    cp .env.example .env
    echo "✓ Created .env from template"
else
    echo "✓ .env file already exists"
fi

# Create logs directory
if [ ! -d logs ]; then
    mkdir -p logs
    echo "✓ Created logs directory"
else
    echo "✓ Logs directory exists"
fi

# Install main dependencies
echo ""
echo "Installing agent-control-interface dependencies..."
if [ ! -d node_modules ]; then
    npm install --silent
    echo "✓ Installed main dependencies"
else
    echo "✓ Main dependencies already installed"
fi

# Install mcp-observability dependencies
echo ""
echo "Installing mcp-observability dependencies..."
if [ -d mcp-observability ]; then
    if [ ! -d mcp-observability/node_modules ]; then
        (cd mcp-observability && npm install --silent)
        echo "✓ Installed mcp-observability dependencies"
    else
        echo "✓ mcp-observability dependencies already installed"
    fi
else
    echo "⚠ Warning: mcp-observability directory not found"
    echo "  The system will use mock data for testing"
fi

# Check network configuration
echo ""
echo "Network Configuration:"
echo "----------------------"
if command -v ip &> /dev/null; then
    CONTAINER_IP=$(ip -4 addr show | grep inet | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
    echo "Container IP: $CONTAINER_IP"
fi
echo "Service will bind to: 0.0.0.0:9500"

# Verify setup
echo ""
echo "Verifying setup..."
echo "------------------"

# Check if port is available
if lsof -i :9500 &> /dev/null; then
    echo "⚠ Warning: Port 9500 is already in use"
    echo "  Run './stop.sh' to stop any existing instance"
else
    echo "✓ Port 9500 is available"
fi

# Summary
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "The Agent Control Interface is now configured as a"
echo "self-contained module with bundled mcp-observability."
echo ""
echo "Configuration:"
echo "  - Port: 9500 (TCP)"
echo "  - MCP: mcp-observability (bundled)"
echo "  - Config: .env file"
echo ""
echo "To start the server:"
echo "  ./start.sh"
echo ""
echo "To test the connection:"
echo "  ./start.sh test"
echo ""
echo "For interactive testing:"
echo "  node tests/test-client.js localhost 9500 interactive"
echo ""