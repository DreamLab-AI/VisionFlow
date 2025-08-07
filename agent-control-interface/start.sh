#!/bin/bash

# Agent Control Interface Startup Script
# 
# This script initializes and starts the Agent Control Interface
# which provides telemetry to the VisionFlow visualization system.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Agent Control Interface Startup"
echo "======================================"
echo ""

# Check if running in Docker container
if [ -f /.dockerenv ]; then
    echo "✓ Running in Docker container"
else
    echo "⚠ Warning: Not running in Docker container"
    echo "  Network connectivity may be limited"
fi

# Check network configuration
echo ""
echo "Network Configuration:"
echo "----------------------"
if command -v ip &> /dev/null; then
    echo "Container IP: $(ip -4 addr show | grep inet | grep -v 127.0.0.1 | awk '{print $2}' | head -1)"
fi
echo "Binding to: 0.0.0.0:9500"
echo ""

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install --silent
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Check for MCP tools availability
echo ""
echo "Checking MCP Tool Availability:"
echo "--------------------------------"

# Check mcp-observability
if [ -d "/workspace/mcp-observability" ]; then
    echo "✓ mcp-observability found"
    if [ ! -d "/workspace/mcp-observability/node_modules" ]; then
        echo "  Installing mcp-observability dependencies..."
        (cd /workspace/mcp-observability && npm install --silent)
    fi
else
    echo "✗ mcp-observability not found (will use mock data)"
fi

# Check for claude-flow
if command -v claude-flow &> /dev/null || [ -f "/app/claude-flow" ]; then
    echo "✓ claude-flow available"
else
    echo "✗ claude-flow not found (will use mock data)"
fi

# Set environment variables
export NODE_ENV=${NODE_ENV:-development}
export LOG_LEVEL=${LOG_LEVEL:-info}
export AGENT_CONTROL_PORT=${AGENT_CONTROL_PORT:-9500}

echo ""
echo "Environment:"
echo "------------"
echo "NODE_ENV: $NODE_ENV"
echo "LOG_LEVEL: $LOG_LEVEL"
echo "PORT: $AGENT_CONTROL_PORT"
echo ""

# Handle different run modes
case "${1:-}" in
    debug)
        echo "Starting in DEBUG mode..."
        export LOG_LEVEL=debug
        node src/index.js --debug
        ;;
    test)
        echo "Running test client..."
        node tests/test-client.js
        ;;
    background)
        echo "Starting in BACKGROUND mode..."
        nohup node src/index.js > logs/agent-control.log 2>&1 &
        PID=$!
        echo "Started with PID: $PID"
        echo $PID > agent-control.pid
        echo ""
        echo "To stop: ./stop.sh"
        echo "To view logs: tail -f logs/agent-control.log"
        ;;
    *)
        echo "Starting Agent Control Interface..."
        echo "======================================"
        echo ""
        echo "Ready to accept connections from VisionFlow"
        echo "TCP Server listening on port 9500"
        echo ""
        echo "Press Ctrl+C to stop"
        echo ""
        exec node src/index.js
        ;;
esac