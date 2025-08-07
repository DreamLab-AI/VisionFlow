#!/bin/bash

# Validation script for Agent Control Interface
# Performs a quick health check of the system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Agent Control Interface Validation"
echo "======================================"
echo ""

# Function to check if process is running
check_process() {
    if ps aux | grep -v grep | grep -q "$1"; then
        return 0
    else
        return 1
    fi
}

# Check dependencies
echo "Checking Dependencies:"
echo "---------------------"

if [ -f package.json ]; then
    echo "✓ package.json found"
else
    echo "✗ package.json missing"
    exit 1
fi

if [ -d node_modules ]; then
    echo "✓ Main dependencies installed"
else
    echo "✗ Main dependencies not installed - run ./setup.sh"
    exit 1
fi

if [ -d mcp-observability/node_modules ]; then
    echo "✓ mcp-observability dependencies installed"
else
    echo "⚠ mcp-observability dependencies not installed"
fi

# Check configuration
echo ""
echo "Checking Configuration:"
echo "----------------------"

if [ -f .env ]; then
    echo "✓ .env configuration file exists"
    # Source the env file to check values
    export $(cat .env | grep -v '^#' | xargs)
    echo "  Port: ${AGENT_CONTROL_PORT:-9500}"
    echo "  MCP Path: ${MCP_OBSERVABILITY_PATH:-mcp-observability}"
else
    echo "⚠ .env file not found - using defaults"
fi

# Check file structure
echo ""
echo "Checking File Structure:"
echo "-----------------------"

REQUIRED_FILES=(
    "src/index.js"
    "src/json-rpc-handler.js"
    "src/telemetry-aggregator.js"
    "src/mcp-bridge.js"
    "src/logger.js"
    "tests/test-client.js"
    "start.sh"
    "stop.sh"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file missing"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "⚠ Some required files are missing"
    exit 1
fi

# Quick syntax check
echo ""
echo "Checking JavaScript Syntax:"
echo "---------------------------"

if node -c src/index.js 2>/dev/null; then
    echo "✓ Main server syntax valid"
else
    echo "✗ Syntax error in main server"
    exit 1
fi

if node -c src/json-rpc-handler.js 2>/dev/null; then
    echo "✓ JSON-RPC handler syntax valid"
else
    echo "✗ Syntax error in JSON-RPC handler"
    exit 1
fi

# Check if server is already running
echo ""
echo "Checking Server Status:"
echo "----------------------"

if check_process "node src/index.js"; then
    echo "⚠ Server appears to be running"
    echo "  Use './stop.sh' to stop it"
else
    echo "✓ Server is not running"
fi

if lsof -i :9500 &> /dev/null; then
    echo "⚠ Port 9500 is in use"
else
    echo "✓ Port 9500 is available"
fi

# Summary
echo ""
echo "======================================"
echo "Validation Complete"
echo "======================================"
echo ""

if [ "$ALL_PRESENT" = true ]; then
    echo "✓ All checks passed!"
    echo ""
    echo "The Agent Control Interface appears to be properly configured."
    echo "This is a self-contained module with bundled mcp-observability."
    echo ""
    echo "Next steps:"
    echo "  1. Start the server: ./start.sh"
    echo "  2. Test connection: ./start.sh test"
    echo "  3. Check logs: tail -f logs/agent-control.log"
    exit 0
else
    echo "✗ Some checks failed"
    echo ""
    echo "Please run './setup.sh' to fix any issues."
    exit 1
fi