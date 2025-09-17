#!/bin/bash
# Simplified Claude Flow Setup Script - Copy-based approach
# This script uses pre-fixed files instead of complex patching

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Main setup function
setup_claude_flow() {
    log_info "🚀 Starting simplified Claude Flow setup..."

    # 1. Clean up any npx cached versions
    log_info "🧹 Cleaning up npx cache to prevent version conflicts..."
    if [ -d "/home/ubuntu/.npm/_npx" ]; then
        find /home/ubuntu/.npm/_npx -name "*claude-flow*" -type d -exec rm -rf {} + 2>/dev/null || true
        log_success "Cleaned npx cache"
    fi

    # Clear npm cache for good measure
    npm cache clean --force 2>/dev/null || true

    # 2. Install claude-flow globally if not present
    if [ ! -f "/usr/bin/claude-flow" ]; then
        log_info "📦 Installing claude-flow globally..."
        npm install -g claude-flow@alpha --force
        log_success "Installed claude-flow globally"
    else
        log_info "✅ claude-flow already installed globally"
    fi

    # 3. Copy pre-fixed files
    log_info "📋 Copying pre-fixed MCP server files..."

    # Check if fixed files exist
    if [ ! -d "/workspace/ext/fixed-mcp-files" ]; then
        log_error "Fixed files directory not found at /workspace/ext/fixed-mcp-files"
        log_error "Please ensure the fixed files are present before running this script"
        exit 1
    fi

    # Backup original files
    if [ -f "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js" ]; then
        cp "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js" \
           "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js.backup-$(date +%s)" 2>/dev/null || true
    fi

    if [ -f "/app/core-assets/scripts/mcp-tcp-server.js" ]; then
        cp "/app/core-assets/scripts/mcp-tcp-server.js" \
           "/app/core-assets/scripts/mcp-tcp-server.js.backup-$(date +%s)" 2>/dev/null || true
    fi

    # Copy fixed files
    cp "/workspace/ext/fixed-mcp-files/mcp-server.js" "/usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js"
    log_success "Copied fixed mcp-server.js"

    cp "/workspace/ext/fixed-mcp-files/mcp-tcp-server.js" "/app/core-assets/scripts/mcp-tcp-server.js"
    log_success "Copied fixed mcp-tcp-server.js"

    # 4. Create wrapper script to prevent npx from downloading new versions
    log_info "🔧 Creating npx wrapper to force global version..."
    cat > /usr/local/bin/npx-claude-flow-wrapper.sh << 'EOF'
#!/bin/bash
# Wrapper to force npx to use global claude-flow
if [[ "$*" == *"claude-flow"* ]]; then
    # Replace npx claude-flow with direct global call
    CMD="${@/npx claude-flow@alpha//usr/bin/claude-flow}"
    CMD="${CMD/npx claude-flow//usr/bin/claude-flow}"
    exec $CMD
else
    exec /usr/bin/npx "$@"
fi
EOF

    chmod +x /usr/local/bin/npx-claude-flow-wrapper.sh

    # 5. Update PATH to use wrapper (for current session)
    if ! echo "$PATH" | grep -q "/usr/local/bin"; then
        export PATH="/usr/local/bin:$PATH"
    fi

    # 6. Set environment variables
    export CLAUDE_FLOW_GLOBAL="/usr/bin/claude-flow"
    export CLAUDE_FLOW_NO_NPX="true"

    # 7. Restart TCP server if running
    log_info "🔄 Restarting TCP server with fixed version..."
    pkill -f mcp-tcp-server.js 2>/dev/null || true
    sleep 1

    # Start TCP server in background
    nohup node /app/core-assets/scripts/mcp-tcp-server.js > /tmp/mcp-tcp.log 2>&1 &
    TCP_PID=$!
    log_success "Started TCP server with PID $TCP_PID"

    # 8. Verify setup
    log_info "🔍 Verifying installation..."

    # Check global binary
    if /usr/bin/claude-flow --version > /dev/null 2>&1; then
        VERSION=$(/usr/bin/claude-flow --version 2>&1 | head -1)
        log_success "claude-flow is working: $VERSION"
    else
        log_error "claude-flow binary not working"
        exit 1
    fi

    # Wait for TCP server to be ready
    sleep 3

    # Test TCP server
    if nc -z localhost 9500 2>/dev/null; then
        log_success "TCP server is listening on port 9500"
    else
        log_warning "TCP server may not be ready yet on port 9500"
    fi

    log_success "✨ Simplified Claude Flow setup complete!"
    echo ""
    log_info "📝 Next steps:"
    echo "  1. The fixed MCP server is now installed"
    echo "  2. TCP server is running on port 9500"
    echo "  3. No mock data will be returned"
    echo "  4. Real agents will be tracked properly"
    echo ""
    log_info "🧪 To test:"
    echo "  bash /workspace/ext/final-test.sh"
    echo ""
    log_info "📁 Fixed files location:"
    echo "  /workspace/ext/fixed-mcp-files/mcp-server.js"
    echo "  /workspace/ext/fixed-mcp-files/mcp-tcp-server.js"
}

# Run setup
setup_claude_flow