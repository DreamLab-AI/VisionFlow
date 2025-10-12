#!/bin/bash
# Agentic Flow Workstation Initialization Script
# Runs on container startup to prepare the development environment

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   🚀 Initializing Agentic Flow CachyOS Workstation            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Environment Check
# =============================================================================

echo -e "${BLUE}🔍 Checking environment...${NC}"

# Check API keys
echo ""
echo "API Keys:"
[ -n "$ANTHROPIC_API_KEY" ] && echo -e "  ${GREEN}✓${NC} ANTHROPIC_API_KEY" || echo "  ✗ ANTHROPIC_API_KEY (optional)"
[ -n "$OPENAI_API_KEY" ] && echo -e "  ${GREEN}✓${NC} OPENAI_API_KEY" || echo "  ✗ OPENAI_API_KEY (optional)"
[ -n "$GOOGLE_GEMINI_API_KEY" ] && echo -e "  ${GREEN}✓${NC} GOOGLE_GEMINI_API_KEY" || echo "  ✗ GOOGLE_GEMINI_API_KEY (optional)"
[ -n "$OPENROUTER_API_KEY" ] && echo -e "  ${GREEN}✓${NC} OPENROUTER_API_KEY" || echo "  ✗ OPENROUTER_API_KEY (optional)"
[ -n "$E2B_API_KEY" ] && echo -e "  ${GREEN}✓${NC} E2B_API_KEY" || echo "  ✗ E2B_API_KEY (optional)"

# =============================================================================
# Management API Initialization
# =============================================================================

echo ""
echo -e "${BLUE}📡 Starting Management API Server...${NC}"

# Ensure logs directory exists
mkdir -p "$HOME/logs" "$HOME/logs/tasks"

# Start Management API with pm2
cd "$HOME/management-api"
if pm2 start server.js --name management-api --log "$HOME/logs/management-api.log" --time > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Management API started on port 9090"
    echo "    API Key: Set via MANAGEMENT_API_KEY environment variable"
    echo "    Logs: $HOME/logs/management-api.log"

    # Configure pm2 to start on container restart
    pm2 save > /dev/null 2>&1
else
    echo -e "  ${YELLOW}⚠${NC} Failed to start Management API"
fi

# =============================================================================
# MCP Server Initialization
# =============================================================================

if [ "$MCP_AUTO_START" = "true" ]; then
    echo ""
    echo -e "${BLUE}🔧 Starting MCP servers...${NC}"

    # Start MCP servers in background
    (
        sleep 5  # Give time for environment to settle
        agentic-flow mcp start > /tmp/mcp-startup.log 2>&1 &
        echo "MCP servers starting in background..."
    ) &
fi

# =============================================================================
# Network Connectivity Tests
# =============================================================================

echo ""
echo -e "${BLUE}🌐 Testing network connectivity...${NC}"

# Test Xinference connectivity
if [ "$ENABLE_XINFERENCE" = "true" ]; then
    echo -n "  Xinference (172.18.0.11:9997): "
    if curl -s --connect-timeout 3 http://172.18.0.11:9997/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Connected${NC}"
        XINFERENCE_MODELS=$(curl -s http://172.18.0.11:9997/v1/models | jq -r '.data[]?.id' 2>/dev/null | wc -l)
        [ $XINFERENCE_MODELS -gt 0 ] && echo "    Available models: $XINFERENCE_MODELS"
    else
        echo -e "${YELLOW}✗ Not reachable${NC}"
        echo "    (Ensure RAGFlow network is connected)"
    fi
fi

# Test external API connectivity
echo -n "  Anthropic API: "
if curl -s --connect-timeout 3 https://api.anthropic.com > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Reachable${NC}"
else
    echo -e "${YELLOW}✗ Not reachable${NC}"
fi

echo -n "  OpenAI API: "
if curl -s --connect-timeout 3 https://api.openai.com > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Reachable${NC}"
else
    echo -e "${YELLOW}✗ Not reachable${NC}"
fi

echo -n "  Google Gemini API: "
if curl -s --connect-timeout 3 https://generativelanguage.googleapis.com > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Reachable${NC}"
else
    echo -e "${YELLOW}✗ Not reachable${NC}"
fi

# =============================================================================
# GPU Check
# =============================================================================

echo ""
echo -e "${BLUE}🖥️  Checking GPU availability...${NC}"

if [ "$GPU_ACCELERATION" = "true" ]; then
    if nvidia-smi > /dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown")
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "Unknown")
        echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected: $GPU_NAME"
        echo "    Total memory: $GPU_MEM"

        # Check CUDA
        if [ -d "/usr/local/cuda" ] || command -v nvcc > /dev/null 2>&1; then
            CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//' || echo "Unknown")
            echo "    CUDA version: $CUDA_VERSION"
        fi
    elif rocm-smi > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} AMD GPU detected"
        rocm-smi --showproductname 2>/dev/null | head -3
    else
        echo -e "  ${YELLOW}⚠${NC} No GPU detected (CPU-only mode)"
        echo "    ONNX will use CPU inference (~6 tokens/sec)"
    fi
else
    echo "  GPU acceleration disabled"
fi

# =============================================================================
# Workspace Setup
# =============================================================================

echo ""
echo -e "${BLUE}📁 Setting up workspace...${NC}"

# Create workspace directories
mkdir -p "$WORKSPACE"/{projects,temp,agents}
mkdir -p "$HOME/models"
mkdir -p "$HOME/.claude-flow"/{memory,metrics,logs}
mkdir -p "$HOME/.config/agentic-flow"

echo "  Workspace: $WORKSPACE"
echo "  Models: $HOME/models"
echo "  Memory: $HOME/.claude-flow"

# =============================================================================
# Agentic Flow Verification
# =============================================================================

echo ""
echo -e "${BLUE}🤖 Verifying Agentic Flow installation...${NC}"

# Check if agentic-flow is installed
if command -v agentic-flow > /dev/null 2>&1; then
    VERSION=$(agentic-flow --version 2>/dev/null || echo "unknown")
    echo -e "  ${GREEN}✓${NC} Agentic Flow installed: $VERSION"

    # Count available agents
    AGENT_COUNT=$(agentic-flow --list 2>/dev/null | grep -c "^\s\+-" || echo "0")
    echo "    Available agents: $AGENT_COUNT"
else
    echo -e "  ${YELLOW}⚠${NC} Agentic Flow not found in PATH"
    echo "    Installing from source..."
    cd /tmp/agentic-flow 2>/dev/null && npm link || echo "    Installation deferred"
fi

# =============================================================================
# Final Setup
# =============================================================================

echo ""
echo -e "${BLUE}⚙️  Finalizing configuration...${NC}"

# Set up git if not configured
if [ ! -f "$HOME/.gitconfig" ]; then
    echo "  Configuring git..."
    git config --global user.name "Agentic Flow User"
    git config --global user.email "user@agentic-flow.local"
    git config --global init.defaultBranch main
fi

# Create helpful symlinks
ln -sf "$WORKSPACE" "$HOME/ws" 2>/dev/null || true

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Workstation initialization complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "🎯 Quick Start Commands:"
echo ""
echo "  Management API:"
echo "    curl -H \"Authorization: Bearer \$MANAGEMENT_API_KEY\" \\"
echo "         http://localhost:9090/v1/status   # System status"
echo "    curl -H \"Authorization: Bearer \$MANAGEMENT_API_KEY\" \\"
echo "         -X POST http://localhost:9090/v1/tasks \\"
echo "         -d '{\"agent\":\"coder\",\"task\":\"...\"}' # Create task"
echo ""
echo "  Basic Usage:"
echo "    agentic-flow --agent coder --task 'Build REST API'"
echo "    af --list                              # List all agents"
echo ""
echo "  Provider Selection:"
echo "    af-gemini --agent coder --task '...'   # Google Gemini"
echo "    af-openai --agent coder --task '...'   # OpenAI GPT-4o"
echo "    af-claude --agent coder --task '...'   # Anthropic Claude"
echo "    af-local --agent coder --task '...'    # Xinference (free)"
echo "    af-offline --agent coder --task '...'  # ONNX (offline)"
echo ""
echo "  Intelligent Router:"
echo "    af-optimize --agent coder --task '...' # Auto-select best model"
echo "    af-perf --agent coder --task '...'     # Optimize for performance"
echo "    af-cost --agent coder --task '...'     # Optimize for cost"
echo ""
echo "  Testing:"
echo "    test-providers                         # Test all providers"
echo "    test-gpu                               # Check GPU status"
echo "    check-keys                             # Verify API keys"
echo ""

echo "📚 Documentation:"
echo "    ~/README.workstation.md                # Full documentation"
echo "    afh                                    # Agentic Flow help"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep container running with interactive shell
exec /usr/bin/zsh
