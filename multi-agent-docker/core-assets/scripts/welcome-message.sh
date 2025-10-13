#!/bin/sh
# This script is sourced by /etc/bash.bashrc to display a welcome message.

# Skip welcome message if setup has been completed
if [ -f /workspace/.setup_completed ]; then
    return 0 2>/dev/null || exit 0
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   🚀 Agentic Flow CachyOS Workstation                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📦 Workspace: /home/devuser/workspace"
echo "🔧 MCP Tools: Available"
echo "💾 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 2>/dev/null || echo 'N/A')"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 Quick Start:"
echo "  dsp                                    # Launch Claude Code"
echo "  af --agent coder --task 'Build API'   # Run agent"
echo "  af-gemini --agent coder --task '...'  # Force Gemini"
echo ""
echo "🔧 Useful Commands:"
echo "  test-providers    # Test all model providers"
echo "  test-gemini-flow  # Test Gemini-Flow orchestration"
echo "  check-keys        # Check API key status"
echo "  mcp-list          # List all MCP tools"
echo "  test-gpu          # Check GPU status"
echo ""
echo "🐝 Gemini-Flow (66-Agent Swarms):"
echo "  gf-swarm          # Deploy 66 specialized agents"
echo "  gf-deploy <obj>   # Deploy custom swarm"
echo "  gf-monitor        # Monitor A2A protocols"
echo ""
echo "📚 Claude Code Plugin:"
echo "  Inside Claude Code (dsp), run:"
echo "    /plugin ruvnet/claude-flow"
echo "  This adds multi-agent orchestration to Claude Code"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""