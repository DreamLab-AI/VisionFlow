#!/bin/bash
# dsp - Shortcut for claude-safe --dangerously-skip-permissions
# Quick access wrapper for Claude Code with permission skipping and MCP env

# Load environment variables for MCP servers from container config
load_mcp_env() {
    # Load from container's docker env (already set by docker-compose)
    # These are inherited from the container's environment
    export CONTEXT7_API_KEY="${CONTEXT7_API_KEY:-}"
    export BRAVE_API_KEY="${BRAVE_API_KEY:-}"
    export GITHUB_TOKEN="${GITHUB_TOKEN:-}"
    export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
    export GOOGLE_GEMINI_API_KEY="${GOOGLE_GEMINI_API_KEY:-}"
    export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
    export ZAI_CONTAINER_URL="http://claude-zai-service:9600"
    export DISPLAY="${DISPLAY:-:1}"
}

if [ "$(id -u)" -eq 0 ]; then
    # Running as root, switch to devuser and preserve working directory
    cd "$(pwd)" 2>/dev/null || cd /home/devuser/workspace/project
    # Load environment variables
    load_mcp_env
    # Pass all environment variables explicitly to devuser shell
    exec su -m devuser -c "
        export HOME=/home/devuser
        export CONTEXT7_API_KEY='${CONTEXT7_API_KEY}'
        export BRAVE_API_KEY='${BRAVE_API_KEY}'
        export GITHUB_TOKEN='${GITHUB_TOKEN}'
        export GOOGLE_API_KEY='${GOOGLE_API_KEY}'
        export OPENAI_API_KEY='${OPENAI_API_KEY}'
        export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY}'
        export GOOGLE_GEMINI_API_KEY='${GOOGLE_GEMINI_API_KEY}'
        export OPENROUTER_API_KEY='${OPENROUTER_API_KEY}'
        export ZAI_CONTAINER_URL='http://claude-zai-service:9600'
        export DISPLAY='${DISPLAY:-:1}'
        cd '$(pwd)' && claude --dangerously-skip-permissions $*
    "
else
    # Already running as devuser
    load_mcp_env
    exec claude --dangerously-skip-permissions "$@"
fi
