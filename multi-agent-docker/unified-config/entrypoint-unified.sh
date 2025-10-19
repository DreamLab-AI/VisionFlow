#!/bin/bash
# Unified Container Entrypoint
# Handles multi-user setup, credential distribution, and service initialization

set -e

echo "========================================"
echo "  TURBO FLOW UNIFIED CONTAINER"
echo "========================================"
echo ""

# ============================================================================
# Phase 1: Directory Setup
# ============================================================================

echo "[1/9] Setting up directories..."

# Ensure all required directories exist
mkdir -p /home/devuser/{workspace,models,agents,.claude/skills,.config,.cache,logs}
mkdir -p /home/gemini-user/{workspace,.config,.cache}
mkdir -p /home/openai-user/{workspace,.config,.cache}
mkdir -p /home/zai-user/{workspace,.config,.cache}
mkdir -p /var/log /var/log/supervisor /run/dbus /run/user/1000 /tmp/.X11-unix /tmp/.ICE-unix
chmod 1777 /tmp/.X11-unix /tmp/.ICE-unix
chmod 700 /run/user/1000
chown devuser:devuser /run/user/1000

# Set permissions
chown -R devuser:devuser /home/devuser
chown -R gemini-user:gemini-user /home/gemini-user
chown -R openai-user:openai-user /home/openai-user
chown -R zai-user:zai-user /home/zai-user

echo "✅ Directories created and permissions set"

# ============================================================================
# Phase 2: Credential Distribution from Environment
# ============================================================================

echo "[2/9] Distributing credentials to users..."

# devuser - Claude Code configuration
if [ -n "$ANTHROPIC_API_KEY" ]; then
    sudo -u devuser bash -c "mkdir -p ~/.config/claude && cat > ~/.config/claude/config.json" <<EOF
{
  "apiKey": "$ANTHROPIC_API_KEY",
  "defaultModel": "claude-sonnet-4"
}
EOF
    echo "✅ Claude API key configured for devuser"
fi

# devuser - Z.AI API key for web-summary skill
if [ -n "$ZAI_API_KEY" ]; then
    sudo -u devuser bash -c "mkdir -p ~/.config/zai && cat > ~/.config/zai/api.json" <<EOF
{
  "apiKey": "$ZAI_API_KEY"
}
EOF
    echo "✅ Z.AI API key configured for devuser (web-summary skill)"
fi

# gemini-user - Google Gemini configuration
if [ -n "$GOOGLE_GEMINI_API_KEY" ]; then
    sudo -u gemini-user bash -c "mkdir -p ~/.config/gemini && cat > ~/.config/gemini/config.json" <<EOF
{
  "apiKey": "$GOOGLE_GEMINI_API_KEY",
  "defaultModel": "gemini-2.0-flash"
}
EOF
    export GOOGLE_API_KEY="$GOOGLE_GEMINI_API_KEY"
    echo "✅ Gemini API key configured for gemini-user"
fi

# openai-user - OpenAI configuration
if [ -n "$OPENAI_API_KEY" ]; then
    sudo -u openai-user bash -c "mkdir -p ~/.config/openai && cat > ~/.config/openai/config.json" <<EOF
{
  "apiKey": "$OPENAI_API_KEY",
  "organization": "$OPENAI_ORG_ID"
}
EOF
    echo "✅ OpenAI API key configured for openai-user"
fi

# zai-user - Z.AI service configuration
if [ -n "$ANTHROPIC_API_KEY" ] && [ -n "$ANTHROPIC_BASE_URL" ]; then
    sudo -u zai-user bash -c "mkdir -p ~/.config/zai && cat > ~/.config/zai/config.json" <<EOF
{
  "apiKey": "$ANTHROPIC_API_KEY",
  "baseUrl": "$ANTHROPIC_BASE_URL",
  "port": 9600,
  "workerPoolSize": ${CLAUDE_WORKER_POOL_SIZE:-4},
  "maxQueueSize": ${CLAUDE_MAX_QUEUE_SIZE:-50}
}
EOF
    echo "✅ Z.AI configuration created for zai-user"
fi

# GitHub token for all users
if [ -n "$GITHUB_TOKEN" ]; then
    for user in devuser gemini-user openai-user; do
        sudo -u $user bash -c "mkdir -p ~/.config/gh && cat > ~/.config/gh/config.yml" <<EOF
git_protocol: https
editor: vim
prompt: enabled
pager:
oauth_token: $GITHUB_TOKEN
EOF
    done
    echo "✅ GitHub token configured for all users"
fi

# ============================================================================
# Phase 3: Copy Host Claude Configuration (if available)
# ============================================================================

echo "[3/9] Checking for host Claude configuration..."

if [ -d "/mnt/host-claude" ] && [ -f "/mnt/host-claude/config.json" ]; then
    cp -r /mnt/host-claude/* /home/devuser/.claude/ 2>/dev/null || true
    chown -R devuser:devuser /home/devuser/.claude
    echo "✅ Host Claude configuration copied"
else
    echo "ℹ️  No host Claude configuration found (this is normal)"
fi

# ============================================================================
# Phase 4: Initialize DBus
# ============================================================================

echo "[4/9] Initializing DBus..."

# Clean up any stale PID files from previous runs
rm -f /run/dbus/pid /var/run/dbus/pid

# DBus will be started by supervisord
echo "✅ DBus configured (supervisord will start)"

# ============================================================================
# Phase 5: Setup Claude Skills
# ============================================================================

echo "[5/9] Setting up Claude Code skills..."

# Make skill tools executable
find /home/devuser/.claude/skills -name "*.py" -exec chmod +x {} \;
find /home/devuser/.claude/skills -name "*.js" -exec chmod +x {} \;
find /home/devuser/.claude/skills -name "*.sh" -exec chmod +x {} \;

# Count skills
SKILL_COUNT=$(find /home/devuser/.claude/skills -name "SKILL.md" | wc -l)
echo "✅ $SKILL_COUNT Claude Code skills available"

# ============================================================================
# Phase 6: Setup Agents
# ============================================================================

echo "[6/9] Setting up Claude agents..."

AGENT_COUNT=$(find /home/devuser/agents -name "*.md" 2>/dev/null | wc -l)
if [ "$AGENT_COUNT" -gt 0 ]; then
    echo "✅ $AGENT_COUNT agent templates available"
else
    echo "⚠️  No agent templates found"
fi

# ============================================================================
# Phase 7: Generate SSH Host Keys
# ============================================================================

echo "[7/9] Generating SSH host keys..."

if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A
    echo "✅ SSH host keys generated"
else
    echo "ℹ️  SSH host keys already exist"
fi

# ============================================================================
# Phase 8: Display Connection Information
# ============================================================================

echo "[8/9] Container ready! Connection information:"
echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│                   CONNECTION DETAILS                        │"
echo "├─────────────────────────────────────────────────────────────┤"
echo "│ SSH:             ssh devuser@<container-ip> -p 22           │"
echo "│                  Password: turboflow                        │"
echo "│                                                             │"
echo "│ VNC:             vnc://<container-ip>:5901                  │"
echo "│                  Password: turboflow                        │"
echo "│                  Display: :1                                │"
echo "│                                                             │"
echo "│ code-server:     http://<container-ip>:8080                 │"
echo "│                  (No authentication required)              │"
echo "│                                                             │"
echo "│ Management API:  http://<container-ip>:9090                 │"
echo "│                  Health: /health                            │"
echo "│                  Status: /api/v1/status                     │"
echo "│                                                             │"
echo "│ Z.AI Service:    http://localhost:9600 (internal only)      │"
echo "│                  Accessible via ragflow network            │"
echo "├─────────────────────────────────────────────────────────────┤"
echo "│ Users:                                                      │"
echo "│   devuser (1000)      - Claude Code, development           │"
echo "│   gemini-user (1001)  - Google Gemini CLI, gemini-flow     │"
echo "│   openai-user (1002)  - OpenAI Codex                       │"
echo "│   zai-user (1003)     - Z.AI service                       │"
echo "├─────────────────────────────────────────────────────────────┤"
echo "│ Skills:           $SKILL_COUNT custom Claude Code skills             │"
echo "│ Agents:           $AGENT_COUNT agent templates                       │"
echo "├─────────────────────────────────────────────────────────────┤"
echo "│ tmux Session:     workspace (8 windows)                     │"
echo "│   Attach with:    tmux attach-session -t workspace         │"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

# ============================================================================
# Phase 9: Start Supervisord
# ============================================================================

echo "[9/9] Starting supervisord (all services)..."
echo ""

# Display what will start
echo "Starting services:"
echo "  ✓ DBus daemon"
echo "  ✓ SSH server (port 22)"
echo "  ✓ VNC server (port 5901)"
echo "  ✓ XFCE4 desktop"
echo "  ✓ Management API (port 9090)"
echo "  ✓ code-server (port 8080)"
echo "  ✓ Claude Z.AI service (port 9600)"
echo "  ✓ Gemini-flow daemon"
echo "  ✓ tmux workspace auto-start"
echo ""
echo "========================================"
echo "  ALL SYSTEMS READY - STARTING NOW"
echo "========================================"
echo ""

# Start supervisord (will run in foreground)
exec /opt/venv/bin/supervisord -n -c /etc/supervisord.conf
