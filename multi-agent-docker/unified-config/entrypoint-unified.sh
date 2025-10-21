#!/bin/bash
# Unified Container Entrypoint - Enhanced Edition
# Handles multi-user setup, credential distribution, service initialization, and CLAUDE.md enhancement

set -e

echo "========================================"
echo "  TURBO FLOW UNIFIED CONTAINER"
echo "========================================"
echo ""

# ============================================================================
# Phase 1: Directory Setup
# ============================================================================

echo "[1/10] Setting up directories..."

# Ensure all required directories exist
mkdir -p /home/devuser/{workspace,models,agents,.claude/skills,.config,.cache,logs,.local/share}
mkdir -p /home/gemini-user/{workspace,.config,.cache,.gemini-flow}
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

echo "✓ Directories created and permissions set"

# ============================================================================
# Phase 2: Credential Distribution from Environment
# ============================================================================

echo "[2/10] Distributing credentials to users..."

# devuser - Claude Code configuration
if [ -n "$ANTHROPIC_API_KEY" ]; then
    sudo -u devuser bash -c "mkdir -p ~/.config/claude && cat > ~/.config/claude/config.json" <<EOF
{
  "apiKey": "$ANTHROPIC_API_KEY",
  "defaultModel": "claude-sonnet-4"
}
EOF
    echo "✓ Claude API key configured for devuser"
fi

# devuser - Z.AI API key for web-summary skill
if [ -n "$ZAI_API_KEY" ]; then
    sudo -u devuser bash -c "mkdir -p ~/.config/zai && cat > ~/.config/zai/api.json" <<EOF
{
  "apiKey": "$ZAI_API_KEY"
}
EOF
    echo "✓ Z.AI API key configured for devuser (web-summary skill)"
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
    echo "✓ Gemini API key configured for gemini-user"
fi

# openai-user - OpenAI configuration
if [ -n "$OPENAI_API_KEY" ]; then
    sudo -u openai-user bash -c "mkdir -p ~/.config/openai && cat > ~/.config/openai/config.json" <<EOF
{
  "apiKey": "$OPENAI_API_KEY",
  "organization": "$OPENAI_ORG_ID"
}
EOF
    echo "✓ OpenAI API key configured for openai-user"
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
    echo "✓ Z.AI configuration created for zai-user"
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
    echo "✓ GitHub token configured for all users"
fi

# ============================================================================
# Phase 3: Copy Host Claude Configuration (if available)
# ============================================================================

echo "[3/10] Checking for host Claude configuration..."

if [ -d "/mnt/host-claude" ] && [ -f "/mnt/host-claude/config.json" ]; then
    cp -r /mnt/host-claude/* /home/devuser/.claude/ 2>/dev/null || true
    chown -R devuser:devuser /home/devuser/.claude
    echo "✓ Host Claude configuration copied"
else
    echo "ℹ️  No host Claude configuration found (this is normal)"
fi

# ============================================================================
# Phase 4: Initialize DBus
# ============================================================================

echo "[4/10] Initializing DBus..."

# Clean up any stale PID files from previous runs
rm -f /run/dbus/pid /var/run/dbus/pid

# DBus will be started by supervisord
echo "✓ DBus configured (supervisord will start)"

# ============================================================================
# Phase 5: Setup Claude Skills
# ============================================================================

echo "[5/10] Setting up Claude Code skills..."

# Make skill tools executable
find /home/devuser/.claude/skills -name "*.py" -exec chmod +x {} \;
find /home/devuser/.claude/skills -name "*.js" -exec chmod +x {} \;
find /home/devuser/.claude/skills -name "*.sh" -exec chmod +x {} \;

# Count skills
SKILL_COUNT=$(find /home/devuser/.claude/skills -name "SKILL.md" | wc -l)
echo "✓ $SKILL_COUNT Claude Code skills available"

# ============================================================================
# Phase 6: Setup Agents
# ============================================================================

echo "[6/10] Setting up Claude agents..."

AGENT_COUNT=$(find /home/devuser/agents -name "*.md" 2>/dev/null | wc -l)
if [ "$AGENT_COUNT" -gt 0 ]; then
    echo "✓ $AGENT_COUNT agent templates available"
else
    echo "ℹ️  No agent templates found"
fi

# ============================================================================
# Phase 6.5: Initialize Claude Flow & Clean NPX Cache
# ============================================================================

echo "[6.5/10] Initializing Claude Flow..."

# Clean any stale NPX caches from all users to prevent corruption
rm -rf /home/devuser/.npm/_npx/* 2>/dev/null || true
rm -rf /home/gemini-user/.npm/_npx/* 2>/dev/null || true
rm -rf /home/openai-user/.npm/_npx/* 2>/dev/null || true
rm -rf /home/zai-user/.npm/_npx/* 2>/dev/null || true
rm -rf /root/.npm/_npx/* 2>/dev/null || true

# Run claude-flow init --force as devuser
sudo -u devuser bash -c "cd /home/devuser && claude-flow init --force" 2>/dev/null || echo "ℹ️  Claude Flow init skipped (not critical)"

# Fix hooks to use global claude-flow instead of npx (prevents cache corruption)
if [ -f /home/devuser/.claude/settings.json ]; then
    sed -i 's|npx claude-flow@alpha|claude-flow|g' /home/devuser/.claude/settings.json
    chown devuser:devuser /home/devuser/.claude/settings.json
    echo "✓ Hooks updated to use global claude-flow"
fi

echo "✓ Claude Flow initialized and NPX cache cleared"

# ============================================================================
# Phase 6.7: Configure Cross-User Service Access
# ============================================================================

echo "[6.7/10] Configuring cross-user service access..."

# Create shared directory for inter-service sockets
mkdir -p /var/run/agentic-services
chmod 755 /var/run/agentic-services

# Create symlinks for devuser to access isolated services
mkdir -p /home/devuser/.local/share/agentic-sockets
ln -sf /var/run/agentic-services/gemini-mcp.sock /home/devuser/.local/share/agentic-sockets/gemini-mcp.sock 2>/dev/null || true
ln -sf http://localhost:9600 /home/devuser/.local/share/agentic-sockets/zai-api.txt 2>/dev/null || true

# Add environment variable exports to devuser's zshrc for service discovery
sudo -u devuser bash -c 'cat >> ~/.zshrc' <<'ENV_EXPORTS'

# Cross-user service access (auto-configured)
export GEMINI_MCP_SOCKET="/var/run/agentic-services/gemini-mcp.sock"
export ZAI_API_URL="http://localhost:9600"
export ZAI_CONTAINER_URL="http://localhost:9600"
export OPENAI_CODEX_SOCKET="/var/run/agentic-services/openai-codex.sock"
ENV_EXPORTS

# Configure MCP settings for Claude Code
sudo -u devuser bash -c 'mkdir -p ~/.config/claude && cat > ~/.config/claude/mcp_settings.json' <<'MCP_CONFIG'
{
  "mcpServers": {
    "web-summary": {
      "command": "node",
      "args": ["/home/devuser/.claude/skills/web-summary/mcp-server/server.js"],
      "env": {
        "ZAI_CONTAINER_URL": "http://localhost:9600",
        "WEB_SUMMARY_TOOL_PATH": "/home/devuser/.claude/skills/web-summary/tools/web_summary_tool.py"
      }
    },
    "qgis": {
      "command": "python3",
      "args": ["-u", "/home/devuser/.claude/skills/qgis/tools/qgis_mcp.py"],
      "env": {
        "QGIS_HOST": "localhost",
        "QGIS_PORT": "9877"
      }
    },
    "blender": {
      "command": "node",
      "args": ["/home/devuser/.claude/skills/blender/tools/mcp-blender-client.js"],
      "env": {
        "BLENDER_HOST": "localhost",
        "BLENDER_PORT": "9876"
      }
    },
    "imagemagick": {
      "command": "python3",
      "args": ["-u", "/home/devuser/.claude/skills/imagemagick/tools/imagemagick_mcp.py"]
    },
    "kicad": {
      "command": "python3",
      "args": ["-u", "/home/devuser/.claude/skills/kicad/tools/kicad_mcp.py"]
    },
    "ngspice": {
      "command": "python3",
      "args": ["-u", "/home/devuser/.claude/skills/ngspice/tools/ngspice_mcp.py"]
    },
    "pbr-rendering": {
      "command": "python3",
      "args": ["-u", "/home/devuser/.claude/skills/pbr-rendering/tools/pbr_mcp_client.py"],
      "env": {
        "PBR_HOST": "localhost",
        "PBR_PORT": "9878"
      }
    },
    "playwright": {
      "command": "node",
      "args": ["/home/devuser/.claude/skills/playwright/tools/playwright-mcp-local.js"]
    }
  }
}
MCP_CONFIG

chown -R devuser:devuser /home/devuser/.local/share/agentic-sockets
chown -R devuser:devuser /home/devuser/.config/claude

echo "✓ Cross-user service access configured"
echo "  - Gemini MCP socket: /var/run/agentic-services/gemini-mcp.sock"
echo "  - Z.AI API: http://localhost:9600"
echo "  - MCP Servers: 8 skills registered (web-summary, qgis, blender, imagemagick, kicad, ngspice, pbr, playwright)"
echo "  - Environment variables added to devuser's .zshrc"

# ============================================================================
# Phase 7: Generate SSH Host Keys
# ============================================================================

echo "[7/10] Generating SSH host keys..."

if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    ssh-keygen -A
    echo "✓ SSH host keys generated"
else
    echo "ℹ️  SSH host keys already exist"
fi

# ============================================================================
# Phase 8: Enhance CLAUDE.md with Project Context
# ============================================================================

echo "[8/10] Enhancing CLAUDE.md with project-specific context..."

# Append compact project documentation to system CLAUDE.md
sudo -u devuser bash -c 'cat >> /home/devuser/CLAUDE.md' <<'CLAUDE_APPEND'

---

## 🚀 Project-Specific: Turbo Flow Claude

### 610 Claude Sub-Agents
- **Repository**: https://github.com/ChrisRoyse/610ClaudeSubagents
- **Location**: `/home/devuser/agents/*.md` (610+ templates)
- **Usage**: Load specific agents with `cat agents/<agent-name>.md`
- **Key Agents**: doc-planner, microtask-breakdown, github-pr-manager, tdd-london-swarm

### Z.AI Service (Cost-Effective Claude API)
**Port**: 9600 (internal only) | **User**: zai-user | **Worker Pool**: 4 concurrent
```bash
# Health check
curl http://localhost:9600/health

# Chat request
curl -X POST http://localhost:9600/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your prompt here", "timeout": 30000}'

# Switch to zai-user
as-zai
```

### Gemini Flow Commands
```bash
gf-init        # Initialize (protocols: a2a,mcp, topology: hierarchical)
gf-swarm       # 66 agents with intelligent coordination
gf-architect   # 5 system architects
gf-coder       # 12 master coders
gf-status      # Swarm status
gf-monitor     # Protocols and performance
gf-health      # Health check
```

### Multi-User System
| User | UID | Purpose | Switch |
|------|-----|---------|--------|
| devuser | 1000 | Claude Code, primary dev | - |
| gemini-user | 1001 | Google Gemini, gemini-flow | `as-gemini` |
| openai-user | 1002 | OpenAI Codex | `as-openai` |
| zai-user | 1003 | Z.AI service (port 9600) | `as-zai` |

### tmux Workspace (8 Windows)
**Attach**: `tmux attach -t workspace`
| Win | Name | Purpose |
|-----|------|---------|
| 0 | Claude-Main | Primary workspace |
| 1 | Claude-Agent | Agent execution |
| 2 | Services | supervisord monitoring |
| 3 | Development | Python/Rust/CUDA dev |
| 4 | Logs | Service logs (split) |
| 5 | System | htop monitoring |
| 6 | VNC-Status | VNC info |
| 7 | SSH-Shell | General shell |

### Management API
**Base**: http://localhost:9090 | **Auth**: `X-API-Key: <MANAGEMENT_API_KEY>`
```bash
GET  /health              # Health (no auth)
GET  /api/status          # System status
POST /api/tasks           # Create task
GET  /api/tasks/:id       # Task status
GET  /metrics             # Prometheus metrics
GET  /documentation       # Swagger UI
```

### Diagnostic Commands
```bash
# Service status
sudo supervisorctl status

# Container diagnostics
docker exec turbo-flow-unified supervisorctl status
docker stats turbo-flow-unified

# Logs
sudo supervisorctl tail -f management-api
sudo supervisorctl tail -f claude-zai
tail -f /var/log/supervisord.log

# User switching test
as-gemini whoami  # Should output: gemini-user
```

### Service Ports
| Port | Service | Access |
|------|---------|--------|
| 22 | SSH | Public (mapped to 2222) |
| 5901 | VNC | Public |
| 8080 | code-server | Public |
| 9090 | Management API | Public |
| 9600 | Z.AI | Internal only |

**Security**: Default creds are DEVELOPMENT ONLY. Change before production:
- SSH: `devuser:turboflow`
- VNC: `turboflow`
- Management API: `X-API-Key: change-this-secret-key`

### Development Environment Notes

**Container Modification Best Practices**:
- ✅ **DO**: Modify Dockerfile and entrypoint scripts DIRECTLY in the project
- ❌ **DON'T**: Create patching scripts or temporary fixes
- ✅ **DO**: Edit /home/devuser/workspace/project/multi-agent-docker/ files
- ❌ **DON'T**: Use workarounds - fix the root cause

**Isolated Docker Environment**:
- This container is isolated from external build systems
- Only these validation tools work:
  - \`cargo test\` - Rust project testing
  - \`npm run check\` / \`npm test\` - Node.js validation
  - \`pytest\` - Python testing
- **DO NOT** attempt to:
  - Build external projects directly
  - Run production builds inside container
  - Execute deployment scripts
  - Access external build infrastructure
- **Instead**: Test, validate, and export artifacts

**File Organization**:
- Never save working files to root (/)
- Use appropriate subdirectories:
  - /docs - Documentation
  - /scripts - Helper scripts
  - /tests - Test files
  - /config - Configuration
CLAUDE_APPEND

echo "✓ CLAUDE.md enhanced with project context"

# ============================================================================
# Phase 9: Display Connection Information
# ============================================================================

echo "[9/10] Container ready! Connection information:"
echo ""
echo "+-------------------------------------------------------------+"
echo "│                   CONNECTION DETAILS                        │"
echo "+-------------------------------------------------------------│"
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
echo "+-------------------------------------------------------------│"
echo "│ Users:                                                      │"
echo "│   devuser (1000)      - Claude Code, development           │"
echo "│   gemini-user (1001)  - Google Gemini CLI, gemini-flow     │"
echo "│   openai-user (1002)  - OpenAI Codex                       │"
echo "│   zai-user (1003)     - Z.AI service                       │"
echo "+-------------------------------------------------------------│"
echo "│ Skills:           $SKILL_COUNT custom Claude Code skills             │"
echo "│ Agents:           $AGENT_COUNT agent templates                       │"
echo "+-------------------------------------------------------------│"
echo "│ tmux Session:     workspace (8 windows)                     │"
echo "│   Attach with:    tmux attach-session -t workspace         │"
echo "+-------------------------------------------------------------+"
echo ""

# ============================================================================
# Phase 10: Start Supervisord
# ============================================================================

echo "[10/10] Starting supervisord (all services)..."
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
