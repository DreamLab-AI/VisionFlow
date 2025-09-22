#!/bin/bash
# Don't exit on errors during initialization
set +e

echo "=== MCP 3D Environment Starting ==="
echo "Container IP: $(hostname -I)"

# Security initialization
echo "=== Security Initialization ==="

# Check if security tokens are set
if [ -z "$WS_AUTH_TOKEN" ] || [ "$WS_AUTH_TOKEN" = "your-secure-websocket-token-change-me" ]; then
    echo "⚠️  WARNING: Default WebSocket auth token detected. Please update WS_AUTH_TOKEN in .env"
fi

if [ -z "$TCP_AUTH_TOKEN" ] || [ "$TCP_AUTH_TOKEN" = "your-secure-tcp-token-change-me" ]; then
    echo "⚠️  WARNING: Default TCP auth token detected. Please update TCP_AUTH_TOKEN in .env"
fi

if [ -z "$JWT_SECRET" ] || [ "$JWT_SECRET" = "your-super-secret-jwt-key-minimum-32-chars" ]; then
    echo "⚠️  WARNING: Default JWT secret detected. Please update JWT_SECRET in .env"
fi

# Create security log directory
mkdir -p /app/mcp-logs/security
chown -R dev:dev /app/mcp-logs

# Set secure permissions on scripts
chmod 750 /app/core-assets/scripts/*.js 2>/dev/null || true
chown dev:dev /app/core-assets/scripts/*.js 2>/dev/null || true

echo "✅ Security initialization complete"

# Ensure the dev user owns their home directory to prevent permission
# issues with npx, cargo, etc. This is safe to run on every start.
# Skip .claude directory as it's mounted read-only from host
chown dev:dev /home/dev
find /home/dev -maxdepth 1 -not -path "/home/dev/.claude*" -not -path "/home/dev" -exec chown -R dev:dev {} \; 2>/dev/null || true

# Fix sudo permissions (critical for setup scripts)
echo "Fixing sudo permissions..."
chown root:root /usr/bin/sudo
chmod 4755 /usr/bin/sudo

# Claude authentication setup
echo "=== Claude Authentication ==="
if [ -d /home/dev/.claude ] && [ -r /home/dev/.claude/.credentials.json ]; then
    echo "✅ Claude configuration directory mounted from host"
    # Check if OAuth token is also provided via environment
    if [ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
        echo "✅ Claude OAuth token also provided via environment"
    fi
elif [ -n "$CLAUDE_CODE_OAUTH_TOKEN" ]; then
    echo "✅ Claude OAuth token provided via environment"
elif [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✅ Anthropic API key provided via environment"
else
    echo "ℹ️  No Claude authentication detected. See docs/claude-auth.md for setup instructions."
    echo "    For local development: Run 'claude login' on your host machine first."
    echo "    For CI/CD: Set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY in your .env file."
fi

# Create multi-agent symlink for easy access to workspace tools
if [ ! -f /usr/local/bin/multi-agent ]; then
    # Create a multi-agent helper script
    cat > /usr/local/bin/multi-agent << 'EOF'
#!/bin/bash
# Multi-agent workspace helper

case "$1" in
    status)
        /app/core-assets/scripts/check-setup-status.sh
        ;;
    logs)
        tail -f /app/mcp-logs/automated-setup.log
        ;;
    health)
        /app/core-assets/scripts/health-check.sh
        ;;
    services)
        supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status
        ;;
    restart)
        supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart all
        ;;
    test-mcp)
        echo '{"jsonrpc":"2.0","id":"test","method":"tools/list","params":{}}' | nc localhost 9500
        ;;
    *)
        echo "Multi-Agent Docker Helper"
        echo "Usage: multi-agent [command]"
        echo ""
        echo "Commands:"
        echo "  status    - Check setup and service status"
        echo "  logs      - View automated setup logs"
        echo "  health    - Run health check"
        echo "  services  - Show supervisor service status"
        echo "  restart   - Restart all services"
        echo "  test-mcp  - Test MCP TCP connection"
        ;;
esac
EOF
    chmod +x /usr/local/bin/multi-agent
    echo "✅ Created multi-agent command"
fi

# The dev user inside the container is created with the same UID/GID as the
# host user, so a recursive chown on /workspace is not necessary and can
# cause permission errors on bind mounts.

# Ensure required directories exist
mkdir -p /workspace/.supervisor
mkdir -p /workspace/.swarm
mkdir -p /workspace/.swarm/sessions
mkdir -p /workspace/.hive-mind
mkdir -p /app/mcp-logs/security
chown -R dev:dev /workspace/.supervisor /workspace/.swarm /workspace/.hive-mind /app/mcp-logs

# Ensure Rust is available for the dev user
if [ -f "/home/dev/.cargo/env" ]; then
    # Set default toolchain if not set
    sudo -u dev bash -c "source /home/dev/.cargo/env && rustup default stable 2>/dev/null || true"
fi

# Create helpful aliases if .bashrc exists for the user
if [ -f "/home/dev/.bashrc" ]; then
    # Check if aliases already exist to avoid duplicates
    if ! grep -q "MCP Server Management" /home/dev/.bashrc; then
        cat >> /home/dev/.bashrc << 'EOF'

# MCP Server Management
alias mcp-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status'
alias mcp-restart='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart all'
alias mcp-logs='tail -f /app/mcp-logs/*.log'
alias mcp-test-blender='nc -zv localhost 9876'
alias mcp-test-qgis='nc -zv localhost 9877'
alias mcp-test-tcp='nc -zv localhost 9500'
alias mcp-test-ws='nc -zv localhost 3002'
alias mcp-blender-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status blender-mcp-server'
alias mcp-qgis-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status qgis-mcp-server'
alias mcp-tcp-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status mcp-tcp-server'
alias mcp-ws-status='supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status mcp-ws-bridge'
alias mcp-tmux-list='tmux ls'
alias mcp-tmux-attach='tmux attach-session -t'

# Quick server access
alias blender-log='tail -f /app/mcp-logs/blender-mcp-server.log'
alias qgis-log='tail -f /app/mcp-logs/qgis-mcp-server.log'
alias tcp-log='tail -f /app/mcp-logs/mcp-tcp-server.log'
alias ws-log='tail -f /app/mcp-logs/mcp-ws-bridge.log'

# Security and monitoring
alias mcp-health='curl -f http://localhost:9501/health'
alias mcp-security-audit='grep SECURITY /app/mcp-logs/*.log | tail -20'
alias mcp-connections='ss -tulnp | grep -E ":(3002|9500|9876|9877)"'
alias mcp-secure-client='node /app/core-assets/scripts/secure-client-example.js'

# Claude shortcuts
alias dsp='claude --dangerously-skip-permissions'

# Performance monitoring
alias mcp-performance='top -p $(pgrep -f "node.*mcp")'
alias mcp-memory='ps aux | grep -E "node.*mcp" | awk "{print \$1,\$2,\$4,\$6,\$11}"'

# Automation tools
alias setup-status='/app/core-assets/scripts/check-setup-status.sh'
alias setup-logs='tail -f /app/mcp-logs/automated-setup.log'
alias rerun-setup='/app/core-assets/scripts/automated-setup.sh'
EOF
    fi
fi

echo ""
echo "=== MCP Environment Ready ==="
echo "Background services are managed by supervisord."
echo "The WebSocket bridge for external control is on port 3002."
echo ""
echo "To set up a fresh workspace, run:"
echo "  /app/setup-workspace.sh"
echo ""

# Run setup script automatically on first start if marker doesn't exist
if [ ! -f /workspace/.setup_completed ]; then
    echo "First time setup detected. Running setup script..."
    if [ -x /app/setup-workspace.sh ]; then
        /app/setup-workspace.sh --quiet || {
            echo "⚠️  Setup script failed. You may need to run it manually."
        }
    fi
fi

# Verify services will start properly
echo "Verifying service prerequisites..."
if [ ! -f /workspace/scripts/mcp-tcp-server.js ]; then
    echo "⚠️  MCP scripts not found in workspace. Copying from core assets..."
    mkdir -p /workspace/scripts
    cp -r /app/core-assets/scripts/* /workspace/scripts/ 2>/dev/null || true
    chown -R dev:dev /workspace/scripts
fi

# Run comprehensive automated setup in background
if [ -x /app/core-assets/scripts/automated-setup.sh ]; then
    echo "Running automated setup process..."
    nohup /app/core-assets/scripts/automated-setup.sh > /app/mcp-logs/automated-setup.log 2>&1 &
    
    # Give setup a moment to start
    sleep 2
    
    # Show setup progress
    echo "Setup running in background. Check progress: tail -f /app/mcp-logs/automated-setup.log"
else
    echo "⚠️  Automated setup script not found"
fi

# Execute supervisord as the main process
exec /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf