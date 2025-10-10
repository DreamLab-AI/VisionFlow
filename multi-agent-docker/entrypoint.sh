#!/bin/bash
set -e
trap 'echo "[ENTRYPOINT ERROR] Caught error at line $LINENO, continuing..." >> /var/log/multi-agent/entrypoint.log' ERR

# Persistent logging
ENTRYPOINT_LOG="/var/log/multi-agent/entrypoint.log"
mkdir -p "$(dirname "$ENTRYPOINT_LOG")"
exec 1> >(tee -a "$ENTRYPOINT_LOG")
exec 2>&1

echo "🚀 Initializing Multi-Agent Environment..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting entrypoint.sh"

# Ensure the dev user owns their home directory to prevent permission
# issues with npx, cargo, etc. Run in background to avoid blocking startup.
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting home directory permissions for user 'dev' (background)..."
(
    # Only fix specific critical directories, avoid traversing into mounts
    chown -R dev:dev /home/dev/.config 2>/dev/null || true
    chown -R dev:dev /home/dev/.local 2>/dev/null || true
    chown -R dev:dev /home/dev/.cargo 2>/dev/null || true
    chown dev:dev /home/dev/.bashrc /home/dev/.profile 2>/dev/null || true

    # Grant dev user access to Docker socket
    if [ -S /var/run/docker.sock ]; then
        DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
        if ! getent group "$DOCKER_GID" >/dev/null 2>&1; then
            groupadd -g "$DOCKER_GID" docker-host
        fi
        usermod -aG "$DOCKER_GID" dev 2>/dev/null || true
    fi
) &
PERM_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Permission fixing started in background (PID: $PERM_PID)"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fixing critical permissions..."
# Fix sudo permissions (critical for setup scripts)
chown root:root /usr/bin/sudo && chmod 4755 /usr/bin/sudo

# Ensure dev user is in sudoers
echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev
chmod 440 /etc/sudoers.d/dev
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Critical permissions fixed"

# Ensure required directories exist and have correct permissions for supervisord
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparing supervisor directories..."
mkdir -p /workspace/.supervisor
mkdir -p /workspace/.swarm
mkdir -p /workspace/.hive-mind
mkdir -p /app/mcp-logs/security
chown -R dev:dev /workspace /app/mcp-logs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Supervisor directories ready"

echo ""
echo "=== MCP Environment Ready ==="
echo "Starting background services..."
echo ""

# Clean up stale X11 locks (must be done BEFORE supervisor starts)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning X11 lock files..."
rm -f /tmp/.X*-lock /tmp/.X11-unix/X* 2>/dev/null || true
echo "[$(date '+%Y-%m-%d %H:%M:%S')] X11 locks cleaned"

# Detect and configure GPU
if [ -f /app/scripts/gpu-detect.sh ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running GPU detection..."
    source /app/scripts/gpu-detect.sh --detect
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU detection complete"
fi

echo ""
echo "✨ Automatic setup will begin in a few seconds..."
echo "   (Check progress with: tail -f /workspace/.setup.log)"
echo ""

# Install Chrome extensions
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing Chrome extensions..."
if [ -f /app/scripts/install-chrome-extensions.sh ]; then
    /app/scripts/install-chrome-extensions.sh &
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Chrome extension installation started in background"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Chrome extension installer not found"
fi

# Create wrapper for claude-flow to ensure database isolation
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up claude-flow wrapper..."
if [ -f "/app/node_modules/.bin/claude-flow" ]; then
    # Install wrapper script that isolates root user database access
    ln -sf /app/scripts/claude-flow-wrapper.sh /usr/bin/claude-flow
    chmod +x /app/scripts/claude-flow-wrapper.sh
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installed claude-flow wrapper (prevents database conflicts)"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: claude-flow not found in /app/node_modules/.bin/"
fi

# Ensure Claude CLI is available (installed via npm as @anthropic-ai/claude-code)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude CLI installed globally via npm"

# Supervisord will be started by CMD (not here)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Entrypoint initialization complete, supervisord will start via CMD"

# Initialize Claude in background after a short delay
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting setup background job..."
(
    exec >/workspace/.setup.log 2>&1
    echo "=== Automatic Setup Log - $(date) ==="

    sleep 5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initializing Claude configuration..."
    # Run claude init as dev user to ensure proper permissions
    su - dev -c 'claude --dangerously-skip-permissions /exit' || true

    # Configure Claude MCP with isolated databases
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configuring Claude MCP database isolation..."
    if [ -f /app/scripts/configure-claude-mcp.sh ]; then
        /app/scripts/configure-claude-mcp.sh || echo "⚠️ MCP config failed, continuing..."
    fi

    # Initialize hive-mind session infrastructure
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initializing hive-mind session manager..."
    if [ -f /app/scripts/hive-session-manager.sh ]; then
        /app/scripts/hive-session-manager.sh init || echo "⚠️ Session manager init failed, continuing..."
    fi

    # Generate topics.json from markdown directory if available
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for markdown directory to generate topics.json..."
    if [ -d /workspace/ext/data/markdown ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Found markdown directory, generating topics.json..."
        if [ -f /app/scripts/generate-topics-from-markdown.py ]; then
            mkdir -p /app/core-assets/config
            python3 /app/scripts/generate-topics-from-markdown.py \
                /workspace/ext/data/markdown \
                /app/core-assets/config/topics.json || {
                echo "⚠️ Topics generation failed, continuing..."
            }
            # Verify generation succeeded
            if [ -f /app/core-assets/config/topics.json ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Generated topics.json successfully"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ topics.json not found after generation"
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ generate-topics-from-markdown.py not found, skipping..."
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No markdown directory found, skipping topics.json generation"
    fi

    # Check if workspace setup hasn't been done yet
    if [ ! -f /workspace/.setup_completed ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running workspace setup..."
        su - dev -c '/app/setup-workspace.sh' || {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Automatic setup failed. Run manually: /app/setup-workspace.sh"
        }
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Workspace already set up, skipping..."
    fi

    echo "=== Setup complete at $(date) ==="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setup background job finished"
) &
SETUP_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setup job started with PID: $SETUP_PID"

# Execute the command (supervisord or override)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing command: $@"
exec "$@"