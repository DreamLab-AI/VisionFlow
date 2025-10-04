#!/bin/bash
set -ex

echo "🚀 Initializing Multi-Agent Environment with GUI..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting entrypoint.sh"

# Setup VNC and X server
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting X server and VNC..."
export DISPLAY=:1
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
Xvfb :1 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 3

if ps -p $XVFB_PID > /dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Xvfb started successfully (PID: $XVFB_PID)"
else
    echo "ERROR: Xvfb failed to start"
    exit 1
fi

# Start XFCE desktop
startxfce4 &
sleep 3

# Start VNC server
x11vnc -display :1 -nopw -forever -xkb -listen 0.0.0.0 -rfbport 5901 -verbose &
VNC_PID=$!
sleep 3

if ps -p $VNC_PID > /dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] VNC server started (PID: $VNC_PID)"
else
    echo "WARNING: VNC server may not have started"
fi

# Install Blender MCP addon
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing Blender MCP addon..."
BLENDER_VERSION=$(/opt/blender-4.5/blender --version | head -n1 | grep -oP '(?<=Blender )\d+\.\d+' || echo "4.5")
ADDON_DIR="/home/dev/.config/blender/${BLENDER_VERSION}/scripts/addons"
mkdir -p "$ADDON_DIR"
cp /home/dev/addon.py "$ADDON_DIR/addon.py"
chown -R dev:dev /home/dev/.config/blender

# Start GUI applications in background
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting GUI applications..."
su - dev -c "DISPLAY=:1 /opt/blender-4.5/blender --python /home/dev/autostart.py" &
su - dev -c "DISPLAY=:1 qgis" &
sleep 5

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GUI setup complete"

# Ensure the dev user owns their home directory to prevent permission
# issues with npx, cargo, etc. Run in background to avoid blocking startup.
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting home directory permissions for user 'dev' (background)..."
(
    # Max depth 1 first (fast)
    find /home/dev -maxdepth 1 ! -name '.claude*' -exec chown dev:dev {} \; 2>/dev/null || true
    # Deeper directories in background (slow, skip .claude entirely)
    find /home/dev -mindepth 2 -maxdepth 5 -not -path "/home/dev/.claude*" -exec chown dev:dev {} \; 2>/dev/null || true
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
mkdir -p /app/mcp-logs/security
chown -R dev:dev /workspace/.supervisor /workspace/.swarm /app/mcp-logs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Supervisor directories ready"

echo ""
echo "=== MCP Environment Ready ==="
echo "Starting background services..."
echo ""

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

# Create symlink for claude-flow from the installed node_modules
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up claude-flow symlink..."
if [ -f "/app/node_modules/.bin/claude-flow" ]; then
    ln -sf /app/node_modules/.bin/claude-flow /usr/bin/claude-flow
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Created claude-flow symlink from node_modules"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: claude-flow not found in /app/node_modules/.bin/"
fi

# Start supervisord in the background for all cases
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting supervisord in background..."
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &
SUPERVISORD_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Supervisord started with PID: $SUPERVISORD_PID"

# Give supervisord a moment to start
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for supervisord to initialize..."
sleep 2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Supervisord initialization wait complete"

# Initialize Claude in background after a short delay
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting setup background job..."
(
    exec >/workspace/.setup.log 2>&1
    echo "=== Automatic Setup Log - $(date) ==="

    sleep 5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initializing Claude configuration..."
    # Run claude init as dev user to ensure proper permissions
    su - dev -c 'claude --dangerously-skip-permissions /exit' || true

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

# If a command is passed to the entrypoint (like /bin/bash), execute it.
# Otherwise, just wait for supervisord
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking for command arguments..."
if [ "$#" -gt 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing command: $@"
    exec "$@"
else
    # Keep the container running by following supervisor logs
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No command specified, tailing supervisord logs..."
    exec tail -f /app/mcp-logs/supervisord.log
fi