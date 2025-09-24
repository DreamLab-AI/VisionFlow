#!/bin/bash
set -e

echo "🚀 Initializing Multi-Agent Environment..."

# Ensure the dev user owns their home directory to prevent permission
# issues with npx, cargo, etc. This is safe to run on every start.
# Skip mounted files that might be read-only
echo "Setting home directory permissions for user 'dev'..."
find /home/dev -maxdepth 1 ! -name '.claude*' -exec chown dev:dev {} \; 2>/dev/null || true
# For deeper directories, but skip .claude directory entirely
find /home/dev -mindepth 2 -not -path "/home/dev/.claude*" -exec chown -R dev:dev {} \; 2>/dev/null || true

echo "Fixing critical permissions..."
# Fix sudo permissions (critical for setup scripts)
chown root:root /usr/bin/sudo && chmod 4755 /usr/bin/sudo

# Ensure dev user is in sudoers
echo "dev ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/dev
chmod 440 /etc/sudoers.d/dev

# Ensure required directories exist and have correct permissions for supervisord
echo "Preparing supervisor directories..."
mkdir -p /workspace/.supervisor
mkdir -p /workspace/.swarm
mkdir -p /app/mcp-logs/security
chown -R dev:dev /workspace/.supervisor /workspace/.swarm /app/mcp-logs

echo ""
echo "=== MCP Environment Ready ==="
echo "Starting background services..."
echo ""
echo "✨ Automatic setup will begin in a few seconds..."
echo "   (Check progress with: tail -f /workspace/.setup.log)"
echo ""

# Start supervisord in the background for all cases
echo "Starting supervisord in background..."
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &

# Give supervisord a moment to start
sleep 2

# Initialize Claude in background after a short delay
(
    exec >/workspace/.setup.log 2>&1
    echo "=== Automatic Setup Log - $(date) ==="
    
    sleep 5
    echo "Initializing Claude configuration..."
    # Run claude init as dev user to ensure proper permissions
    su - dev -c 'claude --dangerously-skip-permissions /exit' || true
    
    # Check if workspace setup hasn't been done yet
    if [ ! -f /workspace/.setup_completed ]; then
        echo "Running workspace setup..."
        su - dev -c '/app/setup-workspace.sh' || {
            echo "⚠️  Automatic setup failed. Run manually: /app/setup-workspace.sh"
        }
    else
        echo "Workspace already set up, skipping..."
    fi
    
    echo "=== Setup complete at $(date) ==="
) &

# If a command is passed to the entrypoint (like /bin/bash), execute it.
# Otherwise, just wait for supervisord
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    # Keep the container running by following supervisor logs
    exec tail -f /var/log/supervisor/supervisord.log
fi