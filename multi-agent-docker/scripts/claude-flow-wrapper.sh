#!/bin/bash
# claude-flow wrapper - ensures database isolation based on user context
set -e

# Detect execution context
CURRENT_USER=$(whoami)

if [ "$CURRENT_USER" = "root" ]; then
    # Root user: use isolated directory to prevent database conflicts
    ISOLATED_DIR="/workspace/.swarm/root-cli-instance"
    mkdir -p "$ISOLATED_DIR"
    chown -R dev:dev "$ISOLATED_DIR" 2>/dev/null || true

    # Execute as dev user in isolated directory
    exec su -s /bin/bash dev -c "
        cd '$ISOLATED_DIR'
        exec /app/node_modules/.bin/claude-flow \"\$@\"
    " -- "$@"
else
    # Non-root user: run normally
    exec /app/node_modules/.bin/claude-flow "$@"
fi
