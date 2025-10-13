#!/bin/bash
# Claude Code wrapper - ensures it runs as devuser, not root
# This prevents the security check from failing

if [ "$(id -u)" -eq 0 ]; then
    # Running as root, switch to devuser and preserve working directory
    cd "$(pwd)" 2>/dev/null || cd /home/devuser/workspace/project
    exec su devuser -c "cd '$(pwd)' && claude $*"
else
    # Already running as devuser
    exec claude "$@"
fi
