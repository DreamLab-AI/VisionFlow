#!/bin/bash
# Container entrypoint - handles user setup and Claude directory

set -e

# Ensure directories exist
mkdir -p /home/devuser/logs /var/log

# Initialize DBus for Chromium
mkdir -p /run/dbus
if [ ! -f /run/dbus/pid ]; then
    echo "Starting DBus daemon..."
    dbus-daemon --system --fork 2>/dev/null || true
fi

# Fix Claude directory permissions if mounted
if [ -d "/home/devuser/.claude" ]; then
    echo "Setting up Claude directory permissions..."
    chown -R devuser:devuser /home/devuser/.claude 2>/dev/null || true
fi

# Fix project directory permissions if mounted
if [ -d "/home/devuser/workspace/project" ]; then
    echo "Setting up project directory permissions..."
    chown -R devuser:devuser /home/devuser/workspace/project 2>/dev/null || true
fi

# Start supervisord as root (required for VNC and service management)
echo "Starting supervisord..."
exec /opt/venv/bin/supervisord -n -c /etc/supervisord.conf
