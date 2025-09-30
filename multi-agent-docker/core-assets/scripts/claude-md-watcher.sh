#!/bin/bash
# CLAUDE.md Resilient Watcher
# Monitors CLAUDE.md and re-applies system tools manifest if overwritten

CLAUDE_MD="/workspace/CLAUDE.md"
MARKER="<!-- SYSTEM_TOOLS_MANIFEST -->"
CHECK_INTERVAL=30

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [CLAUDE.md Watcher] $1"
}

check_and_repair() {
    if [ ! -f "$CLAUDE_MD" ]; then
        log "CLAUDE.md not found, waiting..."
        return 1
    fi

    if ! grep -q "$MARKER" "$CLAUDE_MD"; then
        log "⚠️  System tools manifest missing, re-applying..."
        /app/core-assets/scripts/claude-md-patcher.sh
        if [ $? -eq 0 ]; then
            log "✅ System tools manifest restored"
        else
            log "❌ Failed to restore manifest"
        fi
    fi
}

# Initial check
log "Starting CLAUDE.md watcher (interval: ${CHECK_INTERVAL}s)"
sleep 5  # Wait for initial setup
check_and_repair

# Continuous monitoring
while true; do
    sleep $CHECK_INTERVAL
    check_and_repair
done