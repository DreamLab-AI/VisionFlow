#!/bin/bash
# Supervisor Recovery Script - Detects and recovers from supervisor failures
# Run as a cron job or systemd timer

set -e

SUPERVISOR_SOCK="/workspace/.supervisor/supervisor.sock"
SUPERVISOR_PID="/workspace/.supervisor/supervisord.pid"
LOG_FILE="/var/log/multi-agent/supervisor-recovery.log"

log() {
    echo "[SUPERVISOR-RECOVERY $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_supervisor() {
    # Check if supervisord process is running
    if [ -f "$SUPERVISOR_PID" ]; then
        local PID=$(cat "$SUPERVISOR_PID")
        if kill -0 "$PID" 2>/dev/null; then
            # Process exists, check if socket is responding
            if supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status >/dev/null 2>&1; then
                return 0  # Supervisor is healthy
            else
                log "WARNING: Supervisor process exists but socket not responding"
                return 1
            fi
        else
            log "WARNING: Supervisor PID file exists but process is dead"
            return 1
        fi
    else
        log "WARNING: Supervisor PID file missing"
        return 1
    fi
}

restart_supervisor() {
    log "Attempting to restart supervisord..."

    # Clean up stale files
    rm -f "$SUPERVISOR_PID" "$SUPERVISOR_SOCK"

    # Start supervisord
    /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf

    # Wait for startup
    sleep 3

    if check_supervisor; then
        log "✓ Supervisord successfully restarted"
        return 0
    else
        log "✗ Failed to restart supervisord"
        return 1
    fi
}

check_critical_services() {
    # List of critical services that should always be running
    local CRITICAL_SERVICES=(
        "mcp-ws-bridge"
        "mcp-tcp-server"
        "vnc"
        "xfce"
    )

    local FAILED_SERVICES=()

    for service in "${CRITICAL_SERVICES[@]}"; do
        if ! supervisorctl -c /etc/supervisor/conf.d/supervisord.conf status "$service" 2>/dev/null | grep -q "RUNNING"; then
            FAILED_SERVICES+=("$service")
        fi
    done

    if [ ${#FAILED_SERVICES[@]} -gt 0 ]; then
        log "WARNING: Critical services not running: ${FAILED_SERVICES[*]}"

        # Attempt to restart failed services
        for service in "${FAILED_SERVICES[@]}"; do
            log "Restarting service: $service"
            supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart "$service" || true
        done
    fi
}

# Main recovery routine
main() {
    mkdir -p "$(dirname "$LOG_FILE")"

    log "Starting supervisor health check..."

    if ! check_supervisor; then
        log "Supervisor is not healthy, attempting recovery..."

        # Try restart up to 3 times with exponential backoff
        for attempt in 1 2 3; do
            log "Recovery attempt $attempt/3"

            if restart_supervisor; then
                log "Recovery successful on attempt $attempt"
                break
            fi

            if [ $attempt -lt 3 ]; then
                local wait_time=$((2 ** attempt))
                log "Waiting ${wait_time}s before next attempt..."
                sleep $wait_time
            else
                log "✗ All recovery attempts failed"
                exit 1
            fi
        done
    else
        log "Supervisor is healthy"
    fi

    # Check critical services
    check_critical_services

    log "Health check complete"
}

main "$@"
