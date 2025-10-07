#!/bin/bash
# Database Health Check - Monitor for DB locks and contention

set -e

DB_HEALTH_OK=0
DB_HEALTH_WARN=1
DB_HEALTH_FAIL=2

check_db_locks() {
    local WARNINGS=0
    local ERRORS=0

    # Check for stale WAL files (indicates abandoned connections)
    local STALE_WALS=$(find /workspace/.swarm -name "*.db-wal" -mmin +30 2>/dev/null | wc -l)
    if [ "$STALE_WALS" -gt 5 ]; then
        echo "WARNING: $STALE_WALS stale WAL files found" >&2
        ((WARNINGS++))
    fi

    # Check session index for excessive running sessions
    if [ -f "/workspace/.swarm/sessions/index.json" ]; then
        local RUNNING_SESSIONS=$(jq -r '.sessions | to_entries[] | select(.value.status == "running") | .key' /workspace/.swarm/sessions/index.json 2>/dev/null | wc -l)
        if [ "$RUNNING_SESSIONS" -gt 20 ]; then
            echo "ERROR: $RUNNING_SESSIONS running sessions (limit: 20)" >&2
            ((ERRORS++))
        elif [ "$RUNNING_SESSIONS" -gt 10 ]; then
            echo "WARNING: $RUNNING_SESSIONS running sessions" >&2
            ((WARNINGS++))
        fi
    fi

    # Check for database lock contention
    if [ -f "/workspace/.swarm/memory.db" ]; then
        # Try to connect with short timeout
        if ! timeout 2 sqlite3 /workspace/.swarm/memory.db "PRAGMA quick_check;" >/dev/null 2>&1; then
            echo "ERROR: Main memory.db is locked or corrupted" >&2
            ((ERRORS++))
        fi
    fi

    # Return appropriate status
    if [ "$ERRORS" -gt 0 ]; then
        return $DB_HEALTH_FAIL
    elif [ "$WARNINGS" -gt 0 ]; then
        return $DB_HEALTH_WARN
    fi

    return $DB_HEALTH_OK
}

check_mcp_processes() {
    # Check if MCP TCP server is running
    if ! pgrep -f "mcp-tcp-server.js" >/dev/null; then
        echo "ERROR: MCP TCP server not running" >&2
        return $DB_HEALTH_FAIL
    fi

    # Check if claude-flow MCP is responsive
    if ! timeout 3 nc -z localhost 9500 2>/dev/null; then
        echo "WARNING: MCP TCP port 9500 not responding" >&2
        return $DB_HEALTH_WARN
    fi

    return $DB_HEALTH_OK
}

# Main health check
main() {
    local EXIT_CODE=$DB_HEALTH_OK

    # Check database health
    if ! check_db_locks; then
        local DB_STATUS=$?
        [ "$DB_STATUS" -gt "$EXIT_CODE" ] && EXIT_CODE=$DB_STATUS
    fi

    # Check MCP processes
    if ! check_mcp_processes; then
        local MCP_STATUS=$?
        [ "$MCP_STATUS" -gt "$EXIT_CODE" ] && EXIT_CODE=$MCP_STATUS
    fi

    # Output status
    case $EXIT_CODE in
        $DB_HEALTH_OK)
            echo "Database health: OK"
            ;;
        $DB_HEALTH_WARN)
            echo "Database health: WARNING"
            ;;
        $DB_HEALTH_FAIL)
            echo "Database health: FAILED"
            ;;
    esac

    return $EXIT_CODE
}

main "$@"
