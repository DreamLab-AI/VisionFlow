#!/bin/bash
# Session Cleanup Script - Removes stale sessions and DB locks
# Run periodically via cron or supervisord

set -e

SESSIONS_ROOT="/workspace/.swarm/sessions"
SESSIONS_INDEX="${SESSIONS_ROOT}/index.json"
SESSIONS_LOCK="${SESSIONS_ROOT}/.lock"
STALE_THRESHOLD_MINUTES=${STALE_THRESHOLD_MINUTES:-30}
PRUNE_AGE_HOURS=${PRUNE_AGE_HOURS:-24}

log() {
    echo "[SESSION-CLEANUP $(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Check if session is active (has recent file modifications)
is_session_active() {
    local SESSION_DIR="$1"
    local THRESHOLD_SECONDS=$((STALE_THRESHOLD_MINUTES * 60))

    # Check for recent file modifications in session dir
    local RECENT_FILES=$(find "$SESSION_DIR" -type f -mmin -"$STALE_THRESHOLD_MINUTES" 2>/dev/null | wc -l)

    if [ "$RECENT_FILES" -gt 0 ]; then
        return 0  # Active
    fi

    # Check if any process is using the session's database
    if [ -f "${SESSION_DIR}/.hive-mind/hive.db" ]; then
        if fuser "${SESSION_DIR}/.hive-mind/hive.db" 2>/dev/null | grep -q '[0-9]'; then
            return 0  # DB is in use, session is active
        fi
    fi

    return 1  # Inactive
}

# Clean stale WAL/SHM files
cleanup_wal_files() {
    log "Cleaning stale WAL/SHM files..."

    local CLEANED=0
    while IFS= read -r -d '' DB_FILE; do
        local WAL_FILE="${DB_FILE}-wal"
        local SHM_FILE="${DB_FILE}-shm"

        # Check if DB is not in use
        if ! fuser "$DB_FILE" >/dev/null 2>&1; then
            # Remove WAL/SHM if they exist and DB is not locked
            if [ -f "$WAL_FILE" ]; then
                rm -f "$WAL_FILE" && log "  Removed: $WAL_FILE" && ((CLEANED++))
            fi
            if [ -f "$SHM_FILE" ]; then
                rm -f "$SHM_FILE" && log "  Removed: $SHM_FILE" && ((CLEANED++))
            fi
        fi
    done < <(find "$SESSIONS_ROOT" -name "*.db" -type f -print0 2>/dev/null)

    log "Cleaned $CLEANED WAL/SHM files"
}

# Mark abandoned sessions as completed
mark_abandoned_sessions() {
    log "Checking for abandoned sessions..."

    if [ ! -f "$SESSIONS_INDEX" ]; then
        log "No sessions index found"
        return
    fi

    local MARKED=0

    # Get all running sessions
    local SESSIONS=$(jq -r '.sessions | to_entries[] | select(.value.status == "running") | .key' "$SESSIONS_INDEX" 2>/dev/null)

    for SESSION_ID in $SESSIONS; do
        local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_ID}"

        if [ ! -d "$SESSION_DIR" ]; then
            log "  Session dir missing for $SESSION_ID, marking as failed"
            (
                flock -x 200
                local TEMP_INDEX=$(mktemp)
                jq --arg uuid "$SESSION_ID" \
                   '.sessions[$uuid].status = "failed" | .sessions[$uuid].ended = "'$(date -Iseconds)'"' \
                   "$SESSIONS_INDEX" > "$TEMP_INDEX"
                mv "$TEMP_INDEX" "$SESSIONS_INDEX"
            ) 200>"$SESSIONS_LOCK"
            ((MARKED++))
            continue
        fi

        # Check if session is still active
        if ! is_session_active "$SESSION_DIR"; then
            log "  Marking abandoned session: $SESSION_ID"

            # Update status in index
            (
                flock -x 200
                local TEMP_INDEX=$(mktemp)
                jq --arg uuid "$SESSION_ID" \
                   '.sessions[$uuid].status = "completed" | .sessions[$uuid].ended = "'$(date -Iseconds)'"' \
                   "$SESSIONS_INDEX" > "$TEMP_INDEX"
                mv "$TEMP_INDEX" "$SESSIONS_INDEX"
            ) 200>"$SESSIONS_LOCK"

            # Update session.json
            if [ -f "${SESSION_DIR}/session.json" ]; then
                jq '.status = "completed" | .ended = "'$(date -Iseconds)'"' \
                   "${SESSION_DIR}/session.json" > "${SESSION_DIR}/session.json.tmp"
                mv "${SESSION_DIR}/session.json.tmp" "${SESSION_DIR}/session.json"
            fi

            ((MARKED++))
        fi
    done

    log "Marked $MARKED abandoned sessions as completed"
}

# Prune old sessions
prune_old_sessions() {
    log "Pruning sessions older than ${PRUNE_AGE_HOURS}h..."

    local PRUNED=0
    local CUTOFF_DATE=$(date -d "${PRUNE_AGE_HOURS} hours ago" -Iseconds 2>/dev/null || date -v-${PRUNE_AGE_HOURS}H -Iseconds)

    if [ ! -f "$SESSIONS_INDEX" ]; then
        return
    fi

    # Get sessions older than cutoff
    local OLD_SESSIONS=$(jq -r --arg cutoff "$CUTOFF_DATE" \
        '.sessions | to_entries[] | select(.value.created < $cutoff and (.value.status == "completed" or .value.status == "failed")) | .key' \
        "$SESSIONS_INDEX" 2>/dev/null)

    for SESSION_ID in $OLD_SESSIONS; do
        local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_ID}"

        if [ -d "$SESSION_DIR" ]; then
            log "  Pruning old session: $SESSION_ID"

            # Archive session data before deletion (optional)
            if [ -n "$ARCHIVE_DIR" ] && [ -d "$ARCHIVE_DIR" ]; then
                tar -czf "${ARCHIVE_DIR}/session-${SESSION_ID}.tar.gz" -C "$SESSIONS_ROOT" "$SESSION_ID" 2>/dev/null || true
            fi

            # Remove session directory
            rm -rf "$SESSION_DIR"

            # Remove from index
            (
                flock -x 200
                local TEMP_INDEX=$(mktemp)
                jq --arg uuid "$SESSION_ID" 'del(.sessions[$uuid])' "$SESSIONS_INDEX" > "$TEMP_INDEX"
                mv "$TEMP_INDEX" "$SESSIONS_INDEX"
            ) 200>"$SESSIONS_LOCK"

            ((PRUNED++))
        fi
    done

    log "Pruned $PRUNED old sessions"
}

# Vacuum databases to reclaim space
vacuum_databases() {
    log "Vacuuming databases..."

    local VACUUMED=0

    while IFS= read -r -d '' DB_FILE; do
        # Skip if DB is in use
        if ! fuser "$DB_FILE" >/dev/null 2>&1; then
            if sqlite3 "$DB_FILE" "VACUUM;" 2>/dev/null; then
                log "  Vacuumed: $DB_FILE"
                ((VACUUMED++))
            fi
        fi
    done < <(find "$SESSIONS_ROOT" -name "*.db" -type f -print0 2>/dev/null)

    log "Vacuumed $VACUUMED databases"
}

# Main cleanup routine
main() {
    log "Starting session cleanup..."
    log "Config: stale_threshold=${STALE_THRESHOLD_MINUTES}min, prune_age=${PRUNE_AGE_HOURS}h"

    if [ ! -d "$SESSIONS_ROOT" ]; then
        log "Sessions directory not found: $SESSIONS_ROOT"
        exit 0
    fi

    # Perform cleanup tasks
    cleanup_wal_files
    mark_abandoned_sessions
    prune_old_sessions

    # Optional: vacuum databases (can be resource-intensive)
    if [ "$VACUUM_DBS" = "true" ]; then
        vacuum_databases
    fi

    # Print summary
    log "Cleanup complete"

    if [ -f "$SESSIONS_INDEX" ]; then
        local ACTIVE=$(jq -r '.sessions | to_entries[] | select(.value.status == "running") | .key' "$SESSIONS_INDEX" 2>/dev/null | wc -l)
        local TOTAL=$(jq -r '.sessions | length' "$SESSIONS_INDEX" 2>/dev/null)
        log "Status: $ACTIVE active sessions, $TOTAL total sessions"
    fi
}

# Run cleanup
main "$@"
