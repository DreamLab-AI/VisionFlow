#!/bin/bash
# Hive-Mind Session Manager
# Manages isolated sessions with UUID tracking for external coordination

set -e

SESSIONS_ROOT="/workspace/.swarm/sessions"
SESSIONS_INDEX="/workspace/.swarm/sessions/index.json"
SESSIONS_LOCK="/workspace/.swarm/sessions/.lock"

# Ensure sessions infrastructure exists
init_sessions() {
    mkdir -p "${SESSIONS_ROOT}"
    if [ ! -f "${SESSIONS_INDEX}" ]; then
        echo '{"sessions": {}, "created": "'$(date -Iseconds)'"}' > "${SESSIONS_INDEX}"
    fi
    chown -R dev:dev "${SESSIONS_ROOT}" 2>/dev/null || true
}

# Generate UUID v4
generate_uuid() {
    # Use /proc/sys/kernel/random/uuid if available, otherwise fallback
    if [ -f /proc/sys/kernel/random/uuid ]; then
        cat /proc/sys/kernel/random/uuid
    else
        # Fallback: Generate UUID-like string using openssl
        openssl rand -hex 16 | sed 's/\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)/\1\2\3\4-\5\6-\7\8-\9\10-\11\12\13\14\15\16/'
    fi
}

# Create new session
create_session() {
    local TASK_DESC="$1"
    local PRIORITY="${2:-medium}"
    local METADATA="$3"

    init_sessions

    # Generate session UUID
    local SESSION_UUID=$(generate_uuid)
    local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_UUID}"
    local OUTPUT_DIR="/workspace/ext/hive-sessions/${SESSION_UUID}"

    # Create directories
    mkdir -p "${SESSION_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    # Session metadata file
    cat > "${SESSION_DIR}/session.json" <<EOF
{
  "session_id": "${SESSION_UUID}",
  "task": "${TASK_DESC}",
  "priority": "${PRIORITY}",
  "created": "$(date -Iseconds)",
  "status": "created",
  "working_dir": "${SESSION_DIR}",
  "output_dir": "${OUTPUT_DIR}",
  "database": "${SESSION_DIR}/.swarm/memory.db",
  "log_file": "/var/log/multi-agent/hive-${SESSION_UUID}.log",
  "metadata": ${METADATA:-null}
}
EOF

    # Create symlink to output
    ln -sf "${OUTPUT_DIR}" "${SESSION_DIR}/output"

    # Update index with file locking
    (
        flock -x 200
        local TEMP_INDEX=$(mktemp)
        jq --arg uuid "$SESSION_UUID" \
           --arg task "$TASK_DESC" \
           --arg created "$(date -Iseconds)" \
           '.sessions[$uuid] = {
               "task": $task,
               "status": "created",
               "created": $created,
               "dir": "'${SESSION_DIR}'"
           }' "${SESSIONS_INDEX}" > "${TEMP_INDEX}"
        mv "${TEMP_INDEX}" "${SESSIONS_INDEX}"
    ) 200>"${SESSIONS_LOCK}"

    chown -R dev:dev "${SESSION_DIR}" "${OUTPUT_DIR}" 2>/dev/null || true

    # Return session UUID and metadata
    echo "${SESSION_UUID}"
    cat "${SESSION_DIR}/session.json" >&2
}

# Start hive-mind in session
start_session() {
    local SESSION_UUID="$1"
    shift  # Remaining args are hive-mind args

    local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_UUID}"

    if [ ! -d "${SESSION_DIR}" ]; then
        echo "ERROR: Session ${SESSION_UUID} not found" >&2
        return 1
    fi

    # Update session status
    update_session_status "${SESSION_UUID}" "starting"

    # Read session metadata
    local SESSION_JSON="${SESSION_DIR}/session.json"
    local LOG_FILE=$(jq -r '.log_file' "${SESSION_JSON}")
    local OUTPUT_DIR=$(jq -r '.output_dir' "${SESSION_JSON}")
    local TASK=$(jq -r '.task' "${SESSION_JSON}")

    mkdir -p "$(dirname "$LOG_FILE")"
    chown -R dev:dev "$(dirname "$LOG_FILE")" 2>/dev/null || true

    echo "[SESSION-MGR] Starting session: ${SESSION_UUID}" | tee -a "$LOG_FILE"
    echo "[SESSION-MGR] Working directory: ${SESSION_DIR}" | tee -a "$LOG_FILE"
    echo "[SESSION-MGR] Output directory: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
    echo "[SESSION-MGR] Task: ${TASK}" | tee -a "$LOG_FILE"
    echo "[SESSION-MGR] Command: claude-flow hive-mind spawn \"${TASK}\" --claude" | tee -a "$LOG_FILE"

    # Update status to running
    update_session_status "${SESSION_UUID}" "running"

    # Execute hive-mind spawn as dev user in isolated directory
    cd "${SESSION_DIR}"

    su -s /bin/bash dev -c "
        set -e
        cd '${SESSION_DIR}'
        export HIVE_MIND_SESSION_ID='${SESSION_UUID}'
        export HIVE_MIND_OUTPUT_DIR='${OUTPUT_DIR}'
        /app/node_modules/.bin/claude-flow hive-mind spawn '${TASK}' --claude 2>&1 | tee -a '${LOG_FILE}'
        EXIT_CODE=\${PIPESTATUS[0]}
        exit \${EXIT_CODE}
    " -- "$@"

    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        update_session_status "${SESSION_UUID}" "completed"
    else
        update_session_status "${SESSION_UUID}" "failed"
    fi

    return $EXIT_CODE
}

# Update session status
update_session_status() {
    local SESSION_UUID="$1"
    local STATUS="$2"
    local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_UUID}"

    if [ ! -d "${SESSION_DIR}" ]; then
        echo "ERROR: Session ${SESSION_UUID} not found" >&2
        return 1
    fi

    # Update session.json
    local TEMP_FILE=$(mktemp)
    jq --arg status "$STATUS" \
       --arg updated "$(date -Iseconds)" \
       '.status = $status | .updated = $updated' \
       "${SESSION_DIR}/session.json" > "${TEMP_FILE}"
    mv "${TEMP_FILE}" "${SESSION_DIR}/session.json"

    # Update index
    (
        flock -x 200
        local TEMP_INDEX=$(mktemp)
        jq --arg uuid "$SESSION_UUID" \
           --arg status "$STATUS" \
           --arg updated "$(date -Iseconds)" \
           '.sessions[$uuid].status = $status |
            .sessions[$uuid].updated = $updated' \
           "${SESSIONS_INDEX}" > "${TEMP_INDEX}"
        mv "${TEMP_INDEX}" "${SESSIONS_INDEX}"
    ) 200>"${SESSIONS_LOCK}"
}

# Get session info
get_session() {
    local SESSION_UUID="$1"
    local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_UUID}"

    if [ ! -d "${SESSION_DIR}" ]; then
        echo '{"error": "Session not found"}'
        return 1
    fi

    cat "${SESSION_DIR}/session.json"
}

# List all sessions
list_sessions() {
    init_sessions

    if [ ! -f "${SESSIONS_INDEX}" ]; then
        echo '{"sessions": {}}'
        return
    fi

    cat "${SESSIONS_INDEX}"
}

# Get session status
get_status() {
    local SESSION_UUID="$1"
    local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_UUID}"

    if [ ! -d "${SESSION_DIR}" ]; then
        echo "not_found"
        return 1
    fi

    jq -r '.status' "${SESSION_DIR}/session.json"
}

# Get session output directory
get_output_dir() {
    local SESSION_UUID="$1"
    local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_UUID}"

    if [ ! -d "${SESSION_DIR}" ]; then
        echo ""
        return 1
    fi

    jq -r '.output_dir' "${SESSION_DIR}/session.json"
}

# Get session log file
get_log_file() {
    local SESSION_UUID="$1"
    local SESSION_DIR="${SESSIONS_ROOT}/${SESSION_UUID}"

    if [ ! -d "${SESSION_DIR}" ]; then
        echo ""
        return 1
    fi

    jq -r '.log_file' "${SESSION_DIR}/session.json"
}

# Cleanup old sessions
cleanup_sessions() {
    local MAX_AGE_HOURS="${1:-24}"
    local CLEANED=0

    init_sessions

    # Find sessions older than MAX_AGE_HOURS
    find "${SESSIONS_ROOT}" -maxdepth 1 -type d -name "*-*-*-*-*" -mtime "+$((MAX_AGE_HOURS / 24))" | while read SESSION_DIR; do
        local SESSION_UUID=$(basename "$SESSION_DIR")
        local STATUS=$(jq -r '.status // "unknown"' "${SESSION_DIR}/session.json" 2>/dev/null || echo "unknown")

        # Only cleanup completed or failed sessions
        if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
            echo "Cleaning up session: ${SESSION_UUID} (status: ${STATUS})"

            # Remove from index
            (
                flock -x 200
                local TEMP_INDEX=$(mktemp)
                jq --arg uuid "$SESSION_UUID" 'del(.sessions[$uuid])' \
                   "${SESSIONS_INDEX}" > "${TEMP_INDEX}"
                mv "${TEMP_INDEX}" "${SESSIONS_INDEX}"
            ) 200>"${SESSIONS_LOCK}"

            # Archive or remove session directory
            # (Keep output in ext/ for external access)
            rm -rf "${SESSION_DIR}"
            CLEANED=$((CLEANED + 1))
        fi
    done

    echo "Cleaned up ${CLEANED} old sessions"
}

# Main command dispatcher
case "$1" in
    create)
        create_session "$2" "$3" "$4"
        ;;
    start)
        shift
        SESSION_UUID="$1"
        shift
        start_session "${SESSION_UUID}" "$@"
        ;;
    status)
        get_status "$2"
        ;;
    get)
        get_session "$2"
        ;;
    list)
        list_sessions
        ;;
    output-dir)
        get_output_dir "$2"
        ;;
    log)
        get_log_file "$2"
        ;;
    update-status)
        update_session_status "$2" "$3"
        ;;
    cleanup)
        cleanup_sessions "$2"
        ;;
    init)
        init_sessions
        ;;
    *)
        cat <<EOF
Hive-Mind Session Manager

Usage:
  $0 create <task> [priority] [metadata_json]
      Create new session and return UUID

  $0 start <uuid> [hive-mind args...]
      Start hive-mind spawn in session

  $0 status <uuid>
      Get session status (created|starting|running|completed|failed)

  $0 get <uuid>
      Get full session metadata JSON

  $0 list
      List all sessions

  $0 output-dir <uuid>
      Get session output directory path

  $0 log <uuid>
      Get session log file path

  $0 update-status <uuid> <status>
      Update session status

  $0 cleanup [max_age_hours]
      Remove old completed/failed sessions (default: 24h)

  $0 init
      Initialize sessions infrastructure

Examples:
  # Create and start session in one command
  UUID=\$(docker exec multi-agent-container /app/scripts/hive-session-manager.sh create "build rust app" high)
  docker exec multi-agent-container /app/scripts/hive-session-manager.sh start \$UUID

  # Check status
  docker exec multi-agent-container /app/scripts/hive-session-manager.sh status \$UUID

  # Get output directory
  docker exec multi-agent-container /app/scripts/hive-session-manager.sh output-dir \$UUID

  # Read logs
  docker exec multi-agent-container cat \$(docker exec multi-agent-container /app/scripts/hive-session-manager.sh log \$UUID)

Session Structure:
  /workspace/.swarm/sessions/<UUID>/
    ├── session.json          # Session metadata
    ├── .swarm/memory.db      # Isolated database
    └── output/               # Symlink to /workspace/ext/hive-sessions/<UUID>

  /workspace/ext/hive-sessions/<UUID>/
    └── (all hive-mind output artifacts)

Database Isolation:
  Each session runs in its own working directory, creating its own
  .swarm/memory.db. This prevents SQLite lock conflicts between
  concurrent hive-mind spawns.

EOF
        exit 1
        ;;
esac
