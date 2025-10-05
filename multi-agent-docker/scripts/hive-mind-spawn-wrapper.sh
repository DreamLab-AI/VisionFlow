#!/bin/bash
# Hive-Mind Spawn Wrapper with Database Isolation
# Creates unique working directory for each spawn to prevent SQLite lock conflicts

set -e

# Generate unique session ID
SESSION_ID="hive-$(date +%s)-$(openssl rand -hex 4)"
SESSION_DIR="/workspace/.swarm/sessions/${SESSION_ID}"
OUTPUT_DIR="/workspace/ext/${SESSION_ID}"

# Create isolated directories
mkdir -p "${SESSION_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Ensure dev user owns directories
chown -R dev:dev "${SESSION_DIR}" "${OUTPUT_DIR}" 2>/dev/null || true

echo "[HIVE-WRAPPER] Session: ${SESSION_ID}"
echo "[HIVE-WRAPPER] Work dir: ${SESSION_DIR}"
echo "[HIVE-WRAPPER] Output dir: ${OUTPUT_DIR}"

# Create symlink from session dir to output dir for easy access
ln -sf "${OUTPUT_DIR}" "${SESSION_DIR}/output"

# Log file for this session
LOG_FILE="/var/log/multi-agent/hive-mind-${SESSION_ID}.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Execute hive-mind spawn in isolated directory
cd "${SESSION_DIR}"

echo "[HIVE-WRAPPER] Spawning: $@" | tee -a "$LOG_FILE"

# Run as dev user in isolated directory
exec su -s /bin/bash dev -c "
  cd '${SESSION_DIR}'
  export HIVE_MIND_SESSION_ID='${SESSION_ID}'
  export HIVE_MIND_OUTPUT_DIR='${OUTPUT_DIR}'
  /app/node_modules/.bin/claude-flow hive-mind spawn \"\$@\" --claude 2>&1 | tee -a '${LOG_FILE}'
" -- "$@"
