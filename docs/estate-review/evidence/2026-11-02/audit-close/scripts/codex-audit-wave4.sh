#!/usr/bin/env bash
# Cross-vendor check of wave 4's four headline claims, with both ends of each
# contract excerpted at the pins. Usage: codex-audit-wave4.sh <out-file>
set -u
OUT=$1
A=/home/devuser/workspace/agentbox; AR=c57d9c0508f4
P=/home/devuser/workspace/project; PR=86cdd7635582
S=/home/devuser/workspace/sidestr-rs; SR=814b9e54cd16
show() { echo "--- $2:$3-$4 @ $1"; git -C "$5" show "$1:$2" | sed -n "$3,$4p" | nl -ba -v "$3"; }
{
  cat <<'EOF'
You are a cross-vendor auditor. For each of the four CLAIMS below, judge it only from the source excerpts given (read at pinned revisions). Answer HOLDS, WRONG (say exactly why), or CANNOT-JUDGE (say what is missing). Do not use tools.

CLAIM CONTRACTS-F1: agentbox's live agent-event forwarder opens its WebSocket with no credential, and a release VisionClaw build refuses an unauthenticated agent-events socket (401), because is_insecure_defaults_allowed() is false in release.
CLAIM CONTRACTS-F2: the forwarder sends the bare event, not the JSON-RPC notifications/agent_action envelope VisionClaw's ingest requires, so each frame is rejected as malformed.
CLAIM CONTRACTS-F3: agentbox's GET /v1/tasks response omits task, provider and status, which VisionClaw's TaskInfo requires, so deserialising list_tasks fails.
CLAIM SIDESTR-E2: sidestr-evm stores contract code under its declared code_hash without rehashing, and snapshot restore checks only version, config and the state root, so code bytes are never verified against their hash.

EOF
  echo "=== agentbox forwarder"; show $AR management-api/utils/agent-event-ws-subscriber.js 40 50 "$A"; show $AR management-api/utils/agent-event-ws-subscriber.js 90 100 "$A"; show $AR management-api/utils/agent-event-ws-subscriber.js 140 150 "$A"
  echo "=== agentbox publisher"; show $AR management-api/utils/agent-event-publisher.js 158 168 "$A"
  echo "=== VisionClaw ingest"; show $PR src/agent_events/ingest.rs 56 70 "$P"; show $PR src/agent_events/ingest.rs 95 105 "$P"; show $PR src/agent_events/ingest.rs 140 152 "$P"; show $PR src/agent_events/ingest.rs 258 306 "$P"
  echo "=== VisionClaw schema"; show $PR src/agent_events/schema.rs 1 40 "$P"
  echo "=== agentbox tasks route"; show $AR management-api/routes/tasks.js 160 200 "$A"
  echo "=== VisionClaw TaskInfo"; show $PR src/services/management_api_client.rs 70 100 "$P"
  echo "=== sidestr world"; show $SR sidestr-evm/src/world.rs 100 125 "$S"; show $SR sidestr-evm/src/world.rs 150 205 "$S"
  echo "=== sidestr restore"; show $SR sidestr-evm/src/state.rs 300 330 "$S"
} | timeout 900 codex exec --skip-git-repo-check -m gpt-6-astra --sandbox read-only - > "$OUT" 2>&1
echo "exit=$?" >> "$OUT"
