#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "MISSING $path"
    return 1
  fi
  echo "OK      $path"
}

echo "VisionFlow mesh smoke-test preflight"
echo "Workspace: $WORKSPACE"
echo

require_file "$WORKSPACE/project/docs/adr/ADR-073-private-nostr-relay-mesh-topology.md"
require_file "$WORKSPACE/project/docs/adr/ADR-075-is-envelope-message-contract.md"
require_file "$WORKSPACE/agentbox/docs/developer/sovereign-mesh.md"
require_file "$WORKSPACE/nostr-rust-forum/docs/architecture.md"
require_file "$WORKSPACE/solid-pod-rs/README.md"
require_file "$WORKSPACE/dreamlab-ai-website/forum-config/dreamlab.toml"

echo
echo "Configuration signals"
if rg -n 'mode *= *"standalone"|peer_relays *= *\\[\\]|allowed_remote_dids *= *\\[\\]' "$WORKSPACE/dreamlab-ai-website/forum-config/dreamlab.toml"; then
  echo "NOTE    DreamLab website mesh config appears standalone/empty by default."
fi

if rg -n '7777|nostr-rs-relay|sovereign_mesh|federation' "$WORKSPACE/agentbox/docker-compose.yml" "$WORKSPACE/agentbox/agentbox.toml" 2>/dev/null; then
  echo "NOTE    agentbox relay/federation settings found; verify external exposure before full smoke."
else
  echo "WARN    agentbox relay/federation settings not found in expected files."
fi

echo
echo "Preflight complete. This does not start services or send Nostr events."

