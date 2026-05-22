#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"

# ── Colours (disabled if not a terminal) ──────────────────────────────
if [[ -t 1 ]]; then
  GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'
else
  GREEN=''; RED=''; YELLOW=''; CYAN=''; NC=''
fi

# ── Counters ──────────────────────────────────────────────────────────
PASS=0; FAIL=0; WARN=0

ok()   { PASS=$((PASS+1)); printf "${GREEN}OK${NC}      %s\n" "$1"; }
miss() { FAIL=$((FAIL+1)); printf "${RED}MISSING${NC} %s\n" "$1"; }
warn() { WARN=$((WARN+1)); printf "${YELLOW}WARN${NC}    %s\n" "$1"; }
info() { printf "${CYAN}INFO${NC}    %s\n" "$1"; }

# ── Helpers ───────────────────────────────────────────────────────────
require_file() {
  local path="$1"
  local label="${2:-$path}"
  if [[ -f "$path" ]]; then
    ok "$label"
    return 0
  else
    miss "$label"
    return 1
  fi
}

require_dir() {
  local path="$1"
  local label="${2:-$path}"
  if [[ -d "$path" ]]; then
    ok "$label"
    return 0
  else
    miss "$label"
    return 1
  fi
}

grep_quiet() {
  # grep_quiet PATTERN FILE — returns 0 if pattern found, 1 otherwise
  grep -qE "$1" "$2" 2>/dev/null
}

echo "================================================================"
echo "VisionFlow mesh smoke-test preflight"
echo "Date:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Workspace: $WORKSPACE"
echo "================================================================"
echo

# ── Summary table accumulators ────────────────────────────────────────
declare -a SUMMARY_SUBSTRATE=()
declare -a SUMMARY_MOUNTED=()
declare -a SUMMARY_MESH_READY=()
declare -a SUMMARY_MODE=()

add_summary() {
  SUMMARY_SUBSTRATE+=("$1")
  SUMMARY_MOUNTED+=("$2")
  SUMMARY_MESH_READY+=("$3")
  SUMMARY_MODE+=("$4")
}

# ══════════════════════════════════════════════════════════════════════
# 1. VisionClaw (host project)
# ══════════════════════════════════════════════════════════════════════
echo "--- VisionClaw (host project) ---"
VC_MOUNTED="no"; VC_MESH="no"; VC_MODE="n/a"

if require_dir "$WORKSPACE/project" "project/ (VisionClaw repo)"; then
  VC_MOUNTED="yes"

  # IS-Envelope spec (ADR-075)
  if require_file "$WORKSPACE/project/docs/adr/ADR-075-is-envelope-message-contract.md" \
       "ADR-075 IS-Envelope message contract"; then
    VC_MESH="spec-owner"
  fi

  # Relay mesh topology ADR
  require_file "$WORKSPACE/project/docs/adr/ADR-073-private-nostr-relay-mesh-topology.md" \
    "ADR-073 relay mesh topology" || true

  # IRI parser for did:nostr (exclude target/)
  if find "$WORKSPACE/project/src" -not -path '*/target/*' -name '*.rs' -print0 2>/dev/null \
     | xargs -0 grep -lE 'did:nostr|iri.*parse' 2>/dev/null | head -1 | grep -q .; then
    ok "IRI parser for did:nostr found in src/"
  else
    warn "IRI parser for did:nostr not found in src/ (may be in a different path)"
  fi
fi

add_summary "VisionClaw" "$VC_MOUNTED" "$VC_MESH" "$VC_MODE"
echo

# ══════════════════════════════════════════════════════════════════════
# 2. nostr-rust-forum
# ══════════════════════════════════════════════════════════════════════
echo "--- nostr-rust-forum ---"
NRF_MOUNTED="no"; NRF_MESH="no"; NRF_MODE="unknown"

if require_dir "$WORKSPACE/nostr-rust-forum" "nostr-rust-forum/ repo"; then
  NRF_MOUNTED="yes"

  # governance.rs with kinds 31400-31405 (exclude target/ build artifacts)
  GOV_RS=$(find "$WORKSPACE/nostr-rust-forum" -not -path '*/target/*' -path '*/governance.rs' -print -quit 2>/dev/null || true)
  if [[ -n "$GOV_RS" ]]; then
    ok "governance.rs found at ${GOV_RS#$WORKSPACE/}"
    # Check for governance kind constants
    HAS_KINDS="yes"
    for kind in 31400 31401 31402 31403 31404 31405; do
      if ! grep_quiet "$kind" "$GOV_RS"; then
        HAS_KINDS="no"
        break
      fi
    done
    if [[ "$HAS_KINDS" == "yes" ]]; then
      ok "governance.rs contains all kinds 31400-31405"
      NRF_MESH="yes"
    else
      warn "governance.rs exists but does not contain all kinds 31400-31405"
    fi
  else
    miss "governance.rs (expected in crates/nostr-bbs-core/src/)"
  fi

  # Architecture docs
  require_file "$WORKSPACE/nostr-rust-forum/docs/architecture.md" \
    "nostr-rust-forum architecture docs" || true

  # NIP-42 relay gate (exclude target/)
  if find "$WORKSPACE/nostr-rust-forum" -not -path '*/target/*' -name '*.rs' -print0 2>/dev/null \
     | xargs -0 grep -lE 'NIP.?42|nip42|AUTH|relay.*gate' 2>/dev/null | head -1 | grep -q .; then
    ok "NIP-42 relay gate references found"
  else
    warn "NIP-42 relay gate references not found in source"
  fi

  # Default mode: federated if NIP-05 default (exclude target/)
  if find "$WORKSPACE/nostr-rust-forum" -not -path '*/target/*' \( -name '*.rs' -o -name '*.toml' \) -print0 2>/dev/null \
     | xargs -0 grep -lE 'nip.?05|federat' 2>/dev/null | head -1 | grep -q .; then
    NRF_MODE="federated"
    info "NIP-05/federation references found (default: federated)"
  else
    NRF_MODE="unknown"
    warn "Could not determine default mode"
  fi
fi

add_summary "nostr-rust-forum" "$NRF_MOUNTED" "$NRF_MESH" "$NRF_MODE"
echo

# ══════════════════════════════════════════════════════════════════════
# 3. dreamlab-ai-website
# ══════════════════════════════════════════════════════════════════════
echo "--- dreamlab-ai-website ---"
DW_MOUNTED="no"; DW_MESH="no"; DW_MODE="unknown"

if require_dir "$WORKSPACE/dreamlab-ai-website" "dreamlab-ai-website/ repo"; then
  DW_MOUNTED="yes"

  # Forum config
  FORUM_CFG="$WORKSPACE/dreamlab-ai-website/forum-config/dreamlab.toml"
  if require_file "$FORUM_CFG" "forum-config/dreamlab.toml"; then
    # Check for governance kinds
    if grep_quiet '3140[0-5]' "$FORUM_CFG"; then
      ok "Forum config references governance kinds"
      DW_MESH="yes"
    else
      warn "Forum config does not reference governance kinds 31400-31405"
    fi

    # Default mode
    if grep_quiet 'mode *= *"standalone"' "$FORUM_CFG" || \
       grep_quiet 'peer_relays *= *\[\]' "$FORUM_CFG"; then
      DW_MODE="standalone"
      warn "Forum config appears standalone (empty peer_relays or standalone mode)"
    else
      DW_MODE="federated"
      info "Forum config appears federated"
    fi
  fi

  # Governance dashboard
  if find "$WORKSPACE/dreamlab-ai-website" -not -path '*/node_modules/*' -not -path '*/.next/*' -type f \( -name '*.tsx' -o -name '*.ts' -o -name '*.jsx' -o -name '*.js' \) -print0 2>/dev/null \
     | xargs -0 grep -lE '/governance|governance.*dashboard' 2>/dev/null | head -1 | grep -q .; then
    ok "Governance dashboard route found"
  else
    warn "Governance dashboard route not found in source"
  fi

  # NIP-98 signed responses
  if find "$WORKSPACE/dreamlab-ai-website" -not -path '*/node_modules/*' -not -path '*/.next/*' -type f \( -name '*.ts' -o -name '*.js' \) -print0 2>/dev/null \
     | xargs -0 grep -lE 'nip.?98|NIP.?98|schnorr' 2>/dev/null | head -1 | grep -q .; then
    ok "NIP-98 references found"
  else
    warn "NIP-98 references not found in source"
  fi
fi

add_summary "dreamlab-ai-website" "$DW_MOUNTED" "$DW_MESH" "$DW_MODE"
echo

# ══════════════════════════════════════════════════════════════════════
# 4. agentbox
# ══════════════════════════════════════════════════════════════════════
echo "--- agentbox ---"
AB_MOUNTED="no"; AB_MESH="no"; AB_MODE="unknown"

if require_dir "$WORKSPACE/agentbox" "agentbox/ repo"; then
  AB_MOUNTED="yes"

  # agentbox.toml federation config
  AB_TOML="$WORKSPACE/agentbox/agentbox.toml"
  if require_file "$AB_TOML" "agentbox.toml"; then
    if grep_quiet 'federation' "$AB_TOML"; then
      ok "Federation config section found in agentbox.toml"
      if grep_quiet 'mode *= *"standalone"' "$AB_TOML"; then
        AB_MODE="standalone"
        info "agentbox federation.mode = standalone (default)"
      elif grep_quiet 'mode *= *"client"' "$AB_TOML"; then
        AB_MODE="federated"
        info "agentbox federation.mode = client"
      else
        AB_MODE="standalone"
        info "agentbox federation mode not explicitly set (assuming standalone)"
      fi
    else
      AB_MODE="standalone"
      warn "No federation config section in agentbox.toml"
    fi
  fi

  # Embedded relay at :7777
  if grep -rqE '7777|nostr.?rs.?relay' "$WORKSPACE/agentbox/agentbox.toml" \
       "$WORKSPACE/agentbox/config/" 2>/dev/null; then
    ok "Embedded relay (port 7777) references found"
  else
    warn "Embedded relay (port 7777) references not found"
  fi

  # relay-consumer.js (pod-bridge, kinds 31400-31405)
  RELAY_CONSUMER=$(find "$WORKSPACE/agentbox" -path '*/nostr-bridge/relay-consumer.js' -print -quit 2>/dev/null || true)
  if [[ -n "$RELAY_CONSUMER" ]]; then
    ok "relay-consumer.js found at ${RELAY_CONSUMER#$WORKSPACE/}"
    if grep_quiet '3140[0-5]' "$RELAY_CONSUMER"; then
      ok "relay-consumer.js references governance kinds"
      AB_MESH="yes"
    else
      warn "relay-consumer.js exists but does not reference governance kinds"
    fi
  else
    miss "mcp/nostr-bridge/relay-consumer.js"
  fi

  # Sovereign mesh docs
  require_file "$WORKSPACE/agentbox/docs/developer/sovereign-mesh.md" \
    "agentbox sovereign-mesh docs" || true
fi

add_summary "agentbox" "$AB_MOUNTED" "$AB_MESH" "$AB_MODE"
echo

# ══════════════════════════════════════════════════════════════════════
# 5. solid-pod-rs
# ══════════════════════════════════════════════════════════════════════
echo "--- solid-pod-rs ---"
SP_MOUNTED="no"; SP_MESH="no"; SP_MODE="unknown"

if require_dir "$WORKSPACE/solid-pod-rs" "solid-pod-rs/ repo"; then
  SP_MOUNTED="yes"

  require_file "$WORKSPACE/solid-pod-rs/README.md" "solid-pod-rs README" || true

  # NIP-98 verify module (exclude target/)
  NIP98_FILE=$(find "$WORKSPACE/solid-pod-rs" -not -path '*/target/*' \( -path '*/auth*nip98*' -o -path '*/nip98*' \) 2>/dev/null \
    | head -1 || true)
  if [[ -n "$NIP98_FILE" ]]; then
    ok "NIP-98 auth module found at ${NIP98_FILE#$WORKSPACE/}"
    SP_MESH="yes"
  else
    # Broader search
    if find "$WORKSPACE/solid-pod-rs" -not -path '*/target/*' -name '*.rs' -print0 2>/dev/null \
       | xargs -0 grep -lE 'verify_schnorr|nip.?98|NIP.?98' 2>/dev/null | head -1 | grep -q .; then
      ok "NIP-98 verify_schnorr references found in source"
      SP_MESH="yes"
    else
      miss "NIP-98 verify module (expected auth::nip98::verify_schnorr_signature)"
    fi
  fi

  # CORS support
  if find "$WORKSPACE/solid-pod-rs" -not -path '*/target/*' \( -name '*.rs' -o -name '*.toml' \) -print0 2>/dev/null \
     | xargs -0 grep -lE 'cors|CORS' 2>/dev/null | head -1 | grep -q .; then
    ok "CORS support references found"
  else
    warn "CORS support references not found"
  fi

  # Default mode
  SP_MODE="standalone"
  info "solid-pod-rs defaults to standalone with native mesh support"
fi

add_summary "solid-pod-rs" "$SP_MOUNTED" "$SP_MESH" "$SP_MODE"
echo

# ══════════════════════════════════════════════════════════════════════
# Summary Table
# ══════════════════════════════════════════════════════════════════════
echo "================================================================"
echo "SUMMARY TABLE"
echo "================================================================"
printf "%-22s %-10s %-14s %-12s\n" "substrate" "mounted" "mesh-ready" "default-mode"
printf "%-22s %-10s %-14s %-12s\n" "---------------------" "--------" "------------" "-----------"

for i in "${!SUMMARY_SUBSTRATE[@]}"; do
  printf "%-22s %-10s %-14s %-12s\n" \
    "${SUMMARY_SUBSTRATE[$i]}" \
    "${SUMMARY_MOUNTED[$i]}" \
    "${SUMMARY_MESH_READY[$i]}" \
    "${SUMMARY_MODE[$i]}"
done

echo
echo "================================================================"
echo "TOTALS: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${WARN} warnings${NC}"
echo "================================================================"
echo
echo "Preflight complete. This does not start services or send Nostr events."
echo "Paste this output into docs/protocol/mesh-smoke-test.md 'Preflight Results' section."
