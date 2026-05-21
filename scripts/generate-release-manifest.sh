#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"

repo_json() {
  local name="$1"
  local path="$2"
  local abs="$WORKSPACE/$path"
  if [[ ! -d "$abs/.git" ]]; then
    printf '    {"name":"%s","path":"../%s","head":"%040d","branch":"missing","dirty":true}' "$name" "$path" 0
    return
  fi

  local head branch dirty
  head="$(git -C "$abs" rev-parse HEAD)"
  branch="$(git -C "$abs" branch --show-current)"
  if [[ -n "$(git -C "$abs" status --short)" ]]; then
    dirty=true
  else
    dirty=false
  fi
  printf '    {"name":"%s","path":"../%s","head":"%s","branch":"%s","dirty":%s}' "$name" "$path" "$head" "${branch:-detached}" "$dirty"
}

cat <<JSON
{
  "manifest_version": 1,
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "local-draft",
  "repositories": [
$(repo_json "VisionFlow" "VisionFlow"),
$(repo_json "VisionClaw" "project"),
$(repo_json "agentbox" "agentbox"),
$(repo_json "solid-pod-rs" "solid-pod-rs"),
$(repo_json "nostr-rust-forum" "nostr-rust-forum"),
$(repo_json "dreamlab-ai-website" "dreamlab-ai-website")
  ],
  "compatibility": {
    "identity": "did:nostr with 64-char lowercase x-only secp256k1 pubkeys; NIP-98 HTTP auth expected across mesh participants",
    "mesh": "Designed around NIP-42 relay writes and IS-Envelope routing; current default deployments may remain standalone",
    "pod": "solid-pod-rs is canonical for native pods; Cloudflare Workers pod tier has documented feature differences",
    "governance": "Agent Control Surface event kinds 31400-31405 are the human decision plane",
    "verification": "Run npm run verify in VisionFlow plus substrate-specific CI before promoting candidate to released"
  }
}
JSON

