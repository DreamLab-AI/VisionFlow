#!/usr/bin/env bash
# ADR census across the VisionFlow estate. Prints one TSV row per ADR that is
# not fully closed (decision accepted/superseded/rejected AND implementation
# complete AND activation live), plus a per-repo summary on stderr.
# Usage: adr-census.sh [--summary]
set -u
W=${WORKSPACE:-$HOME/workspace}
declare -A DIRS=(
  [VisionFlow]=VisionFlow/docs/adr
  [VisionClaw]=project/docs/adr
  [agentbox]=agentbox/docs/adr
  [nostr-rust-forum]=nostr-rust-forum/docs/adr
  [solid-pod-rs]=solid-pod-rs/crates/solid-pod-rs/docs/adr
  [dreamlab-ai-website]=dreamlab-ai-website/docs/adr
  [visionGraph]=visionGraph/docs/adr
  [sidestr-rs]=sidestr-rs/docs/adr
  [codescape]=codescape/docs/adr
  [visiontouch-rs]=visiontouch-rs/docs/adr
  [ontomark]=ontomark/docs/adr
)
fm() { awk -v k="$2" 'NR==1&&$0!="---"{exit} NR>1&&$0=="---"{exit} $0 ~ "^"k":"{sub("^"k": *","");sub(/ *#.*/,"");gsub(/["'\'']/,"");print;exit}' "$1"; }
for repo in "${!DIRS[@]}"; do
  d=$W/${DIRS[$repo]}; [ -d "$d" ] || continue
  tot=0; open=0; prop=0
  for f in "$d"/ADR-*.md; do
    [ -f "$f" ] || continue; tot=$((tot+1))
    ds=$(fm "$f" decision_status); is=$(fm "$f" implementation_status); as=$(fm "$f" activation_status)
    if [ -z "$ds" ]; then
      # Bullet-style header (ontomark: "- Status: accepted"): one status axis only
      ds=$(grep -m1 -iE '^(- )?\**status\**:' "$f" | sed -E 's/^(- )?\**[Ss]tatus\**: *//; s/[ ,.;(].*//' | tr 'A-Z' 'a-z')
      [ "$ds" = accepted ] && continue
    fi
    case "$ds" in superseded*|rejected*|deprecated*) continue;; esac
    if [ "$ds" = accepted ] && [ "$is" = complete ] && [ "$as" = live ]; then continue; fi
    open=$((open+1)); [ "$ds" = proposed ] && prop=$((prop+1))
    [ "${1:-}" = --summary ] || printf '%s\t%s\t%s\t%s\t%s\n' "$repo" "$(basename "$f" .md | cut -c1-70)" "${ds:-?}" "${is:-?}" "${as:-?}"
  done
  printf '%-20s total=%-4s open=%-4s proposed=%s\n' "$repo" "$tot" "$open" "$prop" >&2
done
