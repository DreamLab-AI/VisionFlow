#!/usr/bin/env bash
# harness-audit.sh — Pairing audit for harness template JSON files
#
# Reads every *.json template in docs/engineering/templates/, extracts
# guides, sensors, and pairings, then reports coverage and unpaired items.
#
# Exit codes:
#   0  Overall pairing ratio >= TARGET_RATIO (default 80%)
#   1  Below target, or no templates found
#
# Dependencies: jq (https://stedolan.github.io/jq/)
#
# Usage:
#   bash scripts/harness-audit.sh               # run from repo root
#   bash scripts/harness-audit.sh --target 60    # override target ratio
#   bash scripts/harness-audit.sh --dir /path    # override template dir
#   bash scripts/harness-audit.sh --help         # show this message

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_DIR="${TEMPLATE_DIR:-$REPO_ROOT/docs/engineering/templates}"
TARGET_RATIO="${TARGET_RATIO:-80}"

# ── argument parsing ────────────────────────────────────────────────────
usage() {
  sed -n '2,/^$/s/^# //p' "${BASH_SOURCE[0]}"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)   usage ;;
    --target|-t) TARGET_RATIO="$2"; shift 2 ;;
    --dir|-d)    TEMPLATE_DIR="$2"; shift 2 ;;
    *)           echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# ── preflight checks ───────────────────────────────────────────────────
if ! command -v jq &>/dev/null; then
  echo "ERROR: jq is required but not found. Install with: apt-get install jq" >&2
  exit 2
fi

if [[ ! -d "$TEMPLATE_DIR" ]]; then
  echo "ERROR: Template directory not found: $TEMPLATE_DIR" >&2
  exit 1
fi

shopt -s nullglob
TEMPLATES=("$TEMPLATE_DIR"/*.json)
shopt -u nullglob

if [[ ${#TEMPLATES[@]} -eq 0 ]]; then
  echo "ERROR: No JSON templates found in $TEMPLATE_DIR" >&2
  exit 1
fi

# ── per-template analysis ──────────────────────────────────────────────
TOTAL_GUIDES=0
TOTAL_SENSORS=0
TOTAL_PAIRED=0
TOTAL_CONTROLS=0
TOTAL_PLANNED=0
UNPAIRED_GUIDES=()
UNPAIRED_SENSORS=()
PLANNED_CONTROLS=()

# Table header
printf "\nHARNESS PAIRING AUDIT\n"
printf "=====================\n"
printf "%-30s | %6s | %7s | %6s | %6s | %s\n" \
  "Topology" "Guides" "Sensors" "Paired" "Ratio" "Maturity"

DIVIDER="$(printf '%0.s-' {1..80})"

for tpl in "${TEMPLATES[@]}"; do
  # Validate JSON first
  if ! jq empty "$tpl" 2>/dev/null; then
    echo "WARNING: Invalid JSON, skipping: $tpl" >&2
    continue
  fi

  # Extract topology name and maturity
  TOPOLOGY=$(jq -r '.topology // "unknown"' "$tpl")
  MATURITY=$(jq -r '.maturity // "unknown"' "$tpl")

  # Extract guide IDs
  mapfile -t GUIDE_IDS < <(jq -r '.guides[]?.id // empty' "$tpl")
  NUM_GUIDES=${#GUIDE_IDS[@]}

  # Extract sensor IDs
  mapfile -t SENSOR_IDS < <(jq -r '.sensors[]?.id // empty' "$tpl")
  NUM_SENSORS=${#SENSOR_IDS[@]}

  # Extract paired guide and sensor IDs from pairings
  mapfile -t PAIRED_GUIDE_IDS < <(jq -r '.pairings[]?.guide_id // empty' "$tpl")
  mapfile -t PAIRED_SENSOR_IDS < <(jq -r '.pairings[]?.sensor_id // empty' "$tpl")
  NUM_PAIRINGS=${#PAIRED_GUIDE_IDS[@]}

  # Calculate ratio: paired / max(guides, sensors), avoiding division by zero
  MAX_ITEMS=$(( NUM_GUIDES > NUM_SENSORS ? NUM_GUIDES : NUM_SENSORS ))
  if [[ $MAX_ITEMS -gt 0 ]]; then
    RATIO=$(awk "BEGIN { printf \"%.1f\", ($NUM_PAIRINGS / $MAX_ITEMS) * 100 }")
  else
    RATIO="0.0"
  fi

  # Print row
  printf "%-30s | %6d | %7d | %6d | %5s%% | %s\n" \
    "$TOPOLOGY" "$NUM_GUIDES" "$NUM_SENSORS" "$NUM_PAIRINGS" "$RATIO" "$MATURITY"

  # Accumulate totals
  TOTAL_GUIDES=$(( TOTAL_GUIDES + NUM_GUIDES ))
  TOTAL_SENSORS=$(( TOTAL_SENSORS + NUM_SENSORS ))
  TOTAL_PAIRED=$(( TOTAL_PAIRED + NUM_PAIRINGS ))
  TOTAL_CONTROLS=$(( TOTAL_CONTROLS + NUM_GUIDES + NUM_SENSORS ))

  # Source backing: count controls whose source_status is "planned"
  # (aspirational — the source does not yet resolve to a real artifact).
  # An absent source_status defaults to "present".
  while IFS=$'\t' read -r kind cid; do
    [[ -z "$cid" ]] && continue
    TOTAL_PLANNED=$(( TOTAL_PLANNED + 1 ))
    PLANNED_CONTROLS+=("[$TOPOLOGY] $kind $cid")
  done < <(jq -r '
    (.guides[]?  | select(.source_status == "planned") | "guide\t\(.id)"),
    (.sensors[]? | select(.source_status == "planned") | "sensor\t\(.id)")
  ' "$tpl")

  # Find unpaired guides: guide IDs not referenced in any pairing's guide_id
  for gid in "${GUIDE_IDS[@]}"; do
    FOUND=0
    for pgid in "${PAIRED_GUIDE_IDS[@]}"; do
      if [[ "$gid" == "$pgid" ]]; then
        FOUND=1
        break
      fi
    done
    if [[ $FOUND -eq 0 ]]; then
      UNPAIRED_GUIDES+=("[$TOPOLOGY] $gid")
    fi
  done

  # Find unpaired sensors: sensor IDs not referenced in any pairing's sensor_id
  for sid in "${SENSOR_IDS[@]}"; do
    FOUND=0
    for psid in "${PAIRED_SENSOR_IDS[@]}"; do
      if [[ "$sid" == "$psid" ]]; then
        FOUND=1
        break
      fi
    done
    if [[ $FOUND -eq 0 ]]; then
      UNPAIRED_SENSORS+=("[$TOPOLOGY] $sid")
    fi
  done
done

# ── totals ──────────────────────────────────────────────────────────────
TOTAL_MAX=$(( TOTAL_GUIDES > TOTAL_SENSORS ? TOTAL_GUIDES : TOTAL_SENSORS ))
if [[ $TOTAL_MAX -gt 0 ]]; then
  TOTAL_RATIO=$(awk "BEGIN { printf \"%.1f\", ($TOTAL_PAIRED / $TOTAL_MAX) * 100 }")
else
  TOTAL_RATIO="0.0"
fi

echo "$DIVIDER"
printf "%-30s | %6d | %7d | %6d | %5s%% |\n" \
  "TOTAL" "$TOTAL_GUIDES" "$TOTAL_SENSORS" "$TOTAL_PAIRED" "$TOTAL_RATIO"

# ── unpaired items ──────────────────────────────────────────────────────
if [[ ${#UNPAIRED_GUIDES[@]} -gt 0 ]]; then
  printf "\nUnpaired Guides:\n"
  for item in "${UNPAIRED_GUIDES[@]}"; do
    echo "  $item"
  done
fi

if [[ ${#UNPAIRED_SENSORS[@]} -gt 0 ]]; then
  printf "\nUnpaired Sensors:\n"
  for item in "${UNPAIRED_SENSORS[@]}"; do
    echo "  $item"
  done
fi

# ── source backing ──────────────────────────────────────────────────────
# The pairing ratio above counts declared guide/sensor ID cross-references,
# NOT whether each control's source resolves to real code. That distinction
# matters: a template can score 100% paired while some sources are aspirational.
SOURCE_BACKED=$(( TOTAL_CONTROLS - TOTAL_PLANNED ))
if [[ $TOTAL_CONTROLS -gt 0 ]]; then
  BACKED_PCT=$(awk "BEGIN { printf \"%.1f\", ($SOURCE_BACKED / $TOTAL_CONTROLS) * 100 }")
else
  BACKED_PCT="0.0"
fi
printf "\nSource backing: %d/%d controls source-backed (%s%% present, %d planned)\n" \
  "$SOURCE_BACKED" "$TOTAL_CONTROLS" "$BACKED_PCT" "$TOTAL_PLANNED"
if [[ ${#PLANNED_CONTROLS[@]} -gt 0 ]]; then
  printf "\nPlanned (source not yet resolvable):\n"
  for item in "${PLANNED_CONTROLS[@]}"; do
    echo "  $item"
  done
fi

# ── verdict ─────────────────────────────────────────────────────────────
echo ""
RATIO_INT=$(awk "BEGIN { printf \"%d\", $TOTAL_RATIO }")
if (( RATIO_INT >= TARGET_RATIO )); then
  echo "PASS: Overall pairing ratio ${TOTAL_RATIO}% meets target ${TARGET_RATIO}%"
  exit 0
else
  echo "FAIL: Overall pairing ratio ${TOTAL_RATIO}% is below target ${TARGET_RATIO}%"
  exit 1
fi
