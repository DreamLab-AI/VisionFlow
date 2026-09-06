#!/usr/bin/env bash
# harness-audit.sh — Pairing and source-backing audit for harness templates
#
# Reads every *.json template in docs/engineering/templates/, extracts guides,
# sensors and pairings, then reports pairing coverage and source backing.
#
# Three defects were closed on 2026-09-05 (engineering ADR-004 closeout; the
# reproductions are in docs/estate-review/evidence/harness-audit-probe.json):
#
#   1. SOURCE PATHS WERE NEVER RESOLVED. A control counted as "source-backed"
#      whenever `source_status` was absent or "present" — an unchecked self-
#      declaration. A template naming a source file that does not exist scored
#      100% source-backed. Sources are now RESOLVED against the substrate
#      checkouts: a declared-present source that does not resolve is
#      "unresolved" and is NOT source-backed.
#
#   2. PAIRING EDGES WERE COUNTED WITH DUPLICATES. The ratio was
#      raw_pairing_count / max(guides, sensors), so repeating one pairing twice
#      in a template scored 200% and passed any target. Edges are now
#      de-duplicated on (guide_id, sensor_id), and coverage counts DISTINCT
#      paired guides and sensors against the controls that exist, so the ratio
#      is bounded by 100% by construction.
#
#   3. SOURCE BACKING WAS COUNTED PER CONTROL, NOT PER SOURCE. Two controls
#      naming the same file counted as two backed controls. Backing is now
#      capped at DISTINCT sources; per-control figures are still reported.
#
# Substrate availability follows the estate's partial-source failure mode
# (ADR-005 §Decision 2, as used by the drift counter): a substrate that is not
# checked out cannot be resolved, so its controls are reported "unverifiable"
# and excluded from the backing denominator rather than being counted as
# failures. --strict-sources turns unverifiable and unresolved into exit 1.
#
# Exit codes:
#   0  Pairing coverage >= TARGET_RATIO (default 80%) and, under
#      --strict-sources, every declared-present source resolved
#   1  Below target, unresolved sources under --strict-sources, or no templates
#   2  Usage error or missing dependency
#
# Dependencies: jq (https://stedolan.github.io/jq/), bash 4+
#
# Usage:
#   bash scripts/harness-audit.sh                     # run from repo root
#   bash scripts/harness-audit.sh --target 60         # override target ratio
#   bash scripts/harness-audit.sh --dir /path         # override template dir
#   bash scripts/harness-audit.sh --strict-sources    # unresolved source => fail
#   bash scripts/harness-audit.sh --json              # machine-readable summary
#   bash scripts/harness-audit.sh --substrate-root visionclaw=/path/to/repo
#   bash scripts/harness-audit.sh --help              # show this message

set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_DIR="${TEMPLATE_DIR:-$REPO_ROOT/docs/engineering/templates}"
TARGET_RATIO="${TARGET_RATIO:-80}"
STRICT_SOURCES=false
JSON_OUT=false

# Substrate name -> checkout root. Defaults assume the standard workspace
# layout (the canon and its siblings side by side); override per substrate with
# --substrate-root NAME=PATH, or wholesale with HARNESS_SUBSTRATE_ROOTS as a
# comma-separated NAME=PATH list. A substrate with no resolvable root is
# reported as unverifiable, never as a silent pass.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
declare -A SUBSTRATE_ROOT=(
  [VisionFlow]="$REPO_ROOT"
  [visionflow]="$REPO_ROOT"
  [visionclaw]="$WORKSPACE_ROOT/project"
  [agentbox]="$WORKSPACE_ROOT/project/agentbox"
  [solid-pod-rs]="$WORKSPACE_ROOT/solid-pod-rs"
  [nostr-rust-forum]="$WORKSPACE_ROOT/nostr-rust-forum"
  [ruvector]="$WORKSPACE_ROOT/ruvector"
)

# ── argument parsing ────────────────────────────────────────────────────
usage() {
  sed -n '2,/^$/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
  exit 0
}

if [[ -n "${HARNESS_SUBSTRATE_ROOTS:-}" ]]; then
  IFS=',' read -r -a _pairs <<< "$HARNESS_SUBSTRATE_ROOTS"
  for _p in "${_pairs[@]}"; do
    [[ "$_p" == *=* ]] || continue
    SUBSTRATE_ROOT["${_p%%=*}"]="${_p#*=}"
  done
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)          usage ;;
    --target|-t)        TARGET_RATIO="$2"; shift 2 ;;
    --dir|-d)           TEMPLATE_DIR="$2"; shift 2 ;;
    --strict-sources)   STRICT_SOURCES=true; shift ;;
    --json)             JSON_OUT=true; shift ;;
    --substrate-root)   SUBSTRATE_ROOT["${2%%=*}"]="${2#*=}"; shift 2 ;;
    *)                  echo "Unknown option: $1" >&2; exit 2 ;;
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

# ── source resolution ───────────────────────────────────────────────────
# A source is "SUBSTRATE:LOCATOR". The locator may carry an #anchor fragment,
# may be a glob, may name a directory, may be a ' + '-joined list of paths that
# must ALL exist, or may be a prose descriptor for a non-filesystem artefact
# (an MCP namespace, say) that no filesystem check can settle.
#
# Echoes one of: resolved | unresolved | descriptor | unverifiable:<substrate>
resolve_source() {
  local src="$1"

  # No substrate prefix, or a URL: nothing on disk to check against.
  if [[ "$src" != *:* ]] || [[ "$src" =~ ^https?:// ]]; then
    echo "descriptor"; return
  fi

  local substrate="${src%%:*}"
  local locator="${src#*:}"

  local root="${SUBSTRATE_ROOT[$substrate]:-}"
  if [[ -z "$root" ]]; then
    echo "unverifiable:$substrate"; return
  fi
  if [[ ! -d "$root" ]]; then
    echo "unverifiable:$substrate"; return
  fi

  # ' + '-joined composite: every part must resolve.
  if [[ "$locator" == *" + "* ]]; then
    local part all=resolved
    local saved_ifs="$IFS"
    IFS='+' read -r -a _parts <<< "$locator"
    IFS="$saved_ifs"
    for part in "${_parts[@]}"; do
      # shellcheck disable=SC2001
      part="$(echo "$part" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
      [[ -z "$part" ]] && continue
      if [[ "$(resolve_one "$root" "$part")" != resolved ]]; then all=unresolved; fi
    done
    echo "$all"; return
  fi

  resolve_one "$root" "$locator"
}

# Resolve a single locator beneath a substrate root.
resolve_one() {
  local root="$1" locator="$2"

  locator="${locator%%#*}"                       # drop #anchor fragments
  locator="${locator%"${locator##*[![:space:]]}"}"  # rtrim

  # A prose descriptor rather than a path (e.g. "governance-precedents namespace").
  if [[ "$locator" == *" "* ]]; then echo "descriptor"; return; fi
  if [[ -z "$locator" ]]; then echo "descriptor"; return; fi

  local abs
  if [[ "$locator" == "~/"* ]]; then
    # Runtime configuration in the operator's home, not inside the checkout.
    abs="$HOME/${locator#\~/}"
  else
    abs="$root/$locator"
  fi

  if [[ "$abs" == *"*"* ]]; then                 # glob: any match resolves it
    local m
    shopt -s nullglob
    # shellcheck disable=SC2206
    m=( $abs )
    shopt -u nullglob
    [[ ${#m[@]} -gt 0 ]] && echo "resolved" || echo "unresolved"
    return
  fi

  if [[ -e "$abs" ]]; then echo "resolved"; else echo "unresolved"; fi
}

# ── accumulators ────────────────────────────────────────────────────────
TOTAL_GUIDES=0
TOTAL_SENSORS=0
TOTAL_PAIRED_CONTROLS=0
TOTAL_EDGES=0
TOTAL_DUP_EDGES=0
TOTAL_DANGLING=0
UNPAIRED_GUIDES=()
UNPAIRED_SENSORS=()
DUPLICATE_EDGES=()
DANGLING_EDGES=()

# Distinct-source accounting, keyed by the source string itself so that two
# controls naming one file are one source, not two.
declare -A SOURCE_STATE=()      # source -> resolved|unresolved|descriptor|unverifiable|planned
declare -A SOURCE_OWNERS=()     # source -> "[topology] kind id; ..."
declare -A RESOLVE_CACHE=()

CTRL_PLANNED=0
CTRL_RESOLVED=0
CTRL_UNRESOLVED=0
CTRL_DESCRIPTOR=0
CTRL_UNVERIFIABLE=0

ROWS_JSON=()

# ── per-template analysis ──────────────────────────────────────────────
printf "\nHARNESS PAIRING AUDIT\n"
printf "=====================\n"
printf "%-30s | %6s | %7s | %5s | %6s | %s\n" \
  "Topology" "Guides" "Sensors" "Edges" "Cover" "Maturity"

DIVIDER="$(printf '%0.s-' {1..80})"

for tpl in "${TEMPLATES[@]}"; do
  if ! jq empty "$tpl" 2>/dev/null; then
    echo "WARNING: Invalid JSON, skipping: $tpl" >&2
    continue
  fi

  TOPOLOGY=$(jq -r '.topology // "unknown"' "$tpl")
  MATURITY=$(jq -r '.maturity // "unknown"' "$tpl")

  mapfile -t GUIDE_IDS  < <(jq -r '.guides[]?.id  // empty' "$tpl")
  mapfile -t SENSOR_IDS < <(jq -r '.sensors[]?.id // empty' "$tpl")
  NUM_GUIDES=${#GUIDE_IDS[@]}
  NUM_SENSORS=${#SENSOR_IDS[@]}

  declare -A IS_GUIDE=() IS_SENSOR=()
  for g in "${GUIDE_IDS[@]}"; do IS_GUIDE["$g"]=1; done
  for s in "${SENSOR_IDS[@]}"; do IS_SENSOR["$s"]=1; done

  # --- Fix 2: de-duplicate pairing edges on (guide_id, sensor_id) -------
  declare -A EDGE_SEEN=() PAIRED_G=() PAIRED_S=()
  NUM_EDGES=0; NUM_DUP=0; NUM_DANGLING=0
  while IFS=$'\t' read -r gid sid; do
    [[ -z "$gid" || -z "$sid" ]] && continue
    local_key="$gid|$sid"
    if [[ -n "${EDGE_SEEN[$local_key]:-}" ]]; then
      NUM_DUP=$(( NUM_DUP + 1 ))
      DUPLICATE_EDGES+=("[$TOPOLOGY] $gid -> $sid")
      continue
    fi
    EDGE_SEEN["$local_key"]=1
    # An edge referencing an ID that does not exist cannot evidence coverage.
    if [[ -z "${IS_GUIDE[$gid]:-}" || -z "${IS_SENSOR[$sid]:-}" ]]; then
      NUM_DANGLING=$(( NUM_DANGLING + 1 ))
      DANGLING_EDGES+=("[$TOPOLOGY] $gid -> $sid")
      continue
    fi
    NUM_EDGES=$(( NUM_EDGES + 1 ))
    PAIRED_G["$gid"]=1
    PAIRED_S["$sid"]=1
  done < <(jq -r '.pairings[]? | "\(.guide_id // "")\t\(.sensor_id // "")"' "$tpl")

  # --- Coverage: distinct paired controls / declared controls ----------
  # Bounded by 100% by construction: PAIRED_G ⊆ GUIDE_IDS, PAIRED_S ⊆ SENSOR_IDS.
  NUM_PAIRED_CONTROLS=$(( ${#PAIRED_G[@]} + ${#PAIRED_S[@]} ))
  NUM_CONTROLS=$(( NUM_GUIDES + NUM_SENSORS ))
  if [[ $NUM_CONTROLS -gt 0 ]]; then
    COVER=$(awk "BEGIN { printf \"%.1f\", ($NUM_PAIRED_CONTROLS / $NUM_CONTROLS) * 100 }")
  else
    COVER="0.0"
  fi

  printf "%-30s | %6d | %7d | %5d | %5s%% | %s\n" \
    "$TOPOLOGY" "$NUM_GUIDES" "$NUM_SENSORS" "$NUM_EDGES" "$COVER" "$MATURITY"

  TOTAL_GUIDES=$(( TOTAL_GUIDES + NUM_GUIDES ))
  TOTAL_SENSORS=$(( TOTAL_SENSORS + NUM_SENSORS ))
  TOTAL_PAIRED_CONTROLS=$(( TOTAL_PAIRED_CONTROLS + NUM_PAIRED_CONTROLS ))
  TOTAL_EDGES=$(( TOTAL_EDGES + NUM_EDGES ))
  TOTAL_DUP_EDGES=$(( TOTAL_DUP_EDGES + NUM_DUP ))
  TOTAL_DANGLING=$(( TOTAL_DANGLING + NUM_DANGLING ))

  ROWS_JSON+=("$(jq -nc \
    --arg t "$TOPOLOGY" --arg m "$MATURITY" \
    --argjson g "$NUM_GUIDES" --argjson s "$NUM_SENSORS" \
    --argjson e "$NUM_EDGES" --argjson d "$NUM_DUP" --argjson x "$NUM_DANGLING" \
    --arg c "$COVER" \
    '{topology:$t,maturity:$m,guides:$g,sensors:$s,distinct_edges:$e,duplicate_edges:$d,dangling_edges:$x,coverage_pct:($c|tonumber)}')")

  # --- Fix 1 + 3: resolve each control's source; account per distinct source
  while IFS=$'\t' read -r kind cid status src; do
    [[ -z "$cid" ]] && continue
    local_state=""
    if [[ "$status" == "planned" ]]; then
      local_state="planned"
    else
      if [[ -z "${RESOLVE_CACHE[$src]:-}" ]]; then
        RESOLVE_CACHE["$src"]="$(resolve_source "$src")"
      fi
      local_state="${RESOLVE_CACHE[$src]}"
      [[ "$local_state" == unverifiable:* ]] && local_state="unverifiable"
    fi

    case "$local_state" in
      planned)      CTRL_PLANNED=$((CTRL_PLANNED+1)) ;;
      resolved)     CTRL_RESOLVED=$((CTRL_RESOLVED+1)) ;;
      unresolved)   CTRL_UNRESOLVED=$((CTRL_UNRESOLVED+1)) ;;
      descriptor)   CTRL_DESCRIPTOR=$((CTRL_DESCRIPTOR+1)) ;;
      unverifiable) CTRL_UNVERIFIABLE=$((CTRL_UNVERIFIABLE+1)) ;;
    esac

    # Distinct-source state. "planned" loses to a resolved sibling only if some
    # other control declares the same source present and it resolves.
    prev="${SOURCE_STATE[$src]:-}"
    if [[ -z "$prev" || ( "$prev" == "planned" && "$local_state" == "resolved" ) ]]; then
      SOURCE_STATE["$src"]="$local_state"
    fi
    SOURCE_OWNERS["$src"]="${SOURCE_OWNERS[$src]:-}[$TOPOLOGY] $kind $cid; "
  done < <(jq -r '
    (.guides[]?  | "guide\t\(.id)\t\(.source_status // "present")\t\(.source // "")"),
    (.sensors[]? | "sensor\t\(.id)\t\(.source_status // "present")\t\(.source // "")")
  ' "$tpl")

  # --- unpaired controls ------------------------------------------------
  for gid in "${GUIDE_IDS[@]}"; do
    [[ -z "${PAIRED_G[$gid]:-}" ]] && UNPAIRED_GUIDES+=("[$TOPOLOGY] $gid")
  done
  for sid in "${SENSOR_IDS[@]}"; do
    [[ -z "${PAIRED_S[$sid]:-}" ]] && UNPAIRED_SENSORS+=("[$TOPOLOGY] $sid")
  done

  unset IS_GUIDE IS_SENSOR EDGE_SEEN PAIRED_G PAIRED_S
done

# ── totals ──────────────────────────────────────────────────────────────
TOTAL_CONTROLS=$(( TOTAL_GUIDES + TOTAL_SENSORS ))
if [[ $TOTAL_CONTROLS -gt 0 ]]; then
  TOTAL_RATIO=$(awk "BEGIN { printf \"%.1f\", ($TOTAL_PAIRED_CONTROLS / $TOTAL_CONTROLS) * 100 }")
else
  TOTAL_RATIO="0.0"
fi

echo "$DIVIDER"
printf "%-30s | %6d | %7d | %5d | %5s%% |\n" \
  "TOTAL" "$TOTAL_GUIDES" "$TOTAL_SENSORS" "$TOTAL_EDGES" "$TOTAL_RATIO"
printf "Coverage counts DISTINCT paired controls (%d of %d) over de-duplicated edges.\n" \
  "$TOTAL_PAIRED_CONTROLS" "$TOTAL_CONTROLS"

# ── pairing hygiene ─────────────────────────────────────────────────────
if [[ ${#UNPAIRED_GUIDES[@]} -gt 0 ]]; then
  printf "\nUnpaired Guides:\n"
  printf '  %s\n' "${UNPAIRED_GUIDES[@]}"
fi
if [[ ${#UNPAIRED_SENSORS[@]} -gt 0 ]]; then
  printf "\nUnpaired Sensors:\n"
  printf '  %s\n' "${UNPAIRED_SENSORS[@]}"
fi
if [[ ${#DUPLICATE_EDGES[@]} -gt 0 ]]; then
  printf "\nDuplicate pairing edges (counted once):\n"
  printf '  %s\n' "${DUPLICATE_EDGES[@]}"
fi
if [[ ${#DANGLING_EDGES[@]} -gt 0 ]]; then
  printf "\nDangling pairing edges (reference a non-existent control, not counted):\n"
  printf '  %s\n' "${DANGLING_EDGES[@]}"
fi

# ── source backing (distinct sources) ───────────────────────────────────
SRC_RESOLVED=0; SRC_UNRESOLVED=0; SRC_PLANNED=0; SRC_DESCRIPTOR=0; SRC_UNVERIFIABLE=0
UNRESOLVED_LIST=(); PLANNED_LIST=(); DESCRIPTOR_LIST=(); UNVERIFIABLE_LIST=()
for src in "${!SOURCE_STATE[@]}"; do
  case "${SOURCE_STATE[$src]}" in
    resolved)     SRC_RESOLVED=$((SRC_RESOLVED+1)) ;;
    unresolved)   SRC_UNRESOLVED=$((SRC_UNRESOLVED+1));   UNRESOLVED_LIST+=("$src  <- ${SOURCE_OWNERS[$src]}") ;;
    planned)      SRC_PLANNED=$((SRC_PLANNED+1));         PLANNED_LIST+=("$src  <- ${SOURCE_OWNERS[$src]}") ;;
    descriptor)   SRC_DESCRIPTOR=$((SRC_DESCRIPTOR+1));   DESCRIPTOR_LIST+=("$src  <- ${SOURCE_OWNERS[$src]}") ;;
    unverifiable) SRC_UNVERIFIABLE=$((SRC_UNVERIFIABLE+1)); UNVERIFIABLE_LIST+=("$src  <- ${SOURCE_OWNERS[$src]}") ;;
  esac
done
TOTAL_SOURCES=${#SOURCE_STATE[@]}

# Denominator excludes substrates that are not checked out (nothing can be
# resolved about them here) — the partial-source failure mode, reported, not
# hidden. Descriptors stay IN the denominator: they are declared as sources and
# a filesystem check genuinely cannot back them.
BACKABLE=$(( TOTAL_SOURCES - SRC_UNVERIFIABLE ))
if [[ $BACKABLE -gt 0 ]]; then
  BACKED_PCT=$(awk "BEGIN { printf \"%.1f\", ($SRC_RESOLVED / $BACKABLE) * 100 }")
else
  BACKED_PCT="0.0"
fi

printf "\nSOURCE BACKING (distinct sources, resolved against substrate checkouts)\n"
printf "  distinct sources declared : %d  (across %d controls)\n" "$TOTAL_SOURCES" "$TOTAL_CONTROLS"
printf "  resolved (source-backed)  : %d\n" "$SRC_RESOLVED"
printf "  unresolved (declared present, path absent) : %d\n" "$SRC_UNRESOLVED"
printf "  planned (declared not yet built)           : %d\n" "$SRC_PLANNED"
printf "  descriptor (not a filesystem path)         : %d\n" "$SRC_DESCRIPTOR"
printf "  unverifiable (substrate not checked out)   : %d\n" "$SRC_UNVERIFIABLE"
printf "  backing = %d/%d = %s%% of resolvable sources\n" \
  "$SRC_RESOLVED" "$BACKABLE" "$BACKED_PCT"
printf "  per-control: %d resolved, %d unresolved, %d planned, %d descriptor, %d unverifiable\n" \
  "$CTRL_RESOLVED" "$CTRL_UNRESOLVED" "$CTRL_PLANNED" "$CTRL_DESCRIPTOR" "$CTRL_UNVERIFIABLE"

emit_list() {
  local title="$1"; shift
  [[ $# -eq 0 ]] && return
  printf "\n%s:\n" "$title"
  printf '  %s\n' "$@"
}
emit_list "UNRESOLVED — declared present but the path does not exist" "${UNRESOLVED_LIST[@]+"${UNRESOLVED_LIST[@]}"}"
emit_list "Planned (source not yet resolvable)"                       "${PLANNED_LIST[@]+"${PLANNED_LIST[@]}"}"
emit_list "Descriptor (no filesystem artefact to check)"              "${DESCRIPTOR_LIST[@]+"${DESCRIPTOR_LIST[@]}"}"
emit_list "Unverifiable (substrate checkout absent)"                  "${UNVERIFIABLE_LIST[@]+"${UNVERIFIABLE_LIST[@]}"}"

# ── verdict ─────────────────────────────────────────────────────────────
echo ""
RATIO_INT=$(awk "BEGIN { printf \"%d\", $TOTAL_RATIO }")
FAIL=0
if (( RATIO_INT >= TARGET_RATIO )); then
  echo "PASS: pairing coverage ${TOTAL_RATIO}% meets target ${TARGET_RATIO}%"
else
  echo "FAIL: pairing coverage ${TOTAL_RATIO}% is below target ${TARGET_RATIO}%"
  FAIL=1
fi
if [[ "$STRICT_SOURCES" == true ]]; then
  if (( SRC_UNRESOLVED > 0 || SRC_UNVERIFIABLE > 0 )); then
    echo "FAIL (--strict-sources): ${SRC_UNRESOLVED} unresolved and ${SRC_UNVERIFIABLE} unverifiable source(s)"
    FAIL=1
  else
    echo "PASS (--strict-sources): every declared-present source resolved"
  fi
fi

if [[ "$JSON_OUT" == true ]]; then
  jq -nc \
    --argjson rows "$(printf '%s\n' "${ROWS_JSON[@]+"${ROWS_JSON[@]}"}" | jq -sc '.')" \
    --argjson totals "$(jq -nc \
      --argjson g "$TOTAL_GUIDES" --argjson s "$TOTAL_SENSORS" \
      --argjson pc "$TOTAL_PAIRED_CONTROLS" --argjson e "$TOTAL_EDGES" \
      --argjson dup "$TOTAL_DUP_EDGES" --argjson dang "$TOTAL_DANGLING" \
      --arg cov "$TOTAL_RATIO" \
      '{guides:$g,sensors:$s,paired_controls:$pc,distinct_edges:$e,duplicate_edges:$dup,dangling_edges:$dang,coverage_pct:($cov|tonumber)}')" \
    --argjson backing "$(jq -nc \
      --argjson t "$TOTAL_SOURCES" --argjson r "$SRC_RESOLVED" --argjson u "$SRC_UNRESOLVED" \
      --argjson p "$SRC_PLANNED" --argjson d "$SRC_DESCRIPTOR" --argjson v "$SRC_UNVERIFIABLE" \
      --arg pct "$BACKED_PCT" \
      '{distinct_sources:$t,resolved:$r,unresolved:$u,planned:$p,descriptor:$d,unverifiable:$v,backed_pct:($pct|tonumber)}')" \
    --argjson ok "$([[ $FAIL -eq 0 ]] && echo true || echo false)" \
    '{ok:$ok,templates:$rows,totals:$totals,source_backing:$backing}' \
    > "${HARNESS_AUDIT_JSON:-/dev/stdout}"
fi

exit "$FAIL"
