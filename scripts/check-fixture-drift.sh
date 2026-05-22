#!/usr/bin/env bash
# check-fixture-drift.sh — Detect drift between canonical VisionClaw protocol
# fixtures and copies in consuming repos across the VisionFlow ecosystem.
#
# Canonical source: VisionClaw monorepo docs/specs/fixtures/
# Consumers: agentbox, solid-pod-rs, nostr-rust-forum, VisionFlow (future)
#
# Exit codes:
#   0  All fixture copies match canonical checksums
#   1  Drift detected (checksum mismatch or missing canonical file)
#   2  Usage error or missing canonical source
#
# Usage:
#   ./check-fixture-drift.sh [OPTIONS] [REPO_PATH ...]
#
# Options:
#   --canonical DIR    Override canonical fixture directory
#   --json             Output machine-readable JSON summary
#   --quiet            Suppress per-file output, only print summary
#   --help             Show this help
#
# If no REPO_PATHs are given, defaults to standard workspace locations.

set -euo pipefail

# --- Defaults ---
CANONICAL_DIR=""
JSON_OUTPUT=false
QUIET=false
REPO_PATHS=()

# --- Colour codes (disabled if not a terminal) ---
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' RESET=''
fi

usage() {
    sed -n '2,/^$/s/^# //p' "$0"
    exit 0
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --canonical)
            CANONICAL_DIR="$2"
            shift 2
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            REPO_PATHS+=("$1")
            shift
            ;;
    esac
done

# --- Resolve canonical directory ---
# Try well-known locations in order
if [[ -z "$CANONICAL_DIR" ]]; then
    for candidate in \
        "/home/devuser/workspace/project/docs/specs/fixtures" \
        "${WORKSPACE:-}/project/docs/specs/fixtures" \
        "$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)/../project/docs/specs/fixtures" \
    ; do
        if [[ -d "$candidate" ]]; then
            CANONICAL_DIR="$(cd "$candidate" && pwd)"
            break
        fi
    done
fi

if [[ -z "$CANONICAL_DIR" || ! -d "$CANONICAL_DIR" ]]; then
    echo "ERROR: Canonical fixture directory not found." >&2
    echo "Pass --canonical DIR or ensure VisionClaw is at a standard workspace path." >&2
    exit 2
fi

# --- Workspace root (used for display labels) ---
WORKSPACE_ROOT="${WORKSPACE:-/home/devuser/workspace}"

# --- Default repo paths if none given ---
if [[ ${#REPO_PATHS[@]} -eq 0 ]]; then
    for repo in \
        "$WORKSPACE_ROOT/project/agentbox/tests/contract/upstream_vectors" \
        "$WORKSPACE_ROOT/solid-pod-rs/crates/solid-pod-rs-nostr/tests/fixtures" \
        "$WORKSPACE_ROOT/solid-pod-rs/crates/solid-pod-rs-didkey/tests/fixtures" \
        "$WORKSPACE_ROOT/nostr-rust-forum/tests/fixtures" \
    ; do
        if [[ -d "$repo" ]]; then
            REPO_PATHS+=("$repo")
        fi
    done
fi

if [[ ${#REPO_PATHS[@]} -eq 0 ]]; then
    echo "ERROR: No fixture directories found to check." >&2
    exit 2
fi

# --- Build canonical checksum index ---
# Map: relative_path -> sha256
declare -A CANONICAL_SUMS
declare -a CANONICAL_FILES

while IFS= read -r -d '' file; do
    rel="${file#"$CANONICAL_DIR/"}"
    # Skip non-fixture files (markdown, txt)
    case "$rel" in
        *.json) ;;
        *) continue ;;
    esac
    checksum="$(sha256sum "$file" | cut -d' ' -f1)"
    CANONICAL_SUMS["$rel"]="$checksum"
    CANONICAL_FILES+=("$rel")
done < <(find "$CANONICAL_DIR" -type f -print0 | sort -z)

CANONICAL_COUNT=${#CANONICAL_FILES[@]}

if [[ $CANONICAL_COUNT -eq 0 ]]; then
    echo "ERROR: No JSON fixture files found in canonical directory: $CANONICAL_DIR" >&2
    exit 2
fi

# --- Check each repo path ---
TOTAL_MATCHED=0
TOTAL_DRIFTED=0
TOTAL_MISSING=0
TOTAL_EXTRA=0
TOTAL_REPOS=0
DRIFT_DETECTED=false

# JSON accumulator
JSON_REPOS="["

log() {
    if [[ "$QUIET" == false ]]; then
        echo -e "$@"
    fi
}

check_repo() {
    local repo_path="$1"
    local repo_label
    repo_label="$(echo "$repo_path" | sed "s|$WORKSPACE_ROOT/||" 2>/dev/null || echo "$repo_path")"

    local matched=0 drifted=0 missing=0 extra=0
    local drifted_files=() missing_files=() extra_files=()

    log ""
    log "${BOLD}${CYAN}=== $repo_label ===${RESET}"

    # Build set of files present in this repo copy
    declare -A repo_files
    while IFS= read -r -d '' file; do
        local rel="${file#"$repo_path/"}"
        case "$rel" in
            *.json) ;;
            *) continue ;;
        esac
        # Skip non-protocol files (e.g. jss-compatible.json, package.json)
        if [[ -z "${CANONICAL_SUMS[$rel]+_}" ]]; then
            # Check if it looks like a protocol fixture (has _meta or is in schemas/)
            # Only flag as extra if the filename pattern matches canonical fixture naming
            case "$rel" in
                schemas/*|*.schema.json)
                    extra=$((extra + 1))
                    extra_files+=("$rel")
                    ;;
                *)
                    # Check if there is any canonical file with the same basename
                    local base
                    base="$(basename "$rel")"
                    local is_protocol=false
                    for cf in "${CANONICAL_FILES[@]}"; do
                        if [[ "$(basename "$cf")" == "$base" ]]; then
                            is_protocol=true
                            break
                        fi
                    done
                    if [[ "$is_protocol" == true ]]; then
                        extra=$((extra + 1))
                        extra_files+=("$rel")
                    fi
                    # Silently skip non-protocol JSON files
                    ;;
            esac
        fi
        repo_files["$rel"]=1
    done < <(find "$repo_path" -type f -print0 | sort -z)

    # Compare each canonical file against the repo copy
    for rel in "${CANONICAL_FILES[@]}"; do
        local repo_file="$repo_path/$rel"
        if [[ ! -f "$repo_file" ]]; then
            missing=$((missing + 1))
            missing_files+=("$rel")
            continue
        fi

        local repo_sum
        repo_sum="$(sha256sum "$repo_file" | cut -d' ' -f1)"
        local canonical_sum="${CANONICAL_SUMS[$rel]}"

        if [[ "$repo_sum" == "$canonical_sum" ]]; then
            matched=$((matched + 1))
            log "  ${GREEN}MATCH${RESET}  $rel"
        else
            drifted=$((drifted + 1))
            drifted_files+=("$rel")
            log "  ${RED}DRIFT${RESET}  $rel"
            log "         canonical: ${canonical_sum:0:16}..."
            log "         local:     ${repo_sum:0:16}..."
        fi
    done

    # Report missing
    for f in "${missing_files[@]}"; do
        log "  ${YELLOW}MISS${RESET}   $f"
    done

    # Report extra (only protocol-like files already flagged)
    for f in "${extra_files[@]}"; do
        log "  ${CYAN}EXTRA${RESET}  $f"
    done

    # Summary line
    log "  --- ${matched} matched, ${drifted} drifted, ${missing} missing, ${extra} extra (of ${CANONICAL_COUNT} canonical)"

    # Accumulate totals
    TOTAL_MATCHED=$((TOTAL_MATCHED + matched))
    TOTAL_DRIFTED=$((TOTAL_DRIFTED + drifted))
    TOTAL_MISSING=$((TOTAL_MISSING + missing))
    TOTAL_EXTRA=$((TOTAL_EXTRA + extra))
    TOTAL_REPOS=$((TOTAL_REPOS + 1))

    if [[ $drifted -gt 0 ]]; then
        DRIFT_DETECTED=true
    fi

    # JSON fragment — handle empty arrays cleanly
    local drifted_json="[]" missing_json="[]" extra_json="[]"
    if [[ ${#drifted_files[@]} -gt 0 ]]; then
        drifted_json="$(printf '%s\n' "${drifted_files[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]')"
    fi
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        missing_json="$(printf '%s\n' "${missing_files[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]')"
    fi
    if [[ ${#extra_files[@]} -gt 0 ]]; then
        extra_json="$(printf '%s\n' "${extra_files[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]')"
    fi

    if [[ $TOTAL_REPOS -gt 1 ]]; then
        JSON_REPOS+=","
    fi
    JSON_REPOS+="$(cat <<ENDJSON
{
  "path": "$repo_path",
  "label": "$repo_label",
  "matched": $matched,
  "drifted": $drifted,
  "missing": $missing,
  "extra": $extra,
  "drifted_files": $drifted_json,
  "missing_files": $missing_json,
  "extra_files": $extra_json
}
ENDJSON
)"
}

for repo in "${REPO_PATHS[@]}"; do
    check_repo "$repo"
done

JSON_REPOS+="]"

# --- Final summary ---
echo ""
echo -e "${BOLD}Fixture Drift Summary${RESET}"
echo "  Canonical source:  $CANONICAL_DIR"
echo "  Canonical files:   $CANONICAL_COUNT"
echo "  Repos checked:     $TOTAL_REPOS"
echo "  Total matched:     $TOTAL_MATCHED"
echo "  Total drifted:     $TOTAL_DRIFTED"
echo "  Total missing:     $TOTAL_MISSING"
echo "  Total extra:       $TOTAL_EXTRA"

if [[ "$DRIFT_DETECTED" == true ]]; then
    echo ""
    echo -e "  ${RED}${BOLD}DRIFT DETECTED${RESET} -- fixture copies diverge from canonical source."
    echo "  Run the refresh workflow per UPSTREAM_PINS.md to re-sync."
fi

# --- JSON output ---
if [[ "$JSON_OUTPUT" == true ]]; then
    cat <<ENDJSON

{
  "canonical_dir": "$CANONICAL_DIR",
  "canonical_count": $CANONICAL_COUNT,
  "repos_checked": $TOTAL_REPOS,
  "total_matched": $TOTAL_MATCHED,
  "total_drifted": $TOTAL_DRIFTED,
  "total_missing": $TOTAL_MISSING,
  "total_extra": $TOTAL_EXTRA,
  "drift_detected": $DRIFT_DETECTED,
  "repos": $JSON_REPOS
}
ENDJSON
fi

# --- Exit code ---
if [[ "$DRIFT_DETECTED" == true ]]; then
    exit 1
fi
exit 0
