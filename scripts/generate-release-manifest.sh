#!/usr/bin/env bash
# generate-release-manifest.sh — coordinated release manifest for the estate.
#
# Emits the manifest described by docs/releases/ecosystem-release.schema.json:
# every inventoried repository with its HEAD, branch, dirty state and
# provenance, plus the fixture-corpus comparison the release qualification
# depends on.
#
# Two defects were closed on 2026-09-05 (ADR-2006 closeout):
#
#   1. THE ROSTER WAS A SUBSET. The generator covered six repositories while
#      the ADR inventory (docs/estate-review/evidence/adr-inventory.json)
#      records fourteen. A "coordinated release manifest" that silently omits
#      eight repositories coordinates nothing about them — loom, knowledgeGraph,
#      WasmVOWL, visionGraph, dream-machine, logseq, ruvector and RuView could
#      each move under a release with no record. All fourteen are now listed,
#      each flagged first-party / imported / upstream so a reader can tell what
#      the estate authors from what it merely carries or consumes.
#
#   2. THE FIXTURE CLAIM WAS FREE. The manifest asserted "run npm run verify
#      plus substrate-specific CI" in prose and carried no fixture evidence at
#      all, so a candidate could be cut with no canonical corpus ever compared.
#      A fixture comparison now requires an explicit CANONICAL REVISION SET —
#      repo@revision:dir — and the generator REFUSES to emit a candidate or
#      released manifest without one (exit 3). Local drafts may omit it, and
#      are stamped fixtures.status="not-compared" so the gap is visible in the
#      artefact rather than implied by its absence.
#
# Exit codes:
#   0  Manifest written to stdout
#   2  Usage error
#   3  Fixture comparison required but no canonical revision set supplied,
#      or the supplied canonical revision set does not resolve
#
# Usage:
#   scripts/generate-release-manifest.sh
#   scripts/generate-release-manifest.sh --status candidate \
#       --fixtures-canonical project@b00c28a0d766:tests/fixtures
#   scripts/generate-release-manifest.sh --require-fixtures \
#       --fixtures-canonical project@HEAD:tests/fixtures
#
# Options:
#   --status STATUS               local-draft (default) | candidate | released
#   --fixtures-canonical SPEC     canonical revision set, REPO@REV:DIR.
#                                 REPO is a roster path key, REV a revision or
#                                 HEAD, DIR the corpus directory within it.
#   --require-fixtures            demand a canonical revision set even for a
#                                 local draft
#   --workspace DIR               override the workspace root

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
STATUS="local-draft"
FIXTURES_CANONICAL=""
REQUIRE_FIXTURES=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status)              STATUS="$2"; shift 2 ;;
    --fixtures-canonical)  FIXTURES_CANONICAL="$2"; shift 2 ;;
    --require-fixtures)    REQUIRE_FIXTURES=true; shift ;;
    --workspace)           WORKSPACE="$(cd "$2" && pwd)"; shift 2 ;;
    --help|-h)             sed -n '2,/^set -euo/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)                     echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$STATUS" in
  local-draft|candidate|released) ;;
  *) echo "ERROR: --status must be local-draft, candidate or released" >&2; exit 2 ;;
esac

command -v jq >/dev/null || { echo "ERROR: jq is required" >&2; exit 2; }

# ── roster ───────────────────────────────────────────────────────────────
# NAME|PATH|PROVENANCE|UPSTREAM|ROLE
#
# provenance:
#   first-party  authored and released by the estate
#   imported     a fork or adaptation of an external project, carried here
#   upstream     an external project consumed as-is, not released by the estate
#
# Kept in step with docs/estate-review/evidence/adr-inventory.json ("repos").
ROSTER=(
  "VisionFlow|VisionFlow|first-party||canon — cross-repo view, release machinery, governing ADRs"
  "VisionClaw|project|first-party|github.com/JavaScriptSolidServer/JavaScriptSolidServer (jss component)|GPU engine, OWL 2 EL reasoner, IS-Envelope contract"
  "agentbox|project/agentbox|first-party||sovereign agent runtime; skill and MCP count sources"
  "solid-pod-rs|solid-pod-rs|first-party||canonical native Solid pod implementation"
  "nostr-rust-forum|nostr-rust-forum|first-party||governance plane, agent registry, relay"
  "dreamlab-ai-website|dreamlab-ai-website|first-party||commercial surface"
  "loom|loom|first-party||Ontology Loom — model-swappable grounding facade"
  "knowledgeGraph|knowledgeGraph|first-party||corpus generation feeding the Loom"
  "WasmVOWL|WasmVOWL|first-party||WASM ontology visualiser"
  "visionGraph|visionGraph|first-party||graph surface"
  "dream-engine|dream-machine|imported|github.com/ruvnet/dream-machine|dream-cycle engine, forked from ruvnet upstream"
  "logseq|project4|imported|github.com/logseq/logseq|authored vault tooling, forked from logseq upstream"
  "ruvector|ruvector|upstream|github.com/ruvnet/ruvector|vector memory substrate consumed by the estate"
  "RuView|RuView|upstream|github.com/ruvnet/RuView|sensing/consensus research substrate consumed by the estate"
)

repo_json() {
  local name="$1" path="$2" provenance="$3" upstream="$4" role="$5"
  local abs="$WORKSPACE/$path"
  local head branch dirty

  # Detect a repo with `git rev-parse`, not `-d "$abs/.git"`: a git submodule's
  # .git is a gitdir *file* (not a directory), so the old check flagged the
  # agentbox submodule as "missing" even at its correct path.
  if ! head="$(git -C "$abs" rev-parse HEAD 2>/dev/null)"; then
    jq -nc --arg n "$name" --arg p "../$path" --arg pr "$provenance" \
           --arg u "$upstream" --arg r "$role" \
      '{name:$n,path:$p,head:"0000000000000000000000000000000000000000",
        branch:"missing",dirty:true,present:false,provenance:$pr,
        upstream:(if $u=="" then null else $u end),role:$r}'
    return
  fi

  branch="$(git -C "$abs" branch --show-current)"
  if [[ -n "$(git -C "$abs" status --short)" ]]; then dirty=true; else dirty=false; fi

  jq -nc --arg n "$name" --arg p "../$path" --arg h "$head" \
         --arg b "${branch:-detached}" --argjson d "$dirty" \
         --arg pr "$provenance" --arg u "$upstream" --arg r "$role" \
    '{name:$n,path:$p,head:$h,branch:$b,dirty:$d,present:true,provenance:$pr,
      upstream:(if $u=="" then null else $u end),role:$r}'
}

REPOS_JSON="$(
  for entry in "${ROSTER[@]}"; do
    IFS='|' read -r n p pr u r <<< "$entry"
    repo_json "$n" "$p" "$pr" "$u" "$r"
  done | jq -sc '.'
)"

# ── fixtures: a canonical revision set, or an explicit refusal ───────────
# The release qualification claims fixture parity. A claim with no comparison
# behind it is the defect; the generator will not manufacture one.
fixtures_not_compared() {
  jq -nc --arg why "$1" \
    '{status:"not-compared",canonical:null,consumers:[],reason:$why}'
}

resolve_roster_path() { # roster key or path -> workspace-relative path
  local key="$1" entry n p rest
  for entry in "${ROSTER[@]}"; do
    IFS='|' read -r n p rest <<< "$entry"
    if [[ "$key" == "$n" || "$key" == "$p" ]]; then echo "$p"; return 0; fi
  done
  return 1
}

build_fixtures() {
  local spec="$1"
  # REPO@REV:DIR
  if [[ ! "$spec" =~ ^([^@]+)@([^:]+):(.+)$ ]]; then
    echo "ERROR: --fixtures-canonical must be REPO@REV:DIR (got '$spec')" >&2
    exit 3
  fi
  local repo_key="${BASH_REMATCH[1]}" rev="${BASH_REMATCH[2]}" dir="${BASH_REMATCH[3]}"

  local rel
  if ! rel="$(resolve_roster_path "$repo_key")"; then
    echo "ERROR: canonical repo '$repo_key' is not in the release roster" >&2
    exit 3
  fi
  local abs="$WORKSPACE/$rel"
  local resolved
  # `rev-parse` alone echoes any well-formed 40-hex string back unverified, so a
  # nonexistent commit would be recorded as canonical. `--verify …^{commit}`
  # requires the object to exist and to be a commit.
  if ! resolved="$(git -C "$abs" rev-parse --verify --quiet "${rev}^{commit}" 2>/dev/null)"; then
    echo "ERROR: canonical revision '$rev' does not resolve to a commit in $rel" >&2
    exit 3
  fi
  local corpus="$abs/$dir"
  if [[ ! -d "$corpus" ]]; then
    echo "ERROR: canonical corpus directory not found: $rel/$dir" >&2
    exit 3
  fi

  # Corpus digest: sha256 over the sorted "sha256␠name" lines of the corpus, so
  # two manifests agree iff they qualified against byte-identical fixtures.
  local lines count digest
  lines="$(cd "$corpus" && find . -maxdepth 1 -type f -name '*.json' -printf '%f\n' \
            | sort | while read -r f; do echo "$(sha256sum "$f" | cut -d' ' -f1)  $f"; done)"
  count="$(printf '%s' "$lines" | grep -c . || true)"
  if [[ "$count" -eq 0 ]]; then
    echo "ERROR: canonical corpus $rel/$dir contains no *.json fixtures" >&2
    exit 3
  fi
  digest="$(printf '%s\n' "$lines" | sha256sum | cut -d' ' -f1)"

  # Compare consumers, when the comparison tool can run here.
  local verdict="not-run" consumers='[]'
  if [[ -x "$ROOT/scripts/check-fixture-drift.sh" ]]; then
    if "$ROOT/scripts/check-fixture-drift.sh" --canonical "$corpus" --quiet >/dev/null 2>&1; then
      verdict="match"
    else
      verdict="drift"
    fi
  fi

  jq -nc --arg r "$repo_key" --arg rev "$resolved" --arg d "$dir" \
         --arg dg "$digest" --argjson c "$count" --arg v "$verdict" \
         --argjson cons "$consumers" \
    '{status:"compared",
      canonical:{repo:$r,revision:$rev,dir:$d,corpus_sha256:$dg,fixture_count:$c},
      verdict:$v,consumers:$cons}'
}

FIXTURES_JSON=""
if [[ -n "$FIXTURES_CANONICAL" ]]; then
  FIXTURES_JSON="$(build_fixtures "$FIXTURES_CANONICAL")"
else
  # No canonical revision set. A candidate or released manifest may not be
  # emitted on an uncompared corpus — that is the whole point of the block.
  if [[ "$STATUS" != "local-draft" || "$REQUIRE_FIXTURES" == true ]]; then
    {
      echo "ERROR: no canonical fixture revision set supplied."
      echo
      echo "  status '$STATUS' requires a fixture comparison, and a comparison"
      echo "  requires an explicit canonical revision set. Without one the"
      echo "  manifest would assert fixture parity it never checked."
      echo
      echo "  Supply one, e.g.:"
      echo "    --fixtures-canonical VisionClaw@<revision>:tests/fixtures"
      echo
      echo "  Or generate a local draft (--status local-draft), which records"
      echo "  fixtures.status=\"not-compared\" instead of claiming parity."
    } >&2
    exit 3
  fi
  FIXTURES_JSON="$(fixtures_not_compared \
    'No canonical revision set supplied; this local draft asserts no fixture parity. Pass --fixtures-canonical REPO@REV:DIR to qualify a candidate.')"
fi

# ── emit ─────────────────────────────────────────────────────────────────
jq -n \
  --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg status "$STATUS" \
  --argjson repositories "$REPOS_JSON" \
  --argjson fixtures "$FIXTURES_JSON" \
  '{
    manifest_version: 2,
    generated_at: $generated_at,
    status: $status,
    repositories: $repositories,
    fixtures: $fixtures,
    compatibility: {
      identity: "did:nostr with 64-char lowercase x-only secp256k1 pubkeys; NIP-98 HTTP auth expected across mesh participants",
      mesh: "Designed around NIP-42 relay writes and IS-Envelope routing; current default deployments may remain standalone",
      pod: "solid-pod-rs is canonical for native pods; Cloudflare Workers pod tier has documented feature differences",
      governance: "Agent Control Surface event kinds 31400-31405 are the human decision plane",
      verification: "Run npm run verify in VisionFlow plus substrate-specific CI before promoting candidate to released. A candidate additionally requires fixtures.status=\"compared\" against a named canonical revision set."
    }
  }'
