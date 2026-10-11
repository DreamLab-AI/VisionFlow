#!/usr/bin/env bash
# release-manifest.test.sh — regression tests for scripts/generate-release-manifest.sh
#
# Pins the two defects closed on 2026-09-05 (ADR-2006 closeout): a roster that
# covered six of fourteen repositories, and a manifest that could be promoted
# to candidate while asserting fixture parity nothing had checked.
#
#   bash tests/gates/release-manifest.test.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
GEN="$REPO_ROOT/scripts/generate-release-manifest.sh"
INVENTORY="$REPO_ROOT/docs/estate-review/evidence/adr-inventory.json"

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL %s\n     %s\n' "$1" "${2:-}"; }
eq()  { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1" "expected '$3', got '$2'"; fi; }

echo "generate-release-manifest.sh regression tests"
echo "============================================="

# ── 1. The roster covers every inventoried repository ────────────────────
echo
echo "[1] the roster covers all inventoried repositories"
DRAFT="$(bash "$GEN" 2>/dev/null)"; RC=$?
eq "generator exits 0 for a local draft" "$RC" "0"
COUNT="$(jq -r '.repositories | length' <<< "$DRAFT")"
WANT="$(jq -r '.repos | length' "$INVENTORY")"
eq "roster size matches the ADR inventory" "$COUNT" "$WANT"
eq "manifest_version is 2" "$(jq -r '.manifest_version' <<< "$DRAFT")" "2"
eq "current release roster contains sixteen repositories" "$COUNT" "16"
eq "inventory contains the same sixteen identities" "$WANT" "16"
eq "inventory identities are unique" "$(jq '[.repos[].repo] | unique | length' "$INVENTORY")" "$WANT"
eq "repository names are unique" "$(jq '[.repositories[].name] | unique | length' <<< "$DRAFT")" "$COUNT"
eq "repository paths are unique" "$(jq '[.repositories[].path] | unique | length' <<< "$DRAFT")" "$COUNT"

for name in ontomark ontomark-obsidian; do
  eq "$name appears exactly once at its workspace path" \
    "$(jq --arg n "$name" '[.repositories[] | select(.name == $n and .path == "../"+$n)] | length' <<< "$DRAFT")" "1"
  eq "$name names its public producer repository" \
    "$(jq -r --arg n "$name" '.repositories[] | select(.name == $n) | .repository_url' <<< "$DRAFT")" \
    "https://github.com/DreamLab-AI/$name"
  eq "$name inventory URL agrees with the generator" \
    "$(jq -r --arg n "$name" '.repos[] | select(.repo == $n) | .repository_url' "$INVENTORY")" \
    "$(jq -r --arg n "$name" '.repositories[] | select(.name == $n) | .repository_url' <<< "$DRAFT")"
done

# Every inventoried repo path appears in the roster.
MISSING=""
while read -r p; do
  jq -e --arg p "../$p" '.repositories[] | select(.path == $p)' <<< "$DRAFT" >/dev/null \
    || MISSING="$MISSING $p"
done < <(jq -r '.repos[].repo' "$INVENTORY")
if [[ -z "$MISSING" ]]; then ok "every inventoried path is present"
else bad "every inventoried path is present" "absent:$MISSING"; fi

# ── 2. Provenance is declared for every repository ───────────────────────
echo
echo "[2] every repository declares its provenance"
BADPROV="$(jq -r '[.repositories[] | select((.provenance|type) != "string"
             or ([.provenance] | inside(["first-party","imported","upstream"]) | not))] | length' <<< "$DRAFT")"
eq "no repository lacks a valid provenance flag" "$BADPROV" "0"
eq "first-party repository count includes the two Ontomark producers" \
   "$(jq '[.repositories[] | select(.provenance=="first-party")] | length' <<< "$DRAFT")" "12"
eq "imported repositories are flagged" \
   "$(jq -r '[.repositories[] | select(.provenance=="imported")] | length' <<< "$DRAFT")" "2"
eq "upstream repositories are flagged" \
   "$(jq -r '[.repositories[] | select(.provenance=="upstream")] | length' <<< "$DRAFT")" "2"
UPSTREAM_NULL="$(jq -r '[.repositories[] | select(.provenance=="upstream" and .upstream==null)] | length' <<< "$DRAFT")"
eq "every upstream repository names its upstream" "$UPSTREAM_NULL" "0"

# ── 3. A local draft cannot pretend it compared fixtures ─────────────────
echo
echo "[3] a local draft with no canonical revision set says so"
eq "fixtures.status is not-compared" "$(jq -r '.fixtures.status' <<< "$DRAFT")" "not-compared"
eq "no canonical block is fabricated"  "$(jq -r '.fixtures.canonical' <<< "$DRAFT")" "null"
if jq -e '.fixtures.reason | length > 0' <<< "$DRAFT" >/dev/null; then
  ok "the gap is stated in the artefact"
else bad "the gap is stated in the artefact" "fixtures.reason empty"; fi

# ── 4. A CANDIDATE WITHOUT A CANONICAL REVISION SET IS REFUSED ───────────
# The core fix: fixture comparison must fail when no canonical revision set is
# supplied, rather than the manifest asserting parity in prose.
echo
echo "[4] a candidate without a canonical revision set is refused"
ERR="$(bash "$GEN" --status candidate 2>&1 >/dev/null)"; RC=$?
eq "exit code is 3" "$RC" "3"
if [[ "$ERR" == *"no canonical fixture revision set supplied"* ]]; then
  ok "the refusal names the missing input"
else bad "the refusal names the missing input" "stderr: $ERR"; fi

echo
echo "[4b] --require-fixtures refuses even a local draft"
bash "$GEN" --require-fixtures >/dev/null 2>&1; RC=$?
eq "exit code is 3" "$RC" "3"

echo
echo "[4c] a released manifest without a canonical revision set is refused"
bash "$GEN" --status released >/dev/null 2>&1; RC=$?
eq "exit code is 3" "$RC" "3"

# ── 5. An unresolvable canonical revision set is refused ─────────────────
echo
echo "[5] a canonical revision set that does not resolve is refused"
bash "$GEN" --status candidate --fixtures-canonical 'VisionClaw@0000000000000000000000000000000000000000:tests/fixtures' >/dev/null 2>&1
eq "unknown revision exits 3" "$?" "3"
bash "$GEN" --status candidate --fixtures-canonical 'NotARepo@HEAD:tests/fixtures' >/dev/null 2>&1
eq "unknown repository exits 3" "$?" "3"
bash "$GEN" --status candidate --fixtures-canonical 'VisionClaw@HEAD:no/such/dir' >/dev/null 2>&1
eq "missing corpus directory exits 3" "$?" "3"
bash "$GEN" --status candidate --fixtures-canonical 'malformed-spec' >/dev/null 2>&1
eq "malformed spec exits 3" "$?" "3"

# ── 6. A supplied canonical revision set is resolved and recorded ────────
echo
echo "[6] a valid canonical revision set produces a comparison record"
CORPUS_REPO="$(jq -r '.repositories[] | select(.name=="VisionClaw") | .path' <<< "$DRAFT")"
if [[ -d "$REPO_ROOT/$CORPUS_REPO/tests/fixtures" ]]; then
  CAND="$(bash "$GEN" --status candidate --fixtures-canonical 'VisionClaw@HEAD:tests/fixtures' 2>/dev/null)"; RC=$?
  eq "generator exits 0" "$RC" "0"
  eq "fixtures.status is compared" "$(jq -r '.fixtures.status' <<< "$CAND")" "compared"
  REV="$(jq -r '.fixtures.canonical.revision' <<< "$CAND")"
  if [[ "$REV" =~ ^[0-9a-f]{40}$ ]]; then ok "HEAD is resolved to a concrete commit"
  else bad "HEAD is resolved to a concrete commit" "got '$REV'"; fi
  DG="$(jq -r '.fixtures.canonical.corpus_sha256' <<< "$CAND")"
  if [[ "$DG" =~ ^[0-9a-f]{64}$ ]]; then ok "a corpus digest is recorded"
  else bad "a corpus digest is recorded" "got '$DG'"; fi
  if jq -e '.fixtures.canonical.fixture_count > 0' <<< "$CAND" >/dev/null; then
    ok "the fixture count is non-zero"
  else bad "the fixture count is non-zero"; fi
  V="$(jq -r '.fixtures.verdict' <<< "$CAND")"
  case "$V" in match|drift|not-run) ok "verdict is one of match/drift/not-run ($V)" ;;
    *) bad "verdict is valid" "got '$V'" ;; esac
else
  echo "  skip  VisionClaw corpus not present in this environment"
fi

# ── 7. Output validates against the committed schema ─────────────────────
echo
echo "[7] the generated manifest validates against the committed schema"
SCHEMA="$REPO_ROOT/docs/releases/ecosystem-release.schema.json"
if command -v npx >/dev/null 2>&1 && npx --no-install ajv --version >/dev/null 2>&1; then
  echo "$DRAFT" > /tmp/manifest-draft.json
  npx --no-install ajv validate -s "$SCHEMA" -d /tmp/manifest-draft.json >/dev/null 2>&1 \
    && ok "ajv validates the draft" || bad "ajv validates the draft"
else
  # No validator installed: assert the schema's own hard constraints directly,
  # so this test still means something in a bare environment.
  MIN="$(jq -r '.properties.repositories.minItems' "$SCHEMA")"
  if (( COUNT >= MIN )); then ok "roster meets schema minItems ($MIN)"
  else bad "roster meets schema minItems" "$COUNT < $MIN"; fi
  eq "schema pins manifest_version 2" "$(jq -r '.properties.manifest_version.const' "$SCHEMA")" "2"
  if jq -e '.required | index("fixtures")' "$SCHEMA" >/dev/null; then
    ok "schema requires the fixtures block"
  else bad "schema requires the fixtures block"; fi
  BADHEAD="$(jq -r '[.repositories[] | select(.head | test("^[0-9a-f]{40}$") | not)] | length' <<< "$DRAFT")"
  eq "every head is a 40-char sha" "$BADHEAD" "0"
fi

# A fixed roster must retain absent producers and resolve their real Git state
# when present. These fixtures are local and contain no estate corpus data.
echo
echo "[8] new producer entries survive missing checkouts and resolve present ones"
TEST_WORKSPACE="$(mktemp -d)"
trap 'rm -rf "$TEST_WORKSPACE"' EXIT
ABSENT="$(bash "$GEN" --workspace "$TEST_WORKSPACE" 2>/dev/null)"; RC=$?
eq "an empty workspace still produces a draft" "$RC" "0"
eq "both absent Ontomark producers are recorded, not omitted" \
  "$(jq '[.repositories[] | select((.name=="ontomark" or .name=="ontomark-obsidian") and .present==false and .branch=="missing" and .dirty==true and .head=="0000000000000000000000000000000000000000")] | length' <<< "$ABSENT")" "2"
for name in ontomark ontomark-obsidian; do
  mkdir -p "$TEST_WORKSPACE/$name/tests/fixtures"
  git -C "$TEST_WORKSPACE/$name" init -q
  printf '{"synthetic":true}\n' > "$TEST_WORKSPACE/$name/tests/fixtures/roster.json"
  git -C "$TEST_WORKSPACE/$name" add tests/fixtures/roster.json
  git -C "$TEST_WORKSPACE/$name" -c user.name='Release gate' -c user.email='release-gate@example.invalid' commit -qm 'Synthetic roster fixture'
done
printf 'dirty fixture\n' > "$TEST_WORKSPACE/ontomark-obsidian/untracked.txt"
PRESENT="$(bash "$GEN" --workspace "$TEST_WORKSPACE" 2>/dev/null)"; RC=$?
eq "the synthetic producer workspace generates successfully" "$RC" "0"
for name in ontomark ontomark-obsidian; do
  eq "$name records the resolved checkout commit" \
    "$(jq -r --arg n "$name" '.repositories[] | select(.name==$n) | .head' <<< "$PRESENT")" \
    "$(git -C "$TEST_WORKSPACE/$name" rev-parse HEAD)"
  eq "$name is recorded as present" \
    "$(jq -r --arg n "$name" '.repositories[] | select(.name==$n) | .present' <<< "$PRESENT")" "true"
done
eq "a clean core checkout stays clean" "$(jq -r '.repositories[] | select(.name=="ontomark") | .dirty' <<< "$PRESENT")" "false"
eq "plugin untracked work is recorded as dirty" "$(jq -r '.repositories[] | select(.name=="ontomark-obsidian") | .dirty' <<< "$PRESENT")" "true"
NEW_CANONICAL="$(bash "$GEN" --workspace "$TEST_WORKSPACE" --status candidate --fixtures-canonical 'ontomark@HEAD:tests/fixtures' 2>/dev/null)"; RC=$?
eq "the core is usable as an explicitly named canonical fixture producer" "$RC" "0"
eq "canonical fixture revision resolves to the core commit" \
  "$(jq -r '.fixtures.canonical.revision' <<< "$NEW_CANONICAL")" \
  "$(git -C "$TEST_WORKSPACE/ontomark" rev-parse HEAD)"
eq "canonical fixture count is observed" "$(jq -r '.fixtures.canonical.fixture_count' <<< "$NEW_CANONICAL")" "1"
eq "schema roster minimum equals current inventory size" \
  "$(jq -r '.properties.repositories.minItems' "$SCHEMA")" "$WANT"
eq "schema roster minimum includes both new producers" \
  "$(jq -r '.properties.repositories.minItems' "$SCHEMA")" "16"
eq "schema permits the public producer URL" \
  "$(jq -r '.properties.repositories.items.properties.repository_url.format' "$SCHEMA")" "uri"

# ── summary ──────────────────────────────────────────────────────────────
echo
echo "============================================="
echo "passed: $PASS   failed: $FAIL"
[[ $FAIL -eq 0 ]] && { echo "RELEASE-MANIFEST-TESTS-OK"; exit 0; }
echo "RELEASE-MANIFEST-TESTS-FAIL"; exit 1
