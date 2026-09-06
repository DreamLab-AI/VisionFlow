#!/usr/bin/env bash
# website-assets.test.sh — regression tests for the website asset inventory.
#
# The defect: website/build.sh staged repo image directories with
# `cp -r ... 2>/dev/null || true`. Every image copy could fail and the build
# still exited 0 and printed BUILD-COMPLETE, so a completed build certified
# nothing about the media in dist/ — and nothing declared which assets the page
# actually required.
#
#   bash tests/gates/website-assets.test.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TOOL="$REPO_ROOT/scripts/website-assets.mjs"
MANIFEST="$REPO_ROOT/website/assets.manifest.json"
DIST="$REPO_ROOT/website/dist"

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); printf '  FAIL %s\n     %s\n' "$1" "${2:-}"; }
eq()  { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1" "expected '$3', got '$2'"; fi; }

echo "website asset-inventory regression tests"
echo "========================================"

# ── 0. Build once ────────────────────────────────────────────────────────
echo
echo "[0] the site builds"
BUILD_OUT="$(bash "$REPO_ROOT/website/build.sh" 2>&1)"; RC=$?
eq "build exits 0" "$RC" "0"
if grep -q '^ASSET-INVENTORY-OK$' <<< "$BUILD_OUT"; then ok "build asserts the asset inventory"
else bad "build asserts the asset inventory" "no ASSET-INVENTORY-OK in build output"; fi
if grep -q '^BUILD-COMPLETE' <<< "$BUILD_OUT"; then ok "completion sentinel still emitted"
else bad "completion sentinel still emitted"; fi

# ── 1. The manifest declares a non-empty required set ────────────────────
echo
echo "[1] the inventory declares required assets"
REQ="$(jq -r '.required | length' "$MANIFEST")"
if (( REQ > 0 )); then ok "manifest declares $REQ required assets"
else bad "manifest declares required assets" "required is empty"; fi
OPT="$(jq -r '.optional | length' "$MANIFEST")"
if (( OPT > 0 )); then ok "manifest declares $OPT optional groups explicitly"
else bad "manifest declares optional groups"; fi

# ── 2. Every required asset is present after a build ─────────────────────
echo
echo "[2] every required asset exists in dist/ after a build"
MISSING=""
while read -r dest; do
  [[ -f "$DIST/$dest" ]] || MISSING="$MISSING $dest"
done < <(jq -r '.required[].dest' "$MANIFEST")
if [[ -z "$MISSING" ]]; then ok "all required assets present"
else bad "all required assets present" "missing:$MISSING"; fi

# ── 3. A MISSING REQUIRED ASSET FAILS THE BUILD ──────────────────────────
# The core fix. Previously any copy could fail silently.
echo
echo "[3] a missing required asset fails verification"
VICTIM="$(jq -r '.required[] | select(.dest | test("^img/showcase/")) | .dest' "$MANIFEST" | head -1)"
if [[ -z "$VICTIM" ]]; then VICTIM="$(jq -r '.required[-1].dest' "$MANIFEST")"; fi
BACKUP="$(mktemp)"
cp "$DIST/$VICTIM" "$BACKUP"
rm -f "$DIST/$VICTIM"
OUT="$(node "$TOOL" verify --receipt /tmp/asset-test-receipt.json 2>&1)"; RC=$?
eq "verify exits 1 when a required asset is absent" "$RC" "1"
if [[ "$OUT" == *"$VICTIM"* ]]; then ok "the failure names the missing asset ($VICTIM)"
else bad "the failure names the missing asset" "$OUT"; fi
if [[ "$OUT" == *"not publishable"* ]]; then ok "the failure says the candidate is not publishable"
else bad "the failure says the candidate is not publishable"; fi
mkdir -p "$(dirname "$DIST/$VICTIM")"; cp "$BACKUP" "$DIST/$VICTIM"; rm -f "$BACKUP"

# ── 4. An EMPTY required asset also fails ────────────────────────────────
echo
echo "[4] a zero-byte required asset fails verification"
cp "$DIST/$VICTIM" "$BACKUP.2" 2>/dev/null || cp "$DIST/$VICTIM" /tmp/victim.bak
: > "$DIST/$VICTIM"
node "$TOOL" verify --receipt /tmp/asset-test-receipt.json >/dev/null 2>&1
eq "verify exits 1 on an empty required asset" "$?" "1"
cp "${BACKUP}.2" "$DIST/$VICTIM" 2>/dev/null || cp /tmp/victim.bak "$DIST/$VICTIM"

# ── 5. The receipt records what was published ────────────────────────────
echo
echo "[5] the build receipt records bytes, counts and hashes"
node "$TOOL" verify --published-revision deadbeefdeadbeefdeadbeefdeadbeefdeadbeef >/dev/null 2>&1
R="$REPO_ROOT/website/build-receipt.json"
if [[ -f "$R" ]]; then ok "receipt written"; else bad "receipt written"; fi
if jq -e '.dist.file_count > 0' "$R" >/dev/null; then ok "records a file count"; else bad "records a file count"; fi
if jq -e '.dist.bytes > 0' "$R" >/dev/null; then ok "records a byte total"; else bad "records a byte total"; fi
if jq -e '.dist.tree_sha256 | test("^[0-9a-f]{64}$")' "$R" >/dev/null; then
  ok "records a dist tree digest"; else bad "records a dist tree digest"; fi
NH="$(jq -r '[.required_assets[] | select(.sha256 | test("^[0-9a-f]{64}$"))] | length' "$R")"
eq "every required asset carries a sha256" "$NH" "$REQ"
eq "records the published revision" \
   "$(jq -r '.revision.published' "$R")" "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
if jq -e '.manifest.sha256 | test("^[0-9a-f]{64}$")' "$R" >/dev/null; then
  ok "records the manifest hash it enforced"; else bad "records the manifest hash"; fi

# ── 6. Optional groups are recorded explicitly, present or absent ────────
echo
echo "[6] optional asset groups are recorded, not silently swallowed"
NOPT="$(jq -r '.staging.optional | length' "$R")"
eq "every optional group appears in the receipt" "$NOPT" "$OPT"
BADSTATE="$(jq -r '[.staging.optional[] | select(.status != "staged" and .status != "source-absent")] | length' "$R")"
eq "each optional group has an explicit status" "$BADSTATE" "0"

# ── 7. Restore a clean, correct build ────────────────────────────────────
echo
echo "[7] a clean rebuild verifies"
bash "$REPO_ROOT/website/build.sh" >/dev/null 2>&1
eq "rebuild exits 0" "$?" "0"

echo
echo "========================================"
echo "passed: $PASS   failed: $FAIL"
[[ $FAIL -eq 0 ]] && { echo "WEBSITE-ASSET-TESTS-OK"; exit 0; }
echo "WEBSITE-ASSET-TESTS-FAIL"; exit 1
