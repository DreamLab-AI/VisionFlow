#!/usr/bin/env bash
# ADR-2020: check-pele-pin.cjs passes on agreement, fails on drift, and --strict fails on a missing sibling.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
chk="$root/scripts/check-pele-pin.cjs"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
fails=0
expect() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1 (exit $2, want $3)"; fails=$((fails+1)); fi; }
d="$tmp/project/agentbox/skills/diagrams-as-code/scripts/vendor/pele"; mkdir -p "$d"
cp "$root/scripts/vendor/pele/VERSION" "$root/scripts/vendor/pele/pele.js" "$d/"
node "$chk" --workspace "$tmp" >/dev/null 2>&1; expect "agreeing sibling passes (others skipped)" $? 0
node "$chk" --workspace "$tmp" --strict >/dev/null 2>&1; expect "--strict fails on absent siblings" $? 1
sed -i 's/^pele [0-9][0-9.]*/pele 9.9.9/' "$d/VERSION"
node "$chk" --workspace "$tmp" >/dev/null 2>&1; expect "version drift fails" $? 1
cp "$root/scripts/vendor/pele/VERSION" "$d/VERSION"; echo "// tampered" >> "$d/pele.js"
node "$chk" --workspace "$tmp" >/dev/null 2>&1; expect "tampered pele.js fails" $? 1

# --- fixtures for the other readers, with a synthetic reference so the test needs no sibling checkout ---
sha() { sha256sum "$1" | cut -d' ' -f1; }
fixture() { # fresh workspace $w with every surface agreeing
  w="$tmp/w$((++n))"; mkdir -p "$w"; ref="$w/ref"; mkdir -p "$ref"
  echo "// pele" > "$ref/pele.js"
  q="$w/visionGraph/quartz/quartz"; mkdir -p "$q/static/pele/1.2.3/chunks" "$q/components/scripts"
  echo "chunk" > "$q/static/pele/1.2.3/chunks/a.js"
  (cd "$q/static/pele/1.2.3" && echo "$(sha chunks/a.js)  ./chunks/a.js" > SHA256SUMS)
  cat > "$ref/VERSION" <<EOT
pele 1.2.3
integrity sha512-AAAA
file     dist/pele.js
sha256   $(sha "$ref/pele.js")
esm_sums_sha256 $(sha "$q/static/pele/1.2.3/SHA256SUMS")
EOT
  cp "$ref/VERSION" "$q/static/pele/1.2.3/VERSION"
  echo 'const PELE_VERSION = "1.2.3";' > "$q/components/scripts/mermaid.inline.ts"
  mkdir -p "$w/project/agentbox/lib" "$w/dreamlab-ai-website"
  printf 'version = "1.2.3";\nhash = "sha512-AAAA";\nfileSha256 = "%s";\n' "$(sha "$ref/pele.js")" > "$w/project/agentbox/lib/pele.nix"
  echo '{"dependencies":{"pele":"1.2.3"}}' > "$w/dreamlab-ai-website/package.json"
  echo '{"packages":{"node_modules/pele":{"version":"1.2.3","integrity":"sha512-AAAA"}}}' > "$w/dreamlab-ai-website/package-lock.json"
}
run() { node "$chk" --workspace "$w" --reference "$ref/VERSION" >"$tmp/out" 2>&1; echo $?; }
n=0
fixture;                                                                   expect "synthetic fixture agrees" "$(run)" 0
fixture; sed -i 's/fileSha256 = "[0-9a-f]*"/fileSha256 = "'$(printf 0%.0s {1..64})'"/' "$w/project/agentbox/lib/pele.nix"; expect "pele.nix fileSha256 drift fails" "$(run)" 1
fixture; sed -i 's/version = "1.2.3"/version = "1.2.4"/' "$w/project/agentbox/lib/pele.nix"; expect "pele.nix version drift fails" "$(run)" 1
fixture; sed -i 's/AAAA/BBBB/' "$w/dreamlab-ai-website/package-lock.json"; expect "lockfile integrity drift fails" "$(run)" 1
fixture; echo tampered > "$q/static/pele/1.2.3/chunks/a.js";               expect "tampered Quartz chunk fails" "$(run)" 1
fixture; rm "$q/static/pele/1.2.3/SHA256SUMS";                             expect "missing SHA256SUMS fails without a stack trace" "$(run)" 1
grep -q "SHA256SUMS missing" "$tmp/out"; expect "missing SHA256SUMS is reported by name" $? 0
fixture; echo tampered > "$q/static/pele/1.2.3/chunks/a.js"; (cd "$q/static/pele/1.2.3" && echo "$(sha chunks/a.js)  ./chunks/a.js" > SHA256SUMS); expect "chunk swapped and SHA256SUMS regenerated fails (sums are pinned)" "$(run)" 1
fixture; mv "$q/static/pele/1.2.3" "$q/static/pele/1.3.0";                 expect "Quartz on a different version directory fails (not skipped)" "$(run)" 1
fixture; mkdir "$q/static/pele/1.0.0";                                     expect "two Quartz version directories fail" "$(run)" 1
fixture; echo 'const PELE_VERSION = "1.3.0";' > "$q/components/scripts/mermaid.inline.ts"; expect "mermaid.inline.ts serving another version fails" "$(run)" 1
exit $fails
