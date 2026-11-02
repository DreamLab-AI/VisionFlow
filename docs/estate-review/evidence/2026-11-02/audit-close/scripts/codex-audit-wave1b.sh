#!/usr/bin/env bash
# Focused rerun of wave 1 samples 3 and 5 with complete evidence.
# Usage: codex-audit-wave1b.sh
set -u
W=/home/devuser/workspace/wt/VisionFlow-audit-wave1
D=$W/docs/estate-review/evidence/2026-11-02/audit-wave1
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
B=$SP/audit-wave1b-bundle.txt
S=/home/devuser/workspace/solid-pod-rs
V=/home/devuser/workspace/project
SR=9b6143ce55b2
VR=86cdd7635582
around() { # repo rev file line
  local a=$(( $4 > 4 ? $4 - 4 : 1 )) b=$(( $4 + 4 ))
  echo "--- $3 lines $a-$b (cited :$4)"
  git -C "$1" show "$2:$3" | sed -n "${a},${b}p" | nl -ba -v "$a" | sed 's/^/  /'
}
{
  echo "=== SAMPLE 3 (rerun): VisionClaw /ontology/derived write gating vs ADR-135 D7"
  echo "Claim: the derived-write route is gated by power_user only; no signed distiller"
  echo "envelope or distiller allow-list check exists on that path in VisionClaw."
  echo "--- route registration and handler (whole file, $VR)"
  git -C $V show $VR:src/handlers/ontology_derived_handler.rs | nl -ba | sed 's/^/  /'
  echo "--- repository-wide search at $VR in src/ and crates/ for signature or allow-list terms"
  git -C $V grep -n -i -E 'schnorr|bip-?340|verify_signature|envelope|distiller|allow_?list' $VR -- src crates \
    | grep -v -i 'test' | head -40 || echo "  (no matches)"
  echo "--- ADR-135 D7 text"
  git -C /home/devuser/workspace/loom show ba21a689a57d:$(git -C /home/devuser/workspace/loom ls-tree -r --name-only ba21a689a57d | grep -m1 'ADR-135') \
    | awk '/D7/{f=1} f{print; n++} n>25{exit}'
  echo
  echo "=== SAMPLE 5 (rerun): panel SP-10.1 with every cited line"
  awk '/^## SP-10.1 /,/^## SP-10.2 /' $W/docs/diagrams/solid-pod-rs/10-failure-modes-and-invariants.md
  for c in auth/replay.rs:275 ldp.rs:1949 multitenant.rs:199 oidc/mod.rs:184 oidc/mod.rs:549 provenance.rs:514 wac/parser.rs:40 wac/resolver.rs:153; do
    around $S $SR "crates/solid-pod-rs/src/${c%%:*}" "${c##*:}"
  done
} > "$B"
echo "bundle: $(wc -c < "$B") bytes"
codex exec -m gpt-6-astra --skip-git-repo-check \
  "You are an independent cross-vendor auditor. Stdin holds two claims from a source-tracing audit with their evidence at pinned revisions. Do not use tools. For each give SUPPORTED, PARTLY SUPPORTED or NOT SUPPORTED with brief reasoning grounded only in what is shown. For SAMPLE 5, judge each node label of the panel against its cited line and list any label the cited line does not support. End with one line: OVERALL <n>/2 supported." \
  < "$B" > "$D/audit-wave1b-codex.txt" 2>&1
echo "codex exit=$?"
n=$(grep -n '^codex$' "$D/audit-wave1b-codex.txt" | tail -1 | cut -d: -f1)
tail -n +"$n" "$D/audit-wave1b-codex.txt" | head -50
