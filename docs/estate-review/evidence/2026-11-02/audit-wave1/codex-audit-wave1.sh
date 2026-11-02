#!/usr/bin/env bash
# Cross-vendor spot-check of a sample of audit wave 1 claims.
# Codex gets the claims and the source lines at the pinned revisions on stdin,
# with no tools. Usage: codex-audit-wave1.sh
set -u
W=/home/devuser/workspace/wt/VisionFlow-audit-wave1
D=$W/docs/estate-review/evidence/2026-11-02/audit-wave1
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
B=$SP/audit-wave1-bundle.txt
S=/home/devuser/workspace/solid-pod-rs
V=/home/devuser/workspace/project
SR=9b6143ce55b2
VR=86cdd7635582
show() { echo "--- $1@$2 lines $3"; git -C "$1" show "$2:$4" | sed -n "$3p" | cat -n | sed "s/^/  /"; }
{
  echo "=== SAMPLE 1: Solid ADR-2010 finding F1 (from 2026-11-02-solid-forge-adrs.md)"
  python3 -I -c "
s=open('$W/docs/estate-review/2026-11-02-solid-forge-adrs.md').read()
i=s.find('### F1'); j=s.find('### F2'); print(s[i:j])"
  show $S $SR 2296,2312 crates/solid-pod-rs-forge/src/mirror.rs
  show $S $SR 5015,5042 crates/solid-pod-rs-server/src/lib.rs
  echo "=== SAMPLE 2: LWS CID iat window (lead-confirmed bug)"
  echo "Claim: validate_time_bounds never rejects an old or future iat, because the two"
  echo "saturating_sub checks are joined with && and at most one can be non-zero."
  echo "Claim: nothing outside the module constructs LwsCidVerifier at this revision."
  show $S $SR 325,352 crates/solid-pod-rs/src/auth/lws_cid.rs
  echo "=== SAMPLE 3: loom ADR-135 D7 (from 2026-11-02-loom-and-dream-machine-adrs.md)"
  echo "Claim: VisionClaw's /ontology/derived write door checks power_user and nothing"
  echo "else; ADR-135 D7 requires a signed distiller envelope plus an allow-list."
  show $V $VR 25,40 src/handlers/ontology_derived_handler.rs
  show $V $VR 180,195 src/handlers/ontology_derived_handler.rs
  echo "=== SAMPLE 4: panel SP-04.7 (topic text as now committed)"
  awk '/^## SP-04.7/,/^## SP-04.8/' $W/docs/diagrams/solid-pod-rs/04-wac-authorisation.md
  show $S $SR 325,350 crates/solid-pod-rs/src/wac/evaluator.rs
  echo "=== SAMPLE 5: panel SP-10.1 (topic text as now committed)"
  awk '/^## SP-10.1 /,/^## SP-10.2 /' $W/docs/diagrams/solid-pod-rs/10-failure-modes-and-invariants.md
  show $S $SR 115,125 crates/solid-pod-rs/src/auth/nip98.rs
  show $S $SR 186,196 crates/solid-pod-rs-server/src/lib.rs
  echo "=== SAMPLE 6: stale rebind of ADR-2164 (from 2026-11-02-stale-claim-rebind.md)"
  python3 -I -c "
s=open('$W/docs/estate-review/2026-11-02-stale-claim-rebind.md').read()
i=s.find('ADR-2164'); print(s[i:i+2500])"
  show $V $VR 728,745 xr-client/scripts/graph_scene.gd
  show $V $VR 49,53 xr-client/project.godot
} > "$B"
echo "bundle: $(wc -c < "$B") bytes"
codex exec -m gpt-6-astra --skip-git-repo-check \
  "You are an independent cross-vendor auditor. Everything you need is on stdin: six claims from a source-tracing audit, each followed by the exact source lines at the pinned revision (numbered relative to the shown range; the header gives the absolute start line). Do not use tools. For each sample give a verdict SUPPORTED, PARTLY SUPPORTED or NOT SUPPORTED, one or two sentences of reasoning grounded only in the shown lines, and note anything the claim overstates or that the shown lines cannot settle. End with one line: OVERALL <n>/6 supported." \
  < "$B" > "$D/audit-wave1-codex.txt" 2>&1
echo "codex exit=$?"
tail -25 "$D/audit-wave1-codex.txt"
