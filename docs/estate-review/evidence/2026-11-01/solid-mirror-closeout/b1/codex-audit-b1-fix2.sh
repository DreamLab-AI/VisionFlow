#!/usr/bin/env bash
# Cross-vendor audit of B1 fix round 2. Usage: codex-audit-b1-fix1.sh
set -u
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication/crates/solid-pod-rs-forge/src
B=$SP/audit-b1-fix2-bundle.txt
{
  echo "=== BRIEF"; cat "$EV/b1-fix2-brief.md"
  echo "=== YOUR EARLIER B1 ANSWER"; n=$(grep -n '^codex$' "$EV/audit-b1-fix1-codex.txt" | tail -1 | cut -d: -f1); tail -n +"$n" "$EV/audit-b1-fix1-codex.txt"
  echo "=== FIX REPORT"; cat "$EV/b1/fix2/report.md"
  echo "=== INTERDIFF (this round)"; cat "$EV/b1/fix2/interdiff-vs-after-fix1.patch"
  echo "=== CURRENT migrate.rs"; cat "$W/migrate.rs"
  echo "=== CURRENT lib.rs (gate + smart-http excerpts)"; grep -n -A40 'fn unpublished_mirror\|fn smart_http_path\|async fn h_git_smart' "$W/lib.rs"
} > "$B"
cd "$SP" && timeout 1500 codex exec --skip-git-repo-check -m gpt-6-astra --sandbox read-only "You are the cross-vendor evidence auditor for Commit B1 fix round 2 of a Rust repository (solid-pod-rs). Do not use tools; everything you need is on stdin. Follow the BRIEF. For each of your earlier items say FIXED, PARTLY FIXED or NOT FIXED with file:line, then list new material defects ranked by severity, and finish with a one-line verdict: SUPPORTED, PARTLY SUPPORTED or NOT SUPPORTED for the claim that no held object can be served after this round to a remote party without write access to the server filesystem (filesystem-writer attacks are recorded as out of scope in the BRIEF; list them separately, not in the verdict)." < "$B" > "$EV/audit-b1-fix2-codex.txt" 2>&1
echo "codex exit=$?" >> "$EV/audit-b1-fix2-codex.txt"
