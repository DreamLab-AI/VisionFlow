#!/usr/bin/env bash
# Cross-vendor audit of Commit B2 (forks). Usage: codex-audit-b2.sh
set -u
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication/crates/solid-pod-rs-forge/src
B=$SP/audit-b2-bundle.txt
{
  echo "You are an independent cross-vendor security auditor. Use NO tools; everything you need is below. Answer in UK English."
  echo "Report each 'Check especially' item from the brief as FIXED/HOLDS, PARTLY or BROKEN, with file:line evidence and a concrete failure scenario."
  echo "List new material defects ranked by severity, then the filesystem-writer and other out-of-model findings separately."
  echo "End with exactly one line: VERDICT: SUPPORTED, PARTLY SUPPORTED or NOT SUPPORTED. The verdict concerns remote attackers only."
  echo; echo "=== REVIEW BRIEF ==="; cat "$EV/review-b2-brief.md"
  echo; echo "=== WORKER REPORT ==="; cat "$EV/b2/report.md"
  echo; echo "=== B2 DIFF (against local main 8ba3372) ==="; cat "$EV/b2/after-b2.patch"
  echo; echo "=== CURRENT migrate.rs ==="; cat -n "$W/migrate.rs"
  echo; echo "=== CURRENT lib.rs (lines 140-340, 600-820) ==="; sed -n '140,340p;600,820p' "$W/lib.rs" | cat -n
} > "$B"
echo "bundle bytes: $(wc -c < "$B")"
cd "$SP" && timeout 1500 codex exec -m gpt-6-astra --skip-git-repo-check --sandbox read-only - < "$B" > "$EV/audit-b2-codex.txt" 2>&1
echo "codex exit=$?" >> "$EV/audit-b2-codex.txt"
