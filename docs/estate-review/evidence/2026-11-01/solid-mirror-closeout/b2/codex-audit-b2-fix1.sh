#!/usr/bin/env bash
# Cross-vendor audit of B2 fix round 1 (forks). Usage: codex-audit-b2-fix1.sh
set -u
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication/crates/solid-pod-rs-forge/src
B=$SP/audit-b2-fix1-bundle.txt
{
  echo "You are an independent cross-vendor security auditor. Use NO tools; everything you need is below. Answer in UK English."
  echo "You audited Commit B2 before (your earlier answer is included). This is fix round 1, which answers that audit and an Opus security review (F1-F3)."
  echo "For each Must-fix item in the fix brief, report FIXED, PARTLY or BROKEN, with file:line evidence and a concrete failure scenario."
  echo "Then list any new material defect this round introduced, ranked by severity, and list filesystem-writer and other out-of-model findings separately."
  echo "End with exactly one line: VERDICT: SUPPORTED, PARTLY SUPPORTED or NOT SUPPORTED. The verdict concerns remote attackers only, for objects present at migration time."
  echo; echo "=== FIX BRIEF ==="; cat "$EV/b2-fix1-brief.md"
  echo; echo "=== OPUS SECURITY REVIEW OF B2 ==="; cat "$EV/review-b2-security.md"
  echo; echo "=== YOUR EARLIER ANSWER (B2) ==="; n=$(grep -n '^codex$' "$EV/audit-b2-codex.txt" | tail -1 | cut -d: -f1); tail -n +"${n:-1}" "$EV/audit-b2-codex.txt"
  echo; echo "=== WORKER REPORT (fix round 1) ==="; cat "$EV/b2/fix1/report.md"
  echo; echo "=== FIX ROUND 1 INTERDIFF (against the B2 patch) ==="; cat "$EV/b2/fix1/interdiff-vs-after-b2.patch"
  echo; echo "=== CURRENT migrate_fork.rs ==="; cat -n "$W/migrate_fork.rs"
  echo; echo "=== CURRENT forks.rs ==="; cat -n "$W/forks.rs"
  echo; echo "=== CURRENT lib.rs (lines 600-820) ==="; sed -n '600,820p' "$W/lib.rs" | cat -n
} > "$B"
echo "bundle bytes: $(wc -c < "$B")"
cd "$SP" && timeout 1500 codex exec -m gpt-6-astra --skip-git-repo-check --sandbox read-only - < "$B" > "$EV/audit-b2-fix1-codex.txt" 2>&1
echo "codex exit=$?" >> "$EV/audit-b2-fix1-codex.txt"
