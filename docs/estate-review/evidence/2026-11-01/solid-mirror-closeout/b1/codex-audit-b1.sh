#!/usr/bin/env bash
# Cross-vendor audit of Commit B1. Usage: codex-audit-b1.sh
set -u
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
W=/home/devuser/workspace/wt/solid-pod-rs-mirror-publication
B=$SP/audit-b1-bundle.txt
{
  echo "You are an independent cross-vendor evidence auditor. Do not use any tools; everything you need is below. Answer in English."
  echo "Task: follow the REVIEW BRIEF. For each numbered priority give FIXED-SAFE / GAP (with file:line and a concrete scenario) / NOT ASSESSABLE. Then list any other material defect. End with one line: VERDICT: SUPPORTED | PARTLY SUPPORTED | NOT SUPPORTED."
  echo "=== REVIEW BRIEF"; cat "$EV/review-b1-brief.md"
  echo "=== LOCKED DECISIONS 21-26"; sed -n '/^21\. /,$p' "$EV/publication-design.md"
  echo "=== WORKER REPORT"; cat "$EV/b1/b1-report.md"
  echo "=== FULL B1 DIFF against d6133e7"; cat "$EV/b1/after-c.patch"
  echo "=== CURRENT lib.rs (gate and h_git_smart)"; cat "$W/crates/solid-pod-rs-forge/src/lib.rs"
} > "$B"
wc -c "$B"
cd "$SP" && timeout 1800 codex exec --skip-git-repo-check -m gpt-6-astra - < "$B" > "$EV/audit-b1-codex.txt" 2>&1
echo "exit=$?" >> "$EV/audit-b1-codex.txt"
