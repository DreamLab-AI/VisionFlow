#!/usr/bin/env bash
# Cross-vendor audit of the round-4 interdiff. Usage: codex-audit-a4.sh
set -u
EV=/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
B=$SP/audit-a4-bundle.txt
A3=$EV/audit-a3-codex.txt
n=$(grep -n '^codex$' "$A3" | tail -1 | cut -d: -f1)
{
  echo "=== ROUND-4 BRIEF ==="; cat "$EV/fix-a4-brief.md"
  echo "=== ROUND-4 REPORT ==="; cat "$EV/fix-a4/fix-a4-report.md"
  echo "=== YOUR ROUND-3 AUDIT ANSWER ==="; tail -n +"${n:-1}" "$A3"
  echo "=== ROUND-4 INTERDIFF ==="; cat "$EV/fix-a4/fix-a4-interdiff.patch"
} > "$B"
wc -c "$B"
cd "$SP" || exit 2
timeout 1500 codex exec -m gpt-6-astra --skip-git-repo-check --sandbox read-only - <<EOF > "$EV/audit-a4-codex.txt" 2>&1
You are the cross-vendor evidence auditor for a Rust security fix. Use no tools; everything you need is below. Answer in UK English, outcome first.

Round 4 claims to close the gaps you left open in round 3 on High 1 (writes into a repository with a mirror record) and High 2 (stamping a store that holds unreachable objects).

For each of High 1 and High 2 give a final verdict: FIXED, PARTLY FIXED, or NOT FIXED, with file:line from the interdiff. Then:
1. Does has_mirror_record now fail closed on every caller?
2. Are the reachability roots exactly heads and tags, and are alternates refused before marking?
3. Is the quarantine sweep now limited to stale directories, with nothing deleted after a failed push?
4. Do the writer tests assert the object set is unchanged?
5. Is any new regression introduced?
Then list anything else you find, ranked by severity. Say plainly if nothing material remains.

$(cat "$B")
EOF
echo "exit=$?" >> "$EV/audit-a4-codex.txt"
