#!/usr/bin/env bash
# Lead check of one wave 3 worker: strict gate per topic, claim coverage
# against the gate's diagram count, then close the worker's tmux pane.
# Usage: check-w3.sh <worker-name> <area> <topic-prefix-regex>
set -u
W=/home/devuser/workspace/wt/VisionFlow-audit-wave3
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
name="$1"; area="$2"; pre="$3"
out="$SP/gate-$name.txt"
cd "$W"
: > "$out"
for t in $(ls "docs/diagrams/$area/" | grep -E "$pre"); do
  node scripts/diagram-index-gen.cjs docs/diagrams --only "$t" --check --cite-check --strict-citations --render > "$out.one" 2>&1
  rc=$?
  cat "$out.one" >> "$out"
  echo "$t exit=$rc $(grep -o 'rendered [0-9]*/[0-9]* diagrams' "$out.one") $(grep -o '[0-9]* verified at the declared revision' "$out.one")"
done
rm -f "$out.one"
python3 -I "$SP/claim-coverage.py" "docs/estate-review/evidence/2026-11-02/audit-wave3/claims-$name.json"
for p in $(tmux list-panes -t agentbox:2 -F '#{pane_id}'); do
  pid=$(tmux display -p -t "$p" '#{pane_pid}')
  n=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -o 'w3-[a-z0-9-]*' | head -1)
  if [ "$n" = "$name" ]; then tmux kill-pane -t "$p"; echo "closed $p ($n)"; fi
done
