#!/usr/bin/env bash
# Lead check of one wave 2 worker: strict gate per topic, then claim coverage
# against the gate's diagram count.
# Usage: check-worker.sh <claims-file-basename> <area> <topic-prefix-regex>
set -u
W=/home/devuser/workspace/wt/VisionFlow-audit-wave2
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
cd "$W"
for t in $(ls "docs/diagrams/$2/" | grep -E "$3"); do
  out=$(node scripts/diagram-index-gen.cjs docs/diagrams --only "$t" --check --cite-check --strict-citations --render 2>&1)
  rc=$?
  echo "$t exit=$rc $(echo "$out" | grep -o '[0-9]* citations verified' | head -1) $(echo "$out" | grep -o 'rendered [0-9]*/[0-9]*' | head -1)"
done
python3 -I "$SP/claim-coverage.py" "docs/estate-review/evidence/2026-11-02/audit-wave2/$1"
