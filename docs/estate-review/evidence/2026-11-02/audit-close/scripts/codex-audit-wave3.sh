#!/usr/bin/env bash
# Cross-vendor spot-check of a 12-panel sample from audit wave 3. Codex gets
# each panel with the source lines it cites, read at the topic's declared
# revision, on stdin, with no tools. Usage: codex-audit-wave3.sh
set -u
SP=/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad
D=/home/devuser/workspace/wt/VisionFlow-audit-wave3/docs/estate-review/evidence/2026-11-02/audit-wave3
OUT=$D/audit-wave3-codex.txt
PROMPT='You are an independent cross-vendor auditor of an estate code audit. Below are 12 diagram panels from a documentation corpus, each followed by the exact source lines it cites, read with git show at the revision the topic declares. Other reviewers have already marked each panel accurate or corrected it. For EACH panel, judge only from the source lines shown: does every mechanism claim in the panel that carries a citation match the cited lines? Answer per panel with one of: HOLDS, WRONG (say which claim and which line contradicts it), or CANNOT-JUDGE (the cited lines are not enough). Do not use tools. Do not judge uncited prose. Finish with a one-line count: N HOLDS, N WRONG, N CANNOT-JUDGE.'
{
  echo "$PROMPT"
  echo
  cat "$SP/wave3-sample-evidence.txt"
} | timeout 900 codex exec --skip-git-repo-check -m gpt-6-astra --sandbox read-only - > "$OUT" 2>&1
echo "exit=$?" >> "$OUT"
