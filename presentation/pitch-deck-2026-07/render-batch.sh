#!/usr/bin/env bash
# Sequential batch render with one retry per slide.
set -u
[ -f "$HOME/.claude/.env" ] && set -a && . "$HOME/.claude/.env" && set +a
NB=/home/devuser/workspace/project/agentbox/skills/art/tools/nb-generate.cjs
cd "$(dirname "$0")"
for p in prompt-files/slide-*.prompt; do
  n=$(basename "$p" .prompt)
  out="slides/${n}.png"
  [ -s "$out" ] && { echo "SKIP $n (exists)"; continue; }
  for attempt in 1 2; do
    echo "RENDER $n attempt $attempt $(date +%H:%M:%S)"
    if node "$NB" --prompt "$(cat "$p")" --out "$out" --model gemini-3-pro-image --size 4K --aspect 16:9 && [ -s "$out" ]; then
      echo "OK $n"; break
    else
      echo "FAIL $n attempt $attempt"; rm -f "$out"; sleep 10
    fi
  done
done
echo "BATCH DONE $(date +%H:%M:%S)"
