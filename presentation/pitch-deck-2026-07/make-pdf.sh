#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
mkdir -p jpg
for p in slides/slide-*.png; do
  n=$(basename "$p" .png)
  magick "$p" -resize 2200x -quality 82 "jpg/${n}.jpg"
done
img2pdf jpg/slide-*.jpg -o visionflow-pitch-deck-2026-07.pdf 2>/dev/null || magick jpg/slide-*.jpg visionflow-pitch-deck-2026-07.pdf
ls -la visionflow-pitch-deck-2026-07.pdf
