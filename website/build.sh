#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# VisionFlow marketing site — pure static build.
# The site is hand-written HTML/CSS/JS with a self-contained WebGL2 mesh
# experience (static/js/mesh-webgl.js). No compile step, no bundler, no WASM.
#
# The build is copy-only (ADR-2002) but it is NOT unconditional. Every asset the
# page genuinely needs is declared in assets.manifest.json; the staging step
# fails loudly on a missing required asset instead of the old
# `cp -r ... 2>/dev/null || true`, which let every image copy fail while the
# build still reported success. Optional asset groups stay optional, but their
# presence or absence is recorded explicitly in the build receipt.

echo "==> Preparing dist/..."
rm -rf dist
mkdir -p dist

echo "==> Copying static assets..."
cp -r static/* dist/
# `static/*` deliberately does not glob dotfiles; tool state such as
# static/.claude-flow/ must never reach the published artefact.
rm -rf dist/.claude-flow

echo "==> Writing CNAME..."
echo "www.visionflow.info" > dist/CNAME

echo "==> Staging manifest assets (required + optional)..."
node ../scripts/website-assets.mjs stage

echo "==> Verifying asset inventory and writing build receipt..."
node ../scripts/website-assets.mjs verify

echo "==> Build complete. Output in dist/"
ls -la dist/

# Terminal completion sentinel — proves build.sh ran to completion from a bare
# `tail` of its own stdout (byte-identity target for dream-build-check.sh).
echo "BUILD-COMPLETE bytes: $(du -sb dist | cut -f1)"
