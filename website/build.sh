#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# VisionFlow marketing site — pure static build.
# The site is hand-written HTML/CSS/JS with a self-contained WebGL2 mesh
# experience (static/js/mesh-webgl.js). No compile step, no bundler, no WASM.

echo "==> Preparing dist/..."
rm -rf dist
mkdir -p dist

echo "==> Copying static assets..."
cp -r static/* dist/

echo "==> Writing CNAME..."
echo "www.visionflow.info" > dist/CNAME

echo "==> Copying repo images..."
mkdir -p dist/img
cp -r ../assets/diagrams/*    dist/img/ 2>/dev/null || true
cp -r ../assets/generated/*   dist/img/ 2>/dev/null || true
cp -r ../assets/heroes/*      dist/img/ 2>/dev/null || true
cp -r ../assets/screenshots/* dist/img/ 2>/dev/null || true

echo "==> Build complete. Output in dist/"
ls -la dist/
