#!/bin/bash
# ============================================================================
# DEPRECATED: Use scripts/launch.sh rebuild-agent instead
# ============================================================================
#
# This script is maintained for backwards compatibility only.
# The canonical build system is now: scripts/launch.sh
#
# Equivalent commands:
#   ./build-unified.sh              → ./scripts/launch.sh rebuild-agent
#   ./build-unified.sh --no-cache   → ./scripts/launch.sh rebuild-agent
#   ./build-unified.sh --skip-comfyui  → ./scripts/launch.sh rebuild-agent --skip-comfyui
#   ./build-unified.sh --comfyui-full  → ./scripts/launch.sh rebuild-agent --comfyui-full
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  REDIRECTING TO CANONICAL BUILD SYSTEM"
echo "============================================="
echo ""
echo "This script has been consolidated into:"
echo "  $PROJECT_ROOT/scripts/launch.sh rebuild-agent"
echo ""

# Map old flags to new flags
ARGS=""
for arg in "$@"; do
    case "$arg" in
        --no-cache)
            # --no-cache is now default for rebuild-agent
            ;;
        --skip-comfyui)
            ARGS="$ARGS --skip-comfyui"
            ;;
        --comfyui-full)
            ARGS="$ARGS --comfyui-full"
            ;;
        --version|-v)
            echo "Agentic Workstation Build System v3.0.0 (2026-01-31)"
            echo "Canonical: scripts/launch.sh rebuild-agent"
            exit 0
            ;;
        --help|-h)
            echo "DEPRECATED: Use scripts/launch.sh rebuild-agent instead"
            echo ""
            echo "Usage: ./scripts/launch.sh rebuild-agent [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-comfyui   Skip ComfyUI deployment check"
            echo "  --comfyui-full   Build ComfyUI with full open3d support"
            echo "  --skip-cachyos   Skip CachyOS container builds"
            echo ""
            exit 0
            ;;
    esac
done

echo "Executing: $PROJECT_ROOT/scripts/launch.sh rebuild-agent $ARGS"
echo ""

# Execute the canonical script
exec "$PROJECT_ROOT/scripts/launch.sh" rebuild-agent $ARGS
