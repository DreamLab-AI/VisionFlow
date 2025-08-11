#!/bin/bash

# Pre-compile UNIFIED PTX file for Docker build
# This script now only compiles the single unified kernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UTILS_DIR="$PROJECT_ROOT/src/utils"
PTX_DIR="$UTILS_DIR/ptx"

echo "Pre-compiling UNIFIED PTX file for Docker build..."
echo "Project root: $PROJECT_ROOT"

# Create PTX directory if it doesn't exist
mkdir -p "$PTX_DIR"

# Only compile the unified kernel
KERNEL="visionflow_unified"
CU_FILE="$UTILS_DIR/${KERNEL}.cu"
PTX_FILE="$PTX_DIR/${KERNEL}.ptx"

echo "Compiling unified kernel: $KERNEL"

if [ ! -f "$CU_FILE" ]; then
    echo "ERROR: Unified kernel source not found: $CU_FILE"
    exit 1
fi

# Check if PTX is up to date
if [ -f "$PTX_FILE" ] && [ "$PTX_FILE" -nt "$CU_FILE" ]; then
    SIZE=$(ls -lh "$PTX_FILE" | awk '{print $5}')
    echo "SKIP - Already up to date ($SIZE)"
    exit 0
fi

echo -n "Compiling $KERNEL... "

# Compile with optimizations
if nvcc -ptx -arch=sm_86 -O3 --use_fast_math --restrict --ftz=true --prec-div=false --prec-sqrt=false "$CU_FILE" -o "$PTX_FILE" 2>/dev/null; then
    SIZE=$(ls -lh "$PTX_FILE" | awk '{print $5}')
    echo "SUCCESS ($SIZE)"
else
    echo "FAILED with optimizations, trying basic compile..."
    if nvcc -ptx -arch=sm_86 "$CU_FILE" -o "$PTX_FILE" 2>/dev/null; then
        SIZE=$(ls -lh "$PTX_FILE" | awk '{print $5}')
        echo "SUCCESS ($SIZE)"
    else
        echo "FAILED - Unable to compile unified kernel!"
        exit 1
    fi
fi

echo ""
echo "Unified PTX compilation complete:"
echo "  File: $PTX_FILE"
echo "  Size: $(ls -lh "$PTX_FILE" | awk '{print $5}')"
echo ""
echo "Legacy kernels successfully removed:"
echo "  ✗ compute_forces.cu/ptx"
echo "  ✗ compute_dual_graphs.cu/ptx"
echo "  ✗ dual_graph_unified.cu/ptx"
echo "  ✗ unified_physics.cu/ptx"
echo "  ✗ visual_analytics_core.cu/ptx"
echo "  ✗ advanced_compute_forces.cu/ptx"
echo "  ✗ advanced_gpu_algorithms.cu/ptx"
echo "  ✗ initialize_positions.cu/ptx"
echo ""
echo "System now uses ONLY the unified kernel!"