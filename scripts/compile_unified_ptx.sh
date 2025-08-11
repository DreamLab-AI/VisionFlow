#!/bin/bash

# Streamlined PTX compilation for the unified kernel only
# This replaces the need to compile 7 separate kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UTILS_DIR="$PROJECT_ROOT/src/utils"
PTX_DIR="$UTILS_DIR/ptx"

echo "========================================="
echo "VisionFlow Unified PTX Compiler"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Create PTX directory if it doesn't exist
mkdir -p "$PTX_DIR"

# The single unified kernel that replaces all others
UNIFIED_KERNEL="visionflow_unified"
CU_FILE="$UTILS_DIR/${UNIFIED_KERNEL}.cu"
PTX_FILE="$PTX_DIR/${UNIFIED_KERNEL}.ptx"

if [ ! -f "$CU_FILE" ]; then
    echo "ERROR: Unified kernel source not found at $CU_FILE"
    echo "Please ensure visionflow_unified.cu exists"
    exit 1
fi

# Check if PTX needs recompilation
if [ -f "$PTX_FILE" ] && [ "$PTX_FILE" -nt "$CU_FILE" ]; then
    echo "✓ PTX is up to date"
    ls -lh "$PTX_FILE"
    exit 0
fi

echo "Compiling unified kernel..."
echo "Source: $CU_FILE"
echo "Target: $PTX_FILE"
echo ""

# Compile with optimizations
if nvcc -ptx \
    -arch=sm_86 \
    -O3 \
    --use_fast_math \
    --restrict \
    --ftz=true \
    --prec-div=false \
    --prec-sqrt=false \
    "$CU_FILE" \
    -o "$PTX_FILE" 2>&1; then
    
    SIZE=$(ls -lh "$PTX_FILE" | awk '{print $5}')
    echo "✓ SUCCESS - Unified kernel compiled ($SIZE)"
    echo ""
    echo "The following legacy kernels are now obsolete:"
    echo "  - compute_forces.cu"
    echo "  - compute_dual_graphs.cu"
    echo "  - dual_graph_unified.cu"
    echo "  - unified_physics.cu"
    echo "  - visual_analytics_core.cu"
    echo "  - advanced_compute_forces.cu (was failing)"
    echo "  - advanced_gpu_algorithms.cu (was failing)"
    echo ""
    echo "All functionality is now in: visionflow_unified.ptx"
else
    echo "✗ FAILED to compile unified kernel"
    echo "Please check for compilation errors above"
    exit 1
fi

echo ""
echo "========================================="
echo "PTX compilation complete!"
echo "========================================="