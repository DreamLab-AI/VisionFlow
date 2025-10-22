#!/bin/bash
# PTX Compilation Verification Script

echo "========================================="
echo "PTX Compilation Verification"
echo "========================================="
echo ""

PTX_DIR="/home/devuser/workspace/project/src/utils/ptx"
FOUND=0
TOTAL=8

echo "Checking: $PTX_DIR"
echo ""

for ptx in dynamic_grid gpu_aabb_reduction gpu_clustering_kernels gpu_landmark_apsp ontology_constraints sssp_compact visionflow_unified visionflow_unified_stability; do
    if [ -f "$PTX_DIR/${ptx}.ptx" ]; then
        SIZE=$(ls -lh "$PTX_DIR/${ptx}.ptx" | awk '{print $5}')
        echo "✅ ${ptx}.ptx ($SIZE)"
        ((FOUND++))
    else
        echo "❌ ${ptx}.ptx (MISSING)"
    fi
done

echo ""
echo "Status: $FOUND/$TOTAL kernels compiled"
du -sh "$PTX_DIR" 2>/dev/null | awk '{print "Total Size:", $1}'

if [ $FOUND -eq $TOTAL ]; then
    echo "✅ All kernels compiled successfully!"
    exit 0
else
    echo "❌ Missing kernels!"
    exit 1
fi
