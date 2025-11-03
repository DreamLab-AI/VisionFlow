#!/bin/bash
# Documentation Cleanup Script
# Removes duplicate and consolidated files

set -e

echo "🗑️  Documentation Cleanup Script"
echo "================================"
echo ""

# Base directory
DOCS_DIR="/home/devuser/workspace/project/docs"

# Files to remove
REMOVE_FILES=(
    "$DOCS_DIR/IMPLEMENTATION_SUMMARY.md"
    "$DOCS_DIR/SEMANTIC_PHYSICS_IMPLEMENTATION.md"
    "$DOCS_DIR/HIERARCHICAL-VISUALIZATION-SUMMARY.md"
    "$DOCS_DIR/QUICK-INTEGRATION-GUIDE.md"
    "$DOCS_DIR/ARCHIVED_HIERARCHICAL_COLLAPSE_IMPLEMENTATION.md"
    "$DOCS_DIR/api/IMPLEMENTATION_SUMMARY.md"
    "$DOCS_DIR/api/QUICK_REFERENCE.md"
    "$DOCS_DIR/api/ontology-hierarchy-endpoint.md"
    "$DOCS_DIR/research/Quick_Reference_Implementation_Guide.md"
)

# Count files
TOTAL=${#REMOVE_FILES[@]}
REMOVED=0
SKIPPED=0

echo "Files to remove: $TOTAL"
echo ""

# Remove each file
for file in "${REMOVE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Removing: $(basename "$file")"
        rm "$file"
        ((REMOVED++))
    else
        echo "⏭️  Skipped (not found): $(basename "$file")"
        ((SKIPPED++))
    fi
done

echo ""
echo "================================"
echo "Summary:"
echo "  - Total files: $TOTAL"
echo "  - Removed: $REMOVED"
echo "  - Skipped: $SKIPPED"
echo ""
echo "✅ Cleanup complete!"
