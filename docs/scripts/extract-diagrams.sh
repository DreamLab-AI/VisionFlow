#!/bin/bash
# Extract actual ASCII flow diagrams (not directory trees)

files_to_check=(
"multi-agent-docker/TERMINAL_GRID.md"
"archive/reports/hive-mind-integration.md"
"explanations/architecture/integration-patterns.md"
"diagrams/client/state/state-management-complete.md"
"guides/features/ontology-sync-enhancement.md"
"guides/features/local-file-sync-strategy.md"
"guides/hierarchy-integration.md"
"DEVELOPER_JOURNEY.md"
"GETTING_STARTED_WITH_UNIFIED_DOCS.md"
)

cd /home/devuser/workspace/project/docs

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "=== $file ==="
        # Look for flow diagram patterns (boxes with arrows)
        if grep -q "┌.*┐" "$file" && grep -q "└.*┘" "$file" && (grep -q "│.*│" "$file" || grep -q "▼" "$file" || grep -q "─>" "$file"); then
            echo "✓ HAS FLOW DIAGRAM"
            # Show first diagram
            awk '/┌.*┐/{p=1} p{print} /└.*┘/{if(p)exit}' "$file" | head -30
        else
            echo "✗ No flow diagram (likely directory tree)"
        fi
        echo ""
    fi
done
