#!/bin/bash
# Analyze ASCII patterns to distinguish tables from diagrams

echo "# ASCII Pattern Analysis Report"
echo "Generated: $(date)"
echo ""
echo "## Summary"
echo ""

total_files=0
table_files=0
diagram_files=0

for file in $(find /home/devuser/workspace/project/docs -name "*.md" -type f); do
    if grep -q "[┌├└│─┬┤┴┼━═╔╗╚╝║╠╣╦╩╬]" "$file"; then
        total_files=$((total_files + 1))

        # Check if it's primarily tables (has table headers | --- |)
        if grep -q "^|.*|.*|$" "$file" && grep -q "^|.*-.*-.*|$" "$file"; then
            # Likely markdown tables
            table_files=$((table_files + 1))
        else
            # Likely ASCII diagrams
            diagram_files=$((diagram_files + 1))
            echo "DIAGRAM: ${file#/home/devuser/workspace/project/docs/}"
        fi
    fi
done

echo ""
echo "Total files with box-drawing chars: $total_files"
echo "Files with markdown tables: $table_files"
echo "Files with ASCII diagrams needing conversion: $diagram_files"
