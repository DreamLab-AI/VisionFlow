#!/bin/bash

# Link Audit Script for VisionFlow Documentation
# This script checks all internal markdown links and identifies broken ones

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
total_links=0
broken_links=0
fixed_links=0

echo "VisionFlow Documentation Link Audit"
echo "=================================="
echo

# Find all markdown files and process them
find /workspace/ext/docs -name "*.md" -not -path "/workspace/ext/docs/_archive/*" | while read -r file; do
    echo "Checking: $(basename "$file")"
    
    # Extract all markdown links with relative paths
    grep -n '\[[^]]*\](\.[^)]*\.md)' "$file" | while IFS=: read -r line_num match; do
        # Extract the link part
        link=$(echo "$match" | grep -o '\[[^]]*\]([^)]*)' | sed 's/.*(\([^)]*\)).*/\1/')
        
        if [ -n "$link" ]; then
            # Convert relative path to absolute path
            current_dir=$(dirname "$file")
            target_file=$(realpath -m "$current_dir/$link")
            
            # Check if target exists
            if [ ! -f "$target_file" ]; then
                echo -e "  ${RED}BROKEN${NC}: Line $line_num - $link"
                echo "    File: $file"
                echo "    Expected: $target_file"
                ((broken_links++))
            else
                echo -e "  ${GREEN}OK${NC}: $link"
            fi
            ((total_links++))
        fi
    done
    echo
done

echo "=================================="
echo "Link Audit Summary:"
echo "Total links checked: $total_links"
echo -e "Broken links found: ${RED}$broken_links${NC}"
echo -e "Status: $((total_links - broken_links)) links OK"