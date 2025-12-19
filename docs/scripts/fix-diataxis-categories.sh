#!/bin/bash

# Diataxis Framework Category Fixer
# Automatically corrects category field in YAML frontmatter to match directory location

set -e

DOCS_ROOT="/home/devuser/workspace/project/docs"
REPORT_FILE="$DOCS_ROOT/scripts/diataxis-compliance-report.txt"
FIXES_APPLIED=0

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "Diataxis Framework Category Fixer"
echo "=========================================="
echo ""

# Check if report exists
if [ ! -f "$REPORT_FILE" ]; then
    echo "Error: Report file not found. Run check-diataxis-compliance.sh first."
    exit 1
fi

# Function to fix category in a file
fix_category() {
    local file="$1"
    local new_category="$2"
    local old_category="$3"

    # Check if file has frontmatter
    if ! head -1 "$file" | grep -q "^---$"; then
        echo "  Skipping $file - no frontmatter"
        return 1
    fi

    # Create temporary file
    local tmpfile=$(mktemp)

    # Process the file
    local in_frontmatter=0
    local category_replaced=0

    while IFS= read -r line; do
        if [ "$line" = "---" ]; then
            if [ $in_frontmatter -eq 0 ]; then
                in_frontmatter=1
                echo "$line" >> "$tmpfile"
            else
                in_frontmatter=0
                # If we didn't replace category, add it before closing frontmatter
                if [ $category_replaced -eq 0 ]; then
                    echo "category: $new_category" >> "$tmpfile"
                    category_replaced=1
                fi
                echo "$line" >> "$tmpfile"
            fi
        elif [ $in_frontmatter -eq 1 ] && echo "$line" | grep -q "^category:"; then
            # Replace the category line
            echo "category: $new_category" >> "$tmpfile"
            category_replaced=1
        else
            echo "$line" >> "$tmpfile"
        fi
    done < "$file"

    # Replace original file
    mv "$tmpfile" "$file"
    return 0
}

# Read fixes from report file
echo "Reading fixes from report..."
echo ""

while IFS='|' read -r relative_path expected_category actual_category; do
    # Skip header lines
    if [[ "$relative_path" == "Total files"* ]] || [[ "$relative_path" == "Compliant"* ]] || \
       [[ "$relative_path" == "Non-compliant"* ]] || [[ "$relative_path" == "No frontmatter"* ]] || \
       [[ "$relative_path" == "Compliance"* ]] || [[ "$relative_path" == "Files requiring fixes"* ]] || \
       [[ "$relative_path" == "Generated"* ]] || [[ "$relative_path" == "Diataxis"* ]] || \
       [ -z "$relative_path" ]; then
        continue
    fi

    file="$DOCS_ROOT/$relative_path"

    if [ ! -f "$file" ]; then
        echo "  Warning: File not found: $relative_path"
        continue
    fi

    echo -e "${YELLOW}Fixing:${NC} $relative_path"
    echo "  Old category: '$actual_category' → New category: '$expected_category'"

    if fix_category "$file" "$expected_category" "$actual_category"; then
        FIXES_APPLIED=$((FIXES_APPLIED + 1))
        echo -e "  ${GREEN}✓ Fixed${NC}"
    else
        echo "  ✗ Failed"
    fi
    echo ""

done < <(tail -n +10 "$REPORT_FILE")

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo -e "${GREEN}Fixes applied: $FIXES_APPLIED${NC}"
echo ""
echo "Run check-diataxis-compliance.sh again to verify compliance."
echo ""

exit 0
