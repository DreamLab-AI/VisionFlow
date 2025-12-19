#!/bin/bash

# Diataxis Framework Compliance Checker
# Verifies that category in YAML frontmatter matches directory location

set -e

DOCS_ROOT="/home/devuser/workspace/project/docs"
TOTAL_FILES=0
COMPLIANT_FILES=0
NON_COMPLIANT_FILES=0
NO_FRONTMATTER=0
declare -a FIXES_NEEDED=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Diataxis Framework Compliance Check"
echo "=========================================="
echo ""

# Function to extract category from YAML frontmatter
get_category() {
    local file="$1"
    # Check if file starts with ---
    if head -1 "$file" | grep -q "^---$"; then
        # Extract category field
        sed -n '/^---$/,/^---$/p' "$file" | grep "^category:" | sed 's/category: *//' | tr -d '\r'
    else
        echo "NO_FRONTMATTER"
    fi
}

# Function to check a directory
check_directory() {
    local dir="$1"
    local expected_category="$2"

    echo -e "${YELLOW}Checking $dir (expected: $expected_category)${NC}"

    # Find all markdown files in directory and subdirectories
    while IFS= read -r file; do
        TOTAL_FILES=$((TOTAL_FILES + 1))

        local actual_category=$(get_category "$file")
        local relative_path="${file#$DOCS_ROOT/}"

        if [ "$actual_category" = "NO_FRONTMATTER" ]; then
            echo -e "  ${RED}✗${NC} $relative_path - NO FRONTMATTER"
            NO_FRONTMATTER=$((NO_FRONTMATTER + 1))
            FIXES_NEEDED+=("$file|$expected_category|NO_FRONTMATTER")
        elif [ "$actual_category" = "$expected_category" ]; then
            echo -e "  ${GREEN}✓${NC} $relative_path"
            COMPLIANT_FILES=$((COMPLIANT_FILES + 1))
        else
            echo -e "  ${RED}✗${NC} $relative_path - Found: '$actual_category', Expected: '$expected_category'"
            NON_COMPLIANT_FILES=$((NON_COMPLIANT_FILES + 1))
            FIXES_NEEDED+=("$file|$expected_category|$actual_category")
        fi
    done < <(find "$dir" -type f -name "*.md" 2>/dev/null)

    echo ""
}

# Check each Diataxis category
check_directory "$DOCS_ROOT/tutorials" "tutorial"
check_directory "$DOCS_ROOT/guides" "guide"
check_directory "$DOCS_ROOT/reference" "reference"
check_directory "$DOCS_ROOT/explanations" "explanation"

# Calculate compliance percentage
if [ $TOTAL_FILES -gt 0 ]; then
    COMPLIANCE_PERCENT=$((COMPLIANT_FILES * 100 / TOTAL_FILES))
else
    COMPLIANCE_PERCENT=0
fi

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total files checked: $TOTAL_FILES"
echo -e "${GREEN}Compliant files: $COMPLIANT_FILES${NC}"
echo -e "${RED}Non-compliant files: $NON_COMPLIANT_FILES${NC}"
echo -e "${YELLOW}Files without frontmatter: $NO_FRONTMATTER${NC}"
echo ""
echo -e "Diataxis Compliance: ${GREEN}${COMPLIANCE_PERCENT}%${NC}"
echo ""

# Output fixes needed
if [ ${#FIXES_NEEDED[@]} -gt 0 ]; then
    echo "=========================================="
    echo "FIXES NEEDED"
    echo "=========================================="
    for fix in "${FIXES_NEEDED[@]}"; do
        IFS='|' read -r file expected actual <<< "$fix"
        relative_path="${file#$DOCS_ROOT/}"
        if [ "$actual" = "NO_FRONTMATTER" ]; then
            echo "ADD FRONTMATTER: $relative_path → category: $expected"
        else
            echo "CHANGE CATEGORY: $relative_path → '$actual' to '$expected'"
        fi
    done
    echo ""
fi

# Save report to file
REPORT_FILE="$DOCS_ROOT/scripts/diataxis-compliance-report.txt"
{
    echo "Diataxis Framework Compliance Report"
    echo "Generated: $(date)"
    echo ""
    echo "Total files: $TOTAL_FILES"
    echo "Compliant: $COMPLIANT_FILES"
    echo "Non-compliant: $NON_COMPLIANT_FILES"
    echo "No frontmatter: $NO_FRONTMATTER"
    echo "Compliance: ${COMPLIANCE_PERCENT}%"
    echo ""
    echo "Files requiring fixes:"
    for fix in "${FIXES_NEEDED[@]}"; do
        IFS='|' read -r file expected actual <<< "$fix"
        relative_path="${file#$DOCS_ROOT/}"
        echo "$relative_path|$expected|$actual"
    done
} > "$REPORT_FILE"

echo "Report saved to: $REPORT_FILE"

exit 0
