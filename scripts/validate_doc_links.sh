#!/bin/bash

# Documentation Link Validator
# Analyzes all markdown files in /docs/ for broken links and incorrect paths

DOCS_DIR="/home/devuser/workspace/project/docs"
PROJECT_ROOT="/home/devuser/workspace/project"
REPORT_FILE="/tmp/doc_link_report.md"

echo "# Documentation Link Analysis Report" > "$REPORT_FILE"
echo "Generated: $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Initialize counters
TOTAL_LINKS=0
BROKEN_LINKS=0
INCORRECT_PATHS=0
EXTERNAL_LINKS=0
VALID_LINKS=0

# Arrays to store findings
declare -a CRITICAL_ISSUES
declare -a HIGH_ISSUES
declare -a MEDIUM_ISSUES
declare -a LOW_ISSUES

echo "## Summary Statistics" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Count all markdown files
MD_COUNT=$(find "$DOCS_DIR" -type f -name "*.md" | wc -l)
echo "- Total markdown files: $MD_COUNT" >> "$REPORT_FILE"

# Extract all markdown links
echo "Extracting all markdown links..."
find "$DOCS_DIR" -type f -name "*.md" -print0 | while IFS= read -r -d '' file; do
    # Extract links in format [text](path)
    grep -n -oE '\[([^\]]+)\]\(([^)]+)\)' "$file" | while IFS=: read -r line_num link; do
        # Extract the path from the link
        path=$(echo "$link" | sed -E 's/\[([^\]]+)\]\(([^)]+)\)/\2/')
        text=$(echo "$link" | sed -E 's/\[([^\]]+)\]\(([^)]+)\)/\1/')

        TOTAL_LINKS=$((TOTAL_LINKS + 1))

        # Skip external URLs
        if [[ "$path" =~ ^https?:// ]]; then
            EXTERNAL_LINKS=$((EXTERNAL_LINKS + 1))
            continue
        fi

        # Skip anchors
        if [[ "$path" =~ ^# ]]; then
            continue
        fi

        # Get directory of current file
        current_dir=$(dirname "$file")

        # Resolve relative path
        if [[ "$path" =~ ^\.\. ]]; then
            # Relative path with ../
            target="$current_dir/$path"
        elif [[ "$path" =~ ^\. ]]; then
            # Relative path with ./
            target="$current_dir/$path"
        elif [[ "$path" =~ ^/ ]]; then
            # Absolute path
            target="$PROJECT_ROOT$path"
        else
            # Relative path without ./
            target="$current_dir/$path"
        fi

        # Normalize path
        target=$(realpath -m "$target" 2>/dev/null || echo "$target")

        # Check if target exists
        if [ ! -e "$target" ]; then
            BROKEN_LINKS=$((BROKEN_LINKS + 1))

            # Categorize by severity
            if [[ "$path" =~ \.md$ ]]; then
                # Broken documentation link - CRITICAL
                CRITICAL_ISSUES+=("**CRITICAL**: Broken doc link in \`${file#$PROJECT_ROOT/}\`:$line_num")
                CRITICAL_ISSUES+=("  - Link text: \"$text\"")
                CRITICAL_ISSUES+=("  - Path: \`$path\`")
                CRITICAL_ISSUES+=("  - Resolved to: \`$target\`")
                CRITICAL_ISSUES+=("")
            elif [[ "$path" =~ /project/src/ ]] || [[ "$path" =~ /project/client/ ]]; then
                # Incorrect code path reference - HIGH
                HIGH_ISSUES+=("**HIGH**: Incorrect code path in \`${file#$PROJECT_ROOT/}\`:$line_num")
                HIGH_ISSUES+=("  - Link text: \"$text\"")
                HIGH_ISSUES+=("  - Path: \`$path\`")
                HIGH_ISSUES+=("  - Should be relative to project root")
                HIGH_ISSUES+=("")
            else
                # Other broken link - MEDIUM
                MEDIUM_ISSUES+=("**MEDIUM**: Broken link in \`${file#$PROJECT_ROOT/}\`:$line_num")
                MEDIUM_ISSUES+=("  - Link text: \"$text\"")
                MEDIUM_ISSUES+=("  - Path: \`$path\`")
                MEDIUM_ISSUES+=("  - Target: \`$target\`")
                MEDIUM_ISSUES+=("")
            fi
        else
            VALID_LINKS=$((VALID_LINKS + 1))
        fi
    done
done

# Write summary
echo "- Total links found: $TOTAL_LINKS" >> "$REPORT_FILE"
echo "- Valid links: $VALID_LINKS" >> "$REPORT_FILE"
echo "- Broken links: $BROKEN_LINKS" >> "$REPORT_FILE"
echo "- External URLs: $EXTERNAL_LINKS" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Write critical issues
if [ ${#CRITICAL_ISSUES[@]} -gt 0 ]; then
    echo "## 🔴 CRITICAL Issues (${#CRITICAL_ISSUES[@]} lines)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    printf '%s\n' "${CRITICAL_ISSUES[@]}" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# Write high priority issues
if [ ${#HIGH_ISSUES[@]} -gt 0 ]; then
    echo "## 🟠 HIGH Priority Issues (${#HIGH_ISSUES[@]} lines)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    printf '%s\n' "${HIGH_ISSUES[@]}" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# Write medium priority issues
if [ ${#MEDIUM_ISSUES[@]} -gt 0 ]; then
    echo "## 🟡 MEDIUM Priority Issues (${#MEDIUM_ISSUES[@]} lines)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    printf '%s\n' "${MEDIUM_ISSUES[@]}" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

echo "Report generated: $REPORT_FILE"
cat "$REPORT_FILE"
