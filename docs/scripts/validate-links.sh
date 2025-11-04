#!/bin/bash
# validate-links.sh
# Comprehensive link validation for documentation
# Checks all internal markdown links and reports broken references

set -e

DOCS_ROOT="/home/devuser/workspace/project/docs"
REPORT_FILE="${REPORT_FILE:-/tmp/link-validation-report.txt}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FAILED=0
CHECKED=0
SKIPPED=0

log_success() {
  echo -e "${GREEN}✓${NC} $1"
}

log_error() {
  echo -e "${RED}✗${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
}

log_info() {
  echo -e "${BLUE}ℹ${NC} $1"
}

# Initialize report
init_report() {
  cat > "$REPORT_FILE" <<EOF
# Link Validation Report
Generated: $(date)
Root: $DOCS_ROOT

## Summary
EOF
}

# Validate a single link
validate_link() {
  local source_file=$1
  local link=$2
  local line_num=$3

  # Skip external links
  if [[ "$link" =~ ^https?:// ]]; then
    ((SKIPPED++))
    return 0
  fi

  # Skip mailto links
  if [[ "$link" =~ ^mailto: ]]; then
    ((SKIPPED++))
    return 0
  fi

  # Skip anchor-only links
  if [[ "$link" =~ ^#.* ]]; then
    # TODO: Could validate anchors exist in target file
    ((SKIPPED++))
    return 0
  fi

  ((CHECKED++))

  # Split link into path and anchor
  local link_path="${link%#*}"
  local anchor="${link#*#}"
  [[ "$anchor" == "$link" ]] && anchor=""

  # Resolve relative path
  local dir=$(dirname "$source_file")
  local target

  # Handle different path formats
  if [[ "$link_path" =~ ^/ ]]; then
    # Absolute path from project root
    target="/home/devuser/workspace/project${link_path}"
  elif [[ "$link_path" =~ ^\.\. ]]; then
    # Relative path with ../
    target=$(realpath -m "$dir/$link_path" 2>/dev/null)
  elif [[ "$link_path" =~ ^\. ]]; then
    # Relative path with ./
    target=$(realpath -m "$dir/$link_path" 2>/dev/null)
  else
    # Relative path without prefix
    target=$(realpath -m "$dir/$link_path" 2>/dev/null)
  fi

  # Check if target exists
  if [ ! -f "$target" ] && [ ! -d "$target" ]; then
    ((FAILED++))
    log_error "$(basename "$source_file"):$line_num → $link"
    echo "BROKEN: $source_file:$line_num" >> "$REPORT_FILE"
    echo "  Link: $link" >> "$REPORT_FILE"
    echo "  Resolved to: $target" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    return 1
  fi

  # If anchor specified, validate it exists (optional, more complex)
  if [ -n "$anchor" ] && [ -f "$target" ]; then
    # Convert anchor to heading format
    local heading=$(echo "$anchor" | sed 's/-/ /g')
    if ! grep -qi "^#.*$heading" "$target"; then
      log_warn "$(basename "$source_file"):$line_num → anchor '$anchor' not found in $(basename "$target")"
      echo "WARNING: Missing anchor in $target" >> "$REPORT_FILE"
      echo "  Source: $source_file:$line_num" >> "$REPORT_FILE"
      echo "  Anchor: #$anchor" >> "$REPORT_FILE"
      echo "" >> "$REPORT_FILE"
    fi
  fi

  return 0
}

# Process a single markdown file
process_file() {
  local file=$1
  local file_basename=$(basename "$file")

  log_info "Checking: $file_basename"

  local line_num=0
  while IFS= read -r line; do
    ((line_num++))

    # Extract all markdown links from line
    while [[ "$line" =~ \]\(([^)]+)\) ]]; do
      local link="${BASH_REMATCH[1]}"
      line="${line#*](${link})}"

      validate_link "$file" "$link" "$line_num"
    done
  done < "$file"
}

# Main validation loop
validate_all() {
  log_info "Starting comprehensive link validation..."
  log_info "Documentation root: $DOCS_ROOT"
  echo ""

  init_report

  local total_files=0

  while IFS= read -r file; do
    ((total_files++))
    process_file "$file"
  done < <(find "$DOCS_ROOT" -name "*.md" -type f | sort)

  # Generate summary
  echo "" >> "$REPORT_FILE"
  cat >> "$REPORT_FILE" <<EOF
## Statistics
- Total files scanned: $total_files
- Total links checked: $CHECKED
- External/skipped links: $SKIPPED
- Broken links found: $FAILED

## Status
EOF

  if [ $FAILED -eq 0 ]; then
    echo "✓ ALL LINKS VALID" >> "$REPORT_FILE"
  else
    echo "✗ VALIDATION FAILED - $FAILED broken links" >> "$REPORT_FILE"
  fi

  # Print summary
  echo ""
  log_info "=========================================="
  log_info "Validation Summary"
  log_info "=========================================="
  log_info "Files scanned:     $total_files"
  log_info "Links checked:     $CHECKED"
  log_info "External/skipped:  $SKIPPED"

  if [ $FAILED -eq 0 ]; then
    log_success "Broken links:      0"
    log_success ""
    log_success "All links valid!"
    log_info "Report saved to: $REPORT_FILE"
    return 0
  else
    log_error "Broken links:      $FAILED"
    log_error ""
    log_error "Validation failed!"
    log_info "Full report saved to: $REPORT_FILE"
    return 1
  fi
}

# Run validation
validate_all
exit $?
