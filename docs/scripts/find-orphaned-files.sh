#!/bin/bash
# find-orphaned-files.sh
# Finds documentation files that are not referenced by any other documentation
# Helps identify files that may need to be added to navigation or deleted

set -e

DOCS_ROOT="/home/devuser/workspace/project/docs"
MIN_REFS=${MIN_REFS:-1}  # Minimum references (1 = only self-reference)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_orphan() {
  echo -e "${RED}[ORPHAN]${NC} $1"
}

# Exclude patterns (files that are OK to have few references)
EXCLUDE_PATTERNS=(
  "README.md"
  "CONTRIBUTING.md"
  "LICENSE.md"
  "CHANGELOG.md"
  "reports/"
  ".claude-flow/"
)

should_exclude() {
  local file=$1

  for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ "$file" == *"$pattern"* ]]; then
      return 0  # Should exclude
    fi
  done

  return 1  # Should not exclude
}

find_orphans() {
  log_info "Searching for orphaned documentation files..."
  log_info "Root: $DOCS_ROOT"
  log_info "Minimum references threshold: $MIN_REFS"
  echo ""

  local orphan_count=0
  local total_files=0
  local excluded_count=0

  # Create temporary reference map
  local temp_refs=$(mktemp)

  log_info "Building reference map..."

  # First pass: count references for each file
  find "$DOCS_ROOT" -name "*.md" -type f | sort | while read -r file; do
    local basename=$(basename "$file")
    local relative_path=${file#$DOCS_ROOT/}

    # Count references (excluding the file itself)
    local ref_count=$(grep -r "$basename" "$DOCS_ROOT" --include="*.md" | \
      grep -v "^$file:" | wc -l)

    echo "$relative_path|$ref_count" >> "$temp_refs"
  done

  # Second pass: report orphans
  log_info "Analyzing references..."
  echo ""

  while IFS='|' read -r file_path ref_count; do
    ((total_files++))

    local full_path="$DOCS_ROOT/$file_path"

    # Skip excluded patterns
    if should_exclude "$file_path"; then
      ((excluded_count++))
      continue
    fi

    # Check if orphaned
    if [ "$ref_count" -le "$MIN_REFS" ]; then
      ((orphan_count++))
      log_orphan "$file_path (${ref_count} references)"

      # Provide context
      if [ -f "$full_path" ]; then
        # Extract title from file
        local title=$(grep -m 1 "^#" "$full_path" 2>/dev/null | sed 's/^#*\s*//' || echo "No title")
        echo "         Title: $title"

        # Check file size
        local size=$(stat -f%z "$full_path" 2>/dev/null || stat -c%s "$full_path" 2>/dev/null || echo "unknown")
        echo "         Size:  $size bytes"

        # Check last modified
        local modified=$(stat -f%Sm "$full_path" 2>/dev/null || stat -c%y "$full_path" 2>/dev/null || echo "unknown")
        echo "         Modified: $modified"

        echo ""
      fi
    fi
  done < "$temp_refs"

  # Cleanup
  rm -f "$temp_refs"

  # Summary
  echo ""
  log_info "=========================================="
  log_info "Orphan Detection Summary"
  log_info "=========================================="
  log_info "Total files scanned: $total_files"
  log_info "Excluded files:      $excluded_count"
  log_warn "Orphaned files:      $orphan_count"

  if [ $orphan_count -eq 0 ]; then
    log_info ""
    log_info "✓ No orphaned files found!"
    return 0
  else
    echo ""
    log_warn "Consider the following actions for orphaned files:"
    log_warn "  1. Add to navigation/index if intentionally standalone"
    log_warn "  2. Link from related documentation"
    log_warn "  3. Move to reports/ or archive/ if historical"
    log_warn "  4. Delete if obsolete"
    return 1
  fi
}

# Additional utility: Find files with zero references
find_unreferenced() {
  log_info "Finding completely unreferenced files..."

  find "$DOCS_ROOT" -name "*.md" -type f | while read -r file; do
    local basename=$(basename "$file")

    # Count ANY references including self
    local total_refs=$(grep -r "$basename" "$DOCS_ROOT" --include="*.md" | wc -l)

    if [ "$total_refs" -eq 0 ]; then
      log_orphan "$file (0 references - completely unreferenced!)"
    fi
  done
}

# Main execution
case "${1:-orphans}" in
  orphans)
    find_orphans
    ;;
  unreferenced)
    find_unreferenced
    ;;
  *)
    echo "Usage: $0 {orphans|unreferenced}"
    echo ""
    echo "  orphans       - Find files with <= MIN_REFS references (default)"
    echo "  unreferenced  - Find files with 0 references"
    echo ""
    echo "Environment variables:"
    echo "  MIN_REFS      - Minimum reference threshold (default: 1)"
    exit 1
    ;;
esac
