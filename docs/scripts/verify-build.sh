#!/bin/bash
# Build Verification Script for Jekyll Documentation Site
# Validates the build output meets quality standards before deployment

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
DOCS_ROOT="$(dirname "$SCRIPT_DIR")"
SITE_DIR="${SITE_DIR:-$DOCS_ROOT/../_site}"
JSON_OUTPUT="${JSON_OUTPUT:-false}"
VERBOSE="${VERBOSE:-false}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
ERRORS=0
WARNINGS=0
CHECKS_PASSED=0

log_info() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${NC}[INFO] $1${NC}"
    fi
}

log_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
    ((CHECKS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
    ((WARNINGS++))
}

log_error() {
    echo -e "${RED}[FAIL] $1${NC}"
    ((ERRORS++))
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --site-dir)
            SITE_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================="
echo "Jekyll Build Verification"
echo "========================================="
echo "Site directory: $SITE_DIR"
echo ""

# Check 1: Build directory exists
if [[ -d "$SITE_DIR" ]]; then
    log_success "Build directory exists"
else
    log_error "Build directory not found: $SITE_DIR"
    exit 1
fi

# Check 2: Index file exists
if [[ -f "$SITE_DIR/index.html" ]]; then
    log_success "Index file exists"
else
    log_error "Index file (index.html) not found"
fi

# Check 3: Minimum number of HTML files
HTML_COUNT=$(find "$SITE_DIR" -name "*.html" -type f | wc -l)
if [[ $HTML_COUNT -ge 10 ]]; then
    log_success "Sufficient HTML files generated ($HTML_COUNT files)"
else
    log_warning "Low HTML file count: $HTML_COUNT (expected >= 10)"
fi

# Check 4: CSS files exist
CSS_COUNT=$(find "$SITE_DIR" -name "*.css" -type f | wc -l)
if [[ $CSS_COUNT -ge 1 ]]; then
    log_success "CSS files present ($CSS_COUNT files)"
else
    log_warning "No CSS files found"
fi

# Check 5: No build errors in HTML (check for Jekyll error messages)
JEKYLL_ERRORS=$(grep -r "Liquid error" "$SITE_DIR" --include="*.html" 2>/dev/null | wc -l || echo "0")
if [[ $JEKYLL_ERRORS -eq 0 ]]; then
    log_success "No Liquid template errors found"
else
    log_error "Found $JEKYLL_ERRORS Liquid template errors in output"
fi

# Check 6: Sitemap generated
if [[ -f "$SITE_DIR/sitemap.xml" ]]; then
    log_success "Sitemap generated"
else
    log_warning "Sitemap not found (sitemap.xml)"
fi

# Check 7: Feed generated
if [[ -f "$SITE_DIR/feed.xml" ]]; then
    log_success "RSS feed generated"
else
    log_warning "RSS feed not found (feed.xml)"
fi

# Check 8: Check for broken internal links in HTML
BROKEN_LINKS=0
while IFS= read -r html_file; do
    # Extract href values pointing to local files
    while IFS= read -r href; do
        # Skip external links, anchors, and special schemes
        if [[ "$href" =~ ^(http|https|mailto:|tel:|#|javascript:) ]]; then
            continue
        fi

        # Resolve relative path
        if [[ "$href" == /* ]]; then
            target="$SITE_DIR$href"
        else
            target="$(dirname "$html_file")/$href"
        fi

        # Remove anchor from path
        target="${target%%#*}"

        # Check if target exists
        if [[ -n "$target" ]] && [[ ! -e "$target" ]] && [[ ! -e "${target}.html" ]] && [[ ! -e "${target}/index.html" ]]; then
            log_info "Broken link in $html_file: $href"
            ((BROKEN_LINKS++))
        fi
    done < <(grep -oP 'href="\K[^"]+' "$html_file" 2>/dev/null || true)
done < <(find "$SITE_DIR" -name "*.html" -type f)

if [[ $BROKEN_LINKS -eq 0 ]]; then
    log_success "No broken internal links found"
elif [[ $BROKEN_LINKS -lt 10 ]]; then
    log_warning "Found $BROKEN_LINKS broken internal links"
else
    log_error "Found $BROKEN_LINKS broken internal links (threshold: 10)"
fi

# Check 9: Verify Mermaid diagrams are present
MERMAID_DIVS=$(grep -r "class=\"mermaid\"" "$SITE_DIR" --include="*.html" 2>/dev/null | wc -l || echo "0")
if [[ $MERMAID_DIVS -gt 0 ]]; then
    log_success "Mermaid diagrams found ($MERMAID_DIVS instances)"
else
    log_info "No Mermaid diagrams in output"
fi

# Check 10: Build size reasonable
SITE_SIZE_KB=$(du -sk "$SITE_DIR" | cut -f1)
SITE_SIZE_MB=$((SITE_SIZE_KB / 1024))
if [[ $SITE_SIZE_MB -lt 100 ]]; then
    log_success "Build size acceptable (${SITE_SIZE_MB}MB)"
elif [[ $SITE_SIZE_MB -lt 500 ]]; then
    log_warning "Build size large (${SITE_SIZE_MB}MB)"
else
    log_error "Build size too large (${SITE_SIZE_MB}MB, max 500MB)"
fi

# Check 11: Verify assets directory
if [[ -d "$SITE_DIR/assets" ]]; then
    log_success "Assets directory present"
else
    log_warning "Assets directory not found"
fi

# Check 12: No empty HTML files
EMPTY_HTML=$(find "$SITE_DIR" -name "*.html" -type f -empty | wc -l)
if [[ $EMPTY_HTML -eq 0 ]]; then
    log_success "No empty HTML files"
else
    log_error "Found $EMPTY_HTML empty HTML files"
fi

# Summary
echo ""
echo "========================================="
echo "Verification Summary"
echo "========================================="
echo "Checks passed: $CHECKS_PASSED"
echo "Warnings: $WARNINGS"
echo "Errors: $ERRORS"
echo ""
echo "Build Statistics:"
echo "  HTML files: $HTML_COUNT"
echo "  CSS files: $CSS_COUNT"
echo "  Site size: ${SITE_SIZE_MB}MB"
echo ""

# JSON output
if [[ "$JSON_OUTPUT" == "true" ]]; then
    cat <<EOF
{
    "checks_passed": $CHECKS_PASSED,
    "warnings": $WARNINGS,
    "errors": $ERRORS,
    "html_files": $HTML_COUNT,
    "css_files": $CSS_COUNT,
    "site_size_mb": $SITE_SIZE_MB,
    "broken_links": $BROKEN_LINKS,
    "mermaid_diagrams": $MERMAID_DIVS,
    "success": $([[ $ERRORS -eq 0 ]] && echo "true" || echo "false")
}
EOF
fi

# Exit code
if [[ $ERRORS -gt 0 ]]; then
    echo -e "${RED}Build verification FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}Build verification PASSED${NC}"
    exit 0
fi
