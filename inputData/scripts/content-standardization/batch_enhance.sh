#!/bin/bash
# Batch Enhancement Script
# Safe wrapper for batch content enhancement

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAGES_DIR="${PAGES_DIR:-../../mainKnowledgeGraph/pages}"
LEVEL="${LEVEL:-1}"
CORPUS_INDEX="${SCRIPT_DIR}/corpus_index.json"
REPORT_DIR="${SCRIPT_DIR}/reports"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "Content Enhancement Batch Processor"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Pages Directory: ${PAGES_DIR}"
echo "  Enhancement Level: ${LEVEL}"
echo "  Corpus Index: ${CORPUS_INDEX}"
echo ""

# Check if corpus index exists
if [ ! -f "${CORPUS_INDEX}" ]; then
    echo -e "${YELLOW}Warning: Corpus index not found${NC}"
    echo "Building corpus index..."
    python3 "${SCRIPT_DIR}/corpus_indexer.py" "${PAGES_DIR}" "${CORPUS_INDEX}"
    echo -e "${GREEN}✓ Corpus index built${NC}"
    echo ""
fi

# Create reports directory
mkdir -p "${REPORT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="${REPORT_DIR}/enhancement_report_${TIMESTAMP}.json"

echo "Processing files..."
echo ""

# Run enhancement
python3 "${SCRIPT_DIR}/enhance_content.py" \
    --directory "${PAGES_DIR}" \
    --level "${LEVEL}" \
    --apply \
    --corpus-index "${CORPUS_INDEX}" \
    --report "${REPORT_FILE}"

echo ""
echo "============================================================"
echo -e "${GREEN}Batch processing complete!${NC}"
echo "============================================================"
echo ""
echo "Report saved to: ${REPORT_FILE}"
echo ""

# Display summary if jq is available
if command -v jq &> /dev/null; then
    echo "Summary:"
    jq '.summary' "${REPORT_FILE}"
fi

echo ""
echo "To revert changes:"
echo "  git reset --hard HEAD~1"
echo ""
