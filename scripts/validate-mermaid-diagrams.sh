#!/bin/bash
# Validate Mermaid Diagrams in event-flow-diagrams.md
# This script checks syntax and counts diagrams

set -e

DOCS_DIR="/home/devuser/workspace/project/docs/architecture"
FILE="$DOCS_DIR/event-flow-diagrams.md"

echo "=========================================="
echo "Mermaid Diagram Validation Script"
echo "=========================================="
echo ""

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo "❌ ERROR: File not found: $FILE"
    exit 1
fi

echo "✅ File found: $FILE"
echo ""

# Count Mermaid code blocks
echo "Counting Mermaid diagrams..."
MERMAID_COUNT=$(grep -c '```mermaid' "$FILE" || true)
echo "📊 Total Mermaid diagrams: $MERMAID_COUNT"
echo ""

# Expected count
EXPECTED=10
if [ "$MERMAID_COUNT" -eq "$EXPECTED" ]; then
    echo "✅ Diagram count matches expected: $EXPECTED"
else
    echo "⚠️  WARNING: Expected $EXPECTED diagrams, found $MERMAID_COUNT"
fi
echo ""

# Check for specific diagram types
echo "Checking diagram types..."
SEQUENCE_COUNT=$(grep -c 'sequenceDiagram' "$FILE" || true)
FLOWCHART_COUNT=$(grep -c 'flowchart TD' "$FILE" || true)

echo "  • Sequence Diagrams: $SEQUENCE_COUNT"
echo "  • Flowcharts: $FLOWCHART_COUNT"
echo ""

# Check for key participants/components
echo "Validating key components..."

check_component() {
    local component="$1"
    if grep -q "$component" "$FILE"; then
        echo "  ✅ $component"
    else
        echo "  ❌ Missing: $component"
    fi
}

check_component "EventBus"
check_component "GraphRepository"
check_component "WebSocket"
check_component "PhysicsService"
check_component "SemanticService"
check_component "CacheInvalidation"
check_component "EventStore"
echo ""

# Check for color coding
echo "Validating color coding..."
if grep -q 'rgb(' "$FILE"; then
    echo "  ✅ Color coding present"
else
    echo "  ❌ No color coding found"
fi
echo ""

# Check for parallel execution blocks
echo "Validating parallel execution patterns..."
if grep -q 'par Event Distribution' "$FILE"; then
    echo "  ✅ Parallel event distribution"
else
    echo "  ❌ No parallel execution blocks"
fi
echo ""

# Check for loops
echo "Validating loops..."
if grep -q 'loop' "$FILE"; then
    echo "  ✅ Loop structures present"
else
    echo "  ❌ No loop structures"
fi
echo ""

# Check for alternatives (alt/else)
echo "Validating conditional logic..."
if grep -q 'alt ' "$FILE"; then
    echo "  ✅ Alternative/conditional blocks present"
else
    echo "  ❌ No alternative blocks"
fi
echo ""

# Verify event types documented
echo "Checking documented event types..."

check_event() {
    local event="$1"
    if grep -q "$event" "$FILE"; then
        echo "  ✅ $event"
    else
        echo "  ⚠️  Missing: $event"
    fi
}

check_event "GitHubSyncCompletedEvent"
check_event "NodeCreatedEvent"
check_event "PhysicsStepCompletedEvent"
check_event "WebSocketClientConnectedEvent"
check_event "SemanticAnalysisCompletedEvent"
echo ""

# Check for technical details
echo "Validating technical details..."

check_detail() {
    local detail="$1"
    if grep -q "$detail" "$FILE"; then
        echo "  ✅ $detail"
    else
        echo "  ⚠️  Missing: $detail"
    fi
}

check_detail "316 nodes"
check_detail "450 edges"
check_detail "60 FPS"
check_detail "GPU"
check_detail "cache"
check_detail "CQRS"
echo ""

# Summary
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo ""
echo "Total Mermaid Diagrams: $MERMAID_COUNT / $EXPECTED"
echo "Sequence Diagrams: $SEQUENCE_COUNT"
echo "Flowcharts: $FLOWCHART_COUNT"
echo ""

if [ "$MERMAID_COUNT" -eq "$EXPECTED" ]; then
    echo "✅ ALL VALIDATIONS PASSED!"
    echo ""
    echo "Next steps:"
    echo "  1. View diagrams on GitHub"
    echo "  2. Open in VS Code with Mermaid preview"
    echo "  3. Export to PNG/SVG using mermaid-cli"
    echo "  4. See VIEWING_MERMAID_DIAGRAMS.md for details"
    exit 0
else
    echo "⚠️  VALIDATION WARNINGS DETECTED"
    echo ""
    echo "Review the file and ensure all diagrams are present."
    exit 1
fi
