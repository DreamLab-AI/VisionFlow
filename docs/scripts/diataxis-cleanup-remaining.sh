#!/bin/bash

echo "=== Diátaxis Phase 2b: Remaining Files Cleanup ==="
echo ""

MOVED=0

# Helper function for safe moves
safe_move() {
    local src="$1"
    local dest="$2"

    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dest")"
        mv "$src" "$dest"
        echo "✓ Moved: $src -> $dest"
        MOVED=$((MOVED + 1))
    else
        echo "✗ Not found: $src"
    fi
}

echo "Step 1: Move audit/report files to archive..."
echo "-------------------------------------------"

# These are temporary reports and audit files
safe_move "docs/ASCII_CONVERSION_REPORT.md" "docs/archive/reports/ascii-conversion-report.md"
safe_move "docs/ascii-to-mermaid-conversion-report.md" "docs/archive/reports/ascii-to-mermaid-conversion.md"
safe_move "docs/DOCUMENTATION_AUDIT_FINAL.md" "docs/archive/reports/documentation-audit-final.md"
safe_move "docs/DOCUMENTATION_INDEX_REPORT.md" "docs/archive/reports/documentation-index-report.md"
safe_move "docs/DOCUMENTATION_ISSUES.md" "docs/archive/reports/documentation-issues.md"
safe_move "docs/LINK-AUDIT-SUMMARY.md" "docs/archive/reports/link-audit-summary.md"
safe_move "docs/LINK_FIXES_REPORT.md" "docs/archive/reports/link-fixes-report.md"
safe_move "docs/MERMAID_FIXES_EXAMPLES.md" "docs/archive/reports/mermaid-fixes-examples.md"
safe_move "docs/MERMAID_FIXES_REPORT.md" "docs/archive/reports/mermaid-fixes-report.md"
safe_move "docs/MERMAID_VALIDATION_COMPLETE.md" "docs/archive/reports/mermaid-validation-complete.md"
safe_move "docs/HIVE_MIND_INTEGRATION_COMPLETE.md" "docs/archive/reports/hive-mind-integration.md"
safe_move "docs/IMPLEMENTATION_COMPLETE.md" "docs/archive/reports/implementation-complete.md"
safe_move "docs/implementation-report-stubs.md" "docs/archive/reports/implementation-report-stubs.md"

echo ""
echo "Step 2: Move design/architecture documents to explanations..."
echo "-------------------------------------------"

safe_move "docs/analytics-visualization-design.md" "docs/explanations/architecture/analytics-visualization.md"
safe_move "docs/semantic-forces-actor-design.md" "docs/explanations/physics/semantic-forces-actor.md"
safe_move "docs/services-layer-complete.md" "docs/explanations/architecture/services-layer.md"
safe_move "docs/enhanced-ontology-parser-implementation.md" "docs/explanations/ontology/enhanced-parser.md"
safe_move "docs/quality-gates-api-audit.md" "docs/archive/audits/quality-gates-api.md"

echo ""
echo "Step 3: Move Neo4j schema documents to reference..."
echo "-------------------------------------------"

safe_move "docs/neo4j-persistence-analysis.md" "docs/reference/database/neo4j-persistence-analysis.md"
safe_move "docs/neo4j-rich-ontology-schema-v2.md" "docs/reference/database/ontology-schema-v2.md"
safe_move "docs/neo4j-user-settings-schema.md" "docs/reference/database/user-settings-schema.md"

echo ""
echo "Step 4: Move feature/guide documents..."
echo "-------------------------------------------"

safe_move "docs/auth-user-settings.md" "docs/guides/features/auth-user-settings.md"
safe_move "docs/nostr-auth-implementation.md" "docs/guides/features/nostr-auth.md"

# Move other root-level docs
safe_move "docs/observability-strategy.md" "docs/explanations/architecture/observability-strategy.md"
safe_move "docs/ontology-file-workflow.md" "docs/explanations/ontology/file-workflow.md"
safe_move "docs/parser-integration-final.md" "docs/explanations/ontology/parser-integration.md"
safe_move "docs/performance-data-collection-baseline.md" "docs/archive/reports/performance-baseline.md"
safe_move "docs/pipeline-verification-complete.md" "docs/archive/reports/pipeline-verification.md"
safe_move "docs/project-status.md" "docs/archive/reports/project-status.md"
safe_move "docs/reasoning-integration-complete.md" "docs/archive/reports/reasoning-integration.md"
safe_move "docs/reasoning-tests-data-complete.md" "docs/archive/reports/reasoning-tests-data.md"
safe_move "docs/reasoning-tests-final.md" "docs/archive/reports/reasoning-tests-final.md"
safe_move "docs/rust-typescript-async-bridge.md" "docs/explanations/architecture/rust-typescript-bridge.md"
safe_move "docs/technical-debt-register.md" "docs/archive/reports/technical-debt.md"
safe_move "docs/webxr-camera-integration.md" "docs/explanations/architecture/webxr-camera.md"
safe_move "docs/xr-client-implementation.md" "docs/explanations/architecture/xr-client.md"

echo ""
echo "Step 5: Move 'fixes' directory to archive..."
echo "-------------------------------------------"

if [ -d "docs/fixes" ]; then
    mkdir -p "docs/archive/fixes"
    for file in docs/fixes/*.md; do
        [ -f "$file" ] && safe_move "$file" "docs/archive/fixes/$(basename "$file")"
    done
    rmdir docs/fixes 2>/dev/null && echo "✓ Removed empty fixes directory"
fi

echo ""
echo "Step 6: Move specialized and working directories..."
echo "-------------------------------------------"

if [ -d "docs/specialized" ]; then
    mkdir -p "docs/archive/specialized"
    for file in docs/specialized/*.md; do
        [ -f "$file" ] && safe_move "$file" "docs/archive/specialized/$(basename "$file")"
    done
fi

if [ -d "docs/working" ]; then
    mkdir -p "docs/archive/working"
    for file in docs/working/*.md; do
        [ -f "$file" ] && safe_move "$file" "docs/archive/working/$(basename "$file")"
    done
fi

echo ""
echo "Step 7: Move operations documents..."
echo "-------------------------------------------"

if [ -d "docs/operations" ]; then
    for file in docs/operations/*.md; do
        [ -f "$file" ] && safe_move "$file" "docs/guides/operations/$(basename "$file")"
    done
    rmdir docs/operations 2>/dev/null && echo "✓ Removed empty operations directory"
fi

echo ""
echo "Step 8: Handle INDEX-QUICK-START..."
echo "-------------------------------------------"

# This should become part of the main README or tutorials
safe_move "docs/INDEX-QUICK-START.md" "docs/archive/INDEX-QUICK-START-old.md"

echo ""
echo "=== Cleanup Summary ==="
echo "Files moved: $MOVED"
echo ""
echo "✓ Phase 2b Complete"
