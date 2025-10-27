#!/bin/bash
# Validation script for Category 1 refactoring
# Run this after completing all Category 1 tasks

set -e

echo "=== Category 1 Validation Script ==="
echo "Checking for resolved contradictions..."
echo ""

FAIL=0
WARN=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1.1: Binary protocol byte count
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1.1: Binary Protocol Standardization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Checking for incorrect '38 bytes' references... "
if grep -r "38 bytes per node" src/ docs/ --include="*.rs" --include="*.md" 2>/dev/null; then
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Found '38 bytes' references (should be 36 bytes)"
    FAIL=$((FAIL + 1))
else
    echo -e "${GREEN}✅ PASS${NC}"
fi

echo -n "Checking for correct '36 bytes' documentation... "
COUNT=$(grep -r "36 bytes" docs/reference/api/binary-protocol.md 2>/dev/null | wc -l)
if [ "$COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ PASS${NC} ($COUNT occurrences found)"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Binary protocol doc doesn't mention 36 bytes"
    FAIL=$((FAIL + 1))
fi

echo -n "Verifying u32 node ID implementation... "
if grep -q "pub id: u32" src/utils/binary_protocol.rs 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   WireNodeDataItemV2 should use u32 for ID"
    FAIL=$((FAIL + 1))
fi

echo ""

# Test 1.2: API port standardization
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1.2: API Port Standardization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Checking for incorrect port 8080 references... "
if grep -r "localhost:8080\|:8080/" docs/ --include="*.md" 2>/dev/null | grep -v "example\|historical"; then
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Found port 8080 references (should be 3030)"
    FAIL=$((FAIL + 1))
else
    echo -e "${GREEN}✅ PASS${NC}"
fi

echo -n "Checking for incorrect port 3001 references... "
if grep -r "localhost:3001\|:3001/" docs/ --include="*.md" 2>/dev/null | grep -v "example\|historical"; then
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Found port 3001 references (should be 3030)"
    FAIL=$((FAIL + 1))
else
    echo -e "${GREEN}✅ PASS${NC}"
fi

echo -n "Verifying SYSTEM_NETWORK_PORT documentation... "
if grep -q "SYSTEM_NETWORK_PORT" docs/reference/api/rest-api.md 2>/dev/null || \
   grep -q "SYSTEM_NETWORK_PORT" docs/API.md 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING${NC}"
    echo "   SYSTEM_NETWORK_PORT env var not documented in API reference"
    WARN=$((WARN + 1))
fi

echo -n "Checking default port in main.rs... "
if grep -q '"3030"' src/main.rs 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   main.rs should default to port 3030"
    FAIL=$((FAIL + 1))
fi

echo ""

# Test 1.3: Deployment documentation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1.3: Deployment Consolidation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Checking for deployment README... "
if [ -f "docs/deployment/README.md" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Missing docs/deployment/README.md (authoritative guide)"
    FAIL=$((FAIL + 1))
fi

echo -n "Verifying Docker NOT claimed for main project... "
if [ ! -f "Dockerfile" ]; then
    if grep -r "Docker deployment.*main\|docker-compose.*main" docs/deployment/ --include="*.md" -i 2>/dev/null | grep -v "NOT implemented\|future\|planned"; then
        echo -e "${RED}❌ FAIL${NC}"
        echo "   Docs claim Docker deployment but no Dockerfile exists"
        FAIL=$((FAIL + 1))
    else
        echo -e "${GREEN}✅ PASS${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (Dockerfile exists, verify docs are accurate)"
fi

echo -n "Checking multi-agent-docker separation... "
if grep -q "multi-agent-docker.*separate\|Turbo Flow Claude" docs/deployment/README.md 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING${NC}"
    echo "   Should clarify multi-agent-docker/ is separate system"
    WARN=$((WARN + 1))
fi

echo ""

# Test 1.4: Developer guide accuracy
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1.4: Developer Guide Update"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Checking for Rust version documentation... "
if grep -q "Rust.*1\.[0-9]*" docs/developer-guide/01-development-setup.md 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Developer setup should specify Rust version"
    FAIL=$((FAIL + 1))
fi

echo -n "Verifying Vitest (not Jest) in testing docs... "
if [ -f "client/package.json" ]; then
    if grep -q "vitest" client/package.json 2>/dev/null; then
        if grep -q "vitest\|Vitest" docs/developer-guide/05-testing.md 2>/dev/null; then
            echo -e "${GREEN}✅ PASS${NC}"
        else
            echo -e "${YELLOW}⚠️  WARNING${NC}"
            echo "   Testing docs should mention Vitest (package.json uses it)"
            WARN=$((WARN + 1))
        fi
    else
        echo -e "${YELLOW}⚠️  SKIP${NC} (No Vitest in package.json)"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (No client/package.json found)"
fi

echo -n "Checking for accurate dependency list... "
if [ -f "Cargo.toml" ] && [ -f "docs/developer-guide/01-development-setup.md" ]; then
    if grep -q "actix-web\|tokio" docs/developer-guide/01-development-setup.md 2>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC}"
        echo "   Developer setup should mention major dependencies"
        WARN=$((WARN + 1))
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC}"
fi

echo -n "Verifying development ports documented (3030, 5173)... "
if grep -q "3030.*backend\|backend.*3030" docs/developer-guide/01-development-setup.md 2>/dev/null && \
   grep -q "5173.*frontend\|frontend.*5173" docs/developer-guide/01-development-setup.md 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING${NC}"
    echo "   Should document backend port 3030 and frontend port 5173"
    WARN=$((WARN + 1))
fi

echo ""

# Test 1.5: Testing documentation accuracy
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1.5: Testing Documentation Accuracy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Checking for testing status document... "
if [ -f "docs/testing-status.md" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Missing docs/testing-status.md (honest status report)"
    FAIL=$((FAIL + 1))
fi

echo -n "Verifying CI status accuracy... "
if [ ! -d ".github/workflows" ]; then
    if grep -r "automated CI\|continuous integration.*implemented" docs/ --include="*.md" -i 2>/dev/null | grep -v "NOT\|planned\|future"; then
        echo -e "${RED}❌ FAIL${NC}"
        echo "   Docs claim CI exists but no .github/workflows/ directory"
        FAIL=$((FAIL + 1))
    else
        echo -e "${GREEN}✅ PASS${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (CI workflows exist, verify docs match)"
fi

echo -n "Checking for 'cargo test' documentation... "
if grep -q "cargo test" docs/developer-guide/05-testing.md 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Testing guide should document 'cargo test' command"
    FAIL=$((FAIL + 1))
fi

echo ""

# Test 1.6: Cross-cutting updates
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test 1.6: Cross-Cutting Updates"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Checking for updated documentation index... "
if [ -f "docs/00-INDEX.md" ]; then
    if grep -q "2025-10-27\|Recently Updated" docs/00-INDEX.md 2>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC}"
        echo "   Index should show recent update date"
        WARN=$((WARN + 1))
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (No docs/00-INDEX.md found)"
fi

echo -n "Checking for refactoring changelog... "
if [ -f "docs/refactoring/CHANGELOG.md" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   Missing docs/refactoring/CHANGELOG.md"
    FAIL=$((FAIL + 1))
fi

echo -n "Verifying authoritative references marked... "
if [ -f "docs/00-INDEX.md" ]; then
    if grep -q "Authoritative\|Single Source of Truth" docs/00-INDEX.md 2>/dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC}"
        echo "   Index should mark authoritative references"
        WARN=$((WARN + 1))
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC}"
fi

echo ""

# Build & test validation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build & Test Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -n "Testing cargo build... "
if cargo build --quiet 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    echo "   'cargo build' failed - code may be broken"
    FAIL=$((FAIL + 1))
fi

echo -n "Testing cargo check... "
if cargo check --quiet 2>/dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC}"
    FAIL=$((FAIL + 1))
fi

echo ""

# Final summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Validation Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $FAIL -eq 0 ] && [ $WARN -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
    echo "Category 1 refactoring complete with no issues."
    echo ""
    exit 0
elif [ $FAIL -eq 0 ]; then
    echo -e "${YELLOW}⚠️  PASSED WITH WARNINGS${NC}"
    echo "Failed: 0 | Warnings: $WARN"
    echo ""
    echo "Category 1 refactoring complete, but some non-critical issues remain."
    echo "Review warnings above and address if needed."
    echo ""
    exit 0
else
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo "Failed: $FAIL | Warnings: $WARN"
    echo ""
    echo "Category 1 refactoring incomplete. Fix failures before proceeding."
    echo ""
    exit 1
fi
