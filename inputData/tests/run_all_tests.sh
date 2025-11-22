#!/bin/bash
#
# Comprehensive Integration Test Runner
# Runs all integration tests for the ontology project
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_DIR="$SCRIPT_DIR/integration"
REPORT_DIR="$INTEGRATION_DIR/reports"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ensure report directory exists
mkdir -p "$REPORT_DIR"

# Test counters
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

# Functions
print_header() {
    echo ""
    echo "================================================================"
    echo "$1"
    echo "================================================================"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}>>> $1${NC}"
    echo "----------------------------------------------------------------"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_failure() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

run_test_suite() {
    local name=$1
    local command=$2
    local working_dir=$3

    TOTAL_SUITES=$((TOTAL_SUITES + 1))

    print_section "Running: $name"

    if [ -n "$working_dir" ]; then
        pushd "$working_dir" > /dev/null
    fi

    if eval "$command"; then
        PASSED_SUITES=$((PASSED_SUITES + 1))
        print_success "$name PASSED"
        return 0
    else
        FAILED_SUITES=$((FAILED_SUITES + 1))
        print_failure "$name FAILED"
        return 1
    fi

    if [ -n "$working_dir" ]; then
        popd > /dev/null
    fi
}

check_dependencies() {
    print_section "Checking Dependencies"

    local missing_deps=0

    # Check Python
    if command -v python3 &> /dev/null; then
        print_success "Python 3: $(python3 --version)"
    else
        print_failure "Python 3 not found"
        missing_deps=$((missing_deps + 1))
    fi

    # Check Node.js
    if command -v node &> /dev/null; then
        print_success "Node.js: $(node --version)"
    else
        print_failure "Node.js not found"
        missing_deps=$((missing_deps + 1))
    fi

    # Check Rust/Cargo (optional)
    if command -v cargo &> /dev/null; then
        print_success "Rust/Cargo: $(cargo --version)"
    else
        print_warning "Rust/Cargo not found (Rust tests will be skipped)"
    fi

    # Check for audit binary
    if [ -f "$PROJECT_ROOT/Ontology-Tools/tools/audit/target/release/audit" ] || \
       [ -f "$PROJECT_ROOT/Ontology-Tools/tools/audit/target/debug/audit" ]; then
        print_success "Audit tool binary found"
    else
        print_warning "Audit tool not built (some tests will be skipped)"
    fi

    # Check for WASM module
    if [ -d "$PROJECT_ROOT/publishing-tools/WasmVOWL/rust-wasm/pkg" ]; then
        print_success "WASM module found"
    else
        print_warning "WASM module not built (some tests will be skipped)"
    fi

    if [ $missing_deps -gt 0 ]; then
        print_failure "Missing required dependencies. Please install them first."
        exit 1
    fi
}

# Main execution
main() {
    print_header "ONTOLOGY INTEGRATION TEST SUITE"
    echo "Project: Logseq Knowledge Graph Ontology"
    echo "Date: $(date)"
    echo "Location: $INTEGRATION_DIR"
    echo ""

    # Check dependencies
    check_dependencies

    # Test 1: Python Converters
    print_header "TEST SUITE 1: Python Converters"
    run_test_suite \
        "Python Converters Integration Tests" \
        "python3 test_python_tools.py -v" \
        "$INTEGRATION_DIR" \
        || true

    # Test 2: JavaScript Pipeline
    print_header "TEST SUITE 2: JavaScript Pipeline"
    run_test_suite \
        "JavaScript Pipeline Integration Tests" \
        "node test_js_pipeline.js" \
        "$INTEGRATION_DIR" \
        || true

    # Test 3: Rust Tools (optional - skip if cargo not available)
    if command -v cargo &> /dev/null; then
        print_header "TEST SUITE 3: Rust Tools"

        # Check if we need to build the test
        RUST_TEST_DIR="$PROJECT_ROOT/Ontology-Tools/tools/audit"
        if [ -d "$RUST_TEST_DIR" ]; then
            # Copy test file to Rust project
            cp "$INTEGRATION_DIR/test_rust_tools.rs" "$RUST_TEST_DIR/tests/"

            run_test_suite \
                "Rust Tools Integration Tests" \
                "cargo test --test test_rust_tools -- --nocapture" \
                "$RUST_TEST_DIR" \
                || true
        else
            print_warning "Rust audit tool directory not found. Skipping Rust tests."
        fi
    else
        print_warning "Cargo not available. Skipping Rust tests."
    fi

    # Test 4: Cross-Tool Interoperability
    print_header "TEST SUITE 4: Cross-Tool Interoperability"
    run_test_suite \
        "Interoperability Integration Tests" \
        "python3 test_interoperability.py -v" \
        "$INTEGRATION_DIR" \
        || true

    # Generate consolidated report
    print_header "GENERATING CONSOLIDATED REPORT"

    CONSOLIDATED_REPORT="$REPORT_DIR/consolidated-report.json"

    cat > "$CONSOLIDATED_REPORT" <<EOF
{
  "testRun": {
    "timestamp": "$(date -Iseconds)",
    "projectRoot": "$PROJECT_ROOT",
    "testDirectory": "$INTEGRATION_DIR"
  },
  "summary": {
    "totalSuites": $TOTAL_SUITES,
    "passedSuites": $PASSED_SUITES,
    "failedSuites": $FAILED_SUITES,
    "successRate": $(awk "BEGIN {printf \"%.1f\", ($PASSED_SUITES / $TOTAL_SUITES) * 100}")
  },
  "suites": {
    "pythonConverters": "$([ -f "$REPORT_DIR/python-converters-report.json" ] && echo "completed" || echo "pending")",
    "javascriptPipeline": "$([ -f "$REPORT_DIR/javascript-pipeline-report.json" ] && echo "completed" || echo "pending")",
    "rustTools": "$(command -v cargo &> /dev/null && echo "completed" || echo "skipped")",
    "interoperability": "$([ -f "$REPORT_DIR/interoperability-report.json" ] && echo "completed" || echo "pending")"
  },
  "reports": {
    "pythonConverters": "$([ -f "$REPORT_DIR/python-converters-report.json" ] && echo "$REPORT_DIR/python-converters-report.json" || echo "not generated")",
    "javascriptPipeline": "$([ -f "$REPORT_DIR/javascript-pipeline-report.json" ] && echo "$REPORT_DIR/javascript-pipeline-report.json" || echo "not generated")",
    "interoperability": "$([ -f "$REPORT_DIR/interoperability-report.json" ] && echo "$REPORT_DIR/interoperability-report.json" || echo "not generated")"
  },
  "testData": {
    "domains": 6,
    "filesPerDomain": 3,
    "totalTestFiles": 18
  }
}
EOF

    # Display final summary
    print_header "FINAL TEST SUMMARY"
    echo "Total Test Suites: $TOTAL_SUITES"
    echo -e "Passed: ${GREEN}$PASSED_SUITES${NC}"
    echo -e "Failed: ${RED}$FAILED_SUITES${NC}"
    echo ""

    if [ $PASSED_SUITES -eq $TOTAL_SUITES ]; then
        echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
        EXIT_CODE=0
    elif [ $PASSED_SUITES -gt 0 ]; then
        echo -e "${YELLOW}⚠ SOME TESTS FAILED${NC}"
        EXIT_CODE=1
    else
        echo -e "${RED}✗ ALL TESTS FAILED${NC}"
        EXIT_CODE=2
    fi

    echo ""
    echo "Consolidated Report: $CONSOLIDATED_REPORT"
    echo ""

    # Display individual reports
    print_section "Individual Reports"

    if [ -f "$REPORT_DIR/python-converters-report.json" ]; then
        echo "  - Python Converters: $REPORT_DIR/python-converters-report.json"
    fi

    if [ -f "$REPORT_DIR/javascript-pipeline-report.json" ]; then
        echo "  - JavaScript Pipeline: $REPORT_DIR/javascript-pipeline-report.json"
    fi

    if [ -f "$REPORT_DIR/interoperability-report.json" ]; then
        echo "  - Interoperability: $REPORT_DIR/interoperability-report.json"
    fi

    echo ""
    print_header "TEST RUN COMPLETE"

    exit $EXIT_CODE
}

# Run main function
main "$@"
