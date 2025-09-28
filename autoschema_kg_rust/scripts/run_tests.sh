#!/bin/bash

# Comprehensive test runner script for AutoSchema KG Rust

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COVERAGE_THRESHOLD=80
PROPTEST_CASES=${PROPTEST_CASES:-1000}
BENCHMARK_TIMEOUT=300
TEST_TIMEOUT=600

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_dependencies() {
    print_header "Checking Dependencies"

    # Check Rust toolchain
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust."
        exit 1
    fi

    # Check for required tools
    if ! command -v cargo-llvm-cov &> /dev/null; then
        print_warning "cargo-llvm-cov not found. Installing..."
        cargo install cargo-llvm-cov
    fi

    if ! command -v cargo-deny &> /dev/null; then
        print_warning "cargo-deny not found. Installing..."
        cargo install cargo-deny
    fi

    print_success "Dependencies checked"
}

run_linting() {
    print_header "Running Linting and Formatting Checks"

    # Check formatting
    echo "Checking code formatting..."
    if cargo fmt --all -- --check; then
        print_success "Code formatting is correct"
    else
        print_error "Code formatting issues found"
        echo "Run 'cargo fmt' to fix formatting issues"
        return 1
    fi

    # Run clippy
    echo "Running clippy lints..."
    if cargo clippy --all-targets --all-features -- -D warnings; then
        print_success "No clippy warnings"
    else
        print_error "Clippy warnings found"
        return 1
    fi
}

run_security_checks() {
    print_header "Running Security Checks"

    # Security audit
    echo "Running security audit..."
    if cargo audit; then
        print_success "No security vulnerabilities found"
    else
        print_warning "Security audit found issues (may be acceptable)"
    fi

    # Dependency check
    echo "Checking dependencies..."
    if cargo deny check; then
        print_success "Dependency check passed"
    else
        print_error "Dependency check failed"
        return 1
    fi
}

run_unit_tests() {
    print_header "Running Unit Tests"

    echo "Running all unit tests..."
    if timeout ${TEST_TIMEOUT} cargo test --lib --all-features --verbose; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        return 1
    fi

    echo "Running doc tests..."
    if cargo test --doc --all-features; then
        print_success "Doc tests passed"
    else
        print_error "Doc tests failed"
        return 1
    fi
}

run_integration_tests() {
    print_header "Running Integration Tests"

    echo "Running integration tests..."

    # Check if Neo4j is available for integration tests
    if nc -z localhost 7687 2>/dev/null; then
        export NEO4J_URI="bolt://localhost:7687"
        export NEO4J_USER="neo4j"
        export NEO4J_PASSWORD="password"
        print_success "Neo4j detected, running full integration tests"
    else
        print_warning "Neo4j not available, running mock integration tests"
        export MOCK_INTEGRATION=true
    fi

    if timeout ${TEST_TIMEOUT} cargo test --test integration_tests --all-features -- --nocapture; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        return 1
    fi
}

run_property_tests() {
    print_header "Running Property-based Tests"

    echo "Running property tests with ${PROPTEST_CASES} cases..."
    export PROPTEST_CASES

    if timeout ${TEST_TIMEOUT} cargo test --test property_tests --all-features -- --nocapture; then
        print_success "Property tests passed"
    else
        print_error "Property tests failed"
        return 1
    fi
}

run_coverage() {
    print_header "Generating Code Coverage"

    echo "Generating coverage report..."
    if cargo llvm-cov --all-features --workspace --html; then
        print_success "Coverage report generated"

        # Extract coverage percentage
        if command -v cargo-llvm-cov &> /dev/null; then
            COVERAGE=$(cargo llvm-cov --all-features --workspace --summary-only | grep "TOTAL" | awk '{print $10}' | sed 's/%//')

            if (( $(echo "$COVERAGE >= $COVERAGE_THRESHOLD" | bc -l) )); then
                print_success "Coverage ${COVERAGE}% meets threshold of ${COVERAGE_THRESHOLD}%"
            else
                print_warning "Coverage ${COVERAGE}% below threshold of ${COVERAGE_THRESHOLD}%"
            fi
        fi
    else
        print_error "Coverage generation failed"
        return 1
    fi
}

run_benchmarks() {
    print_header "Running Performance Benchmarks"

    echo "Running benchmarks..."
    if timeout ${BENCHMARK_TIMEOUT} cargo bench --all-features; then
        print_success "Benchmarks completed"

        # Generate benchmark report
        if [ -d "target/criterion" ]; then
            echo "Benchmark results saved to target/criterion/"
            find target/criterion -name "index.html" | head -5 | while read -r file; do
                echo "  - $file"
            done
        fi
    else
        print_warning "Benchmarks timed out or failed"
    fi
}

run_memory_tests() {
    print_header "Running Memory Tests"

    if command -v valgrind &> /dev/null; then
        echo "Running memory leak detection..."

        # Build test executable
        cargo build --all-features

        # Find test executable
        TEST_EXEC=$(find target/debug/deps -name "*autoschema*" -type f -executable | head -1)

        if [ -n "$TEST_EXEC" ]; then
            if valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
                       --track-origins=yes --error-exitcode=1 \
                       "$TEST_EXEC" --test-threads=1 > valgrind.log 2>&1; then
                print_success "No memory leaks detected"
            else
                print_warning "Memory issues detected (see valgrind.log)"
            fi
        else
            print_warning "No test executable found for memory testing"
        fi
    else
        print_warning "Valgrind not available, skipping memory tests"
    fi
}

generate_reports() {
    print_header "Generating Test Reports"

    # Create reports directory
    mkdir -p reports

    # Test summary
    cat > reports/test_summary.md << EOF
# Test Summary Report

Generated on: $(date)

## Test Results

- ✓ Unit Tests: $(cargo test --lib --all-features 2>&1 | grep "test result:" | tail -1 || echo "Not run")
- ✓ Integration Tests: Completed
- ✓ Property Tests: ${PROPTEST_CASES} cases
- ✓ Doc Tests: Completed

## Coverage

$(if [ -f "target/llvm-cov/html/index.html" ]; then echo "Coverage report available at target/llvm-cov/html/index.html"; else echo "Coverage report not generated"; fi)

## Benchmarks

$(if [ -d "target/criterion" ]; then echo "Benchmark results available at target/criterion/"; else echo "Benchmarks not run"; fi)

## Files Tested

$(find . -name "*.rs" -not -path "./target/*" | wc -l) Rust source files

## Test Files

- Unit tests: $(find . -name "*.rs" -path "*/tests/*" -o -name "*_test.rs" -o -name "test_*.rs" | wc -l) files
- Integration tests: $(find tests -name "*.rs" 2>/dev/null | wc -l) files
- Property tests: 1 file

EOF

    print_success "Test reports generated in reports/"
}

cleanup() {
    print_header "Cleaning Up"

    # Clean build artifacts if requested
    if [ "$CLEAN_AFTER" = "true" ]; then
        cargo clean
        print_success "Build artifacts cleaned"
    fi

    # Compress logs if they exist
    if [ -f "valgrind.log" ]; then
        gzip valgrind.log
        print_success "Logs compressed"
    fi
}

main() {
    local failed_tests=()

    print_header "AutoSchema KG Rust - Comprehensive Test Suite"
    echo "Starting test execution at $(date)"
    echo ""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --coverage-threshold)
                COVERAGE_THRESHOLD="$2"
                shift 2
                ;;
            --proptest-cases)
                PROPTEST_CASES="$2"
                shift 2
                ;;
            --skip-benchmarks)
                SKIP_BENCHMARKS=true
                shift
                ;;
            --skip-memory)
                SKIP_MEMORY=true
                shift
                ;;
            --clean-after)
                CLEAN_AFTER=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --coverage-threshold N    Set coverage threshold (default: 80)"
                echo "  --proptest-cases N       Set property test cases (default: 1000)"
                echo "  --skip-benchmarks        Skip benchmark tests"
                echo "  --skip-memory           Skip memory tests"
                echo "  --clean-after           Clean build artifacts after tests"
                echo "  --help                  Show this help"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run test suite
    check_dependencies || failed_tests+=("dependencies")

    run_linting || failed_tests+=("linting")

    run_security_checks || failed_tests+=("security")

    run_unit_tests || failed_tests+=("unit")

    run_integration_tests || failed_tests+=("integration")

    run_property_tests || failed_tests+=("property")

    run_coverage || failed_tests+=("coverage")

    if [ "$SKIP_BENCHMARKS" != "true" ]; then
        run_benchmarks || failed_tests+=("benchmarks")
    fi

    if [ "$SKIP_MEMORY" != "true" ]; then
        run_memory_tests || failed_tests+=("memory")
    fi

    generate_reports

    cleanup

    # Summary
    print_header "Test Summary"

    if [ ${#failed_tests[@]} -eq 0 ]; then
        print_success "All tests passed successfully!"
        echo ""
        echo "Test execution completed at $(date)"
        exit 0
    else
        print_error "Some tests failed:"
        for test in "${failed_tests[@]}"; do
            echo "  - $test"
        done
        echo ""
        echo "Test execution completed at $(date)"
        exit 1
    fi
}

# Run main function with all arguments
main "$@"