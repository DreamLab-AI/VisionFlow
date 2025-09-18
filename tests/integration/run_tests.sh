#!/bin/bash
"""
Integration Test Runner Script

Bash script to set up environment and run integration tests.
"""

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}Integration Test Runner${NC}"
echo "========================"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python version: $PYTHON_VERSION"
    else
        print_error "Python 3 not found. Please install Python 3.8 or later."
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing test dependencies..."
    
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"
        print_status "Dependencies installed"
    else
        print_warning "requirements.txt not found, installing basic dependencies"
        python3 -m pip install pytest pytest-asyncio requests websocket-client
    fi
}

# Check services
check_services() {
    print_status "Checking service availability..."
    
    # Check TCP server
    if nc -z localhost 9500 2>/dev/null; then
        print_status "TCP server (9500) - ✓ Available"
    else
        print_warning "TCP server (9500) - ✗ Not available"
    fi
    
    # Check WebSocket bridge
    if nc -z localhost 3002 2>/dev/null; then
        print_status "WebSocket bridge (3002) - ✓ Available"
    else
        print_warning "WebSocket bridge (3002) - ✗ Not available"
    fi
    
    # Check health endpoint
    if curl -s "http://localhost:9501/health" > /dev/null 2>&1; then
        print_status "Health endpoint (9501) - ✓ Available"
    else
        print_warning "Health endpoint (9501) - ✗ Not available"
    fi
    
    # Check GPU container
    if docker ps --filter "name=mcp-gui-tools" --format "table {{.Names}}" | grep -q "mcp-gui-tools"; then
        print_status "GPU container - ✓ Running"
    else
        print_warning "GPU container - ✗ Not running"
    fi
}

# Run specific test suite
run_test_suite() {
    local test_file="$1"
    print_status "Running test suite: $test_file"
    
    cd "$SCRIPT_DIR"
    
    if python3 -m pytest "$test_file" -v --tb=short; then
        print_status "✓ $test_file - PASSED"
        return 0
    else
        print_error "✗ $test_file - FAILED"
        return 1
    fi
}

# Run all tests
run_all_tests() {
    print_status "Running all integration tests..."
    
    cd "$SCRIPT_DIR"
    
    local failed_tests=0
    local total_tests=0
    
    # List of test files
    local test_files=(
        "tcp_persistence_test.py"
        "gpu_stability_test.py"
        "client_polling_test.py"
        "security_validation_test.py"
    )
    
    echo ""
    echo "Test Suites to Run:"
    for test_file in "${test_files[@]}"; do
        echo "  - $test_file"
        ((total_tests++))
    done
    echo ""
    
    # Run each test suite
    for test_file in "${test_files[@]}"; do
        echo -e "${BLUE}Running: $test_file${NC}"
        echo "----------------------------------------"
        
        if [ -f "$test_file" ]; then
            if run_test_suite "$test_file"; then
                echo ""
            else
                ((failed_tests++))
                echo ""
            fi
        else
            print_warning "Test file not found: $test_file"
            ((failed_tests++))
        fi
    done
    
    # Summary
    echo -e "${BLUE}Test Summary${NC}"
    echo "============"
    echo "Total test suites: $total_tests"
    echo "Failed test suites: $failed_tests"
    echo "Passed test suites: $((total_tests - failed_tests))"
    
    if [ $failed_tests -eq 0 ]; then
        print_status "All tests passed! ✓"
        return 0
    else
        print_error "$failed_tests test suite(s) failed ✗"
        return 1
    fi
}

# Generate comprehensive report
generate_report() {
    print_status "Generating comprehensive test report..."
    
    cd "$SCRIPT_DIR"
    
    if python3 test_runner.py; then
        print_status "Test report generated successfully"
        
        # Show report location
        if [ -f "latest_test_report.md" ]; then
            echo ""
            print_status "Latest report: $SCRIPT_DIR/latest_test_report.md"
            echo ""
            echo "Report preview:"
            echo "---------------"
            head -20 "latest_test_report.md"
            echo "..."
        fi
    else
        print_error "Failed to generate test report"
        return 1
    fi
}

# Cleanup function
cleanup() {
    print_status "Cleaning up test artifacts..."
    cd "$SCRIPT_DIR"
    
    # Remove temporary files
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_status "Cleanup completed"
}

# Main execution
main() {
    local command="${1:-all}"
    
    case "$command" in
        "setup")
            check_python
            install_dependencies
            check_services
            ;;
        "tcp")
            run_test_suite "tcp_persistence_test.py"
            ;;
        "gpu")
            run_test_suite "gpu_stability_test.py"
            ;;
        "polling")
            run_test_suite "client_polling_test.py"
            ;;
        "security")
            run_test_suite "security_validation_test.py"
            ;;
        "all")
            check_python
            install_dependencies
            check_services
            echo ""
            run_all_tests
            ;;
        "report")
            generate_report
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  setup     - Install dependencies and check services"
            echo "  tcp       - Run TCP persistence tests only"
            echo "  gpu       - Run GPU stability tests only"
            echo "  polling   - Run client polling tests only"
            echo "  security  - Run security validation tests only"
            echo "  all       - Run all test suites (default)"
            echo "  report    - Generate comprehensive test report"
            echo "  cleanup   - Clean up test artifacts"
            echo "  help      - Show this help message"
            echo ""
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run main function with all arguments
main "$@"