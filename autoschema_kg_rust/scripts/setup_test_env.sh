#!/bin/bash

# Setup test environment for AutoSchema KG Rust

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

install_rust_tools() {
    print_header "Installing Rust Tools"

    # Install required tools for testing
    local tools=(
        "cargo-llvm-cov"
        "cargo-deny"
        "cargo-audit"
        "cargo-outdated"
        "cargo-tarpaulin"
        "cargo-criterion"
    )

    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            echo "Installing $tool..."
            cargo install "$tool" || print_warning "Failed to install $tool"
        else
            print_success "$tool already installed"
        fi
    done
}

setup_neo4j() {
    print_header "Setting Up Neo4j Test Database"

    if command -v docker &> /dev/null; then
        echo "Starting Neo4j container for testing..."

        # Stop existing container if running
        docker stop neo4j-test 2>/dev/null || true
        docker rm neo4j-test 2>/dev/null || true

        # Start Neo4j container
        docker run -d \
            --name neo4j-test \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/testpassword \
            -e NEO4J_PLUGINS='["apoc"]' \
            -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
            neo4j:5.15

        # Wait for Neo4j to be ready
        echo "Waiting for Neo4j to start..."
        local retries=30
        while [ $retries -gt 0 ]; do
            if docker exec neo4j-test cypher-shell -u neo4j -p testpassword "RETURN 1" &>/dev/null; then
                print_success "Neo4j is ready"
                break
            fi
            retries=$((retries - 1))
            sleep 2
        done

        if [ $retries -eq 0 ]; then
            print_error "Neo4j failed to start"
            return 1
        fi

        # Create test environment variables
        cat > .env.test << EOF
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=testpassword
EOF

        print_success "Neo4j test environment configured"
    else
        print_warning "Docker not available, Neo4j tests will use mocks"
        echo "MOCK_NEO4J=true" > .env.test
    fi
}

create_test_data() {
    print_header "Creating Test Data"

    mkdir -p test_data

    # Create sample CSV files
    cat > test_data/companies.csv << 'EOF'
id,name,founder,founded_year,industry,headquarters
1,"Apple Inc","Steve Jobs",1976,"Technology","Cupertino, CA"
2,"Google","Larry Page",1998,"Technology","Mountain View, CA"
3,"Microsoft","Bill Gates",1975,"Technology","Redmond, WA"
4,"Amazon","Jeff Bezos",1994,"E-commerce","Seattle, WA"
5,"Tesla","Elon Musk",2003,"Automotive","Austin, TX"
EOF

    # Create sample JSON files
    cat > test_data/products.json << 'EOF'
{
  "products": [
    {
      "id": 1,
      "name": "iPhone",
      "company": "Apple Inc",
      "category": "Smartphone",
      "release_year": 2007
    },
    {
      "id": 2,
      "name": "Search Engine",
      "company": "Google",
      "category": "Web Service",
      "release_year": 1998
    },
    {
      "id": 3,
      "name": "Windows",
      "company": "Microsoft",
      "category": "Operating System",
      "release_year": 1985
    }
  ]
}
EOF

    # Create sample markdown files
    cat > test_data/readme.md << 'EOF'
# Technology Companies

## Apple Inc.
- **Founded**: 1976
- **Founder**: Steve Jobs, Steve Wozniak
- **Headquarters**: Cupertino, California
- **Industry**: Technology

Apple Inc. is an American multinational technology company that specializes in consumer electronics, software and online services.

## Google
- **Founded**: 1998
- **Founders**: Larry Page, Sergey Brin
- **Headquarters**: Mountain View, California
- **Parent**: Alphabet Inc.

Google is a multinational technology company focusing on search engine technology, online advertising, cloud computing, and more.

## Products

### Apple Products
- iPhone
- iPad
- Mac
- Apple Watch

### Google Products
- Search
- Gmail
- Android
- YouTube
EOF

    # Create sample text files
    cat > test_data/tech_article.txt << 'EOF'
The Evolution of Technology Companies

The technology industry has been transformed by several pioneering companies over the past few decades. Apple Inc., founded by Steve Jobs and Steve Wozniak in 1976, revolutionized personal computing and later the mobile phone industry with the iPhone.

Google, established by Larry Page and Sergey Brin in 1998, transformed how we access and search for information on the internet. Their search algorithm became the foundation for the world's most popular search engine.

Microsoft, led by Bill Gates since 1975, dominated the personal computer market with the Windows operating system and Microsoft Office suite.

These companies continue to shape the future of technology through innovation in artificial intelligence, cloud computing, and emerging technologies.
EOF

    print_success "Test data created in test_data/"
}

setup_environment_variables() {
    print_header "Setting Up Environment Variables"

    # Create development environment file
    cat > .env.development << 'EOF'
# Development Environment Variables
RUST_LOG=debug
RUST_BACKTRACE=1

# Test Configuration
TEST_MODE=true
PROPTEST_CASES=100
BENCHMARK_TIMEOUT=60

# LLM Configuration (for testing)
OPENAI_API_KEY=sk-test-key-for-testing
ANTHROPIC_API_KEY=test-key-for-testing

# Vector Store Configuration
VECTOR_DIMENSIONS=384
INDEX_TYPE=hnsw

# Neo4j Configuration (will be overridden by .env.test)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
EOF

    # Create production environment template
    cat > .env.production.template << 'EOF'
# Production Environment Variables Template
# Copy to .env.production and fill in real values

RUST_LOG=info
RUST_BACKTRACE=0

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Vector Store Configuration
VECTOR_DIMENSIONS=768
INDEX_TYPE=hnsw

# Neo4j Configuration
NEO4J_URI=bolt://your_neo4j_host:7687
NEO4J_USER=your_neo4j_user
NEO4J_PASSWORD=your_neo4j_password

# Performance Settings
MAX_WORKERS=8
BATCH_SIZE=50
CACHE_SIZE=1000
EOF

    print_success "Environment files created"
}

create_test_scripts() {
    print_header "Creating Test Helper Scripts"

    # Create quick test script
    cat > scripts/quick_test.sh << 'EOF'
#!/bin/bash
# Quick test runner for development

set -e

echo "Running quick tests..."

# Run only unit tests with coverage
cargo llvm-cov --lib --all-features --html

# Run clippy
cargo clippy --all-targets --all-features -- -W clippy::all

echo "Quick tests completed!"
echo "Coverage report: target/llvm-cov/html/index.html"
EOF

    # Create performance test script
    cat > scripts/perf_test.sh << 'EOF'
#!/bin/bash
# Performance testing script

set -e

echo "Running performance tests..."

# Run benchmarks
cargo bench --all-features

# Generate performance report
echo "Performance tests completed!"
echo "Benchmark results: target/criterion/"
EOF

    # Create memory test script
    cat > scripts/memory_test.sh << 'EOF'
#!/bin/bash
# Memory testing script

set -e

if ! command -v valgrind &> /dev/null; then
    echo "Installing valgrind..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y valgrind
    elif command -v brew &> /dev/null; then
        brew install valgrind
    else
        echo "Please install valgrind manually"
        exit 1
    fi
fi

echo "Running memory tests..."

# Build debug version
cargo build --all-features

# Find test executable
TEST_EXEC=$(find target/debug/deps -name "*autoschema*" -type f -executable | head -1)

if [ -n "$TEST_EXEC" ]; then
    valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
             --track-origins=yes --verbose \
             "$TEST_EXEC" --test-threads=1
else
    echo "No test executable found"
    exit 1
fi
EOF

    chmod +x scripts/*.sh

    print_success "Test helper scripts created"
}

install_system_dependencies() {
    print_header "Installing System Dependencies"

    # Detect OS and install dependencies
    if command -v apt-get &> /dev/null; then
        echo "Detected Ubuntu/Debian, installing dependencies..."
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            pkg-config \
            libssl-dev \
            netcat-openbsd \
            curl \
            git
        print_success "Ubuntu/Debian dependencies installed"

    elif command -v brew &> /dev/null; then
        echo "Detected macOS, installing dependencies..."
        brew install \
            openssl \
            pkg-config \
            netcat
        print_success "macOS dependencies installed"

    elif command -v dnf &> /dev/null; then
        echo "Detected Fedora, installing dependencies..."
        sudo dnf install -y \
            gcc \
            pkg-config \
            openssl-devel \
            nmap-ncat \
            curl \
            git
        print_success "Fedora dependencies installed"

    else
        print_warning "Unknown OS, please install build tools manually"
    fi
}

verify_setup() {
    print_header "Verifying Setup"

    local errors=0

    # Check Rust toolchain
    if cargo --version &>/dev/null; then
        print_success "Cargo is working"
    else
        print_error "Cargo is not working"
        errors=$((errors + 1))
    fi

    # Check if project builds
    if cargo check --all-features &>/dev/null; then
        print_success "Project builds successfully"
    else
        print_error "Project build failed"
        errors=$((errors + 1))
    fi

    # Check test data
    if [ -d "test_data" ] && [ -f "test_data/companies.csv" ]; then
        print_success "Test data is available"
    else
        print_error "Test data is missing"
        errors=$((errors + 1))
    fi

    # Check environment files
    if [ -f ".env.development" ]; then
        print_success "Environment files created"
    else
        print_error "Environment files missing"
        errors=$((errors + 1))
    fi

    if [ $errors -eq 0 ]; then
        print_success "Setup verification completed successfully!"
        return 0
    else
        print_error "Setup verification found $errors error(s)"
        return 1
    fi
}

cleanup_previous_setup() {
    print_header "Cleaning Previous Setup"

    # Stop any running Neo4j containers
    docker stop neo4j-test 2>/dev/null || true
    docker rm neo4j-test 2>/dev/null || true

    # Clean old test data
    rm -rf test_data_old
    if [ -d "test_data" ]; then
        mv test_data test_data_old
    fi

    # Clean old environment files
    rm -f .env.test.old .env.development.old
    if [ -f ".env.test" ]; then
        mv .env.test .env.test.old
    fi
    if [ -f ".env.development" ]; then
        mv .env.development .env.development.old
    fi

    print_success "Previous setup cleaned"
}

main() {
    print_header "AutoSchema KG Rust - Test Environment Setup"
    echo "Setting up comprehensive testing environment..."
    echo ""

    # Parse command line arguments
    local SKIP_DOCKER=false
    local SKIP_SYSTEM=false
    local CLEAN_FIRST=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-system)
                SKIP_SYSTEM=true
                shift
                ;;
            --clean-first)
                CLEAN_FIRST=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-docker    Skip Docker/Neo4j setup"
                echo "  --skip-system    Skip system dependencies"
                echo "  --clean-first    Clean previous setup first"
                echo "  --help          Show this help"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    if [ "$CLEAN_FIRST" = "true" ]; then
        cleanup_previous_setup
    fi

    if [ "$SKIP_SYSTEM" != "true" ]; then
        install_system_dependencies
    fi

    install_rust_tools

    if [ "$SKIP_DOCKER" != "true" ]; then
        setup_neo4j
    fi

    create_test_data

    setup_environment_variables

    create_test_scripts

    verify_setup

    print_header "Setup Complete!"
    echo "Your test environment is ready!"
    echo ""
    echo "Next steps:"
    echo "1. Run './scripts/run_tests.sh' to execute the full test suite"
    echo "2. Run './scripts/quick_test.sh' for fast development testing"
    echo "3. Check test coverage at target/llvm-cov/html/index.html"
    echo ""
    echo "Environment files created:"
    echo "- .env.development (for testing)"
    echo "- .env.production.template (copy and customize for production)"
    if [ -f ".env.test" ]; then
        echo "- .env.test (Neo4j test configuration)"
    fi
    echo ""
    echo "Test data available in test_data/"
}

# Run main function with all arguments
main "$@"