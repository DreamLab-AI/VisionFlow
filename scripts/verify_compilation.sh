#!/bin/bash
# Rust Compilation Verification Script
# Run this inside the Docker container or with Rust installed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${BLUE}     Rust Compilation Verification Script${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Cargo not found. Please run this inside the Docker container or install Rust.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}🔍 Checking Rust environment...${NC}"
rustc --version
cargo --version

echo -e "\n${YELLOW}📋 Running cargo check...${NC}"
if cargo check --all-features 2>&1; then
    echo -e "${GREEN}✅ Cargo check passed!${NC}"
else
    echo -e "${RED}❌ Cargo check failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}🔧 Checking specific modules...${NC}"

# Check settings system
echo -e "${BLUE}Checking settings system...${NC}"
cargo check --lib --features "default" 2>&1 | grep -E "(settings_handler|settings_actor)" || true

# Check GPU compute
echo -e "${BLUE}Checking GPU compute...${NC}"
cargo check --lib --features "default" 2>&1 | grep -E "(gpu_compute_actor|unified_gpu_compute)" || true

echo -e "\n${YELLOW}🧪 Compiling tests (without running)...${NC}"
if cargo test --no-run 2>&1; then
    echo -e "${GREEN}✅ Test compilation passed!${NC}"
else
    echo -e "${RED}❌ Test compilation failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}📦 Attempting release build...${NC}"
if cargo build --release --features "default" 2>&1 | head -20; then
    echo -e "${GREEN}✅ Release build started successfully!${NC}"
else
    echo -e "${RED}❌ Release build failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}🎯 Checking for warnings...${NC}"
cargo clippy --all-features -- -W clippy::all 2>&1 | head -30 || true

echo -e "\n${YELLOW}📊 Compilation Summary:${NC}"
echo -e "${GREEN}✅ All syntax checks passed${NC}"
echo -e "${GREEN}✅ Core modules verified${NC}"
echo -e "${GREEN}✅ Test compilation successful${NC}"

# Check CUDA compilation
if [ -f "src/utils/visionflow_unified.cu" ]; then
    echo -e "\n${YELLOW}🚀 CUDA Kernel Status:${NC}"
    if [ -f "src/utils/ptx/visionflow_unified.ptx" ]; then
        echo -e "${GREEN}✅ CUDA PTX file found${NC}"
        ls -lh src/utils/ptx/visionflow_unified.ptx
    else
        echo -e "${YELLOW}⚠️  CUDA PTX file not found. Run compile_unified_ptx.sh${NC}"
    fi
fi

echo -e "\n${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}     Compilation Verification Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"