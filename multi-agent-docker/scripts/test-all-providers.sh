#!/bin/bash
# Test All Model Providers for Agentic Flow
# Validates connectivity and basic functionality for each provider

set -e

echo "🧪 Testing All Model Providers"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test task
TASK="Write a Python function that prints 'Hello World'"
MAX_TOKENS=50

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Function to test a provider
test_provider() {
    local name=$1
    local provider=$2
    local model=$3
    local required_key=$4

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if required API key is set
    if [ -n "$required_key" ] && [ -z "${!required_key}" ]; then
        echo -e "${YELLOW}⚠️  SKIPPED: $required_key not set${NC}"
        ((SKIPPED++))
        return
    fi

    # Run test
    echo "Command: agentic-flow --agent coder --task \"$TASK\" --provider $provider --max-tokens $MAX_TOKENS"
    echo ""

    if agentic-flow --agent coder --task "$TASK" --provider "$provider" --max-tokens "$MAX_TOKENS" 2>&1 | head -20; then
        echo ""
        echo -e "${GREEN}✅ $name test PASSED${NC}"
        ((PASSED++))
    else
        echo ""
        echo -e "${RED}❌ $name test FAILED${NC}"
        ((FAILED++))
    fi
}

# Function to test Xinference connectivity
test_xinference_connectivity() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: Xinference Connectivity"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if curl -s --connect-timeout 5 http://172.18.0.11:9997/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Xinference is reachable at http://172.18.0.11:9997${NC}"
        echo ""
        echo "Available models:"
        curl -s http://172.18.0.11:9997/v1/models | jq -r '.data[]?.id' 2>/dev/null | head -5 || echo "  (model list unavailable)"
        return 0
    else
        echo -e "${RED}❌ Xinference not reachable${NC}"
        echo "  Ensure RAGFlow network is connected"
        return 1
    fi
}

# Function to test GPU
test_gpu() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: GPU Availability"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if nvidia-smi > /dev/null 2>&1; then
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    elif rocm-smi > /dev/null 2>&1; then
        echo -e "${GREEN}✅ AMD GPU detected${NC}"
        rocm-smi
    else
        echo -e "${YELLOW}⚠️  No GPU detected (CPU-only mode)${NC}"
    fi
}

# =============================================================================
# Main Test Sequence
# =============================================================================

# Check GPU first
test_gpu

# Check Xinference connectivity
test_xinference_connectivity
XINFERENCE_AVAILABLE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Starting Provider Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Google Gemini
test_provider \
    "Google Gemini" \
    "gemini" \
    "gemini-2.5-flash" \
    "GOOGLE_GEMINI_API_KEY"

# 2. OpenAI
test_provider \
    "OpenAI GPT-4o" \
    "openai" \
    "gpt-4o" \
    "OPENAI_API_KEY"

# 3. Anthropic Claude
test_provider \
    "Anthropic Claude" \
    "anthropic" \
    "claude-3-5-sonnet-20241022" \
    "ANTHROPIC_API_KEY"

# 4. OpenRouter
test_provider \
    "OpenRouter (Llama 3.1)" \
    "openrouter" \
    "meta-llama/llama-3.1-8b-instruct" \
    "OPENROUTER_API_KEY"

# 5. Xinference (if available)
if [ $XINFERENCE_AVAILABLE -eq 0 ]; then
    test_provider \
        "Xinference (Local)" \
        "xinference" \
        "auto" \
        ""
else
    echo ""
    echo -e "${YELLOW}⚠️  SKIPPED: Xinference not available${NC}"
    ((SKIPPED++))
fi

# 6. ONNX (offline)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing: ONNX (Offline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if ONNX model exists
if [ -f "/home/devuser/models/phi-4.onnx" ] || [ -d "/home/devuser/models/phi-4" ]; then
    echo "Command: agentic-flow --agent coder --task \"$TASK\" --provider onnx --local-only --max-tokens $MAX_TOKENS"
    echo ""

    if agentic-flow --agent coder --task "$TASK" --provider onnx --local-only --max-tokens "$MAX_TOKENS" 2>&1 | head -20; then
        echo ""
        echo -e "${GREEN}✅ ONNX test PASSED${NC}"
        ((PASSED++))
    else
        echo ""
        echo -e "${RED}❌ ONNX test FAILED${NC}"
        ((FAILED++))
    fi
else
    echo -e "${YELLOW}⚠️  SKIPPED: ONNX model not found at /home/devuser/models/phi-4.onnx${NC}"
    echo "  Download with: npx agentic-flow --provider onnx --download-model"
    ((SKIPPED++))
fi

# 7. Intelligent Router Test
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing: Intelligent Router"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Command: agentic-flow --agent coder --task \"$TASK\" --optimize --priority performance --max-tokens $MAX_TOKENS"
echo ""

if agentic-flow --agent coder --task "$TASK" --optimize --priority performance --max-tokens "$MAX_TOKENS" 2>&1 | head -20; then
    echo ""
    echo -e "${GREEN}✅ Router test PASSED${NC}"
    ((PASSED++))
else
    echo ""
    echo -e "${RED}❌ Router test FAILED${NC}"
    ((FAILED++))
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✅ Passed:  $PASSED${NC}"
echo -e "${RED}❌ Failed:  $FAILED${NC}"
echo -e "${YELLOW}⚠️  Skipped: $SKIPPED${NC}"
echo ""

TOTAL=$((PASSED + FAILED + SKIPPED))
echo "Total tests: $TOTAL"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All available providers are working!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}⚠️  Some tests failed. Check the output above for details.${NC}"
    exit 1
fi
