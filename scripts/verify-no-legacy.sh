#!/bin/bash
# verify-no-legacy.sh - Verification script to ensure ALL legacy code has been removed
#
# Usage: ./scripts/verify-no-legacy.sh
# Exit Code: 0 = Success (no legacy), 1 = Failure (legacy found)

set -e

PROJECT_ROOT="/home/devuser/workspace/project"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Legacy Code Verification Script"
echo "======================================"
echo ""

# Track overall status
OVERALL_STATUS=0

# Function to check for pattern and report
check_pattern() {
    local pattern=$1
    local description=$2
    local exclude_pattern=$3

    echo -n "Checking for $description... "

    if [ -n "$exclude_pattern" ]; then
        COUNT=$(grep -rn "$pattern" src/ --include="*.rs" | grep -v "$exclude_pattern" | wc -l)
    else
        COUNT=$(grep -rn "$pattern" src/ --include="*.rs" | wc -l)
    fi

    if [ "$COUNT" -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} (0 references)"
    else
        echo -e "${RED}❌ FAIL${NC} ($COUNT references found)"
        OVERALL_STATUS=1

        # Show first 5 references
        echo "  First references:"
        if [ -n "$exclude_pattern" ]; then
            grep -rn "$pattern" src/ --include="*.rs" | grep -v "$exclude_pattern" | head -5 | sed 's/^/    /'
        else
            grep -rn "$pattern" src/ --include="*.rs" | head -5 | sed 's/^/    /'
        fi
        echo ""
    fi
}

echo "## Phase 1: GraphServiceActor References"
check_pattern "GraphServiceActor" "GraphServiceActor" "^src/actors/graph_actor.rs:"
check_pattern "Addr<GraphServiceActor>" "GraphServiceActor addresses" ""
check_pattern "graph_service_addr.*GraphServiceActor" "graph_service_addr with legacy type" ""
check_pattern "TransitionalGraphSupervisor" "TransitionalGraphSupervisor (should be removed)" ""

echo ""
echo "## Phase 2: PhysicsOrchestratorActor References"
check_pattern "PhysicsOrchestratorActor" "PhysicsOrchestratorActor" "^src/actors/physics_orchestrator_actor.rs:"
check_pattern "Addr<PhysicsOrchestratorActor>" "PhysicsOrchestratorActor addresses" ""

echo ""
echo "## Phase 3: GPU Actor References"
check_pattern "GPUManagerActor" "GPUManagerActor" "^src/actors/gpu/gpu_manager_actor.rs:"
check_pattern "ForceComputeActor" "ForceComputeActor" "^src/actors/gpu/force_compute_actor.rs:"
check_pattern "ClusteringActor" "ClusteringActor" "^src/actors/gpu/clustering_actor.rs:"
check_pattern "AnomalyDetectionActor" "AnomalyDetectionActor" "^src/actors/gpu/anomaly_detection_actor.rs:"
check_pattern "GPUResourceActor" "GPUResourceActor" "^src/actors/gpu/gpu_resource_actor.rs:"
check_pattern "StressMajorizationActor" "StressMajorizationActor" "^src/actors/gpu/stress_majorization_actor.rs:"
check_pattern "OntologyConstraintActor" "OntologyConstraintActor" "^src/actors/gpu/ontology_constraint_actor.rs:"
check_pattern "ConstraintActor" "ConstraintActor (GPU)" "^src/actors/gpu/constraint_actor.rs:"

echo ""
echo "## Phase 4: Legacy Files Should Not Exist"

# Check if legacy files exist
check_file_exists() {
    local file=$1
    local description=$2

    echo -n "Checking if $description exists... "

    if [ -f "$file" ]; then
        echo -e "${RED}❌ FAIL${NC} (file should be deleted)"
        OVERALL_STATUS=1
    else
        echo -e "${GREEN}✅ PASS${NC} (file deleted)"
    fi
}

check_file_exists "src/actors/graph_actor.rs" "GraphServiceActor file"
check_file_exists "src/actors/graph_messages.rs" "GraphServiceActor messages"
check_file_exists "src/actors/graph_service_supervisor.rs" "GraphServiceSupervisor"
check_file_exists "src/actors/physics_orchestrator_actor.rs" "PhysicsOrchestratorActor"
check_file_exists "src/actors/gpu/gpu_manager_actor.rs" "GPUManagerActor"
check_file_exists "src/actors/gpu/force_compute_actor.rs" "ForceComputeActor"
check_file_exists "src/actors/gpu/clustering_actor.rs" "ClusteringActor"
check_file_exists "src/actors/gpu/anomaly_detection_actor.rs" "AnomalyDetectionActor"
check_file_exists "src/actors/gpu/gpu_resource_actor.rs" "GPUResourceActor"
check_file_exists "src/actors/gpu/stress_majorization_actor.rs" "StressMajorizationActor"
check_file_exists "src/actors/gpu/ontology_constraint_actor.rs" "OntologyConstraintActor"
check_file_exists "src/actors/gpu/constraint_actor.rs" "GPU ConstraintActor"

echo ""
echo "## Cargo Check"
echo -n "Running cargo check... "

if cargo check 2>&1 | tee /tmp/cargo-check-migration.log > /dev/null; then
    ERRORS=$(grep -c "^error" /tmp/cargo-check-migration.log || true)
    WARNINGS=$(grep -c "^warning" /tmp/cargo-check-migration.log || true)

    if [ "$ERRORS" -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} (0 errors, $WARNINGS warnings)"
    else
        echo -e "${RED}❌ FAIL${NC} ($ERRORS errors, $WARNINGS warnings)"
        OVERALL_STATUS=1
        echo "  Errors:"
        grep "^error" /tmp/cargo-check-migration.log | head -10 | sed 's/^/    /'
    fi
else
    echo -e "${RED}❌ FAIL${NC} (cargo check failed)"
    OVERALL_STATUS=1
fi

echo ""
echo "## Test Suite"
echo -n "Running cargo test... "

if cargo test --quiet 2>&1 | tee /tmp/cargo-test-migration.log > /dev/null; then
    echo -e "${GREEN}✅ PASS${NC} (all tests passed)"
else
    echo -e "${RED}❌ FAIL${NC} (some tests failed)"
    OVERALL_STATUS=1
    echo "  See /tmp/cargo-test-migration.log for details"
fi

echo ""
echo "======================================"
echo "## Summary"
echo "======================================"

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ SUCCESS: All legacy code has been removed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run performance benchmarks: cargo bench"
    echo "  2. Check code coverage: cargo tarpaulin"
    echo "  3. Review and update documentation"
    echo "  4. Celebrate! 🎉"
    exit 0
else
    echo -e "${RED}❌ FAILURE: Legacy code still exists${NC}"
    echo ""
    echo "Please review the failures above and:"
    echo "  1. Remove remaining legacy references"
    echo "  2. Delete legacy files"
    echo "  3. Fix any cargo errors"
    echo "  4. Re-run this script"
    exit 1
fi
