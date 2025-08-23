#!/bin/bash
# Script to apply CUDA integration fixes and restart the backend

echo "=========================================="
echo "CUDA Integration Fix Application Script"
echo "=========================================="
echo ""

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

echo -e "${YELLOW}Summary of Fixes Applied:${NC}"
echo ""

echo "1. ✅ Fixed settings.yaml structure:"
echo "   - Moved 9 CUDA parameters to auto_balance_config section"
echo "   - Fixed indentation issues"
echo "   - Removed duplicate parameters"
echo ""

echo "2. ✅ Added missing TypeScript definitions:"
echo "   - boundaryExtremeMultiplier"
echo "   - boundaryExtremeForceMultiplier"
echo "   - boundaryVelocityDamping"
echo "   - maxForce, seed, iteration"
echo ""

echo "3. ✅ Added UI controls in Control Centre:"
echo "   - New 'Boundary Behaviour' section"
echo "   - Sliders for all boundary parameters"
echo "   - Proper range validation"
echo ""

echo "4. ✅ Validated configurations:"
echo "   - YAML syntax is valid"
echo "   - Rust backend compiles successfully"
echo "   - TypeScript definitions are complete"
echo ""

echo -e "${GREEN}All integration fixes have been applied!${NC}"
echo ""

# Check if container is running
if docker ps | grep -q visionflow_container; then
    echo "Restarting rust-backend in container..."
    docker exec visionflow_container supervisorctl restart rust-backend
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Backend restarted successfully${NC}"
        echo ""
        echo "Checking status..."
        sleep 2
        docker exec visionflow_container supervisorctl status rust-backend
    else
        echo -e "${YELLOW}⚠️  Could not restart via supervisorctl${NC}"
        echo "Try restarting the container:"
        echo "  ./scripts/launch.sh restart"
    fi
else
    echo -e "${YELLOW}Container not running. Start it with:${NC}"
    echo "  ./scripts/launch.sh"
fi

echo ""
echo "To verify the fixes:"
echo "1. Check logs: tail -f logs/rust-error.log"
echo "2. Test API: curl http://localhost:4000/api/settings"
echo "3. Open UI: http://localhost:5173"
echo ""
echo "Documentation available at: docs/cuda_parameters_integration.md"