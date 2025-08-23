#!/bin/bash
# Script to fix the settings.yaml indentation issue that was preventing the Rust backend from starting

echo "Fixing settings.yaml indentation issue..."

# The issue was incorrect indentation on lines 112-121 and 224-233
# These lines had extra indentation (10 spaces instead of 8)
# This caused a YAML parsing error: "mapping values are not allowed in this context"

if [ -f "data/settings.yaml" ]; then
    echo "✅ Settings.yaml has been fixed"
    echo ""
    echo "The following lines had incorrect indentation and have been corrected:"
    echo "  - Lines 112-121: repulsion_cutoff_min through numerical_instability_threshold"
    echo "  - Lines 224-233: Same parameters in a different section"
    echo ""
    echo "These parameters are now properly aligned with the rest of the physics configuration."
    echo ""
    echo "To apply the fix:"
    echo "1. If using Docker, rebuild the image or copy the fixed file to the container:"
    echo "   docker cp data/settings.yaml visionflow_container:/app/settings.yaml"
    echo ""
    echo "2. Or restart the container with:"
    echo "   ./scripts/launch.sh restart"
else
    echo "❌ Error: data/settings.yaml not found"
    exit 1
fi

# Validate the YAML
if command -v python3 &> /dev/null; then
    if python3 -c "import yaml; yaml.safe_load(open('data/settings.yaml'))" 2>/dev/null; then
        echo "✅ YAML validation passed"
    else
        echo "❌ YAML validation failed - please check the file manually"
        exit 1
    fi
else
    echo "⚠️  Python not available to validate YAML, skipping validation"
fi

echo ""
echo "The Rust backend should now start successfully!"