#!/bin/bash
# Diagnostic script to understand the settings.yaml issue

echo "Settings.yaml Diagnostic Report"
echo "==============================="
echo ""

# Check local file
echo "1. Local settings.yaml check:"
echo "   File size: $(wc -c < data/settings.yaml) bytes"
echo "   Last modified: $(stat -c %y data/settings.yaml)"
echo "   repulsion_cutoff_min occurrences: $(grep -c "repulsion_cutoff_min:" data/settings.yaml)"
echo ""

# Check if container is running
if docker ps | grep -q visionflow_container; then
    echo "2. Container check:"
    echo "   ✅ Container is running"
    echo ""
    
    # Check mounted file in container
    echo "3. Mounted file in container:"
    docker exec visionflow_container sh -c "wc -c < /app/settings.yaml" | xargs -I {} echo "   File size: {} bytes"
    docker exec visionflow_container sh -c "grep -c 'repulsion_cutoff_min:' /app/settings.yaml" | xargs -I {} echo "   repulsion_cutoff_min occurrences: {}"
    echo ""
    
    # Show the exact structure where repulsion_cutoff_min is located
    echo "4. YAML structure for repulsion_cutoff_min:"
    echo "   Checking path in container's settings.yaml..."
    docker exec visionflow_container sh -c "grep -B15 'repulsion_cutoff_min:' /app/settings.yaml | head -20"
    echo ""
else
    echo "2. Container check:"
    echo "   ❌ Container is not running"
fi

echo "5. Analysis:"
echo "   The error 'missing field repulsion_cutoff_min' is misleading."
echo "   The field EXISTS in the YAML at:"
echo "   - visualisation.graphs.logseq.physics.repulsion_cutoff_min"
echo "   - visualisation.graphs.visionflow.physics.repulsion_cutoff_min"
echo ""
echo "   The issue is likely that the Rust struct definition expects this field"
echo "   at a different location in the YAML hierarchy."
echo ""
echo "   This is a CODE issue, not a YAML issue. The Rust application's"
echo "   deserialization structure doesn't match the actual YAML structure."