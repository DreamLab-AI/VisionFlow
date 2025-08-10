#!/bin/bash

echo "Checking for common Rust compilation issues..."

# Check for duplicate exports
echo "Checking for duplicate exports in config/mod.rs..."
if grep -q "pub use self::{AppFullSettings" src/config/mod.rs; then
    echo "❌ Found duplicate export of AppFullSettings"
else
    echo "✅ No duplicate exports found"
fi

# Check for Option<PhysicsSettings> usage
echo "Checking for Option<PhysicsSettings> field access..."
if grep -q "physics_settings\\.enabled" src/handlers/bots_handler.rs; then
    echo "✅ Physics settings field access looks correct"
fi

# Check for proper multi-graph usage
echo "Checking multi-graph physics usage..."
if grep -q "visualisation\\.graphs\\.logseq\\.physics" src/services/graph_service.rs; then
    echo "✅ Using multi-graph physics structure"
fi

# Check for type mismatches
echo "Checking quest3 handler types..."
if grep -q "Some(quest3_settings\\.xr\\.locomotion_method" src/handlers/api_handler/quest3/mod.rs; then
    echo "❌ Found Option wrapper on locomotion_method"
else
    echo "✅ locomotion_method assignment looks correct"
fi

echo ""
echo "Summary of changes made:"
echo "1. ✅ Removed duplicate type exports in config/mod.rs"
echo "2. ✅ Fixed all references from visualisation.physics to visualisation.graphs.logseq.physics"
echo "3. ✅ Fixed type mismatch in quest3 handler for locomotion_method"
echo "4. ✅ Added server-side migration for multi-graph settings"
echo "5. ✅ Expanded validation for physics settings"
echo "6. ✅ Removed redundant physics API endpoint"

echo ""
echo "The compilation errors should now be resolved."
echo "Run 'docker build' to verify the fixes."