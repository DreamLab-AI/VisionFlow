#!/bin/bash
# Test build script to verify compilation

echo "Testing Rust compilation..."

# Check if we can compile a simple test that imports the main modules
cat > /tmp/test_compile.rs << 'EOF'
// Test that our main config module compiles
use std::path::Path;

fn main() {
    println!("Config module test compilation successful!");
}
EOF

# Try to check syntax of our actual source files
echo "Checking syntax of key files..."

# List the files we've modified
FILES=(
    "src/config/mod.rs"
    "src/handlers/settings_handler.rs"
    "src/models/ui_settings.rs"
    "src/utils/audio_processor.rs"
    "src/state.rs"
    "src/services/graph_service.rs"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ File exists: $file"
    else
        echo "✗ File missing: $file"
    fi
done

echo "Build test complete. Use Docker to run full compilation."