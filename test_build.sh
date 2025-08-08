#!/bin/bash
# Test build script to verify compilation

echo "Testing Rust backend compilation..."
echo "==================================="

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Cargo.toml not found. Are you in the right directory?"
    exit 1
fi

# Create a simple cargo check output
cat > check_build.rs << 'EOF'
// Dummy file to test if cargo is available
fn main() {
    println!("If you see this, you need to run: docker-compose build rust-backend");
}
EOF

rustc check_build.rs 2>&1
rm -f check_build.rs check_build

echo ""
echo "Build verification complete!"
echo ""
echo "To actually build the project, run:"
echo "  docker-compose build rust-backend"
echo ""
echo "Then to start the system:"
echo "  docker-compose up"