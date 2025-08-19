#!/bin/bash
# Security audit script for VisionFlow dependencies

set -e

echo "=== VisionFlow Security Audit ==="
echo "Checking for known vulnerabilities in dependencies..."
echo ""

# Check if cargo-audit is installed
if ! command -v cargo-audit &> /dev/null; then
    echo "cargo-audit not found. Installing..."
    cargo install cargo-audit || echo "Failed to install cargo-audit"
fi

# Check if we're in the correct directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Cargo.toml not found. Please run from project root."
    exit 1
fi

# Run security audit
echo "Running cargo-audit..."
if command -v cargo-audit &> /dev/null; then
    cargo audit || echo "Audit completed with findings"
else
    echo "cargo-audit not available"
fi

echo ""
echo "Checking for outdated dependencies..."
if ! command -v cargo-outdated &> /dev/null; then
    echo "cargo-outdated not found. Installing..."
    cargo install cargo-outdated || echo "Failed to install cargo-outdated"
fi

if command -v cargo-outdated &> /dev/null; then
    cargo outdated || echo "Outdated check completed"
else
    echo "cargo-outdated not available"
fi

echo ""
echo "Security audit complete!"
echo ""
echo "Recommendations:"
echo "1. Review any vulnerabilities reported by cargo-audit"
echo "2. Consider updating outdated dependencies marked as 'compatible'"
echo "3. Test thoroughly after any security updates"
echo "4. Monitor https://rustsec.org for new advisories"