#!/bin/bash
# Bulk migration of common unwrap patterns
# This script handles the most common, safe-to-automate patterns

set -e

echo "=== Bulk Unwrap Migration ==="
echo "Working directory: $(pwd)"
echo ""

# Counter for changes
TOTAL_CHANGES=0

# Pattern 1: RwLock read/write with expect
echo "Migrating RwLock .read().unwrap() -> .read().expect()..."
CHANGES=$(find src -name "*.rs" ! -path "*/tests/*" -exec grep -l "\.read()\.unwrap()" {} \; | wc -l)
find src -name "*.rs" ! -path "*/tests/*" -exec sed -i 's/\.read()\.unwrap()/\.read().expect("RwLock poisoned")/g' {} \;
echo "  ✓ Modified $CHANGES files"
TOTAL_CHANGES=$((TOTAL_CHANGES + CHANGES))

echo "Migrating RwLock .write().unwrap() -> .write().expect()..."
CHANGES=$(find src -name "*.rs" ! -path "*/tests/*" -exec grep -l "\.write()\.unwrap()" {} \; | wc -l)
find src -name "*.rs" ! -path "*/tests/*" -exec sed -i 's/\.write()\.unwrap()/\.write().expect("RwLock poisoned")/g' {} \;
echo "  ✓ Modified $CHANGES files"
TOTAL_CHANGES=$((TOTAL_CHANGES + CHANGES))

# Pattern 2: Mutex lock with expect
echo "Migrating Mutex .lock().unwrap() -> .lock().expect()..."
CHANGES=$(find src -name "*.rs" ! -path "*/tests/*" -exec grep -l "\.lock()\.unwrap()" {} \; | wc -l)
find src -name "*.rs" ! -path "*/tests/*" -exec sed -i 's/\.lock()\.unwrap()/\.lock().expect("Mutex poisoned")/g' {} \;
echo "  ✓ Modified $CHANGES files"
TOTAL_CHANGES=$((TOTAL_CHANGES + CHANGES))

# Pattern 3: SystemTime duration_since
echo "Migrating SystemTime unwraps..."
CHANGES=$(find src -name "*.rs" ! -path "*/tests/*" -exec grep -l "duration_since(UNIX_EPOCH)\.unwrap()" {} \; | wc -l)
find src -name "*.rs" ! -path "*/tests/*" -exec sed -i 's/duration_since(UNIX_EPOCH)\.unwrap()/duration_since(UNIX_EPOCH).unwrap_or(Duration::from_secs(0))/g' {} \;
echo "  ✓ Modified $CHANGES files"
TOTAL_CHANGES=$((TOTAL_CHANGES + CHANGES))

# Add Duration import where needed
echo "Adding Duration imports where needed..."
find src -name "*.rs" ! -path "*/tests/*" -exec grep -l "Duration::from_secs" {} \; | while read file; do
    if ! grep -q "use std::time::Duration" "$file"; then
        # Add import after other std::time imports or at the start of use statements
        if grep -q "use std::time::" "$file"; then
            sed -i '/use std::time::/a use std::time::Duration;' "$file" 2>/dev/null || true
        fi
    fi
done

echo ""
echo "=== Migration Summary ==="
echo "Total file modifications: $TOTAL_CHANGES"
echo ""

# Count remaining unwraps (excluding tests)
REMAINING=$(grep -r "\.unwrap()" src --include="*.rs" | grep -v test | grep -v "// SAFETY" | wc -l)
echo "Remaining unwraps (excluding tests): $REMAINING"
echo ""
echo "✅ Bulk migration complete!"
