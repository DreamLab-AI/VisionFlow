#!/bin/bash
# Automated replacement of Utc::now() with centralized time utilities

set -e

echo "=== Time Utilities Centralization Script ==="
echo "Target: Replace 305+ Utc::now() calls with utils::time functions"
echo ""

# Find all Rust files except time.rs itself
FILES=$(find src -name "*.rs" -type f ! -path "*/utils/time.rs")

TOTAL_REPLACED=0
FILES_MODIFIED=0

for file in $FILES; do
    # Skip if file doesn't contain Utc::now()
    if ! grep -q "Utc::now()" "$file"; then
        continue
    fi

    echo "Processing: $file"

    # Create backup
    cp "$file" "${file}.bak"

    # Track if this file was modified
    MODIFIED=0

    # Add time import if chrono is used but time import is missing
    if grep -q "use chrono" "$file" && ! grep -q "use crate::utils::time" "$file"; then
        # Find the last use statement and add after it
        sed -i '/^use /a use crate::utils::time;' "$file"
        MODIFIED=1
    fi

    # Replace various patterns
    # Pattern 1: Utc::now()
    if sed -i 's/Utc::now()/time::now()/g' "$file"; then
        COUNT=$(diff "${file}.bak" "$file" | grep -c "^>" || true)
        if [ "$COUNT" -gt 0 ]; then
            echo "  - Replaced $COUNT occurrences of Utc::now() with time::now()"
            TOTAL_REPLACED=$((TOTAL_REPLACED + COUNT))
            MODIFIED=1
        fi
    fi

    # Pattern 2: chrono::Utc::now()
    if sed -i 's/chrono::Utc::now()/time::now()/g' "$file"; then
        COUNT=$(diff "${file}.bak" "$file" | grep -c "^>" || true)
        if [ "$COUNT" -gt 0 ]; then
            echo "  - Replaced $COUNT occurrences of chrono::Utc::now() with time::now()"
            TOTAL_REPLACED=$((TOTAL_REPLACED + COUNT))
            MODIFIED=1
        fi
    fi

    # Pattern 3: .to_rfc3339() -> time::format_iso8601()
    # This is more complex, handle case by case

    # Pattern 4: .timestamp_millis() on Utc::now()
    if sed -i 's/time::now()\.timestamp_millis()/time::timestamp_millis()/g' "$file"; then
        MODIFIED=1
    fi

    # Pattern 5: .timestamp() on Utc::now()
    if sed -i 's/time::now()\.timestamp()/time::timestamp_seconds()/g' "$file"; then
        MODIFIED=1
    fi

    if [ "$MODIFIED" -eq 1 ]; then
        FILES_MODIFIED=$((FILES_MODIFIED + 1))
    else
        # Restore backup if no changes
        mv "${file}.bak" "$file"
    fi
done

echo ""
echo "=== Summary ==="
echo "Files modified: $FILES_MODIFIED"
echo "Total replacements: $TOTAL_REPLACED"
echo ""
echo "Verifying remaining Utc::now() calls (excluding time.rs)..."
REMAINING=$(grep -r "Utc::now()" src --include="*.rs" | grep -v "src/utils/time.rs" | grep -v "use chrono" | wc -l)
echo "Remaining Utc::now() calls: $REMAINING"

echo ""
echo "Done! Run 'cargo check' to verify compilation."
