#!/bin/bash
# Automated migration of Number::from_f64().unwrap() to safe_json_number()

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Migrating JSON Number unwraps to safe_json_number ==="
echo "Project root: $PROJECT_ROOT"

# List of files to migrate (excluding result_helpers.rs which already uses it internally)
FILES=(
    "src/actors/optimized_settings_actor.rs"
    "src/utils/unified_gpu_compute.rs"
    "src/handlers/api_handler/analytics/community.rs"
    "src/config/path_access.rs"
    "src/performance/settings_benchmark.rs"
)

for file in "${FILES[@]}"; do
    filepath="$PROJECT_ROOT/$file"

    if [ ! -f "$filepath" ]; then
        echo "⚠️  File not found: $filepath"
        continue
    fi

    echo "📝 Processing: $file"

    # Check if file already has the import
    if ! grep -q "use crate::utils::result_helpers::safe_json_number" "$filepath"; then
        echo "  → Adding import..."
        # Find the last 'use' statement and add after it
        sed -i '/^use /!b;:a;n;/^use /ba;i\use crate::utils::result_helpers::safe_json_number;' "$filepath" 2>/dev/null || \
        # If that fails, try adding after first use block
        sed -i '1,/^use /s/^\(use .*\)$/\1\nuse crate::utils::result_helpers::safe_json_number;/' "$filepath"
    else
        echo "  ✓ Import already exists"
    fi

    # Replace all Number::from_f64(...).unwrap() with safe_json_number(...)
    # This handles various spacing patterns
    perl -i -pe 's/serde_json::Number::from_f64\((.*?)\)\.unwrap\(\)/safe_json_number($1)/g' "$filepath"
    perl -i -pe 's/Number::from_f64\((.*?)\)\.unwrap\(\)/safe_json_number($1)/g' "$filepath"

    echo "  ✅ Migrated"
done

echo ""
echo "=== Migration Summary ==="
grep -r "Number::from_f64.*\.unwrap()" src/ --include="*.rs" | grep -v test | grep -v result_helpers.rs | wc -l | xargs echo "Remaining Number unwraps:"
echo ""
echo "✅ Migration complete!"
