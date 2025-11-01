#!/bin/bash

# Remove comments from TypeScript files in /client/src

TARGET_DIR="/mnt/mldata/githubs/AR-AI-Knowledge-Graph/client/src"

find "$TARGET_DIR" -type f \( -name "*.ts" -o -name "*.tsx" \) | while read -r file; do
    echo "Processing: $file"

    # Create temporary file
    temp_file=$(mktemp)

    # Remove comments using sed:
    # 1. Remove single-line comments (// ...)
    # 2. Remove multi-line comments (/* ... */)
    # 3. Preserve URLs and regex patterns
    sed -E '
        # Remove single-line comments (but not URLs)
        s|([^:])//.*|\1|g
        # Remove multi-line comments (simplified - works for single-line /* */ comments)
        s|/\*.*\*/||g
    ' "$file" > "$temp_file"

    # Use perl for more sophisticated multi-line comment removal
    perl -i -p0e 's|/\*.*?\*/||gs' "$temp_file"

    # Move temp file back to original
    mv "$temp_file" "$file"
done

echo "Comment removal complete!"
