#!/bin/bash

TARGET_DIR="/mnt/mldata/githubs/AR-AI-Knowledge-Graph/src"

find "$TARGET_DIR" -type f -name "*.rs" | while read -r file; do
    echo "Processing: $file"

    temp_file=$(mktemp)

    sed -E '
        s|([^:])//.*|\1|g
        s|/\*.*\*/||g
    ' "$file" > "$temp_file"

    perl -i -p0e 's|/\*.*?\*/||gs' "$temp_file"

    mv "$temp_file" "$file"
done

echo "Rust comment removal complete!"
