#!/bin/bash

# Fix truncated difficulty levels
find . -name "*.md" -type f -exec sed -i 's/^difficulty-level: advance$/difficulty-level: advanced/g' {} \;
find . -name "*.md" -type f -exec sed -i 's/^difficulty-level: interm$/difficulty-level: intermediate/g' {} \;

echo "Fixed truncated difficulty levels"
