#!/bin/bash

FILES=(
"working/mermaid-remediation-report.md"
"archive/sprint-logs/p1-1-summary.md"
"archive/sprint-logs/p1-2-pagerank.md"
"archive/docs/guides/xr-setup.md"
"archive/reports/2025-12-02-restructuring-complete.md"
"archive/reports/2025-12-02-user-settings-summary.md"
"archive/specialized/client-components-reference.md"
"explanations/system-overview.md"
"explanations/ontology/intelligent-pathfinding-system.md"
"explanations/ontology/ontology-typed-system.md"
"scripts/README.md"
"guides/ontology-semantic-forces.md"
"guides/graphserviceactor-migration.md"
"guides/semantic-features-implementation.md"
"guides/agent-orchestration.md"
"guides/ontology-storage-guide.md"
)

for FILE in "${FILES[@]}"; do
  if [ ! -f "$FILE" ]; then
    echo "Creating frontmatter for: $FILE"
    # Add minimal frontmatter to files without it
    TITLE=$(basename "$FILE" .md | tr '-' ' ' | tr '_' ' ')
    cat > "$FILE.tmp" <<EOF
---
title: "$TITLE"
description: "Documentation file"
category: explanation
tags:
  - documentation
updated-date: 2025-12-19
difficulty-level: intermediate
---

EOF
    cat "$FILE" >> "$FILE.tmp" 2>/dev/null || true
    mv "$FILE.tmp" "$FILE"
  fi
done

# Replace invalid tags in all files
find . -name "*.md" -type f -exec sed -i 's/^\s\+-\s\+websocket$/  - api/g' {} \;
find . -name "*.md" -type f -exec sed -i 's/^\s\+-\s\+neo4j$/  - database/g' {} \;
find . -name "*.md" -type f -exec sed -i 's/^\s\+-\s\+rust$/  - backend/g' {} \;
find . -name "*.md" -type f -exec sed -i 's/^\s\+-\s\+rest$/  - api/g' {} \;
find . -name "*.md" -type f -exec sed -i 's/^\s\+-\s\+react$/  - frontend/g' {} \;
find . -name "*.md" -type f -exec sed -i 's/^\s\+-\s\+client$/  - frontend/g' {} \;
find . -name "*.md" -type f -exec sed -i 's/^\s\+-\s\+guide$/  - tutorial/g' {} \;

echo "Cleanup complete"
