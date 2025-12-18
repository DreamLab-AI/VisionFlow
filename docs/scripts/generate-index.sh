#!/bin/bash
# Index Generator - Creates INDEX.md with navigation
set -euo pipefail

DOCS_ROOT="${DOCS_ROOT:-$(dirname "$(dirname "$(realpath "$0")")")}"
INDEX_FILE="$DOCS_ROOT/INDEX.md"

cat > "$INDEX_FILE" <<'EOF'
# Documentation Index

**Version:** 2.0.0
**Last Updated:** 2025-12-18

Welcome to the unified documentation corpus. This index provides complete navigation across all documentation categories.

---

## Quick Navigation

- [Architecture](./architecture/README.md) - System design and architectural decisions
- [Development](./development/README.md) - Development guides and workflows
- [Deployment](./deployment/README.md) - Deployment procedures and infrastructure
- [API Reference](./api/README.md) - API documentation and references
- [User Guides](./guides/README.md) - End-user documentation
- [Reference](./reference/README.md) - Technical reference materials

---

## Documentation Structure

```mermaid
graph TD
    A[Documentation Root] --> B[Architecture]
    A --> C[Development]
    A --> D[Deployment]
    A --> E[API Reference]
    A --> F[User Guides]
    A --> G[Reference]

    B --> B1[System Design]
    B --> B2[Component Architecture]
    B --> B3[Integration Patterns]

    C --> C1[Setup Guides]
    C --> C2[Development Workflows]
    C --> C3[Testing Strategies]

    D --> D1[Docker Deployment]
    D --> D2[CI/CD Pipelines]
    D --> D3[Infrastructure]

    E --> E1[REST APIs]
    E --> E2[WebSocket APIs]
    E --> E3[MCP Tools]

    F --> F1[Getting Started]
    F --> F2[How-To Guides]
    F --> F3[Troubleshooting]

    G --> G1[CLI Reference]
    G --> G2[Configuration]
    G --> G3[Glossary]
```

---

## Category Overview

### Architecture Documentation

System design, architectural patterns, and technical decisions.

EOF

# Generate category sections
for category_dir in "$DOCS_ROOT"/*/; do
    if [[ -d "$category_dir" && $(basename "$category_dir") != "scripts" && $(basename "$category_dir") != ".github" ]]; then
        category=$(basename "$category_dir")
        category_title=$(echo "$category" | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) tolower(substr($i,2))}1')

        cat >> "$INDEX_FILE" <<EOF

#### $category_title

EOF

        # List all markdown files in category
        find "$category_dir" -name "*.md" -type f | sort | while read -r file; do
            relative_path="${file#$DOCS_ROOT/}"
            filename=$(basename "$file" .md)

            # Try to extract title from front matter
            title=$(awk '/^---$/,/^---$/{if(/^title:/) {sub(/^title:[[:space:]]*/, ""); print; exit}}' "$file")
            if [[ -z "$title" ]]; then
                title=$(echo "$filename" | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) tolower(substr($i,2))}1')
            fi

            echo "- [$title](./$relative_path)" >> "$INDEX_FILE"
        done
    fi
done

cat >> "$INDEX_FILE" <<'EOF'

---

## Documentation Standards

All documentation in this corpus follows these standards:

- **Front Matter**: Every document has structured metadata
- **Mermaid Diagrams**: Visual diagrams use Mermaid syntax
- **Cross-Linking**: Documents are interconnected via relative links
- **Version Control**: All changes tracked via Git
- **Validation**: Automated CI/CD ensures quality

---

## Contributing

See [CONTRIBUTION.md](./CONTRIBUTION.md) for guidelines on contributing to this documentation.

## Maintenance

See [MAINTENANCE.md](./MAINTENANCE.md) for procedures on maintaining this documentation corpus.

---

*This index is automatically generated. Run `./scripts/generate-index.sh` to update.*
EOF

echo "Index generated: $INDEX_FILE"
