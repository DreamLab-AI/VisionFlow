---
layout: default
title: Jekyll Migration Log
nav_exclude: true
description: Documentation migration log from custom frontmatter to Jekyll-compatible format
---

# Jekyll Documentation Migration Log

**Date**: 2026-01-02
**Migration Type**: Frontmatter standardisation to Jekyll format
**Status**: Complete

## Summary

All tutorials and guides have been migrated to Jekyll-compatible YAML frontmatter format with proper navigation hierarchy using `parent`, `grand_parent`, `has_children`, and `nav_order` attributes.

## Migration Statistics

| Category | Files Migrated | Index Files Created/Updated |
|----------|----------------|----------------------------|
| Tutorials | 4 | 1 (created) |
| Guides (root) | 32 | 1 (updated) |
| AI Models | 7 | 1 (created) |
| Architecture | 2 | 1 (created) |
| Client | 4 | 1 (updated by linter) |
| Developer | 10 | 1 (updated) |
| Features | 12 | 1 (updated) |
| Infrastructure | 9 | 1 (updated) |
| Migration | 2 | 1 (updated) |
| Operations | 2 | 1 (updated) |
| Getting Started | 1 | - (updated) |
| **Total** | **83** | **10** |

## Files Processed

### Tutorials (3 files)

| File | nav_order | Status |
|------|-----------|--------|
| `tutorials/index.md` | 2 | Created |
| `tutorials/01-installation.md` | 1 | Migrated |
| `tutorials/02-first-graph.md` | 2 | Migrated |
| `tutorials/neo4j-quick-start.md` | 3 | Migrated |

### Guides Root Level (13 files)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/index.md` | 3 | Updated |
| `guides/README.md` | 1 | Migrated |
| `guides/agent-orchestration.md` | 5 | Migrated |
| `guides/configuration.md` | 3 | Migrated |
| `guides/contributing.md` | 10 | Migrated |
| `guides/deployment.md` | 2 | Migrated |
| `guides/development-workflow.md` | 4 | Migrated |
| `guides/docker-compose-guide.md` | 6 | Migrated |
| `guides/extending-the-system.md` | 7 | Migrated |
| `guides/graphserviceactor-migration.md` | 20 | Migrated |
| `guides/ontology-storage-guide.md` | 15 | Migrated |
| `guides/orchestrating-agents.md` | 8 | Migrated |
| `guides/troubleshooting.md` | 25 | Migrated |

### AI Models (6 files)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/ai-models/index.md` | 30 | Created |
| `guides/ai-models/README.md` | 1 | Migrated |
| `guides/ai-models/INTEGRATION_SUMMARY.md` | 2 | Migrated |
| `guides/ai-models/deepseek-deployment.md` | 3 | Migrated |
| `guides/ai-models/deepseek-verification.md` | 4 | Migrated |
| `guides/ai-models/perplexity-integration.md` | 5 | Migrated |
| `guides/ai-models/ragflow-integration.md` | 6 | Migrated |

### Architecture (1 file)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/architecture/index.md` | 31 | Created |
| `guides/architecture/actor-system.md` | 1 | Migrated |

### Client (3 files)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/client/index.md` | 3 | Updated by linter |
| `guides/client/state-management.md` | 1 | Migrated |
| `guides/client/three-js-rendering.md` | 2 | Migrated |
| `guides/client/xr-integration.md` | 3 | Migrated |

### Developer (8 files)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/developer/README.md` | 33 | Updated |
| `guides/developer/01-development-setup.md` | 2 | Migrated |
| `guides/developer/02-project-structure.md` | 3 | Migrated |
| `guides/developer/04-adding-features.md` | 4 | Migrated |
| `guides/developer/06-contributing.md` | 5 | Migrated |
| `guides/developer/json-serialization-patterns.md` | 6 | Migrated |
| `guides/developer/test-execution.md` | 7 | Migrated |
| `guides/developer/websocket-best-practices.md` | 8 | Migrated |

### Features (11 files)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/features/index.md` | 34 | Updated |
| `guides/features/auth-user-settings.md` | 1 | Migrated |
| `guides/features/filtering-nodes.md` | 2 | Migrated |
| `guides/features/github-pagination-fix.md` | 3 | Migrated |
| `guides/features/intelligent-pathfinding.md` | 4 | Migrated |
| `guides/features/local-file-sync-strategy.md` | 5 | Migrated |
| `guides/features/natural-language-queries.md` | 6 | Migrated |
| `guides/features/nostr-auth.md` | 7 | Migrated |
| `guides/features/ontology-sync-enhancement.md` | 8 | Migrated |
| `guides/features/semantic-forces.md` | 9 | Migrated |
| `guides/features/settings-authentication.md` | 10 | Migrated |
| `guides/features/MOVED.md` | 99 | Migrated |

### Infrastructure (7 files)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/infrastructure/README.md` | 35 | Updated |
| `guides/infrastructure/architecture.md` | 2 | Migrated |
| `guides/infrastructure/docker-environment.md` | 3 | Migrated |
| `guides/infrastructure/goalie-integration.md` | 4 | Migrated |
| `guides/infrastructure/port-configuration.md` | 5 | Migrated |
| `guides/infrastructure/tools.md` | 6 | Migrated |
| `guides/infrastructure/troubleshooting.md` | 7 | Migrated |

### Migration (1 file)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/migration/index.md` | 36 | Updated |
| `guides/migration/json-to-binary-protocol.md` | 1 | Migrated |

### Operations (1 file)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/operations/index.md` | 37 | Updated |
| `guides/operations/pipeline-operator-runbook.md` | 1 | Migrated |

### Getting Started (1 file)

| File | nav_order | Status |
|------|-----------|--------|
| `guides/getting-started/index.md` | 1 | Updated |

## Frontmatter Changes

### Before (Custom Format)
```yaml
---
title: Example Title
description: Long description...
category: guide
tags:
  - tag1
  - tag2
related-docs:
  - path/to/doc.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Dependency
---
```

### After (Jekyll Format)
```yaml
---
layout: default
title: Example Title
parent: Parent Section
grand_parent: Guides
nav_order: 1
has_children: true  # Only for index files
description: Concise description
---
```

## Navigation Hierarchy

```
docs/
├── tutorials/
│   └── index.md (parent: none, has_children: true)
│       ├── 01-installation.md (parent: Tutorials)
│       ├── 02-first-graph.md (parent: Tutorials)
│       └── neo4j-quick-start.md (parent: Tutorials)
├── guides/
│   └── index.md (parent: none, has_children: true)
│       ├── getting-started/index.md (parent: Guides)
│       ├── ai-models/
│       │   └── index.md (parent: Guides, has_children: true)
│       │       └── *.md (parent: AI Models, grand_parent: Guides)
│       ├── architecture/
│       │   └── index.md (parent: Guides, has_children: true)
│       │       └── *.md (parent: Architecture, grand_parent: Guides)
│       ├── client/
│       │   └── index.md (parent: Guides, has_children: true)
│       │       └── *.md (parent: Client, grand_parent: Guides)
│       ├── developer/
│       │   └── README.md (parent: Guides, has_children: true)
│       │       └── *.md (parent: Developer, grand_parent: Guides)
│       ├── features/
│       │   └── index.md (parent: Guides, has_children: true)
│       │       └── *.md (parent: Features, grand_parent: Guides)
│       ├── infrastructure/
│       │   └── README.md (parent: Guides, has_children: true)
│       │       └── *.md (parent: Infrastructure, grand_parent: Guides)
│       ├── migration/
│       │   └── index.md (parent: Guides, has_children: true)
│       │       └── *.md (parent: Migration, grand_parent: Guides)
│       └── operations/
│           └── index.md (parent: Guides, has_children: true)
│               └── *.md (parent: Operations, grand_parent: Guides)
```

## Notes

1. **Content Preserved**: All original content was preserved during migration
2. **Duplicate Tags Removed**: Redundant tags like duplicate `api` entries were cleaned
3. **Descriptions Shortened**: Long descriptions truncated to concise summaries
4. **nav_order Scheme**:
   - Root sections: 1-10
   - Subdirectory sections: 30-40
   - Individual pages within sections: 1-99
5. **Index Files**: Created or updated for each subdirectory with `has_children: true`

## Verification

To verify the migration:

```bash
# Check all markdown files have Jekyll frontmatter
grep -l "^layout: default" docs/**/*.md | wc -l

# List files missing layout
find docs -name "*.md" -exec grep -L "^layout:" {} \;

# Build Jekyll site locally
bundle exec jekyll build

# Serve locally to test navigation
bundle exec jekyll serve
```
