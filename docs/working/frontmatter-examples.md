---
title: Front Matter Examples
description: Real examples of front matter implementation across different documentation categories
category: reference
tags:
  - documentation
  - metadata
  - examples
  - reference
  - best-practices
updated-date: 2025-12-18
difficulty-level: beginner
related-docs:
  - working/frontmatter-implementation-summary.md
  - working/frontmatter-validation.md
---

# Front Matter Examples

This document showcases real examples of front matter implementation across different documentation categories.

## Tutorial Example

**File**: `tutorials/01-installation.md`

```yaml
---
title: Installation Guide
description: Step-by-step guide to installing VisionFlow on Linux, macOS, and Windows with Docker and manual setup options.
category: tutorial
tags:
  - installation
  - getting-started
  - setup
  - docker
  - tutorial
related-docs:
  - tutorials/02-first-graph.md
  - guides/docker-environment-setup.md
  - reference/CONFIGURATION_REFERENCE.md
updated-date: 2025-12-18
difficulty-level: beginner
dependencies:
  - Docker installation
  - Git
---
```

**Category**: `tutorial` - Learning-oriented, teaches through doing
**Difficulty**: `beginner` - Entry point for new users
**Tags**: Mix of installation, setup, and tutorial tags
**Related**: Links to next tutorial and supporting guides

---

## How-To Example

**File**: `guides/deployment.md`

```yaml
---
title: Deployment Guide
description: Production deployment strategies for VisionFlow including Docker Compose, Kubernetes, and cloud platforms.
category: howto
tags:
  - deployment
  - docker
  - kubernetes
  - production
  - guide
related-docs:
  - guides/docker-compose-guide.md
  - guides/infrastructure/architecture.md
  - reference/CONFIGURATION_REFERENCE.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Docker installation
  - Basic Kubernetes knowledge
---
```

**Category**: `howto` - Problem-oriented, solving deployment tasks
**Difficulty**: `intermediate` - Requires some experience
**Tags**: Deployment-focused with infrastructure tags
**Dependencies**: Lists prerequisites explicitly

---

## Reference Example

**File**: `reference/api/rest-api-reference.md`

```yaml
---
title: REST API Complete Reference
description: Complete REST API documentation including endpoints, request/response formats, authentication, and error handling.
category: reference
tags:
  - api
  - rest
  - reference
  - endpoints
  - http
related-docs:
  - reference/api/01-authentication.md
  - reference/api/03-websocket.md
  - reference/error-codes.md
  - guides/developer/websocket-best-practices.md
updated-date: 2025-12-18
difficulty-level: intermediate
---
```

**Category**: `reference` - Information-oriented, technical description
**Difficulty**: `intermediate` - Requires API knowledge
**Tags**: API-focused reference tags
**Related**: Links to related API documentation

---

## Explanation Example

**File**: `explanations/architecture/database-architecture.md`

```yaml
---
title: Database Architecture
description: Comprehensive overview of VisionFlow's hybrid database architecture combining Neo4j graph database with SQLite for optimal performance.
category: explanation
tags:
  - architecture
  - database
  - neo4j
  - design
  - patterns
related-docs:
  - reference/database/schemas.md
  - reference/database/ontology-schema-v2.md
  - guides/neo4j-integration.md
  - explanations/system-overview.md
updated-date: 2025-12-18
difficulty-level: advanced
---
```

**Category**: `explanation` - Understanding-oriented, clarifying concepts
**Difficulty**: `advanced` - Deep architectural knowledge
**Tags**: Architecture and database design tags
**Related**: Links to schemas and integration guides

---

## Complex Example with Dependencies

**File**: `guides/developer/01-development-setup.md`

```yaml
---
title: Development Environment Setup
description: Complete guide to setting up a local VisionFlow development environment with all required tools, dependencies, and configurations.
category: howto
tags:
  - development
  - setup
  - environment
  - docker
  - rust
related-docs:
  - guides/developer/02-project-structure.md
  - guides/developer/04-adding-features.md
  - guides/contributing.md
  - reference/CONFIGURATION_REFERENCE.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Docker Desktop 20.10+
  - Rust toolchain (latest stable)
  - Node.js 18+
  - Git
  - Visual Studio Code (recommended)
---
```

**Category**: `howto` - Step-by-step setup guide
**Difficulty**: `intermediate` - Requires technical knowledge
**Tags**: Development environment tags
**Dependencies**: Detailed prerequisite list with versions
**Related**: Links to next steps in developer journey

---

## Minimal Example (Working Document)

**File**: `working/DOCS_ROOT_CLEANUP.md`

```yaml
---
title: Documentation Root Cleanup
description: Cleanup plan for removing deprecated files and organizing documentation structure.
category: reference
tags:
  - documentation
  - cleanup
  - organization
updated-date: 2025-12-18
difficulty-level: intermediate
---
```

**Category**: `reference` - Internal planning document
**Difficulty**: `intermediate` - Maintenance task
**Tags**: Minimal but sufficient (3 tags)
**Dependencies**: None (optional field omitted)
**Related**: None listed (working document)

---

## Migration/Archive Example

**File**: `archive/INDEX-QUICK-START-old.md`

```yaml
---
title: Quick Start Index (Archived)
description: Original quick start navigation index, archived after documentation reorganization in December 2025.
category: reference
tags:
  - archive
  - navigation
  - deprecated
  - documentation
  - reference
related-docs:
  - README.md
  - QUICK_NAVIGATION.md
  - archive/README.md
updated-date: 2025-12-18
difficulty-level: beginner
---
```

**Category**: `reference` - Archived navigation
**Difficulty**: `beginner` - Historical reference
**Tags**: Clearly marked as archived
**Related**: Links to current navigation and archive index

---

## Multi-Domain Example

**File**: `guides/vircadia-xr-complete-guide.md`

```yaml
---
title: Vircadia XR Complete Guide
description: Comprehensive guide to integrating VisionFlow with Vircadia for immersive XR experiences including setup, networking, and multi-user coordination.
category: howto
tags:
  - xr
  - vircadia
  - immersive
  - networking
  - guide
related-docs:
  - guides/client/xr-integration.md
  - concepts/architecture/xr-immersive-system.md
  - guides/vircadia-multi-user-guide.md
  - reference/protocols/binary-websocket.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Vircadia client installed
  - VisionFlow server running
  - WebSocket understanding
  - XR device (optional)
---
```

**Category**: `howto` - Integration guide
**Difficulty**: `advanced` - Complex XR integration
**Tags**: XR and networking focused
**Dependencies**: Specific software and knowledge requirements
**Related**: Comprehensive cross-references

---

## Field-by-Field Explanation

### `title`
- **Source**: Extracted from first H1 heading or filename
- **Format**: Title case, descriptive
- **Required**: Yes
- **Example**: `"Installation Guide"`

### `description`
- **Source**: First paragraph or summary
- **Format**: 1-2 sentences, 200 characters max
- **Required**: Yes
- **Example**: `"Step-by-step guide to installing VisionFlow..."`

### `category`
- **Source**: Inferred from path and content
- **Format**: One of: `tutorial`, `howto`, `reference`, `explanation`
- **Required**: Yes
- **Framework**: Diátaxis
- **Example**: `tutorial`

### `tags`
- **Source**: Generated from standardized vocabulary
- **Format**: Array of 3-5 lowercase, hyphenated strings
- **Required**: Yes
- **Vocabulary**: 45 standardized tags
- **Example**: `["installation", "getting-started", "docker"]`

### `related-docs`
- **Source**: Link graph analysis + sibling files
- **Format**: Array of relative file paths
- **Required**: No (but highly recommended)
- **Max**: 5 related documents
- **Example**: `["tutorials/02-first-graph.md"]`

### `updated-date`
- **Source**: Auto-generated on creation/update
- **Format**: `YYYY-MM-DD` (ISO 8601)
- **Required**: Yes
- **Example**: `2025-12-18`

### `difficulty-level`
- **Source**: Inferred from content and path
- **Format**: One of: `beginner`, `intermediate`, `advanced`
- **Required**: Yes
- **Example**: `intermediate`

### `dependencies`
- **Source**: Extracted from prerequisites section
- **Format**: Array of strings
- **Required**: No (optional field)
- **Example**: `["Docker installation", "Node.js 18+"]`

---

## Best Practices

### ✅ DO

- Keep descriptions concise (1-2 sentences)
- Use standardized tags from vocabulary
- Link to related documentation
- Update `updated-date` when content changes
- List specific version requirements in dependencies
- Choose appropriate difficulty level
- Use proper Diátaxis category

### ❌ DON'T

- Don't use custom/unstandardized tags
- Don't exceed 5 tags
- Don't link to non-existent files in related-docs
- Don't forget to update the date
- Don't misclassify categories
- Don't use vague dependencies like "basic knowledge"
- Don't omit required fields

---

## Validation Checklist

Before committing documentation:

- [ ] All required fields present
- [ ] Category is valid Diátaxis type
- [ ] 3-5 tags from standardized vocabulary
- [ ] Related docs all exist
- [ ] Date format is YYYY-MM-DD
- [ ] Difficulty level appropriate
- [ ] Dependencies specific and actionable
- [ ] Description is clear and concise

---

## Automated Validation

Run validation script before committing:

```bash
node scripts/validate-frontmatter.js
```

**Expected Output**:
```
============================================================
Front Matter Validation Summary
============================================================
Total Files:         298
With Front Matter:   297
Valid:               296
Errors:              0
Warnings:            0
============================================================

Coverage: 99.7%
```

---

## See Also

- [Front Matter Implementation Summary](./frontmatter-implementation-summary.md) - Complete implementation report
- [Front Matter Validation Report](./frontmatter-validation.md) - Current validation status
- [README](../README.md) - Documentation home

---

**Generated**: 2025-12-18
**Coverage**: 99.7% (297/298 files)
**Status**: Production Ready ✅
