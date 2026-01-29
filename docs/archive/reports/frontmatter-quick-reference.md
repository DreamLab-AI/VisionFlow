---
title: Front Matter Quick Reference
description: Cheat sheet for proper front matter formatting in VisionFlow documentation
category: reference
tags:
  - documentation
  - reference
  - tutorial
updated-date: 2025-12-30
---

# Front Matter Quick Reference

**One-page guide** for correct YAML front matter formatting.

---

## Basic Template

```yaml
---
title: Document Title
description: One-line description of the document
category: [tutorial|howto|reference|explanation]
tags:
  - primary-tag
  - secondary-tag
updated-date: 2025-12-30
---
```

---

## Required Fields

All 4 fields are **required**:

| Field | Type | Example |
|-------|------|---------|
| `title` | String | "Getting Started with Neo4j" |
| `description` | String | "Learn how to set up Neo4j for VisionFlow" |
| `category` | String | "tutorial", "howto", "reference", or "explanation" |
| `tags` | Array | ["neo4j", "setup", "database"] |

---

## Category Reference (Diataxis Framework)

Choose ONE category based on document purpose:

### Tutorial
**Purpose**: Help users learn fundamental concepts
- Introduction and orientation
- Learning paths
- Hands-on walkthroughs

**Example**:
```yaml
category: tutorial
```

**Files**: `getting-started/installation.md`, `01-GETTING_STARTED.md`

---

### How-to
**Purpose**: Provide step-by-step instructions for specific tasks
- Task-oriented guides
- Setup instructions
- Configuration steps

**Example**:
```yaml
category: howto
```

**Files**: `guides/features/auth-user-settings.md`, `guides/docker-compose-guide.md`

---

### Reference
**Purpose**: Provide technical specifications and complete information
- API documentation
- Database schemas
- Configuration options
- Quick lookups

**Example**:
```yaml
category: reference
```

**Files**: `reference/api/README.md`, `reference/protocols/README.md`

---

### Explanation
**Purpose**: Explain design decisions and architectural concepts
- Why decisions were made
- Design patterns
- Conceptual overviews
- Trade-off analysis

**Example**:
```yaml
category: explanation
```

**Files**: `concepts/hexagonal-architecture.md`, `architecture/overview.md`

---

## Standard Tag Vocabulary (45 tags)

Use ONLY these tags (case-sensitive):

**Core/Domain**:
- `api` - REST/GraphQL APIs
- `architecture` - System design
- `authentication` - Auth mechanisms
- `configuration` - Config files/setup
- `database` - Database systems
- `deployment` - Deployment processes
- `development` - Development practices
- `documentation` - Doc-related
- `features` - Feature documentation
- `graph` - Graph structures
- `integration` - Integration guides
- `ontology` - Ontology systems
- `operations` - Operational runbooks
- `security` - Security practices

**Technology Stack**:
- `docker` - Docker containers
- `gpu` - GPU/CUDA computing
- `kubernetes` - K8s orchestration
- `neo4j` - Neo4j database
- `nostr` - Nostr protocol
- `rust` - Rust programming
- `solid` - SOLID pods
- `three.js` - Three.js library
- `typescript` - TypeScript code
- `webgl` - WebGL rendering

**Infrastructure**:
- `monitoring` - Monitoring/observability
- `protocol` - Protocol specifications
- `websocket` - WebSocket protocol

**Patterns & Practices**:
- `getting-started` - Getting started guides
- `howto` - How-to guides
- `installation` - Installation steps
- `performance` - Performance topics
- `reference` - Reference docs
- `setup` - Setup/configuration
- `testing` - Testing practices
- `troubleshooting` - Troubleshooting guides
- `tutorial` - Tutorials
- `visualization` - Data visualization
- `xr` - Extended reality

---

## Tag Selection Guide

**Choose tags that best describe the document**:

1. **Primary domain tag** (required):
   - Choose 1-2 most relevant tags from Core/Domain section

2. **Technology tags** (optional):
   - Add if document discusses specific technologies

3. **Pattern tags** (optional):
   - Add if document provides guidance (howto, tutorial, etc.)

---

## Examples by Document Type

### Installation Guide
```yaml
---
title: Installing VisionFlow
description: Step-by-step installation instructions for VisionFlow
category: tutorial
tags:
  - installation
  - setup
  - docker
updated-date: 2025-12-30
---
```

### API Reference
```yaml
---
title: REST API Reference
description: Complete REST API endpoint documentation
category: reference
tags:
  - api
  - reference
  - authentication
updated-date: 2025-12-30
---
```

### Feature Guide
```yaml
---
title: Using Semantic Forces
description: How to leverage semantic forces in your graphs
category: howto
tags:
  - features
  - ontology
  - graph
updated-date: 2025-12-30
---
```

### Architecture Document
```yaml
---
title: Hexagonal Architecture
description: Explanation of VisionFlow's hexagonal architecture design
category: explanation
tags:
  - architecture
  - design
updated-date: 2025-12-30
---
```

### Multi-Topic Document
```yaml
---
title: Neo4j Integration Guide
description: Complete guide to integrating Neo4j with VisionFlow
category: howto
tags:
  - neo4j
  - database
  - integration
  - setup
updated-date: 2025-12-30
---
```

---

## Optional Fields

These fields are recommended but optional:

```yaml
---
title: Document Title
description: Description
category: reference
tags:
  - tag1
  - tag2
related-docs:              # Optional: Links to related docs
  - other-doc.md
  - guides/feature.md
difficulty-level: beginner # Optional: beginner|intermediate|advanced|expert
updated-date: 2025-12-30   # Optional: ISO date format YYYY-MM-DD
dependencies:              # Optional: Prerequisites
  - Docker installation
  - Node.js v18+
---
```

### Optional Field Details

**related-docs**: Links to related documentation files
```yaml
related-docs:
  - getting-started/installation.md
  - guides/configuration.md
  - architecture/overview.md
```

**difficulty-level**: Target audience skill level
```yaml
difficulty-level: intermediate  # Choose: beginner|intermediate|advanced|expert
```

**updated-date**: Last update date in ISO format
```yaml
updated-date: 2025-12-30  # Format: YYYY-MM-DD
```

**dependencies**: Prerequisite knowledge or software
```yaml
dependencies:
  - Docker installation
  - Node.js v18+
  - Basic Neo4j knowledge
```

---

## Common Mistakes & Fixes

### Mistake 1: Invalid Category
```yaml
# WRONG
category: guide

# CORRECT - Choose from: tutorial, howto, reference, explanation
category: howto
```

### Mistake 2: Non-Standard Tags
```yaml
# WRONG
tags:
  - design
  - patterns
  - backend

# CORRECT
tags:
  - architecture
  - development
```

### Mistake 3: Missing Required Fields
```yaml
# WRONG
---
title: My Document
category: reference
---

# CORRECT - Must include all 4 required fields
---
title: My Document
description: What this document covers
category: reference
tags:
  - relevant-tag
---
```

### Mistake 4: Duplicate Tags
```yaml
# WRONG - Don't repeat tags
tags:
  - api
  - api
  - testing

# CORRECT - Each tag once
tags:
  - api
  - testing
```

### Mistake 5: Wrong Formatting
```yaml
# WRONG - Not proper YAML
---
title=My Document
description=About this document
category=reference
tags: api, testing
---

# CORRECT - Proper YAML formatting
---
title: My Document
description: About this document
category: reference
tags:
  - api
  - testing
---
```

---

## Validation Checklist

Before committing, verify:

- [ ] Front matter starts with `---` on line 1
- [ ] All 4 required fields present (title, description, category, tags)
- [ ] Category is one of: tutorial, howto, reference, explanation
- [ ] All tags are from standard vocabulary (45 tags)
- [ ] At least 1 tag selected
- [ ] No duplicate tags
- [ ] Proper YAML formatting (colons after field names, dash-space for lists)
- [ ] Front matter ends with `---`

---

## Quick Lookup: Tags by Topic

### By Technology
**Neo4j**: neo4j, database, graph
**GPU**: gpu, rust, performance
**WebSocket**: websocket, protocol, integration
**Docker**: docker, deployment, setup
**Authentication**: authentication, security, api
**Three.js**: three.js, visualization, webgl

### By Document Type
**Getting Started**: tutorial, getting-started, setup
**How-to**: howto, setup, installation
**Reference**: reference, api, database, protocol
**Architecture**: explanation, architecture, design

### By Audience
**Users**: tutorial, getting-started, features
**Developers**: development, api, architecture, testing
**Operators**: operations, deployment, monitoring, troubleshooting
**Architects**: explanation, architecture, design

---

## When to Update Front Matter

Update the `updated-date` field when:
- Making content changes
- Fixing errors or clarifications
- Updating instructions with new information
- Major revisions

**Do NOT update** for:
- Minor formatting changes
- Typo fixes (unless significant)
- Whitespace changes

---

## Need Help?

- **Diataxis Framework**: https://diataxis.fr/
- **YAML Syntax**: https://en.wikipedia.org/wiki/YAML
- **VisionFlow Docs**: /docs/README.md
- **Report Issues**: See docs/reports/frontmatter-validation.md

---

## Examples from VisionFlow

### Good Example 1
```yaml
---
title: Getting Started with VisionFlow
description: Entry points for new users, developers, architects, and operators
category: tutorial
tags:
  - getting-started
updated-date: 2025-12-19
difficulty-level: intermediate
---
```

### Good Example 2
```yaml
---
title: VisionFlow Architecture Overview
description: VisionFlow is built on three core architectural principles
category: reference
tags:
  - architecture
  - design
  - patterns
  - structure
  - api
related-docs:
  - architecture/developer-journey.md
  - TECHNOLOGY_CHOICES.md
  - guides/developer/01-development-setup.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Docker installation
  - Node.js runtime
  - Neo4j database
---
```

---

**Last Updated**: December 30, 2025
**Validation Status**: 89.9% coverage (338/376 files)

