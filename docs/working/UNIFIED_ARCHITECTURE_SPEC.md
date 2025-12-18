# Unified Information Architecture Specification

**Version:** 1.0
**Date:** 2025-12-18
**Status:** APPROVED FOR IMPLEMENTATION
**Target Completion:** Phase 3 (2-3 weeks)

---

## Executive Summary

This specification defines the unified information architecture for the VisionFlow documentation corpus, consolidating 298 markdown files into a coherent, discoverable, and maintainable structure based on the Diátaxis framework.

### Current State Analysis

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Total files | 298 | 220-230 | 23% reduction |
| Orphaned files | 124 (41.6%) | 0 (0%) | 100% elimination |
| Frontmatter coverage | 72.5% | 98%+ | 35% increase |
| Duplicate groups | 14 groups | 0 groups | 100% elimination |
| Link validity | ~60% | 98%+ | 63% increase |
| Directory depth | Inconsistent | 3 levels max | Standardized |

### Strategic Approach

**Foundation:** Diátaxis Framework (4 quadrants)
- **Tutorials** (Learning-oriented): Step-by-step lessons
- **Guides** (Task-oriented): Practical how-to documentation
- **Explanations** (Understanding-oriented): Conceptual deep-dives
- **Reference** (Information-oriented): Technical specifications

**Top-Level Organization:** 7 sections (max)
1. Getting Started
2. Guides
3. Explanations
4. Reference
5. Architecture
6. Examples
7. Archive

---

## 1. Unified Directory Structure

### 1.1 Top-Level Architecture

```
docs/
├── README.md                           # Main entry point (730 lines, comprehensive)
├── ARCHITECTURE_OVERVIEW.md            # System architecture hub (1761 lines)
├── DEVELOPER_JOURNEY.md                # Learning path navigator
├── QUICK_NAVIGATION.md                 # Fast-access index
│
├── getting-started/                    # Quadrant 1: TUTORIALS
│   ├── README.md                       # Tutorials index
│   ├── 01-installation.md             # Environment setup
│   ├── 02-first-graph.md              # First knowledge graph
│   ├── 03-neo4j-quickstart.md         # Database quick start
│   ├── 04-agent-basics.md             # Agent system intro
│   └── 05-deployment-basics.md        # Deployment fundamentals
│
├── guides/                             # Quadrant 2: HOW-TO GUIDES
│   ├── README.md                       # Guides index
│   ├── features/                       # Feature guides
│   │   ├── README.md
│   │   ├── semantic-forces.md
│   │   ├── physics-simulation.md
│   │   ├── xr-immersive.md
│   │   ├── voice-interaction.md
│   │   └── multi-workspace.md
│   ├── ontology/                       # Ontology guides
│   │   ├── README.md
│   │   ├── neo4j-setup.md
│   │   ├── semantic-forces-guide.md
│   │   ├── reasoning-integration.md
│   │   └── visualization.md
│   ├── infrastructure/                 # Infrastructure guides
│   │   ├── README.md
│   │   ├── docker-environment.md
│   │   ├── deployment.md
│   │   └── troubleshooting.md
│   ├── developer/                      # Developer guides
│   │   ├── README.md
│   │   ├── websocket-best-practices.md
│   │   ├── testing-guide.md
│   │   ├── telemetry-logging.md
│   │   └── debugging.md
│   ├── ai-models/                      # AI model guides
│   │   ├── README.md
│   │   ├── comfyui-integration.md
│   │   ├── blender-mcp.md
│   │   ├── deepseek-deployment.md
│   │   └── model-deployment.md
│   └── client/                         # Client development
│       ├── README.md
│       ├── state-management.md
│       ├── three-js-rendering.md
│       └── xr-integration.md
│
├── explanations/                       # Quadrant 3: UNDERSTANDING
│   ├── README.md                       # Explanations index
│   ├── architecture/                   # Architecture concepts
│   │   ├── README.md
│   │   ├── hexagonal-cqrs.md
│   │   ├── adapter-patterns.md
│   │   ├── services-architecture.md
│   │   ├── integration-patterns.md
│   │   ├── core/                       # Core architecture
│   │   │   ├── client.md
│   │   │   ├── client-typescript.md
│   │   │   ├── server.md
│   │   │   └── database.md
│   │   ├── components/                 # Component architectures
│   │   │   ├── websocket-protocol.md
│   │   │   ├── actor-system.md
│   │   │   └── gpu-pipeline.md
│   │   └── subsystems/                 # Subsystem architectures
│   │       ├── blender-mcp.md
│   │       ├── ontology-storage.md
│   │       └── analytics.md
│   ├── ontology/                       # Ontology concepts
│   │   ├── README.md
│   │   ├── ontology-overview.md
│   │   ├── neo4j-integration.md
│   │   ├── hierarchical-visualization.md
│   │   ├── semantic-forces.md
│   │   └── graph-algorithms.md
│   ├── physics/                        # Physics simulation
│   │   ├── README.md
│   │   ├── semantic-forces.md
│   │   ├── barnes-hut-algorithm.md
│   │   └── gpu-acceleration.md
│   └── patterns/                       # Design patterns
│       ├── README.md
│       ├── hexagonal-architecture.md
│       ├── cqrs-pattern.md
│       └── actor-model.md
│
├── reference/                          # Quadrant 4: SPECIFICATIONS
│   ├── README.md                       # Reference index
│   ├── api/                            # API specifications
│   │   ├── README.md
│   │   ├── rest-api-complete.md       # PRIMARY: Consolidated REST API
│   │   ├── graphql-api.md
│   │   └── error-codes.md
│   ├── protocols/                      # Protocol specifications
│   │   ├── README.md
│   │   └── binary-websocket.md         # PRIMARY: Consolidated WebSocket
│   ├── database/                       # Database specifications
│   │   ├── README.md
│   │   ├── schemas.md
│   │   └── cypher-queries.md
│   ├── ontology/                       # Ontology specifications
│   │   ├── README.md
│   │   ├── schema.md
│   │   ├── api.md
│   │   └── data-model.md
│   └── client/                         # Client specifications
│       ├── README.md
│       ├── components.md
│       └── state-api.md
│
├── architecture/                       # ADRS & MAJOR DECISIONS
│   ├── README.md                       # Architecture decisions index
│   ├── decisions/                      # Architecture Decision Records
│   │   ├── ADR-001-hexagonal.md
│   │   ├── ADR-002-cqrs.md
│   │   ├── ADR-003-actor-model.md
│   │   └── ADR-004-binary-protocol.md
│   └── phase-reports/                  # Major milestone reports
│       ├── phase1-completion.md
│       └── skills-refactoring-plan.md
│
├── diagrams/                           # VISUAL DOCUMENTATION
│   ├── README.md                       # Diagrams index
│   ├── mermaid-library/                # Mermaid diagram library
│   │   ├── 00-style-guide.md
│   │   ├── 01-system-architecture.md
│   │   ├── 02-data-flow.md
│   │   ├── 03-deployment.md
│   │   └── 04-agent-orchestration.md
│   ├── architecture/                   # Architecture diagrams
│   │   ├── system-overview.md
│   │   └── hexagonal-cqrs.md
│   ├── server/                         # Server diagrams
│   │   ├── actors/
│   │   └── api/
│   ├── client/                         # Client diagrams
│   │   ├── state/
│   │   └── rendering/
│   └── infrastructure/                 # Infrastructure diagrams
│       ├── gpu/
│       ├── websocket/
│       └── testing/
│
├── audits/                             # MIGRATION & AUDIT REPORTS
│   ├── README.md
│   ├── neo4j-settings-migration-audit.md
│   ├── neo4j-migration-action-plan.md
│   └── neo4j-migration-summary.md
│
├── archive/                            # HISTORICAL CONTENT
│   ├── README.md
│   ├── deprecated-patterns/
│   ├── fixes/
│   ├── reports/
│   ├── sprint-logs/
│   ├── analysis/                       # Completed analyses
│   │   └── client-architecture-2025-12.md
│   └── implementation-logs/
│
└── working/                            # TEMPORARY (active development)
    ├── README.md
    └── [temporary analysis files]
```

### 1.2 Directory Ownership & Scope

| Directory | Owner | Scope | Depth | Max Files |
|-----------|-------|-------|-------|-----------|
| `getting-started/` | Tutorials Team | Beginner learning paths | 1 | 10 |
| `guides/` | Documentation Team | Task-oriented guides | 2 | 60 |
| `explanations/` | Architecture Team | Conceptual deep-dives | 3 | 80 |
| `reference/` | API Team | Technical specifications | 2 | 40 |
| `architecture/` | Architecture Team | ADRs & major decisions | 2 | 20 |
| `diagrams/` | All Teams | Visual documentation | 3 | 50 |
| `audits/` | QA Team | Migration & audit reports | 1 | 15 |
| `archive/` | Documentation Team | Historical content | 2 | ∞ |
| `working/` | All Teams | Temporary analyses | 1 | 5 |

---

## 2. Consolidation & Migration Plan

### 2.1 File Consolidation (47 High-Value Opportunities)

#### Phase 1: Quick Wins (Days 1-2)

**DELETE: Exact Duplicates (4 files)**
```bash
rm -rf docs/concepts/                        # Duplicate of explanations/architecture/core/
rm -rf docs/archive/data/pages/             # Duplicate of archive/data/markdown/
```

**STANDARDIZE: README Files (15 → 10)**
```bash
# Standardize case
mv docs/guides/readme.md docs/guides/README.md

# Merge redundant READMEs
# guides/infrastructure/readme.md → merge into guides/README.md
# guides/developer/readme.md → merge into guides/README.md
# explanations/architecture/gpu/readme.md → merge up
# reference/api/readme.md → merge up
```

**ARCHIVE: Completed Working Documents (10-12 files)**
```bash
mv docs/working/CLIENT_ARCHITECTURE_ANALYSIS.md docs/archive/analysis/client-architecture-2025-12.md
# (After incorporating findings into active docs)
```

#### Phase 2: Medium Priority (Days 3-7)

**CONSOLIDATE: API Reference (6 → 1 PRIMARY)**
```bash
# Merge into reference/api/rest-api-complete.md:
# - reference/API_REFERENCE.md (17KB)
# - reference/api-complete-reference.md (25KB)
# - reference/api/rest-api-reference.md (13KB)

# Keep separate (different purposes):
# - explanations/architecture/api-handlers-reference.md (implementation)
# - diagrams/server/api/rest-api-architecture.md (visual)
```

**CONSOLIDATE: WebSocket Protocol (7 → 4)**
```bash
# Merge into reference/protocols/binary-websocket.md:
# - reference/websocket-protocol.md
# - reference/api/03-websocket.md

# Keep separate:
# - explanations/architecture/components/websocket-protocol.md (architecture)
# - diagrams/infrastructure/websocket/binary-protocol-complete.md (diagrams)
# - guides/developer/websocket-best-practices.md (guide)
# - guides/migration/json-to-binary-protocol.md (migration)
```

**ORGANIZE: Guides Directory (31 → 6 subdirectories)**
```bash
mkdir -p docs/guides/{getting-started,features,ontology,developer,infrastructure,ai-models,client}

# Move files to appropriate subdirectories
mv docs/guides/neo4j-integration.md docs/guides/ontology/neo4j-setup.md
mv docs/guides/ontology-*.md docs/guides/ontology/
# ... (see detailed move operations in consolidation-plan.md)
```

**CONSOLIDATE: Testing Documentation (6 → 3)**
```bash
# Merge into guides/developer/testing-guide.md:
# - guides/developer/test-execution.md

# Keep separate:
# - diagrams/infrastructure/testing/test-architecture.md (diagrams)
# - explanations/architecture/reasoning-tests-summary.md (specialized)
```

**CONSOLIDATE: Troubleshooting (2 → 1)**
```bash
# Merge into guides/troubleshooting.md:
# - guides/infrastructure/troubleshooting.md
```

#### Phase 3: Major Reorganization (Days 8-15)

**CONSOLIDATE: Architecture Documentation (12+ → Organized Hierarchy)**
```bash
# Primary entry point
# ARCHITECTURE_OVERVIEW.md (keep at root, merge ARCHITECTURE_COMPLETE.md)

# Organize explanations/architecture/
mkdir -p docs/explanations/architecture/{core,components,subsystems}

# Move subsystem docs from architecture/ to explanations/architecture/subsystems/
mv docs/architecture/blender-mcp-unified-architecture.md \
   docs/explanations/architecture/subsystems/blender-mcp.md

# Incorporate archived specialized content
mv docs/archive/specialized/client-typescript-architecture.md \
   docs/explanations/architecture/core/client-typescript.md

mv docs/archive/specialized/client-components-reference.md \
   docs/reference/client/components.md
```

**ORGANIZE: Ontology Documentation (47 → 3 Sections)**
```bash
# Create structure
mkdir -p docs/guides/ontology
mkdir -p docs/reference/ontology

# Guides (practical)
mv docs/guides/neo4j-integration.md docs/guides/ontology/neo4j-setup.md
# ... (see detailed plan)

# Explanations (conceptual)
# Merge hierarchical visualization docs
# explanations/architecture/hierarchical-visualization.md → explanations/ontology/

# Reference (specifications)
# Create reference/ontology/{schema,api,data-model}.md

# Keep audits separate
# audits/neo4j-*.md (historical records)
```

**ORGANIZE: Client Documentation (8 → 3 Sections)**
```bash
# Core architecture
# explanations/architecture/core/client.md (primary)
# explanations/architecture/core/client-typescript.md (from archive)

# Diagrams
# diagrams/client/{state,rendering,xr}/

# Reference
mkdir -p docs/reference/client
# reference/client/{components,state-api}.md (from archive)

# Guides
mkdir -p docs/guides/client
# guides/client/{state-management,three-js-rendering,xr-integration}.md
```

**CLEAN: Archive Directory**
```bash
# Review categories:
# - archive/specialized/ → Move active content to main docs
# - archive/fixes/ → Keep only historical quick-references
# - archive/docs/guides/ → Delete if content in current guides
# - archive/working/ → Clean out temporary files
# - archive/data/pages/ → DELETE (duplicate)

# Decision criteria:
# Archive if: Superseded, migration complete, temporary analysis finished
# Keep active if: Still current, ongoing reference, only version available
```

### 2.2 File Move Operations (Complete List)

See `directory-restructure-plan.md` for comprehensive move operations.

---

## 3. Navigation Design

### 3.1 Information Flow Paths

#### User Journey #1: New Developer
```
README.md
  → DEVELOPER_JOURNEY.md
    → getting-started/01-installation.md
      → getting-started/02-first-graph.md
        → guides/features/semantic-forces.md
          → explanations/physics/semantic-forces.md
            → reference/api/rest-api-complete.md
```

#### User Journey #2: System Architect
```
README.md
  → ARCHITECTURE_OVERVIEW.md
    → explanations/architecture/hexagonal-cqrs.md
      → diagrams/mermaid-library/01-system-architecture.md
        → reference/api/rest-api-complete.md
          → architecture/decisions/ADR-001-hexagonal.md
```

#### User Journey #3: DevOps Engineer
```
README.md
  → QUICK_NAVIGATION.md
    → guides/infrastructure/deployment.md
      → diagrams/mermaid-library/03-deployment.md
        → guides/infrastructure/troubleshooting.md
          → explanations/architecture/subsystems/
```

### 3.2 Breadcrumb Navigation

Every document includes breadcrumb trail in front matter:

```yaml
---
title: "Neo4j Setup Guide"
breadcrumbs:
  - text: "Home"
    href: "../README.md"
  - text: "Guides"
    href: "../README.md"
  - text: "Ontology"
    href: "README.md"
  - text: "Neo4j Setup"
    href: "neo4j-setup.md"
---
```

### 3.3 Previous/Next Article Chains

Sequential navigation for learning paths:

```yaml
---
title: "02 - First Graph"
previous:
  title: "01 - Installation"
  href: "01-installation.md"
next:
  title: "03 - Neo4j Quick Start"
  href: "03-neo4j-quickstart.md"
---
```

### 3.4 Sidebar Navigation Maps

Category-specific navigation sidebars:

**Tutorials Sidebar** (`getting-started/README.md`)
```markdown
## Tutorials

### Getting Started
1. [Installation](01-installation.md)
2. [First Graph](02-first-graph.md)
3. [Neo4j Quick Start](03-neo4j-quickstart.md)
4. [Agent Basics](04-agent-basics.md)
5. [Deployment Basics](05-deployment-basics.md)

### Next Steps
- [Feature Guides](../guides/features/README.md)
- [Architecture Concepts](../explanations/architecture/README.md)
```

**Guides Sidebar** (`guides/README.md`)
```markdown
## Guides

### By Category
- [Features](features/README.md) - Feature usage guides
- [Ontology](ontology/README.md) - Knowledge graph guides
- [Infrastructure](infrastructure/README.md) - Deployment & DevOps
- [Developer](developer/README.md) - Development workflows
- [AI Models](ai-models/README.md) - AI model integration
- [Client](client/README.md) - Client development

### Popular Guides
1. [Neo4j Setup](ontology/neo4j-setup.md)
2. [Deployment Guide](infrastructure/deployment.md)
3. [WebSocket Best Practices](developer/websocket-best-practices.md)
4. [Testing Guide](developer/testing-guide.md)
```

### 3.5 Related Articles Sections

Every document includes:

```markdown
---

## Related Documentation

### Prerequisites
- [Installation Tutorial](../getting-started/01-installation.md) - Required setup

### See Also
- [Neo4j Integration Guide](../guides/ontology/neo4j-setup.md) - Database setup
- [Database Schemas Reference](../reference/database/schemas.md) - Schema spec

### Deep Dive
- [Hexagonal CQRS Architecture](../explanations/architecture/hexagonal-cqrs.md)
- [Database Architecture](../explanations/architecture/core/database.md)

### Referenced By
- [Developer Journey](../DEVELOPER_JOURNEY.md)
- [Architecture Overview](../ARCHITECTURE_OVERVIEW.md)

---
```

### 3.6 Cross-Cutting Navigation

**Tag-Based Navigation** (in front matter):
```yaml
tags:
  - neo4j
  - ontology
  - database
  - integration
```

**Difficulty-Based Navigation**:
```yaml
difficulty-level: beginner | intermediate | advanced | expert
```

**Priority-Based Ranking**:
```yaml
priority: 1  # 1=highest (landing pages), 5=lowest (deprecated)
```

---

## 4. Implementation Specification

### 4.1 File Move/Rename Operations

**Script:** `docs/scripts/restructure-directories.sh`

```bash
#!/bin/bash
# docs/scripts/restructure-directories.sh

set -e

DOCS_ROOT="/home/devuser/workspace/project/docs"
BACKUP_DIR="/home/devuser/workspace/project/docs-backup-$(date +%Y%m%d-%H%M%S)"

echo "=== VisionFlow Documentation Restructure ==="
echo "Backup: $BACKUP_DIR"

# Backup
echo "Creating backup..."
cp -r "$DOCS_ROOT" "$BACKUP_DIR"

# Phase 1: Delete duplicates
echo "Phase 1: Removing exact duplicates..."
rm -rf "$DOCS_ROOT/concepts/"
rm -rf "$DOCS_ROOT/archive/data/pages/"

# Phase 1: Standardize README case
echo "Standardizing README files..."
mv "$DOCS_ROOT/guides/readme.md" "$DOCS_ROOT/guides/README.md" 2>/dev/null || true

# Phase 2: Create new directories
echo "Phase 2: Creating new directory structure..."
mkdir -p "$DOCS_ROOT/guides/"{getting-started,features,ontology,developer,infrastructure,ai-models,client}
mkdir -p "$DOCS_ROOT/explanations/architecture/"{core,components,subsystems}
mkdir -p "$DOCS_ROOT/explanations/"{ontology,physics,patterns}
mkdir -p "$DOCS_ROOT/reference/"{api,protocols,database,ontology,client}
mkdir -p "$DOCS_ROOT/architecture/"{decisions,phase-reports}
mkdir -p "$DOCS_ROOT/archive/analysis"

# Phase 2: Move guides
echo "Reorganizing guides..."
mv "$DOCS_ROOT/guides/neo4j-integration.md" "$DOCS_ROOT/guides/ontology/neo4j-setup.md" 2>/dev/null || true

# Phase 3: Move architecture docs
echo "Phase 3: Reorganizing architecture..."
mv "$DOCS_ROOT/architecture/blender-mcp-unified-architecture.md" \
   "$DOCS_ROOT/explanations/architecture/subsystems/blender-mcp.md" 2>/dev/null || true

# Phase 3: Incorporate archived specialized content
echo "Restoring active content from archive..."
mv "$DOCS_ROOT/archive/specialized/client-typescript-architecture.md" \
   "$DOCS_ROOT/explanations/architecture/core/client-typescript.md" 2>/dev/null || true

mv "$DOCS_ROOT/archive/specialized/client-components-reference.md" \
   "$DOCS_ROOT/reference/client/components.md" 2>/dev/null || true

# Archive completed working docs
echo "Archiving completed analyses..."
mv "$DOCS_ROOT/working/CLIENT_ARCHITECTURE_ANALYSIS.md" \
   "$DOCS_ROOT/archive/analysis/client-architecture-2025-12.md" 2>/dev/null || true

echo "=== Restructure Complete ==="
echo "Backup location: $BACKUP_DIR"
echo "Next: Run link update script"
```

### 4.2 Link Update Rules

**Script:** `docs/scripts/update-links.py`

```python
#!/usr/bin/env python3
# docs/scripts/update-links.py

import os
import re
from pathlib import Path

DOCS_ROOT = Path("/home/devuser/workspace/project/docs")

# Link mapping: old path → new path
LINK_MAPPING = {
    "concepts/architecture/core/client.md": "explanations/architecture/core/client.md",
    "guides/neo4j-integration.md": "guides/ontology/neo4j-setup.md",
    "architecture/blender-mcp-unified-architecture.md": "explanations/architecture/subsystems/blender-mcp.md",
    "archive/specialized/client-typescript-architecture.md": "explanations/architecture/core/client-typescript.md",
    "archive/specialized/client-components-reference.md": "reference/client/components.md",
    # ... (complete mapping)
}

def update_links_in_file(file_path: Path):
    """Update all markdown links in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    updated_content = content

    # Find all markdown links
    link_pattern = r'\[([^\]]*)\]\(([^)]*\.md)\)'

    def replace_link(match):
        text = match.group(1)
        old_link = match.group(2)

        # Resolve old link relative to current file
        current_dir = file_path.parent
        old_target = (current_dir / old_link).resolve()
        old_relative = old_target.relative_to(DOCS_ROOT)

        # Check if link needs updating
        old_relative_str = str(old_relative)
        if old_relative_str in LINK_MAPPING:
            new_relative_str = LINK_MAPPING[old_relative_str]
            new_target = DOCS_ROOT / new_relative_str

            # Calculate new relative path from current file
            new_link = os.path.relpath(new_target, current_dir)

            return f'[{text}]({new_link})'

        return match.group(0)  # No change

    updated_content = re.sub(link_pattern, replace_link, updated_content)

    if updated_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return True

    return False

def main():
    print("Updating markdown links...")
    updated_count = 0

    for md_file in DOCS_ROOT.rglob("*.md"):
        if any(skip in md_file.parts for skip in ["node_modules", ".venv"]):
            continue

        if update_links_in_file(md_file):
            updated_count += 1
            print(f"  Updated: {md_file.relative_to(DOCS_ROOT)}")

    print(f"\nUpdated {updated_count} files")

if __name__ == "__main__":
    main()
```

### 4.3 Front Matter Requirements

**Standard Front Matter Template**:

```yaml
---
title: "Document Title"
description: "Brief 1-2 sentence description"
category: tutorial | guide | explanation | reference | archive
tags:
  - primary-technology
  - feature-area
  - domain-concept
related-docs:
  - path/to/related-doc1.md
  - path/to/related-doc2.md
breadcrumbs:
  - text: "Home"
    href: "../README.md"
  - text: "Category"
    href: "README.md"
  - text: "Current Doc"
    href: "current-doc.md"
previous:
  title: "Previous Article"
  href: "previous.md"
next:
  title: "Next Article"
  href: "next.md"
updated-date: 2025-12-18
difficulty-level: beginner | intermediate | advanced | expert
priority: 1-5
dependencies:
  - Required software/setup
---
```

**Front Matter Injection Script**: See `link-generation-spec.md` Section 8.A

### 4.4 Navigation Template

**Category Index Template** (`[category]/README.md`):

```markdown
# [Category Name]

[Brief description of category]

## Quick Navigation

### By Topic
- [Topic 1](topic1/README.md) - Description
- [Topic 2](topic2/README.md) - Description

### Popular Documents
1. [Document 1](path/to/doc1.md) - Brief description
2. [Document 2](path/to/doc2.md) - Brief description
3. [Document 3](path/to/doc3.md) - Brief description

### By Difficulty
#### Beginner
- [Doc A](doc-a.md)
- [Doc B](doc-b.md)

#### Intermediate
- [Doc C](doc-c.md)

#### Advanced
- [Doc D](doc-d.md)

## Related Categories
- [Related Category 1](../category1/README.md)
- [Related Category 2](../category2/README.md)

---

**Last Updated**: 2025-12-18
**Documents**: [count]
**Maintainer**: [team]
```

### 4.5 Index Generation Rules

**Auto-Generate Category Indexes**:

```python
#!/usr/bin/env python3
# docs/scripts/generate-indexes.py

from pathlib import Path
import yaml

DOCS_ROOT = Path("/home/devuser/workspace/project/docs")

CATEGORIES = {
    'getting-started': 'Tutorials',
    'guides': 'How-To Guides',
    'explanations': 'Conceptual Explanations',
    'reference': 'Technical Reference',
}

def generate_category_index(category_dir: Path, category_name: str):
    """Generate README.md index for a category"""

    # Scan all markdown files in category
    docs = []
    for md_file in category_dir.rglob("*.md"):
        if md_file.name == "README.md":
            continue

        # Read front matter
        with open(md_file, 'r') as f:
            content = f.read()

        fm_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if fm_match:
            front_matter = yaml.safe_load(fm_match.group(1))
        else:
            front_matter = {}

        docs.append({
            'path': md_file.relative_to(category_dir),
            'title': front_matter.get('title', md_file.stem),
            'description': front_matter.get('description', ''),
            'priority': front_matter.get('priority', 3),
            'difficulty': front_matter.get('difficulty-level', 'intermediate'),
        })

    # Sort by priority, then alphabetically
    docs.sort(key=lambda x: (x['priority'], x['title']))

    # Generate index content
    index_content = f"# {category_name}\n\n"
    index_content += f"Total documents: {len(docs)}\n\n"
    index_content += "## Documents\n\n"

    for doc in docs:
        index_content += f"- [{doc['title']}]({doc['path']}) - {doc['description']}\n"

    index_content += f"\n---\n\n**Last Updated**: 2025-12-18\n"

    # Write to README.md
    index_path = category_dir / "README.md"
    with open(index_path, 'w') as f:
        f.write(index_content)

    print(f"Generated index: {index_path.relative_to(DOCS_ROOT)}")

def main():
    for category_slug, category_name in CATEGORIES.items():
        category_dir = DOCS_ROOT / category_slug
        if category_dir.exists():
            generate_category_index(category_dir, category_name)

if __name__ == "__main__":
    main()
```

---

## 5. Validation Against Principles

### 5.1 100% Discoverability

**Rule:** No orphaned files (all files linked from ≥2 sources)

**Validation:**
```bash
python3 docs/scripts/validate-links.py --check-orphans
```

**Remediation:**
- All orphans linked from parent category index
- All orphans linked from related-docs in front matter
- All orphans tagged for semantic discovery

### 5.2 Consistent Structure

**Rules:**
- Max directory depth: 3 levels
- Consistent naming: kebab-case
- Category-specific subdirectories only
- No mixed content types in same directory

**Validation:**
```bash
# Check directory depth
find docs -name "*.md" -type f | awk -F'/' '{print NF-1}' | sort -n | tail -1
# Should be ≤ 5 (docs/ + category/ + subcategory/ + file.md = 4 levels)

# Check naming consistency
find docs -name "*.md" | grep -v '^[a-z0-9-]*\.md$'
# Should be empty
```

### 5.3 Clear Information Hierarchy

**Rules:**
- Root: 7 top-level sections max
- Each section: 3-10 subsections
- Each subsection: 5-20 documents
- Landing pages for all sections

**Validation:**
```bash
# Check root structure
ls -1 docs/ | wc -l
# Should be ≤ 10 (7 sections + README + ARCHITECTURE + DEVELOPER_JOURNEY)

# Check section sizes
for dir in docs/{getting-started,guides,explanations,reference}; do
  echo "$dir: $(find $dir -name "*.md" | wc -l) files"
done
```

### 5.4 Easy Navigation for All User Types

**User Types:**
1. New Developer → `getting-started/` + `DEVELOPER_JOURNEY.md`
2. System Architect → `ARCHITECTURE_OVERVIEW.md` + `explanations/architecture/`
3. DevOps Engineer → `guides/infrastructure/` + `diagrams/mermaid-library/03-deployment.md`
4. API Consumer → `reference/api/` + `diagrams/server/api/`
5. Contributor → `guides/developer/` + `architecture/decisions/`

**Validation:**
- User journey mapping complete
- Fast-access index exists (`QUICK_NAVIGATION.md`)
- Breadcrumb navigation on all pages
- Related docs linked bidirectionally

---

## 6. Success Metrics

### 6.1 Quantitative Metrics

| Metric | Baseline | Target | Validation |
|--------|----------|--------|------------|
| Total files | 298 | 220-230 | `find docs -name "*.md" \| wc -l` |
| Orphaned files | 124 (41.6%) | 0 (0%) | `python3 scripts/validate-links.py --orphans` |
| Frontmatter coverage | 72.5% | 98%+ | `python3 scripts/validate-front-matter.py` |
| Duplicate groups | 14 | 0 | `fdupes -r docs/` |
| Link validity | ~60% | 98%+ | `python3 scripts/validate-links.py` |
| Directory depth | 5 max | 3 max | `find docs -type d \| awk -F'/' '{print NF}' \| sort -n \| tail -1` |
| Average outbound links | 3-5 | 8-12 | Link graph analysis |
| Average inbound links | 1-2 | 3-5 | Link graph analysis |
| Category index coverage | 60% | 100% | All directories have README.md |

### 6.2 Qualitative Metrics

**Information Architecture:**
- [ ] Clear separation between tutorials/guides/explanations/reference
- [ ] Logical grouping within each category
- [ ] Consistent naming conventions
- [ ] Appropriate directory depth

**Navigation Quality:**
- [ ] Every document reachable in ≤3 clicks from root
- [ ] Breadcrumb navigation on all pages
- [ ] Related docs linked bidirectionally
- [ ] User journey maps validated

**Content Organization:**
- [ ] No confusion about where to file new docs
- [ ] No duplicate content
- [ ] Archive contains only historical content
- [ ] Working directory has <5 temporary files

**Discoverability:**
- [ ] Tag-based navigation effective
- [ ] Search keywords comprehensive
- [ ] Category indexes helpful
- [ ] Fast-access index complete

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Days 1-2)
**Goal:** Remove duplicates, standardize structure

1. Create backup
2. Delete exact duplicates (concepts/, archive/data/pages/)
3. Standardize README files
4. Archive completed working documents
5. Create new directory structure
6. Validate Phase 1

**Deliverables:**
- Backup created
- Duplicates removed
- README files standardized
- Directory skeleton created
- Validation report

### Phase 2: Consolidation (Days 3-7)
**Goal:** Consolidate fragmented content

1. Consolidate API reference documentation
2. Organize guides directory structure
3. Consolidate WebSocket protocol docs
4. Merge testing documentation
5. Consolidate troubleshooting docs
6. Update cross-references
7. Validate Phase 2

**Deliverables:**
- API reference consolidated
- Guides organized into subdirectories
- Protocol docs merged
- Testing docs consolidated
- Cross-references updated
- Validation report

### Phase 3: Major Reorganization (Days 8-15)
**Goal:** Establish final IA structure

1. Consolidate architecture documentation
2. Organize ontology documentation
3. Organize client-side documentation
4. Clean archive directory
5. Generate all category indexes
6. Inject front matter into all docs
7. Generate bidirectional links
8. Create navigation templates
9. Validate complete IA
10. Final review and adjustments

**Deliverables:**
- Architecture docs consolidated
- Ontology docs organized
- Client docs organized
- Archive cleaned
- All indexes generated
- Front matter complete
- Links bidirectional
- Navigation templates created
- Final validation report
- Implementation complete

### Phase 4: Validation & Refinement (Days 16-18)
**Goal:** Ensure quality and completeness

1. Run comprehensive link validation
2. Check frontmatter consistency
3. Verify all orphans resolved
4. Test user journeys
5. Generate link graph visualization
6. Review with stakeholders
7. Final adjustments
8. Documentation freeze

**Deliverables:**
- Link validation report (98%+ valid)
- Orphan report (0 orphans)
- User journey validation
- Link graph visualization
- Stakeholder sign-off
- Documentation frozen

---

## 8. Rollback & Risk Mitigation

### Rollback Plan

**Backup Strategy:**
```bash
# Before Phase 1
tar -czf docs-backup-phase0-$(date +%Y%m%d).tar.gz docs/

# Before Phase 2
tar -czf docs-backup-phase1-$(date +%Y%m%d).tar.gz docs/

# Before Phase 3
tar -czf docs-backup-phase2-$(date +%Y%m%d).tar.gz docs/
```

**Git Strategy:**
```bash
# Each phase in separate branch
git checkout -b docs-restructure-phase1
# ... Phase 1 work
git commit -m "docs: Phase 1 - Remove duplicates and standardize"

git checkout -b docs-restructure-phase2
# ... Phase 2 work
git commit -m "docs: Phase 2 - Consolidate fragmented content"

# If rollback needed
git checkout main
git branch -D docs-restructure-phase2  # Discard phase 2
git checkout docs-restructure-phase1   # Return to phase 1
```

### Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Broken links after moves | High | High | Automated link update script + validation |
| Lost content during merge | Critical | Low | Manual review + diff comparison + backups |
| Confusion during transition | Medium | Medium | Phased rollout + clear communication |
| Incorrect categorization | Medium | Medium | Peer review + Diátaxis framework adherence |
| Performance degradation | Low | Low | Static site generation + CDN caching |
| Regression to old patterns | Low | Medium | Documentation guidelines + templates |

---

## 9. Maintenance Plan

### Ongoing Maintenance

**Weekly Tasks:**
- Run link validation script
- Review working/ directory (archive completed)
- Check for new orphans

**Monthly Tasks:**
- Review archive/ for outdated content
- Update category indexes
- Analyze link graph
- Check for duplicate content

**Quarterly Tasks:**
- Comprehensive structure review
- User journey validation
- Tag taxonomy update
- Documentation audit

### Documentation Decision Matrix

| If creating... | Location | Example |
|----------------|----------|---------|
| Step-by-step tutorial | `getting-started/` | Installation guide |
| Task-oriented guide | `guides/[topic]/` | Neo4j setup |
| Conceptual explanation | `explanations/[topic]/` | Hexagonal architecture |
| API specification | `reference/api/` | REST endpoint details |
| Protocol specification | `reference/protocols/` | WebSocket binary protocol |
| Database schema | `reference/database/` | Neo4j schema |
| Architecture decision | `architecture/decisions/` | ADR-001-hexagonal |
| Visual diagram | `diagrams/[topic]/` | System architecture |
| Audit/migration report | `audits/` | Neo4j migration audit |
| Temporary analysis | `working/` | Feature exploration |

---

## 10. Appendix: Related Specifications

### A. Directory Restructure Plan
See `directory-restructure-plan.md` for:
- Complete file move operations
- Detailed consolidation steps
- Bash scripts for execution

### B. Navigation Design
See `navigation-design.md` for:
- User journey maps
- Breadcrumb templates
- Sidebar navigation specs
- Related articles algorithms

### C. Link Generation
See `link-generation-spec.md` (already exists) for:
- Bidirectional link generation
- Tag-based similarity
- Link validation rules
- Front matter injection

---

**Document Status:** APPROVED FOR IMPLEMENTATION
**Next Action:** Begin Phase 1 (Days 1-2)
**Output Location:** `/home/devuser/workspace/project/docs/working/UNIFIED_ARCHITECTURE_SPEC.md`
**Related Files:**
- `directory-restructure-plan.md` (to be created)
- `navigation-design.md` (to be created)
- `link-generation-spec.md` (exists)
- `consolidation-plan.md` (exists)
- `taxonomy-analysis.md` (exists)
