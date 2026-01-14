# VisionFlow Documentation Structure v2.0

## Executive Summary

This document defines a clean, unified documentation architecture following the Diataxis framework. The new structure reduces 68 directories to 12 top-level directories, eliminates duplicate navigation files, and enforces a maximum depth of 3 levels.

### Current Problems Addressed

| Problem | Current State | Solution |
|---------|---------------|----------|
| Directory sprawl | 68 directories | 12 top-level directories |
| Navigation confusion | 4 nav files (README, INDEX, NAVIGATION, QUICK_NAVIGATION) | Single README.md per directory |
| Overlapping purposes | concepts/, explanations/, guides/ all contain architecture docs | Clear Diataxis separation |
| Legacy files | phase6-*, phase7-* scattered throughout | Archive legacy, rename active |
| Non-docs in docs | .py, .json, .html files | Move to scripts/ or remove |
| Inconsistent naming | MIX_OF_CASE.md vs lowercase.md | Enforce lowercase-with-hyphens |
| Deep nesting | Up to 6 levels deep | Max 3 levels |

---

## Directory Structure

```
docs/
├── README.md                           # Single entry point (replaces INDEX.md, NAVIGATION.md, QUICK_NAVIGATION.md)
├── getting-started/                    # TUTORIALS - Learning-oriented
│   ├── index.md                        # Section overview with learning paths
│   ├── installation.md                 # Docker and native setup
│   ├── first-graph.md                  # Create first visualisation
│   ├── neo4j-quickstart.md             # Database basics
│   └── navigation-basics.md            # 3D interface introduction
│
├── guides/                             # HOW-TO - Task-oriented
│   ├── index.md                        # Section overview by task category
│   ├── configuration.md                # Environment and settings
│   ├── troubleshooting.md              # Common issues and fixes
│   ├── security.md                     # Auth, secrets, hardening
│   ├── deployment.md                   # Production deployment
│   ├── testing.md                      # Test execution and coverage
│   ├── contributing.md                 # Contribution workflow
│   ├── features/                       # Feature-specific how-tos
│   │   ├── index.md
│   │   ├── filtering-nodes.md
│   │   ├── natural-language-queries.md
│   │   ├── pathfinding.md
│   │   ├── semantic-forces.md
│   │   ├── auth-user-settings.md
│   │   ├── nostr-auth.md
│   │   └── solid-integration.md
│   ├── neo4j/                          # Database operations
│   │   ├── index.md
│   │   ├── integration.md
│   │   ├── migration.md
│   │   └── ontology-storage.md
│   ├── agents/                         # AI agent operations
│   │   ├── index.md
│   │   ├── orchestration.md
│   │   ├── skills.md
│   │   └── ai-models.md
│   ├── xr/                             # XR/VR operations
│   │   ├── index.md
│   │   ├── vircadia-setup.md
│   │   └── multi-user.md
│   ├── infrastructure/                 # Infrastructure operations
│   │   ├── index.md
│   │   ├── docker-setup.md
│   │   ├── port-configuration.md
│   │   └── monitoring.md
│   └── developer/                      # Developer workflows
│       ├── index.md
│       ├── setup.md
│       ├── project-structure.md
│       ├── adding-features.md
│       ├── websocket-patterns.md
│       └── client-development.md
│
├── reference/                          # REFERENCE - Information-oriented
│   ├── index.md                        # Section overview
│   ├── api/                            # API specifications
│   │   ├── index.md
│   │   ├── rest-api.md                 # Complete REST reference
│   │   ├── websocket-api.md            # Real-time protocol
│   │   ├── authentication.md           # Auth endpoints
│   │   ├── solid-api.md                # Solid Pod API
│   │   └── semantic-api.md             # NLP query API
│   ├── protocols/                      # Protocol specifications
│   │   ├── index.md
│   │   ├── binary-websocket.md         # 36-byte node format
│   │   └── websocket-v2.md             # Protocol version 2
│   ├── database/                       # Database schemas
│   │   ├── index.md
│   │   ├── schemas.md                  # Neo4j graph schema
│   │   ├── ontology-schema.md          # OWL storage schema
│   │   └── user-settings-schema.md     # User data model
│   ├── error-codes.md                  # Error reference
│   ├── configuration-reference.md      # All config options
│   └── implementation-status.md        # Feature matrix
│
├── architecture/                       # EXPLANATION - Understanding-oriented
│   ├── index.md                        # Architecture overview (replaces ARCHITECTURE_OVERVIEW.md)
│   ├── overview.md                     # What VisionFlow is (replaces OVERVIEW.md)
│   ├── technology-choices.md           # Stack rationale (replaces TECHNOLOGY_CHOICES.md)
│   ├── developer-journey.md            # Codebase learning path
│   ├── system/                         # System architecture
│   │   ├── index.md
│   │   ├── hexagonal-cqrs.md           # Ports and adapters
│   │   ├── data-flow.md                # End-to-end pipeline
│   │   ├── services.md                 # Business logic layer
│   │   ├── multi-agent.md              # Agent coordination
│   │   └── integration-patterns.md     # System integration
│   ├── components/                     # Component deep-dives
│   │   ├── index.md
│   │   ├── server.md                   # Rust/Actix backend
│   │   ├── client.md                   # React/Three.js frontend
│   │   ├── database.md                 # Neo4j architecture
│   │   ├── websocket.md                # Real-time communication
│   │   └── visualisation.md            # 3D rendering pipeline
│   ├── gpu/                            # GPU subsystem
│   │   ├── index.md
│   │   ├── semantic-forces.md          # 39 CUDA kernels
│   │   ├── communication-flow.md       # CPU-GPU transfer
│   │   └── optimisations.md            # Performance tuning
│   ├── ontology/                       # Ontology subsystem
│   │   ├── index.md
│   │   ├── storage.md                  # OWL persistence
│   │   ├── reasoning-pipeline.md       # Whelk inference
│   │   ├── semantic-forces.md          # Physics constraints
│   │   └── parser.md                   # Enhanced OWL parser
│   ├── ports/                          # Hexagonal ports
│   │   ├── index.md
│   │   ├── knowledge-graph.md
│   │   ├── ontology-repository.md
│   │   ├── inference-engine.md
│   │   ├── gpu-physics.md
│   │   └── settings-repository.md
│   ├── decisions/                      # ADRs
│   │   ├── index.md
│   │   └── 0001-neo4j-persistence.md
│   └── diagrams/                       # Architecture diagrams
│       ├── index.md
│       ├── system-overview.mmd
│       ├── data-flow.mmd
│       └── component-interactions.mmd
│
├── development/                        # Contributor documentation
│   ├── index.md
│   ├── setup.md                        # Development environment
│   ├── workflow.md                     # Git, CI/CD, PRs
│   ├── code-style.md                   # Formatting standards
│   ├── testing-strategy.md             # Test philosophy
│   └── release-process.md              # Version and deploy
│
├── operations/                         # Production operations
│   ├── index.md
│   ├── runbook.md                      # Operator playbook
│   ├── monitoring.md                   # Observability setup
│   ├── backup-recovery.md              # Data protection
│   └── incident-response.md            # Incident handling
│
├── assets/                             # Static assets
│   ├── diagrams/                       # Mermaid source files
│   │   └── *.mmd
│   ├── screenshots/                    # UI screenshots
│   │   └── *.png
│   └── images/                         # Other images
│       └── *.svg
│
├── archive/                            # Historical documentation
│   ├── index.md                        # Archive manifest
│   ├── sprints/                        # Sprint reports
│   ├── audits/                         # Audit reports
│   ├── research/                       # Research notes
│   └── legacy/                         # Deprecated docs
│       ├── phase6/
│       └── phase7/
│
└── scripts/                            # Documentation tooling (NON-DOCS)
    ├── validate-links.py               # Link checker
    ├── generate-index.py               # Index generator
    └── convert-diagrams.py             # Diagram converter
```

---

## Diataxis Mapping

### 1. Tutorials (getting-started/)
**Purpose**: Learning-oriented, teaches through doing
**Audience**: New users with no prior VisionFlow experience
**Style**: Step-by-step, hand-holding, assumes nothing

| Current Location | New Location |
|------------------|--------------|
| tutorials/01-installation.md | getting-started/installation.md |
| tutorials/02-first-graph.md | getting-started/first-graph.md |
| tutorials/neo4j-quick-start.md | getting-started/neo4j-quickstart.md |
| guides/navigation-guide.md | getting-started/navigation-basics.md |

**Rationale**: Tutorials are distinct from guides. They exist to teach concepts through a controlled learning experience. The current tutorials/ folder has only 3 files but navigation-guide.md belongs here as it teaches fundamentals.

### 2. How-To Guides (guides/)
**Purpose**: Task-oriented, solves specific problems
**Audience**: Users who know what they want to accomplish
**Style**: Goal-focused, assumes basic familiarity

| Current Location | New Location |
|------------------|--------------|
| guides/configuration.md | guides/configuration.md |
| guides/troubleshooting.md | guides/troubleshooting.md |
| guides/security.md | guides/security.md |
| guides/deployment.md | guides/deployment.md |
| guides/testing-guide.md | guides/testing.md |
| guides/features/*.md | guides/features/*.md |
| guides/neo4j-*.md | guides/neo4j/*.md |
| guides/agent-orchestration.md | guides/agents/orchestration.md |
| guides/orchestrating-agents.md | guides/agents/orchestration.md (merge) |
| guides/multi-agent-skills.md | guides/agents/skills.md |
| guides/vircadia-xr-complete-guide.md | guides/xr/vircadia-setup.md |
| guides/vircadia-multi-user-guide.md | guides/xr/multi-user.md |
| guides/infrastructure/*.md | guides/infrastructure/*.md |
| guides/developer/*.md | guides/developer/*.md |
| guides/client/*.md | guides/developer/client-development.md (merge) |

**Rationale**: Guides remain task-oriented but reorganised into logical subgroups. Duplicate guides (agent-orchestration.md vs orchestrating-agents.md) are merged.

### 3. Reference (reference/)
**Purpose**: Information-oriented, lookup of facts
**Audience**: Developers needing precise specifications
**Style**: Dry, complete, accurate

| Current Location | New Location |
|------------------|--------------|
| reference/api-complete-reference.md | reference/api/rest-api.md |
| reference/api/rest-api-complete.md | reference/api/rest-api.md (merge) |
| reference/api/rest-api-reference.md | reference/api/rest-api.md (merge) |
| reference/api/01-authentication.md | reference/api/authentication.md |
| reference/api/03-websocket.md | reference/api/websocket-api.md |
| reference/protocols/binary-websocket.md | reference/protocols/binary-websocket.md |
| reference/websocket-protocol.md | reference/protocols/websocket-v2.md |
| reference/database/*.md | reference/database/*.md |
| reference/error-codes.md | reference/error-codes.md |
| reference/CONFIGURATION_REFERENCE.md | reference/configuration-reference.md |

**Rationale**: Multiple REST API reference files are merged into one authoritative source. Protocol docs consolidated.

### 4. Explanation (architecture/)
**Purpose**: Understanding-oriented, explains concepts
**Audience**: Architects, senior developers
**Style**: Discursive, explores context and alternatives

| Current Location | New Location |
|------------------|--------------|
| ARCHITECTURE_OVERVIEW.md | architecture/index.md |
| OVERVIEW.md | architecture/overview.md |
| TECHNOLOGY_CHOICES.md | architecture/technology-choices.md |
| DEVELOPER_JOURNEY.md | architecture/developer-journey.md |
| explanations/system-overview.md | architecture/system/index.md |
| explanations/architecture/hexagonal-cqrs.md | architecture/system/hexagonal-cqrs.md |
| explanations/architecture/data-flow-complete.md | architecture/system/data-flow.md |
| explanations/architecture/core/server.md | architecture/components/server.md |
| explanations/architecture/core/client.md | architecture/components/client.md |
| explanations/architecture/database-architecture.md | architecture/components/database.md |
| explanations/architecture/gpu/*.md | architecture/gpu/*.md |
| explanations/ontology/*.md | architecture/ontology/*.md |
| explanations/architecture/ports/*.md | architecture/ports/*.md |
| concepts/architecture/core/server.md | architecture/components/server.md (merge) |

**Rationale**: The concepts/ and explanations/ directories served the same purpose. Consolidated into architecture/ with clear subsections.

---

## File Naming Conventions

### Rules
1. **Lowercase with hyphens**: `my-document-name.md`
2. **No ALL_CAPS filenames**: `ARCHITECTURE_OVERVIEW.md` becomes `architecture/index.md`
3. **No numbered prefixes**: `01-installation.md` becomes `installation.md`
4. **No phase/date prefixes**: `phase6-multiuser-sync.md` moves to archive/
5. **No underscores**: `user_settings.md` becomes `user-settings.md`
6. **index.md for directories**: Each directory has an index.md entry point

### Exceptions
- ADR files retain their numbered format: `0001-neo4j-persistence.md`
- Version suffixes when necessary: `ontology-schema-v2.md`

---

## Navigation Design

### Single Entry Points
Each directory has exactly one `index.md` that serves as:
- Section introduction
- Table of contents
- Links to all child documents
- Context for when to use each document

### Cross-References
- Use relative paths: `../guides/configuration.md`
- Never use absolute paths: `/docs/guides/configuration.md`
- Prefer links to index files for sections: `../architecture/` links to `../architecture/index.md`

### Breadcrumbs
Each document includes a frontmatter navigation hint:

```yaml
---
title: "Installation Guide"
parent: "getting-started/index.md"
section: "Getting Started"
---
```

---

## Migration Plan

### Phase 1: Preparation (Day 1)
1. Create new directory structure (empty)
2. Generate mapping file: old-path -> new-path
3. Create redirect stubs for all moved files
4. Back up current docs/

### Phase 2: Content Migration (Days 2-3)
1. Move tutorials -> getting-started/
2. Reorganise guides/ with subdirectories
3. Merge duplicate reference files
4. Consolidate concepts/ + explanations/ -> architecture/
5. Move legacy files -> archive/

### Phase 3: Cleanup (Day 4)
1. Remove empty directories
2. Delete redirect stubs after link update
3. Remove .py, .json, .html files (move to scripts/)
4. Rename ALL_CAPS files
5. Update all internal links

### Phase 4: Verification (Day 5)
1. Run link validator
2. Verify all documents accessible from README.md
3. Check Mermaid diagram rendering
4. Update external references (CHANGELOG, root README)

---

## Files to Delete/Archive

### Move to archive/legacy/
- phase6-integration-guide.md
- phase6-multiuser-sync-implementation.md
- phase7_broadcast_optimization.md
- phase7_implementation_summary.md
- architecture/phase1-completion.md
- All *-COMPLETE.md, *-ANALYSIS.md reports

### Move to scripts/
- validate_links.py
- validate_links_enhanced.py
- visionflow_strategic_analysis.py
- visionflow_wardley_analysis.py
- scripts/*.py

### Delete (generated/temporary)
- link-audit-categorized.json
- link-audit-fix-report.json
- .claude-flow/metrics/*.json
- visionflow_wardley_map.html

### Merge duplicates
- guides/agent-orchestration.md + guides/orchestrating-agents.md
- reference/api-complete-reference.md + reference/api/rest-api-complete.md + reference/api/rest-api-reference.md
- concepts/architecture/core/server.md + explanations/architecture/core/server.md

---

## Navigation Files Consolidation

### Current State (4 navigation files)
1. README.md - Main entry, 800+ lines
2. INDEX.md - Master index, 665 lines
3. NAVIGATION.md - Navigation guide
4. QUICK_NAVIGATION.md - Quick links

### New State (1 navigation file per directory)
1. docs/README.md - Unified entry point (~300 lines)
   - Quick start (5-minute path)
   - Role-based navigation (4 paths)
   - Section overview with links to index.md files
   - Recent updates
   - Getting help

Each subdirectory's index.md handles its own navigation.

---

## Design Rationale

### Why This Structure?

**1. Diataxis Compliance**
The Diataxis framework separates documentation by user need:
- Learning -> Tutorials (getting-started/)
- Doing -> How-to Guides (guides/)
- Understanding -> Explanation (architecture/)
- Information -> Reference (reference/)

This eliminates the current confusion where concepts/, explanations/, and guides/architecture/ all contain overlapping content.

**2. Maximum 3 Levels Deep**
Deep nesting (current: 6 levels) makes navigation difficult:
- Current: docs/explanations/architecture/components/websocket-protocol.md
- New: docs/architecture/components/websocket.md

**3. Single Responsibility**
Each file should have one purpose:
- Current: 3 REST API reference files with overlapping content
- New: 1 authoritative rest-api.md

**4. Predictable Locations**
Users should be able to guess where content lives:
- "How do I configure X?" -> guides/configuration.md
- "What's the API for Y?" -> reference/api/
- "Why was Z designed this way?" -> architecture/

**5. Separate Code from Docs**
Python scripts in docs/ violate separation of concerns:
- Move to docs/scripts/ (clearly tooling, not content)
- Or move to project-root/scripts/docs/

---

## Success Criteria

After migration, the documentation must:

1. [ ] Have exactly 12 top-level directories
2. [ ] Have no file deeper than 3 levels
3. [ ] Have no duplicate navigation files
4. [ ] Have no ALL_CAPS filenames (except ADRs)
5. [ ] Have no numbered prefixes (except ADRs)
6. [ ] Have 100% valid internal links
7. [ ] Have every directory contain an index.md
8. [ ] Have no .py, .json, .html files outside scripts/
9. [ ] Pass Diataxis audit (each doc fits exactly one category)
10. [ ] Be navigable from README.md to any document in 3 clicks

---

## Appendix: Full File Mapping

See [file-mapping.csv](./file-mapping.csv) for complete old->new path mapping.

---

**Document Version**: 1.0
**Created**: 2026-01-14
**Author**: Documentation Architecture Team
**Status**: Proposed
