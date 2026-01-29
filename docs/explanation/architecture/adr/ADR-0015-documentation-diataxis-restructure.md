---
title: "ADR-0015: Documentation Architecture Restructure (Diataxis + DDD)"
status: proposed
date: 2026-01-29
deciders: ["System Architecture Designer", "Documentation Team"]
category: reference
tags:
  - documentation
  - architecture
  - diataxis
  - ddd
related-docs:
  - docs/archive/reports/diataxis-compliance-final-report.md
  - docs/diagrams/README.md
updated-date: 2026-01-29
difficulty-level: intermediate
---

# ADR-0015: Documentation Architecture Restructure (Diataxis + DDD)

## Status

**Proposed**

## Context

VisionFlow is a complex Rust-based physics simulation platform with:
- Multi-agent docker infrastructure (Claude Flow V3 integration)
- WebSocket real-time streaming with binary protocol
- GPU-accelerated computation (39 CUDA kernels)
- RuVector PostgreSQL memory backend
- Neo4j graph database persistence
- OWL ontology reasoning engine

### Current State Analysis

**Documentation Inventory:**
- Total markdown files: 294
- Top-level directories: 16 (including archive, scripts)
- Diataxis compliance: 100% (achieved 2025-12-19)
- Mermaid diagrams: 200+

**Current Directory Structure:**
```
docs/
+-- README.md                     # Navigation hub
+-- getting-started/              # TUTORIALS (3 files)
+-- guides/                       # HOW-TO (75 files, 11 subdirs)
+-- reference/                    # REFERENCE (31 files)
+-- explanations/                 # EXPLANATION (30 files)
+-- architecture/                 # Mixed content (40 files)
+-- diagrams/                     # Mermaid sources (80+ files)
+-- archive/                      # Deprecated content
+-- research/                     # Analysis documents
+-- multi-agent-docker/           # Docker-specific docs
+-- testing/                      # Test documentation
+-- analysis/                     # Code analysis
+-- scripts/                      # Automation scripts
+-- sprints/                      # Sprint planning
+-- assets/                       # Images, static files
```

### Identified Issues

1. **Category Overlap**: `architecture/` and `explanations/architecture/` contain overlapping content
2. **Missing Use-Cases Section**: Real-world applications scattered across getting-started/overview.md
3. **Concepts vs Explanations**: No clear `concepts/` directory for foundational understanding
4. **Depth Violation**: Some paths exceed 3 levels (e.g., `explanations/architecture/decisions/`)
5. **Naming Inconsistencies**: Mix of SCREAMING_CASE and kebab-case filenames
6. **Orphan Content**: research/, analysis/, sprints/ not linked from main navigation

## Decision

Adopt a restructured documentation architecture that:
1. Maintains strict Diataxis framework alignment
2. Adds dedicated `concepts/` and `use-cases/` sections
3. Consolidates architecture content under `explanations/`
4. Enforces maximum 3-level directory depth
5. Standardizes all filenames to lowercase-kebab-case

### Proposed Structure

```
docs/
+-- README.md                          # Hub with role-based navigation
+-- CHANGELOG.md                       # Documentation changelog
|
+-- getting-started/                   # TUTORIALS - Learning-oriented
|   +-- README.md                      # Tutorial index
|   +-- installation.md                # Docker and native setup
|   +-- first-graph.md                 # Create first visualization
|   +-- neo4j-quickstart.md            # Database basics
|   +-- hello-agent.md                 # NEW: First AI agent deployment
|
+-- guides/                            # HOW-TO - Task-oriented
|   +-- README.md                      # Guide index
|   +-- deployment/                    # Production deployment
|   |   +-- docker-compose.md
|   |   +-- kubernetes.md
|   |   +-- security-hardening.md
|   +-- development/                   # Developer workflows
|   |   +-- setup.md
|   |   +-- adding-features.md
|   |   +-- testing.md
|   |   +-- contributing.md
|   +-- integration/                   # External integrations
|   |   +-- neo4j.md
|   |   +-- github-sync.md
|   |   +-- solid-pods.md
|   +-- operations/                    # Day-to-day ops
|   |   +-- monitoring.md
|   |   +-- troubleshooting.md
|   |   +-- backup-restore.md
|   +-- features/                      # Feature-specific how-tos
|       +-- natural-language-queries.md
|       +-- semantic-forces.md
|       +-- intelligent-pathfinding.md
|
+-- reference/                         # REFERENCE - Information-oriented
|   +-- README.md                      # Reference index
|   +-- api/                           # API specifications
|   |   +-- rest-api.md
|   |   +-- websocket-api.md
|   |   +-- authentication.md
|   +-- configuration/                 # Config reference
|   |   +-- environment-variables.md
|   |   +-- docker-compose-options.md
|   |   +-- neo4j-settings.md
|   +-- protocols/                     # Protocol specs
|   |   +-- binary-websocket.md
|   |   +-- mcp-protocol.md
|   +-- database/                      # Schema documentation
|   |   +-- neo4j-schema.md
|   |   +-- ontology-schema.md
|   +-- cli/                           # CLI reference
|   |   +-- cargo-commands.md
|   |   +-- docker-commands.md
|   +-- error-codes.md                 # Error reference
|   +-- glossary.md                    # NEW: Terminology definitions
|
+-- concepts/                          # EXPLANATION - Understanding-oriented
|   +-- README.md                      # Concepts index
|   +-- physics-engine.md              # Force-directed graph physics
|   +-- constraint-system.md           # Semantic constraint resolution
|   +-- gpu-acceleration.md            # CUDA architecture overview
|   +-- ontology-reasoning.md          # OWL inference concepts
|   +-- actor-model.md                 # Actix actor system
|   +-- hexagonal-architecture.md      # Ports and adapters pattern
|   +-- multi-agent-system.md          # AI agent coordination
|   +-- real-time-sync.md              # WebSocket synchronization
|
+-- use-cases/                         # NEW SECTION - Application scenarios
|   +-- README.md                      # Use-cases index
|   +-- industry-applications.md       # Sector-specific deployments
|   +-- privacy-features.md            # Data sovereignty scenarios
|   +-- decentralization.md            # Distributed deployment
|   +-- research-workflows.md          # Academic use patterns
|   +-- enterprise-knowledge.md        # Corporate KM scenarios
|
+-- architecture/                      # Technical architecture (ADRs)
|   +-- README.md                      # Architecture overview
|   +-- adr/                           # Architecture Decision Records
|   |   +-- ADR-0001-neo4j-persistence.md
|   |   +-- ADR-0015-documentation-restructure.md
|   +-- diagrams/                      # Architecture diagrams only
|       +-- system-context.md
|       +-- data-flow.md
|
+-- diagrams/                          # Mermaid diagram library
|   +-- README.md                      # Diagram index
|   +-- mermaid-library/               # Reusable diagrams
|   +-- infrastructure/                # Infra diagrams
|   +-- data-flow/                     # Data flow diagrams
|
+-- archive/                           # Historical content
    +-- README.md
    +-- audits/
    +-- reports/
    +-- phases/
```

## Rationale

### 1. Why Add `concepts/`?

The Diataxis "Explanation" quadrant covers understanding-oriented content, but the current structure conflates:
- **Concepts** (foundational understanding): "What is force-directed layout?"
- **Architecture** (system design): "How does VisionFlow implement physics?"

Separating these:
- Helps new users understand fundamentals before diving into implementation
- Reduces cognitive load by providing clear conceptual foundations
- Aligns with DDD's focus on ubiquitous language

### 2. Why Add `use-cases/`?

Currently, use-case content is scattered:
- getting-started/overview.md (lines 151-167)
- Various guide introductions

Dedicated section benefits:
- Helps prospects understand applicability
- Provides templates for implementation patterns
- Supports sales/marketing alignment with technical docs
- Maps to DDD's bounded context identification

### 3. Why Consolidate Architecture?

Current state has:
- `docs/architecture/` (40 files)
- `docs/explanations/architecture/` (30 files)

Consolidation approach:
- Move conceptual content to `concepts/`
- Move ADRs to `architecture/adr/`
- Keep only system-level architecture diagrams in `architecture/`

### 4. Why 3-Level Depth Maximum?

Current violations:
- `docs/architecture/adr/ADR-0001-*.md` (4 levels)
- `docs/archive/reports/code-quality/` (4 levels)

3-level maximum:
- Reduces navigation complexity
- Improves discoverability
- Aligns with URL best practices

### 5. Why Lowercase-Kebab-Case?

Current mixed naming:
- `HEXAGONAL_ARCHITECTURE_STATUS.md` (SCREAMING_SNAKE)
- `API_REFERENCE.md` (SCREAMING_SNAKE)
- `binary-websocket.md` (kebab-case)

Standardization benefits:
- Consistent URLs
- Easier scripting/automation
- Platform compatibility (case-sensitive filesystems)

## Migration Plan

### Phase 1: Preparation (Week 1)

1. **Create backup**: `cp -r docs docs-backup-2026-01-29`
2. **Generate link inventory**: Extract all internal links
3. **Create redirect map**: Old path -> New path mapping
4. **Update CI/CD**: Add link validation to pipeline

### Phase 2: Structure Creation (Week 2)

1. Create new directories:
   ```bash
   mkdir -p docs/concepts
   mkdir -p docs/use-cases
   mkdir -p docs/architecture/adr
   ```

2. Move files according to mapping (see File Rename Mapping below)

3. Update README.md hub with new structure

### Phase 3: Content Migration (Weeks 3-4)

1. **Split architecture content**:
   - Conceptual -> `concepts/`
   - ADRs -> `architecture/adr/`
   - Technical diagrams -> `architecture/diagrams/`

2. **Consolidate explanations**:
   - Merge `explanations/architecture/` into `concepts/`
   - Archive redundant content

3. **Create use-cases**:
   - Extract from getting-started/overview.md
   - Add industry-specific templates

### Phase 4: Link Updates (Week 5)

1. Run automated link rewriter
2. Manual verification of cross-references
3. Update external documentation links

### Phase 5: Validation (Week 6)

1. Run Diataxis compliance checker
2. Verify all pages linked from navigation
3. Test all internal links
4. Update search indexes

## File Rename Mapping

### High-Impact Renames

| Current Path | New Path | Reason |
|-------------|----------|--------|
| `docs/architecture/overview.md` | `docs/architecture/README.md` | Consolidation |
| `docs/explanations/architecture/decisions/` | `docs/architecture/adr/` | Flatten + rename |
| `docs/reference/api/README.md` | `docs/reference/api/README.md` | Naming convention |
| `docs/reference/configuration/README.md` | `docs/reference/configuration/README.md` | Naming convention |
| `docs/reference/protocols/README.md` | `docs/reference/protocols/README.md` | Naming convention |
| `docs/reference/database/README.md` | `docs/reference/database/README.md` | Naming convention |
| `docs/reference/error-codes.md` | `docs/reference/error-codes.md` | Naming convention |
| `docs/concepts/hexagonal-architecture.md` | `docs/concepts/hexagonal-architecture.md` | Category + naming |
| `docs/architecture/VIRCADIA_BABYLON_CONSOLIDATION_ANALYSIS.md` | `docs/archive/analysis/vircadia-babylon.md` | Archive + naming |

### Content Splits

| Source File | Split Into | Content Distribution |
|------------|-----------|---------------------|
| `docs/getting-started/overview.md` | `use-cases/README.md` | Lines 151-167 (use cases) |
| | `concepts/README.md` | Lines 107-148 (key capabilities) |
| `docs/concepts/semantic-physics-system.md` | `concepts/physics-engine.md` | Conceptual content |
| | `architecture/diagrams/physics-flow.md` | Technical diagrams |

## Link Update Requirements

### Internal Link Patterns to Update

```regex
# ADR links
\(.*decisions/0001.*\.md\) -> (../architecture/adr/ADR-0001-*.md)

# SCREAMING_CASE files
\([^)]*[A-Z_]{2,}\.md\) -> lowercase-kebab equivalent

# Explanations/architecture consolidation
\(.*explanations/architecture/.*\.md\) -> concepts/ or architecture/
```

### External Links to Verify

- GitHub README links
- API documentation references
- Integration partner documentation

## Diagram Placement Strategy

### Diagram Types and Locations

| Diagram Type | Location | Purpose |
|-------------|----------|---------|
| System context (C4 Level 1) | `architecture/diagrams/` | High-level system boundaries |
| Container (C4 Level 2) | `architecture/diagrams/` | Service decomposition |
| Component (C4 Level 3) | `concepts/` (inline) | Conceptual understanding |
| Sequence diagrams | Relevant guide/concept | Process flows |
| Entity-relationship | `reference/database/` | Schema documentation |
| Data flow | `diagrams/data-flow/` | End-to-end data paths |
| Infrastructure | `diagrams/infrastructure/` | Deployment topology |

### Mermaid Source Management

1. **Inline for simple diagrams** (<30 lines): Embed directly in markdown
2. **Separate files for complex diagrams**: Store in `diagrams/mermaid-library/`
3. **Include via reference**: Use relative includes for reusability

```markdown
<!-- Example include pattern -->
See [System Context Diagram](../diagrams/mermaid-library/01-system-architecture-overview.md#system-context)
```

## Consequences

### Positive

1. **Clearer mental model**: Users understand where to find content
2. **Reduced duplication**: Consolidated architecture content
3. **Better discoverability**: Dedicated use-cases section
4. **Consistent URLs**: Lowercase-kebab naming
5. **Maintainability**: 3-level depth maximum
6. **Diataxis compliance**: Strict quadrant alignment

### Negative

1. **Migration effort**: ~40 hours estimated
2. **External link breakage**: Requires redirect setup
3. **Learning curve**: Team adapts to new structure
4. **Search index rebuild**: Temporary discoverability impact

### Neutral

1. **Archive growth**: Deprecated content preserved
2. **CI/CD changes**: Link validation additions

## Metrics for Success

| Metric | Target | Measurement |
|--------|--------|-------------|
| Internal link integrity | 100% | Automated link checker |
| Diataxis compliance | 100% | Category validation script |
| Depth violations | 0 | Directory depth check |
| Naming violations | 0 | Filename pattern check |
| Orphan pages | 0 | Navigation coverage audit |
| User satisfaction | +20% | Documentation feedback survey |

## References

- [Diataxis Framework](https://diataxis.fr/)
- [C4 Model for Architecture Documentation](https://c4model.com/)
- [Domain-Driven Design Documentation Patterns](https://www.domainlanguage.com/ddd/)
- [Google Technical Writing Style Guide](https://developers.google.com/style)

## Appendix: Automation Scripts

### Link Validation Script

```bash
#!/bin/bash
# docs/scripts/validate-links.sh
find docs -name "*.md" -exec grep -l '\[.*\](.*\.md)' {} \; | \
  while read file; do
    grep -oP '\[.*?\]\(\K[^)]+\.md' "$file" | \
      while read link; do
        dir=$(dirname "$file")
        target=$(realpath --relative-to=. "$dir/$link" 2>/dev/null)
        [ ! -f "$target" ] && echo "BROKEN: $file -> $link"
      done
  done
```

### Diataxis Category Validator

```bash
#!/bin/bash
# docs/scripts/check-diataxis.sh
for dir in tutorials guides reference concepts; do
  find "docs/$dir" -name "*.md" -exec grep -l "category:" {} \; | \
    while read file; do
      cat=$(grep -oP 'category:\s*\K\w+' "$file")
      [ "$cat" != "${dir%s}" ] && echo "MISMATCH: $file has category $cat"
    done
done
```

---

**Decision recorded by**: System Architecture Designer
**Memory key**: `docs-architecture/diataxis-structure-2026-01-29`
