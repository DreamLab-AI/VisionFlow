---
title: Documentation Taxonomy Analysis
description: Distribution of documentation by Diataxis framework:
category: explanation
tags:
  - architecture
  - patterns
  - structure
  - api
  - rest
related-docs:
  - working/ASSET_RESTORATION.md
  - working/CLIENT_ARCHITECTURE_ANALYSIS.md
  - working/CLIENT_DOCS_SUMMARY.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Docker installation
---

# Documentation Taxonomy Analysis

**Analysis Date:** 2025-12-18T21:12:02.572007

**Total Files:** 298
**Total Size:** 10,978,332 bytes (10.47 MB)

## Executive Summary

- **Files with frontmatter:** 216 (72.5%)
- **Orphaned files:** 124 (41.6%)
- **Duplicate groups:** 0
- **Potentially obsolete:** 3

## Diataxis Classification

Distribution of documentation by Diataxis framework:

| Category | Count | Percentage | Purpose |
|----------|-------|------------|---------|
| Explanation | 151 | 50.7% | Understanding-oriented (conceptual background) |
| Howto | 106 | 35.6% | Task-oriented (practical guides) |
| Reference | 17 | 5.7% | Information-oriented (technical descriptions) |
| Unknown | 12 | 4.0% | Unclassified (needs categorization) |
| Tutorial | 12 | 4.0% | Learning-oriented (step-by-step lessons) |

## Directory Structure Analysis

### Top-Level Directory Distribution

| Directory | File Count | Total Size | Avg Size |
|-----------|------------|------------|----------|
| `archive` | 75 | 2,040,173 bytes | 27,202 bytes |
| `guides` | 69 | 1,088,296 bytes | 15,772 bytes |
| `explanations` | 56 | 1,150,328 bytes | 20,542 bytes |
| `reference` | 19 | 263,073 bytes | 13,846 bytes |
| `root` | 18 | 923,383 bytes | 51,299 bytes |
| `working` | 14 | 4,468,490 bytes | 319,178 bytes |
| `diagrams` | 13 | 563,700 bytes | 43,362 bytes |
| `multi-agent-docker` | 12 | 132,781 bytes | 11,065 bytes |
| `architecture` | 6 | 139,707 bytes | 23,284 bytes |
| `audits` | 5 | 67,163 bytes | 13,433 bytes |
| `.claude-flow` | 3 | 1,859 bytes | 620 bytes |
| `tutorials` | 3 | 35,094 bytes | 11,698 bytes |
| `analysis` | 2 | 36,414 bytes | 18,207 bytes |
| `concepts` | 2 | 47,292 bytes | 23,646 bytes |
| `assets` | 1 | 20,579 bytes | 20,579 bytes |

## Taxonomy Issues

### 1. Cross-Cutting Content

Topics that appear in multiple directories (potential for consolidation):


**readme** appears in 15 directories:
  - `.`
  - `archive`
  - `archive/deprecated-patterns`
  - `archive/fixes`
  - `archive/reports`
  - `archive/working`
  - `audits`
  - `diagrams`
  - `explanations/architecture`
  - `explanations/architecture/gpu`
  - `guides`
  - `guides/ai-models`
  - `guides/developer`
  - `guides/infrastructure`
  - `reference/api`

### 2. Orphaned Files (No Backlinks)


124 files have no incoming links from other documentation:


**archive/reports/** (18 orphans):
  - 2025-12-02-restructuring-complete.md
  - 2025-12-02-stub-implementation.md
  - 2025-12-02-user-settings-summary.md
  - ARCHIVE_INDEX.md
  - CLEANUP_SUMMARY.md
  - ... and 13 more

**./** (12 orphans):
  - ARCHITECTURE_COMPLETE.md
  - ASCII_DEPRECATION_COMPLETE.md
  - DOCUMENTATION_MODERNIZATION_COMPLETE.md
  - QA_VALIDATION_FINAL.md
  - QUICK_NAVIGATION.md
  - ... and 7 more

**archive/fixes/** (11 orphans):
  - actor-handlers.md
  - before-after-comparison.md
  - borrow-checker-summary.md
  - borrow-checker.md
  - pagerank-fix.md
  - ... and 6 more

**working/** (8 orphans):
  - ASSET_RESTORATION.md
  - DOCUMENTATION_INDEX_COMPLETE.md
  - ISOLATED_FEATURES_INTEGRATION_COMPLETE.md
  - SCREENSHOT_CAPTURE_TODOS.md
  - TotalContext.txt
  - ... and 3 more

**archive/sprint-logs/** (7 orphans):
  - p0-3-semantic-forces.md
  - p1-1-checklist.md
  - p1-1-configure-complete.md
  - p1-1-summary.md
  - p1-2-pagerank.md
  - ... and 2 more

**multi-agent-docker/** (7 orphans):
  - ANTIGRAVITY.md
  - SKILLS.md
  - TERMINAL_GRID.md
  - comfyui-sam3d-setup.md
  - hyprland-migration-summary.md
  - ... and 2 more

**architecture/** (5 orphans):
  - blender-mcp-unified-architecture.md
  - phase1-completion.md
  - skill-mcp-classification.md
  - skills-refactoring-plan.md
  - visionflow-distributed-systems-assessment.md

**archive/reports/documentation-alignment-2025-12-02/json-reports/** (5 orphans):
  - archive-report.json
  - ascii-report.json
  - link-report.json
  - mermaid-report.json
  - stubs-report.json

**guides/ai-models/** (5 orphans):
  - INTEGRATION_SUMMARY.md
  - deepseek-deployment.md
  - deepseek-verification.md
  - perplexity-integration.md
  - ragflow-integration.md

**archive/data/pages/** (4 orphans):
  - ComfyWorkFlows.md
  - IMPLEMENTATION-SUMMARY.md
  - OntologyDefinition.md
  - implementation-examples.md

### 3. Missing Frontmatter


82 files lack frontmatter metadata (27.5%):

| Directory | Files Missing Frontmatter |
|-----------|---------------------------|
| `.` | 17 |
| `working` | 9 |
| `multi-agent-docker` | 7 |
| `architecture` | 5 |
| `archive/reports/documentation-alignment-2025-12-02/json-reports` | 5 |
| `guides/ai-models` | 4 |
| `.claude-flow/metrics` | 3 |
| `guides/client` | 3 |
| `multi-agent-docker/fixes` | 3 |
| `analysis` | 2 |
| `concepts/architecture/core` | 2 |
| `diagrams` | 2 |
| `multi-agent-docker/development-notes` | 2 |
| `archive` | 1 |
| `archive/implementation-logs` | 1 |

### 4. Deprecated/Archived Content


3 files are marked as deprecated/archived:

- `archive/INDEX-QUICK-START-old.md` (9,713 bytes)
- `archive/deprecated-patterns/03-architecture-WRONG-STACK.md` (14,809 bytes)
- `archive/deprecated-patterns/README.md` (3,214 bytes)

**Archive directory contains 75 files** (2,040,173 bytes):
  - `archive/reports`: 19 files
  - `archive/fixes`: 12 files
  - `archive/sprint-logs`: 7 files
  - `archive/reports/documentation-alignment-2025-12-02/json-reports`: 5 files
  - `archive/data/pages`: 4 files
  - `archive/reports/documentation-alignment-2025-12-02`: 4 files
  - `archive`: 3 files
  - `archive/data/markdown`: 3 files
  - `archive/specialized`: 3 files
  - `archive/deprecated-patterns`: 2 files

### 5. Fragmented Topics Needing Consolidation


Topics split across multiple files that could be consolidated:


**Guide-related documentation:** 77 files (1,239,373 bytes)
  Spread across directories:
    - `guides`: 31 files
    - `guides/features`: 11 files
    - `guides/developer`: 8 files
    - `guides/infrastructure`: 7 files
    - `guides/ai-models`: 6 files

**Agent-related documentation:** 20 files (313,935 bytes)
  Spread across directories:
    - `multi-agent-docker`: 7 files
    - `guides`: 3 files
    - `multi-agent-docker/fixes`: 3 files
    - `multi-agent-docker/development-notes`: 2 files
    - `.claude-flow/metrics`: 1 files

**Docker-related documentation:** 16 files (283,171 bytes)
  Spread across directories:
    - `multi-agent-docker`: 7 files
    - `multi-agent-docker/fixes`: 3 files
    - `guides`: 2 files
    - `multi-agent-docker/development-notes`: 2 files
    - `archive/multi-agent-docker/skills`: 1 files

**Api-related documentation:** 14 files (308,988 bytes)
  Spread across directories:
    - `reference/api`: 7 files
    - `archive/audits`: 1 files
    - `.`: 1 files
    - `diagrams/architecture`: 1 files
    - `diagrams/server/api`: 1 files

**Gpu-related documentation:** 10 files (124,673 bytes)
  Spread across directories:
    - `explanations/architecture/gpu`: 3 files
    - `explanations/architecture/ports`: 2 files
    - `multi-agent-docker/fixes`: 2 files
    - `diagrams/infrastructure/gpu`: 1 files
    - `explanations/architecture`: 1 files

**Setup-related documentation:** 6 files (84,594 bytes)
  Spread across directories:
    - `archive/docs/guides/user`: 1 files
    - `archive/docs/guides`: 1 files
    - `archive/reports/documentation-alignment-2025-12-02`: 1 files
    - `guides/developer`: 1 files
    - `guides`: 1 files

### 6. Missing Documentation Areas


Identified gaps based on analysis:

- **Tutorials**: Only 12 found - need more step-by-step learning paths
- **Reference documentation**: Only 17 found - need comprehensive API/config references
- **Faq**: Only 0 file(s) found
- **Security**: Only 1 file(s) found

## Recommendations


### Immediate Actions

1. **Add frontmatter** to 82 files lacking metadata
2. **Link orphaned content** - 124 files have no backlinks
3. **Review archive/** directory - 75 files consuming 1.95MB

### Structural Improvements

1. **Consolidate fragmented topics** - Merge related documentation scattered across directories
2. **Implement Diataxis taxonomy** - Reorganize into tutorials/, how-to/, reference/, explanation/
3. **Create navigation hierarchy** - Build clear index pages and cross-references
4. **Standardize naming** - Apply consistent filename conventions

### Content Priorities

1. Create more **tutorial content** (currently only 4.0% of docs)
2. Expand **reference documentation** (only 5.7% of docs)
3. Add missing topic areas: troubleshooting, FAQ, migration guides
4. Consolidate explanation content (currently 50.7% - may be excessive)

## File Type Distribution

| Extension | Count | Total Size | Avg Size |
|-----------|-------|------------|----------|
| `.md` | 282 | 4,937,811 bytes | 17,510 bytes |
| `.json` | 12 | 4,983,817 bytes | 415,318 bytes |
| `.txt` | 4 | 1,056,704 bytes | 264,176 bytes |