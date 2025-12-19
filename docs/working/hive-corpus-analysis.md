---
title: "Documentation Corpus Analysis Report"
description: "**Generated**: 2025-12-19 18:06:41   **Analyst**: Corpus Analyzer (Hive Mind Worker)   **Scope**: /home/devuser/workspace/project/docs"
category: explanation
tags:
  - documentation
updated-date: 2025-12-19
difficulty-level: intermediate
---

# Documentation Corpus Analysis Report
**Generated**: 2025-12-19 18:06:41  
**Analyst**: Corpus Analyzer (Hive Mind Worker)  
**Scope**: /home/devuser/workspace/project/docs

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Files** | 301 |
| **Total Directories** | 96 |
| **Duplicate Filenames** | 15 |
| **Orphaned Files** | 59 |
| **Deep Nested Files** | 3 (depth > 3) |
| **Naming Violations** | 32 |
| **Reference Rate** | 83.33% |

---

## 1. Directory Distribution

**Top 20 directories by file count:**

1. `explanations/architecture` - **30 files**
2. `guides` - **30 files**
3. `(root)` - **22 files**
4. `archive/reports` - **19 files**
5. `reference` - **14 files**
6. `archive/fixes` - **12 files**
7. `guides/features` - **11 files**
8. `explanations/ontology` - **8 files**
9. `guides/developer` - **8 files**
10. `archive/sprint-logs` - **7 files**
11. `explanations/architecture/ports` - **7 files**
12. `guides/infrastructure` - **7 files**
13. `multi-agent-docker` - **7 files**
14. `reference/api` - **7 files**
15. `architecture` - **6 files**
16. `diagrams/mermaid-library` - **6 files**
17. `guides/ai-models` - **6 files**
18. `audits` - **5 files**
19. `archive/analysis` - **4 files**
20. `archive/data/pages` - **4 files**


**Total directories analysed**: 96

---

## 2. Duplicate Filenames (15)

Files with the same name appearing in multiple locations:


### `README.md` (15 occurrences)

- `README.md`
- `archive/README.md`
- `archive/deprecated-patterns/README.md`
- `archive/fixes/README.md`
- `archive/reports/README.md`
- `archive/working/README.md`
- `audits/README.md`
- `diagrams/README.md`
- `diagrams/mermaid-library/README.md`
- `explanations/architecture/README.md`
- `explanations/architecture/gpu/README.md`
- `guides/ai-models/README.md`
- `reference/README.md`
- `reference/api/README.md`
- `scripts/README.md`

### `INDEX.md` (2 occurrences)

- `INDEX.md`
- `reference/INDEX.md`

### `websocket-protocol.md` (2 occurrences)

- `explanations/architecture/components/websocket-protocol.md`
- `reference/websocket-protocol.md`

### `neo4j-integration.md` (2 occurrences)

- `explanations/ontology/neo4j-integration.md`
- `guides/neo4j-integration.md`

### `troubleshooting.md` (2 occurrences)

- `guides/infrastructure/troubleshooting.md`
- `guides/troubleshooting.md`

### `client.md` (2 occurrences)

- `concepts/architecture/core/client.md`
- `explanations/architecture/core/client.md`

### `server.md` (2 occurrences)

- `concepts/architecture/core/server.md`
- `explanations/architecture/core/server.md`

### `readme.md` (2 occurrences)

- `guides/developer/readme.md`
- `guides/infrastructure/readme.md`

### `semantic-forces.md` (2 occurrences)

- `explanations/physics/semantic-forces.md`
- `guides/features/semantic-forces.md`

### `hierarchical-visualization.md` (2 occurrences)

- `explanations/architecture/hierarchical-visualization.md`
- `explanations/ontology/hierarchical-visualization.md`

### `quick-reference.md` (2 occurrences)

- `archive/fixes/quick-reference.md`
- `explanations/architecture/quick-reference.md`

### `implementation-examples.md` (2 occurrences)

- `archive/data/markdown/implementation-examples.md`
- `archive/data/pages/implementation-examples.md`

### `IMPLEMENTATION-SUMMARY.md` (2 occurrences)

- `archive/data/markdown/IMPLEMENTATION-SUMMARY.md`
- `archive/data/pages/IMPLEMENTATION-SUMMARY.md`

### `OntologyDefinition.md` (2 occurrences)

- `archive/data/markdown/OntologyDefinition.md`
- `archive/data/pages/OntologyDefinition.md`

### `xr-setup.md` (2 occurrences)

- `archive/docs/guides/user/xr-setup.md`
- `archive/docs/guides/xr-setup.md`


---

## 3. Orphaned Files (59)

Files with no inbound links from other documentation:


### `(root)/`

- GETTING_STARTED_WITH_UNIFIED_DOCS.md
- VALIDATION_CHECKLIST.md

### `architecture/`

- phase1-completion.md
- skill-mcp-classification.md
- skills-refactoring-plan.md

### `archive/`

- ARCHIVE_REPORT.md
- INDEX-QUICK-START-old.md

### `archive/analysis/`

- analysis-summary-2025-12.md
- client-docs-summary-2025-12.md
- historical-context-recovery-2025-12.md

### `archive/audits/`

- quality-gates-api.md

### `archive/data/markdown/`

- IMPLEMENTATION-SUMMARY.md
- OntologyDefinition.md
- implementation-examples.md

### `archive/data/pages/`

- ComfyWorkFlows.md
- IMPLEMENTATION-SUMMARY.md
- OntologyDefinition.md
- implementation-examples.md

### `archive/docs/guides/`

- working-with-gui-sandbox.md

### `archive/fixes/`

- type-corrections-final-summary.md
- type-corrections-progress.md

### `archive/implementation-logs/`

- stress-majorization-implementation.md


**Action Required**: Review orphaned files for:
- Outdated content (candidate for archival)
- Missing links from navigation/index pages
- Legitimate standalone documentation

---

## 4. Deep Nested Files (3)

Files nested more than 3 directories deep:

- `archive/docs/guides/developer/05-testing-guide.md` (depth: 4)
- `archive/docs/guides/user/xr-setup.md` (depth: 4)
- `archive/docs/guides/user/working-with-agents.md` (depth: 4)


**Best Practice**: Keep documentation hierarchy to 3 levels or less for better discoverability.

---

## 5. Naming Convention Violations ({len(data['naming_violations'])})

Files not following kebab-case naming convention:


### CamelCase/TitleCase (7)

- `archive/INDEX-QUICK-START-old.md`
- `archive/data/markdown/IMPLEMENTATION-SUMMARY.md`
- `archive/data/markdown/OntologyDefinition.md`
- `archive/deprecated-patterns/03-architecture-WRONG-STACK.md`
- `multi-agent-docker/ANTIGRAVITY.md`
- `multi-agent-docker/SKILLS.md`
- `reference/INDEX.md`

### snake_case (13)

- `architecture/HEXAGONAL_ARCHITECTURE_STATUS.md`
- `archive/ARCHIVE_REPORT.md`
- `archive/reports/ARCHIVE_INDEX.md`
- `archive/reports/CLEANUP_SUMMARY.md`
- `archive/tests/test_README.md`
- `guides/ai-models/INTEGRATION_SUMMARY.md`
- `multi-agent-docker/TERMINAL_GRID.md`
- `reference/API_REFERENCE.md`
- `reference/CONFIGURATION_REFERENCE.md`
- `reference/DATABASE_SCHEMA_REFERENCE.md`
- _(+3 more)_


**Recommended Convention**: `kebab-case-naming.md`

---

## 6. Taxonomy Overview

**Root-level structure:**

- **analysis/** (2 files)
- **architecture/** (6 files)
- **archive/** (75 files)
  - **analysis/** (4 files)
  - **audits/** (1 files)
  - **data/** (7 files)
  - **deprecated-patterns/** (2 files)
  - **docs/** (5 files)
  - **fixes/** (12 files)
  - **implementation-logs/** (1 files)
  - **multi-agent-docker/** (1 files)
  - **reports/** (26 files)
  - **specialized/** (3 files)
  - **sprint-logs/** (7 files)
  - **tests/** (1 files)
  - **working/** (2 files)
- **assets/** (1 files)
  - **diagrams/** (1 files)
- **audits/** (5 files)
- **concepts/** (2 files)
  - **architecture/** (2 files)
- **diagrams/** (19 files)
  - **architecture/** (1 files)
  - **client/** (3 files)
  - **data-flow/** (1 files)
  - **infrastructure/** (3 files)
  - **mermaid-library/** (6 files)
  - **server/** (3 files)
- **explanations/** (56 files)
  - **architecture/** (45 files)
  - **ontology/** (8 files)
  - **physics/** (2 files)
- **guides/** (68 files)
  - **ai-models/** (6 files)
  - **architecture/** (1 files)
  - **client/** (3 files)
  - **developer/** (8 files)
  - **features/** (11 files)
  - **infrastructure/** (7 files)
  - **migration/** (1 files)
  - **operations/** (1 files)
- **multi-agent-docker/** (12 files)
  - **development-notes/** (2 files)
  - **fixes/** (3 files)
- **reference/** (26 files)
  - **api/** (7 files)
  - **database/** (4 files)
  - **protocols/** (1 files)
- **scripts/** (1 files)
- **tutorials/** (3 files)
- **working/** (3 files)


---

## 7. Recommendations

### High Priority
1. **Review and link orphaned files** - 59 files have no inbound links
2. **Consolidate duplicates** - 15 filenames appear in multiple locations
3. **Standardize naming** - 32 files violate kebab-case convention

### Medium Priority
4. **Flatten deep hierarchies** - 3 files are nested > 3 levels deep
5. **Update directory distribution** - Some directories have high file counts (consider splitting)

### Low Priority
6. **Maintain reference rate** - Current: 83.33% (target: >90%)

---

## Next Steps

This analysis provides the foundation for:
- **Link Validator**: Check broken links and references
- **Mermaid Validator**: Verify diagram syntax
- **Ontology Mapper**: Map documentation to codebase
- **Archive Manager**: Identify candidates for archival

**Status**: ✅ Corpus analysis complete - ready for validator coordination

