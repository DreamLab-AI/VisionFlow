# Filename Standardization - Visual Overview

**Visual guide to the documentation restructuring process**

---

## 📊 Before & After: Directory Structure

### BEFORE Standardization

```
docs/
├── readme.md
├── alignment-report.md                          ← SCREAMING-SNAKE-CASE
├── deprecation-strategy-index.md                ← SCREAMING-SNAKE-CASE
├── documentation-audit-completion-report.md     ← SCREAMING-SNAKE-CASE
├── GRAPHSERVICEACTOR-DEPRECATION-*.md (8 files) ← SCREAMING-SNAKE-CASE
├── link-validation-report.md                    ← SCREAMING-SNAKE-CASE
├── NEO4J-SETTINGS-MIGRATION-*.md                ← SCREAMING-SNAKE-CASE
│
├── concepts/
│   ├── architecture/
│   │   ├── 00-ARCHITECTURE-overview.md          ← SCREAMING-SNAKE-CASE
│   │   ├── cqrs-directive-template.md           ← SCREAMING-SNAKE-CASE
│   │   ├── pipeline-integration.md              ← SCREAMING-SNAKE-CASE
│   │   ├── pipeline-sequence-diagrams.md        ← SCREAMING-SNAKE-CASE
│   │   ├── quick-reference.md                   ← SCREAMING-SNAKE-CASE
│   │   ├── semantic-physics.md                  ← Ambiguous
│   │   ├── semantic-physics-system.md           ← Ambiguous
│   │   └── reasoning-tests-summary.md           ← Ambiguous
│   ├── hierarchical-visualization.md            ← DUPLICATE
│   ├── neo4j-integration.md                     ← DUPLICATE
│   └── ontology-reasoning.md                    ← Ambiguous
│
├── guides/
│   ├── developer/
│   │   ├── 01-development-setup.md              ✓
│   │   ├── 02-project-structure.md              ✓
│   │   ├── 03-architecture.md                   ✓
│   │   ├── 04-adding-features.md                ✓
│   │   ├── 04-testing-status.md                 ← NUMBERING CONFLICT
│   │   ├── 05-testing.md                        ✓
│   │   ├── 06-contributing.md                   ✓
│   │   ├── development-setup.md                 ← DUPLICATE
│   │   ├── adding-a-feature.md                  ← DUPLICATE
│   │   └── 05-05-testing-guide.md                     ← DUPLICATE
│   ├── operations/
│   │   └── pipeline-operator-runbook.md         ← SCREAMING-SNAKE-CASE
│   ├── user/
│   │   └── xr-setup.md                          ← DUPLICATE (different audience)
│   ├── 05-05-testing-guide.md                         ← DUPLICATE
│   ├── xr-setup.md                              ← DUPLICATE (different audience)
│   └── neo4j-integration.md                     ← DUPLICATE
│
├── reference/
│   ├── api/
│   │   ├── 01-authentication.md                 ✓
│   │   ├── 03-websocket.md                      ← GAP (missing 02)
│   │   ├── rest-api-complete.md                 ← Ambiguous
│   │   └── rest-api-reference.md                ← Ambiguous
│   └── semantic-physics-implementation.md       ← Ambiguous
│
├── implementation/
│   └── stress-majorization-implementation.md    ← SCREAMING-SNAKE-CASE
│
└── multi-agent-docker/
    ├── architecture.md                          ← SCREAMING-SNAKE-CASE
    ├── DOCKER-environment.md                    ← SCREAMING-SNAKE-CASE
    ├── GOALIE-integration.md                    ← SCREAMING-SNAKE-CASE
    ├── PORT-configuration.md                    ← SCREAMING-SNAKE-CASE
    ├── tools.md                                 ← SCREAMING-SNAKE-CASE
    └── troubleshooting.md                       ← SCREAMING-SNAKE-CASE
```

### AFTER Standardization

```
docs/
├── readme.md                                    ✓ Standard
├── contributing.md                              ✓ Standard exception
│
├── reports/                                     ★ NEW DIRECTORY
│   ├── audits/
│   │   ├── alignment-report-2025-11-04.md       ← Moved & renamed
│   │   ├── documentation-audit-completion-2025-11-04.md
│   │   └── link-validation-report-2025-11-04.md
│   └── deprecation/
│       ├── graphserviceactor-deprecation-analysis.md
│       ├── graphserviceactor-deprecation-delivery.md
│       ├── graphserviceactor-deprecation-research.md
│       ├── graphserviceactor-deprecation-summary.md
│       ├── graphserviceactor-deprecation-templates.md
│       ├── graphserviceactor-implementation-plan.md
│       ├── graphserviceactor-search-index.md
│       └── deprecation-strategy-index.md
│
├── concepts/
│   ├── architecture/
│   │   ├── 00-architecture-overview.md          ← Renamed (kebab-case)
│   │   ├── cqrs-directive-template.md           ← Renamed
│   │   ├── pipeline-integration.md              ← Renamed
│   │   ├── pipeline-sequence-diagrams.md        ← Renamed
│   │   ├── quick-reference.md                   ← Renamed
│   │   ├── semantic-physics-overview.md         ← Disambiguated
│   │   ├── semantic-physics-architecture.md     ← Disambiguated
│   │   ├── reasoning-test-results.md            ← Renamed (more accurate)
│   │   └── hierarchical-visualization.md        ✓ Keep (unique location)
│   └── ontology-reasoning-concepts.md           ← Disambiguated
│
├── guides/
│   ├── developer/
│   │   ├── 01-development-setup.md              ✓ Merged duplicate into this
│   │   ├── 02-project-structure.md              ✓
│   │   ├── 03-architecture.md                   ✓
│   │   ├── 04-adding-features.md                ✓ Merged duplicate into this
│   │   ├── 05-05-05-testing-guide.md                  ✓ Renamed & merged 3 files
│   │   └── 06-contributing.md                   ✓
│   ├── migration/
│   │   └── neo4j-settings-migration.md          ← Moved from root
│   ├── operations/
│   │   └── pipeline-operator-runbook.md         ← Renamed
│   ├── user/
│   │   └── xr-setup.md                          ✓ User-focused version
│   ├── xr-setup.md                              ✓ Developer version
│   └── neo4j-integration.md                     ✓ Keep (guide focus)
│
├── reference/
│   ├── api/
│   │   ├── 01-authentication.md                 ✓
│   │   ├── 02-rest-api.md                       ★ NEW (fills gap)
│   │   ├── 03-websocket.md                      ✓
│   │   └── rest-api-detailed-spec.md            ← Disambiguated
│   └── semantic-physics-api-reference.md        ← Disambiguated
│
├── implementation/
│   └── stress-majorization-implementation.md    ← Renamed
│
└── multi-agent-docker/
    ├── readme.md                                ✓ Standard exception
    ├── architecture.md                          ← Renamed
    ├── docker-environment.md                    ← Renamed
    ├── goalie-integration.md                    ← Renamed
    ├── port-configuration.md                    ← Renamed
    ├── tools.md                                 ← Renamed
    └── troubleshooting.md                       ← Renamed
```

---

## 🔄 Transformation Flow

### Phase 1: Duplicate Resolution

```
┌─────────────────────────────────────────────────────────────┐
│                    DUPLICATE MERGING                        │
└─────────────────────────────────────────────────────────────┘

guides/developer/
├── development-setup.md (507 lines)  ┐
│                                     ├─→ 01-development-setup.md (631+ lines)
└── 01-development-setup.md (631 lines) ┘

guides/developer/
├── adding-a-feature.md (265 lines)  ┐
│                                    ├─→ 04-adding-features.md (19K+ bytes)
└── 04-adding-features.md (19K)     ┘

guides/developer/
├── 05-05-testing-guide.md (669 lines)     ┐
├── 05-testing.md (3.5K)             ├─→ 05-05-05-testing-guide.md (consolidated)
└── guides/05-05-testing-guide.md (358)    ┘

guides/
├── xr-setup.md (1054 lines)         → KEEP (developer focus)
└── user/xr-setup.md (651 lines)     → KEEP (user focus)
    └─→ Both updated with cross-references
```

### Phase 2: Numbering Fixes

```
┌─────────────────────────────────────────────────────────────┐
│                    SEQUENCE COMPLETION                      │
└─────────────────────────────────────────────────────────────┘

guides/developer/                    reference/api/
├── 01-✓                             ├── 01-authentication.md
├── 02-✓                             ├── 02-[MISSING] ← CREATE
├── 03-✓                             └── 03-websocket.md
├── 04-✓ (conflict resolved)                 ↓
├── 05-✓                             ├── 01-authentication.md
└── 06-✓                             ├── 02-rest-api.md ★ NEW
                                     └── 03-websocket.md
```

### Phase 3: Case Normalization

```
┌─────────────────────────────────────────────────────────────┐
│            SCREAMING-SNAKE-CASE → kebab-case                │
└─────────────────────────────────────────────────────────────┘

Root Level (11 files):
alignment-report.md                  → reports/audits/alignment-report-2025-11-04.md
GRAPHSERVICEACTOR-DEPRECATION-*.md   → reports/deprecation/graphserviceactor-deprecation-*.md
NEO4J-SETTINGS-*.md                  → guides/migration/neo4j-settings-migration.md

Architecture (5 files):
00-ARCHITECTURE-overview.md          → 00-architecture-overview.md
cqrs-directive-template.md           → cqrs-directive-template.md
pipeline-integration.md              → pipeline-integration.md
pipeline-sequence-diagrams.md        → pipeline-sequence-diagrams.md
quick-reference.md                   → quick-reference.md

Multi-Agent-Docker (6 files):
architecture.md                      → architecture.md
DOCKER-environment.md                → docker-environment.md
GOALIE-integration.md                → goalie-integration.md
PORT-configuration.md                → port-configuration.md
tools.md                             → tools.md
troubleshooting.md                   → troubleshooting.md

Other Directories (3 files):
pipeline-operator-runbook.md         → pipeline-operator-runbook.md
stress-majorization-implementation.md → stress-majorization-implementation.md
```

### Phase 4: Disambiguation

```
┌─────────────────────────────────────────────────────────────┐
│              SIMILAR NAMES → DESCRIPTIVE NAMES              │
└─────────────────────────────────────────────────────────────┘

Semantic Physics:
semantic-physics.md                  → semantic-physics-overview.md
semantic-physics-system.md           → semantic-physics-architecture.md
semantic-physics-implementation.md   → semantic-physics-api-reference.md

REST API:
rest-api-reference.md                → 02-rest-api.md (numbered sequence)
rest-api-complete.md                 → rest-api-detailed-spec.md

Other:
reasoning-tests-summary.md           → reasoning-test-results.md
ontology-reasoning.md                → ontology-reasoning-concepts.md
```

---

## 📈 Impact Metrics

### File Count Changes

```
BEFORE                               AFTER
─────────────────────────────────────────────────────────
Root level:         13 files    →    2 files (README, CONTRIBUTING)
reports/:            0 files    →   12 files (new directory)
guides/developer:   10 files    →    6 files (duplicates merged)
reference/api:       4 files    →    5 files (gap filled)
architecture:       15 files    →   15 files (renamed)
multi-agent-docker:  7 files    →    7 files (renamed)

Total Changes: 30+ files affected
Deletions:      7 duplicate files removed
Additions:      1 new directory, 2 new files
Renames:       26 files renamed
Merges:         6 files merged into 3
```

### Naming Convention Distribution

```
BEFORE:
├── kebab-case:           45% (23 files)
├── SCREAMING-SNAKE-CASE: 32% (16 files)
├── Mixed:                15% (8 files)
└── Duplicates:           8% (4 files)

AFTER:
├── kebab-case:           96% (48 files)
└── Standard exceptions:   4% (2 files: README, CONTRIBUTING)
```

---

## 🔗 Reference Update Flow

### How References Get Updated

```
┌──────────────────────────────────────────────────────────────┐
│  1. FILE RENAME/MOVE                                         │
│     guides/developer/05-05-testing-guide.md                        │
│     → guides/developer/05-05-05-testing-guide.md                   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  2. SCRIPT FINDS ALL REFERENCES                              │
│     grep -r "05-05-testing-guide.md" docs/ --include="*.md"        │
│                                                              │
│     Found in:                                                │
│     - development-setup.md (3 references)                    │
│     - 04-adding-features.md (2 references)                   │
│     - readme.md (1 reference)                                │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  3. SCRIPT UPDATES REFERENCES                                │
│     sed -i "s|05-05-testing-guide.md|05-05-05-testing-guide.md|g"        │
│                                                              │
│     Before: [Testing Guide](05-05-testing-guide.md)                │
│     After:  [Testing Guide](05-05-05-testing-guide.md)             │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  4. VALIDATION                                               │
│     ./scripts/validate-links.sh                              │
│                                                              │
│     ✓ All 6 references updated successfully                  │
│     ✓ No broken links found                                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 Success Visualization

### Validation Checkpoints

```
Phase 1 Complete
├── ✓ 7 duplicate files resolved
├── ✓ 0 broken links
├── ✓ Git commit created
└── ✓ Validation passed
    ↓
Phase 2 Complete
├── ✓ Numbering conflicts resolved
├── ✓ Sequences completed
├── ✓ 0 broken links
└── ✓ Validation passed
    ↓
Phase 3 Complete
├── ✓ 26 files renamed to kebab-case
├── ✓ Reports organized in /reports/
├── ✓ 0 SCREAMING-SNAKE-CASE remaining
├── ✓ 0 broken links
└── ✓ Validation passed
    ↓
Phase 4 Complete
├── ✓ 7 files disambiguated
├── ✓ Clear file purposes
├── ✓ 0 broken links
└── ✓ Validation passed
    ↓
Final Validation
├── ✓ All 30+ files processed
├── ✓ 0 broken internal links
├── ✓ 0 orphaned files
├── ✓ All sequences valid
├── ✓ 100% references updated
└── ✓ READY FOR MERGE
```

---

## 📂 New Directory Structure Benefits

### Before: Cluttered Root

```
docs/
├── (13 mixed files at root level)
├── (Inconsistent naming everywhere)
└── (Duplicates scattered across directories)
```

### After: Organized Hierarchy

```
docs/
├── readme.md ────────────────── Entry point
├── getting-started/ ─────────── User onboarding
├── guides/ ──────────────────── How-to guides
│   ├── developer/ (numbered) ── Developer workflow
│   ├── user/ ────────────────── End-user guides
│   ├── operations/ ──────────── Deployment/ops
│   └── migration/ ───────────── Migration guides
├── concepts/ ────────────────── Conceptual docs
│   └── architecture/ ────────── Architecture
├── reference/ ───────────────── Technical reference
│   └── api/ (numbered) ──────── API documentation
├── reports/ ─────────────────── Reports & audits (NEW)
│   ├── audits/ ──────────────── Audit reports
│   └── deprecation/ ─────────── Deprecation tracking
├── implementation/ ──────────── Implementation details
└── scripts/ ─────────────────── Automation scripts (NEW)
```

---

## 🚀 Execution Timeline

```
DAY 1                    DAY 2                    DAY 3
─────────────────────────────────────────────────────────────

9:00  ┌─────────────┐  9:00  ┌─────────────┐  9:00  ┌──────────────┐
      │Pre-flight   │        │Phase 2      │        │Phase 4       │
      │checks       │        │Numbering    │        │Disambiguation│
10:00 ├─────────────┤  10:00 ├─────────────┤  10:00 ├──────────────┤
      │Phase 1.1.1  │        │Phase 3.1    │        │Final ref     │
      │Dev setup    │        │Move reports │        │updates       │
11:00 ├─────────────┤  11:00 ├─────────────┤  11:00 ├──────────────┤
      │Phase 1.1.2  │        │Phase 3.2    │        │Comprehensive │
      │Add features │        │Architecture │        │validation    │
12:00 │   LUNCH     │  12:00 │   LUNCH     │  12:00 │   LUNCH      │
13:00 ├─────────────┤  13:00 ├─────────────┤  13:00 ├──────────────┤
      │Phase 1.1.3  │        │Phase 3.3    │        │Documentation │
      │Testing      │        │Other dirs   │        │updates       │
14:00 ├─────────────┤  14:00 ├─────────────┤  14:00 ├──────────────┤
      │Phase 1.1.4  │        │Reference    │        │Team          │
      │XR setup     │        │updates      │        │notification  │
15:00 ├─────────────┤  15:00 ├─────────────┤  15:00 └──────────────┘
      │Reference    │        │Validation   │        COMPLETE ✓
      │updates      │        │             │
16:00 ├─────────────┤  16:00 └─────────────┘
      │Validation   │        Phase 2 & 3 ✓
17:00 └─────────────┘
      Phase 1 ✓

Total: 6-8 hours over 3 days
```

---

## 🎨 Naming Convention Visual Guide

### ✅ CORRECT: kebab-case

```
✓ semantic-physics-overview.md
✓ pipeline-integration.md
✓ rest-api-detailed-spec.md
✓ 01-development-setup.md
✓ neo4j-settings-migration.md
```

### ❌ INCORRECT: SCREAMING-SNAKE-CASE

```
✗ semantic-physics-overview.md
✗ pipeline-integration.md
✗ rest-api-detailed-spec.md
✗ NEO4j-settings-migration.md
```

### ⚠️ EXCEPTIONS: Standard conventions

```
✓ readme.md          (GitHub/project standard)
✓ contributing.md    (GitHub standard)
✓ license.md         (Legal standard)
✓ changelog.md       (Project standard)
```

---

## 📋 Quick Reference: Phase Actions

```
┌──────────┬─────────────┬──────────────┬──────────┐
│  Phase   │    Files    │     Time     │  Action  │
├──────────┼─────────────┼──────────────┼──────────┤
│  Phase 1 │   7 files   │   2-3 hrs    │  MERGE   │
│  Phase 2 │   2 files   │   30 min     │  CREATE  │
│  Phase 3 │  26 files   │   1-2 hrs    │  RENAME  │
│  Phase 4 │   7 files   │   1 hr       │  RENAME  │
├──────────┼─────────────┼──────────────┼──────────┤
│  TOTAL   │  42+ files  │   6-8 hrs    │  MIXED   │
└──────────┴─────────────┴──────────────┴──────────┘
```

---

**End of Visual Overview**

For detailed execution instructions, see:
- **filename-standardization-execution-plan.md** - Complete technical plan
- **filename-standardization-quick-start.md** - Copy-paste commands
- **filename-standardization-summary.md** - Executive summary
