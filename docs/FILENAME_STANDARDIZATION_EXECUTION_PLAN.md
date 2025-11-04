# Filename Standardization Execution Plan

**Document Version:** 1.0
**Created:** 2025-11-04
**Status:** Ready for Execution
**Total Files Affected:** 30

---

## Executive Summary

This plan standardizes 30 documentation files across 4 phases to eliminate duplicates, fix numbering conflicts, normalize case conventions, and disambiguate similar filenames. The approach prioritizes safety with dependency tracking, link validation, and rollback strategies.

### Success Metrics
- Zero broken internal links after completion
- All filenames follow kebab-case convention
- No duplicate content across files
- Consistent numbering sequences in all directories
- All cross-references updated

---

## Phase 1: Critical Duplicates (7 files)

**Priority:** CRITICAL - Must be completed first
**Estimated Time:** 2-3 hours
**Risk Level:** HIGH (content merge required)

### 1.1 Developer Guide Duplicates (6 files → 3 files)

#### Action 1.1.1: Development Setup Merge
**Current State:**
- `/docs/guides/developer/development-setup.md` (507 lines) - WIP version
- `/docs/guides/developer/01-development-setup.md` (631 lines) - More complete

**Target:** `/docs/guides/developer/01-development-setup.md` (KEEP)

**Steps:**
1. Compare both files line-by-line for unique content
2. Extract any unique sections from `development-setup.md`
3. Merge unique content into `01-development-setup.md`
4. Update last_updated timestamp
5. Delete `development-setup.md`

**References to Update:**
- `/docs/guides/developer/testing-guide.md` (2 references)
- `/docs/guides/developer/development-setup.md` itself (internal links)
- `/docs/DOCUMENTATION_AUDIT_COMPLETION_REPORT.md` (1 reference)

**Validation:**
```bash
grep -r "development-setup.md" /home/devuser/workspace/project/docs --include="*.md"
```

---

#### Action 1.1.2: Adding Features Merge
**Current State:**
- `/docs/guides/developer/adding-a-feature.md` (265 lines)
- `/docs/guides/developer/04-adding-features.md` (19K bytes)

**Target:** `/docs/guides/developer/04-adding-features.md` (KEEP)

**Steps:**
1. Compare content between both files
2. Identify unique sections in `adding-a-feature.md`
3. Merge into `04-adding-features.md`
4. Update metadata and timestamps
5. Delete `adding-a-feature.md`

**References to Update:**
- `/docs/guides/developer/testing-guide.md` (2 references)
- `/docs/guides/developer/development-setup.md` (3 references)
- `/docs/DOCUMENTATION_AUDIT_COMPLETION_REPORT.md` (1 reference)

**Validation:**
```bash
grep -r "adding-a-feature.md" /home/devuser/workspace/project/docs --include="*.md"
```

---

#### Action 1.1.3: Testing Guide Consolidation
**Current State:**
- `/docs/guides/developer/testing-guide.md` (669 lines) - Comprehensive
- `/docs/guides/developer/05-testing.md` (3.5K bytes) - Shorter version
- `/docs/guides/testing-guide.md` (358 lines) - Root guides version

**Target:** `/docs/guides/developer/05-testing-guide.md` (NEW - RENAME & MERGE)

**Steps:**
1. Rename `05-testing.md` to `05-testing-guide.md` for consistency
2. Compare all three testing files
3. Merge comprehensive content from `testing-guide.md` into `05-testing-guide.md`
4. Extract any unique content from `/docs/guides/testing-guide.md`
5. Delete both old `testing-guide.md` files
6. Create redirect note in `/docs/guides/testing-guide.md` location (optional)

**References to Update:**
- `/docs/guides/developer/testing-guide.md` references itself (remove after merge)
- `/docs/guides/developer/development-setup.md` (3 references)
- `/docs/getting-started/02-first-graph-and-agents.md` (potential references)

**Validation:**
```bash
grep -r "testing-guide.md\|05-testing.md" /home/devuser/workspace/project/docs --include="*.md"
```

---

#### Action 1.1.4: XR Setup Duplicate Resolution
**Current State:**
- `/docs/guides/xr-setup.md` (1054 lines) - Comprehensive version
- `/docs/guides/user/xr-setup.md` (651 lines) - User-focused version

**Target:** Keep BOTH with different purposes

**Decision Rationale:**
- `guides/xr-setup.md` - General/developer XR setup (comprehensive)
- `guides/user/xr-setup.md` - End-user simplified XR setup

**Steps:**
1. Analyze content overlap (estimated 40-50%)
2. Differentiate purposes clearly in frontmatter
3. Add cross-references between the two
4. Update `guides/xr-setup.md` title to "XR Setup - Developer Guide"
5. Update `guides/user/xr-setup.md` title to "XR Setup - User Guide"
6. Add mutual cross-links at top of each file

**Frontmatter Updates:**
```yaml
# guides/xr-setup.md
---
title: XR Setup - Developer Guide
audience: developers
related:
  - /docs/guides/user/xr-setup.md
---

# guides/user/xr-setup.md
---
title: XR Setup - User Guide
audience: end-users
related:
  - /docs/guides/xr-setup.md
---
```

**References to Update:**
- `/docs/getting-started/02-first-graph-and-agents.md` (3 references - point to user version)
- `/docs/LINK_VALIDATION_REPORT.md` (1 reference)
- `/docs/DOCUMENTATION_AUDIT_COMPLETION_REPORT.md` (1 reference)

**Validation:**
```bash
grep -r "xr-setup.md" /home/devuser/workspace/project/docs --include="*.md"
```

---

### 1.2 Additional Critical Duplicates (1 identified)

#### Action 1.2.1: Hierarchical Visualization Duplicate
**Current State:**
- `/docs/concepts/hierarchical-visualization.md`
- `/docs/concepts/architecture/hierarchical-visualization.md`

**Analysis Required:**
1. Compare file contents
2. Determine canonical location (likely `concepts/architecture/`)
3. Merge or redirect as appropriate

**Steps:**
1. Read both files and compare
2. If identical: delete `/docs/concepts/hierarchical-visualization.md`, keep architecture version
3. If different: merge unique content into architecture version
4. Search and update all references

**Validation:**
```bash
diff /docs/concepts/hierarchical-visualization.md /docs/concepts/architecture/hierarchical-visualization.md
grep -r "hierarchical-visualization.md" /home/devuser/workspace/project/docs --include="*.md"
```

---

#### Action 1.2.2: Neo4j Integration Duplicate
**Current State:**
- `/docs/concepts/neo4j-integration.md`
- `/docs/guides/neo4j-integration.md`

**Analysis Required:**
1. Determine if one is conceptual overview, other is practical guide
2. Differentiate or merge as appropriate

**Steps:**
1. Compare content focus (concepts vs. implementation)
2. If overlapping: merge into guides version (practical focus)
3. If distinct: update titles to clarify ("Neo4j Integration Concepts" vs. "Neo4j Integration Guide")
4. Update cross-references

**Validation:**
```bash
grep -r "neo4j-integration.md" /home/devuser/workspace/project/docs --include="*.md"
```

---

### Phase 1 Summary
| File Operation | Source | Destination | Action |
|----------------|--------|-------------|--------|
| 1.1.1 | `guides/developer/development-setup.md` | `guides/developer/01-development-setup.md` | MERGE & DELETE |
| 1.1.2 | `guides/developer/adding-a-feature.md` | `guides/developer/04-adding-features.md` | MERGE & DELETE |
| 1.1.3a | `guides/developer/testing-guide.md` | `guides/developer/05-testing-guide.md` | MERGE & DELETE |
| 1.1.3b | `guides/testing-guide.md` | `guides/developer/05-testing-guide.md` | MERGE & DELETE |
| 1.1.3c | `guides/developer/05-testing.md` | `guides/developer/05-testing-guide.md` | RENAME & MERGE |
| 1.1.4 | `guides/xr-setup.md` | (KEEP) | DIFFERENTIATE |
| 1.1.4 | `guides/user/xr-setup.md` | (KEEP) | DIFFERENTIATE |
| 1.2.1 | `concepts/hierarchical-visualization.md` | TBD after analysis | ANALYZE |
| 1.2.2 | `concepts/neo4j-integration.md` | TBD after analysis | ANALYZE |

**Phase 1 Deliverables:**
- 6 files deleted
- 3 files merged and updated
- 2 files differentiated with cross-links
- 2 duplicates analyzed and resolved
- All references updated
- Link validation passed

---

## Phase 2: Numbering Conflicts (2 files)

**Priority:** HIGH
**Estimated Time:** 30 minutes
**Risk Level:** LOW

### 2.1 Developer Guide Numbering Gap

**Current Sequence:**
```
01-development-setup.md       ✓
02-project-structure.md        ✓
03-architecture.md             ✓
04-adding-features.md          ✓
04-testing-status.md           ✗ CONFLICT (duplicate 04)
05-testing-guide.md (to be created) ✓
06-contributing.md             ✓
```

**Issue:** `04-testing-status.md` conflicts with `04-adding-features.md`

**Resolution:**

#### Action 2.1.1: Analyze Testing Status Content
**Steps:**
1. Read `04-testing-status.md` to determine purpose
2. Options:
   - **Option A:** If it's a status report → Move to `/docs/reports/testing-status-[date].md`
   - **Option B:** If it's part of testing guide → Merge into `05-testing-guide.md`
   - **Option C:** If it's ongoing status → Rename to `05-testing-status.md`

**Recommended Action:** Read file first, then decide

**Validation:**
```bash
cat /home/devuser/workspace/project/docs/guides/developer/04-testing-status.md | head -50
```

---

### 2.2 API Reference Numbering Gap

**Current Sequence:**
```
01-authentication.md           ✓
03-websocket.md               ✗ MISSING 02
README.md                      (unnumbered - correct)
neo4j-quick-start.md          (unnumbered - should add?)
rest-api-complete.md          (unnumbered - should add?)
rest-api-reference.md         (unnumbered - should add?)
```

**Issue:** Missing `02-*.md` in sequence

**Resolution:**

#### Action 2.2.1: Determine Missing Content
**Steps:**
1. Review `README.md` to see intended structure
2. Check if REST API content should be `02-rest-api.md`
3. Verify websocket is truly third in logical sequence

**Proposed Sequence:**
```
01-authentication.md           (Auth first - correct)
02-rest-api.md                (REST API second - NEW)
03-websocket.md               (WebSocket third - correct)
04-graphql.md                 (If GraphQL exists - TBD)
README.md                      (Index - unnumbered correct)
neo4j-quick-start.md          (Quick start - unnumbered correct)
rest-api-complete.md          → merge into 02-rest-api.md
rest-api-reference.md         → merge into 02-rest-api.md or keep as appendix
```

**Action Plan:**
1. Create `02-rest-api.md` by merging/consolidating REST content
2. Keep quick-start and complete reference as supplementary (unnumbered)
3. Update README.md with correct sequence

---

### Phase 2 Summary
| File Operation | Source | Destination | Action |
|----------------|--------|-------------|--------|
| 2.1.1 | `guides/developer/04-testing-status.md` | TBD after analysis | ANALYZE & MOVE/MERGE |
| 2.2.1 | `reference/api/rest-api-*.md` (2 files) | `reference/api/02-rest-api.md` | CREATE & CONSOLIDATE |

**Phase 2 Deliverables:**
- Developer guide numbering corrected (01-06 sequence)
- API reference numbering corrected (01-03+ sequence)
- All sequences documented in respective README.md files

---

## Phase 3: Case Normalization (8+ files)

**Priority:** MEDIUM
**Estimated Time:** 1-2 hours
**Risk Level:** MEDIUM (many references)

### 3.1 Root-Level SCREAMING_SNAKE_CASE (11 files)

#### Category A: Deprecation/Report Documents (Should move to /docs/reports/)

| Current Path | Target Path | Reason |
|--------------|-------------|--------|
| `ALIGNMENT_REPORT.md` | `reports/alignment-report-2025-11-04.md` | Status report |
| `DEPRECATION_STRATEGY_INDEX.md` | `reports/deprecation-strategy-index.md` | Planning document |
| `DOCUMENTATION_AUDIT_COMPLETION_REPORT.md` | `reports/documentation-audit-completion-2025-11-04.md` | Completed audit |
| `GRAPHSERVICEACTOR_DEPRECATION_ANALYSIS.md` | `reports/graphserviceactor-deprecation-analysis.md` | Analysis report |
| `GRAPHSERVICEACTOR_DEPRECATION_DELIVERY.md` | `reports/graphserviceactor-deprecation-delivery.md` | Delivery plan |
| `GRAPHSERVICEACTOR_DEPRECATION_RESEARCH.md` | `reports/graphserviceactor-deprecation-research.md` | Research |
| `GRAPHSERVICEACTOR_DEPRECATION_SUMMARY.md` | `reports/graphserviceactor-deprecation-summary.md` | Summary |
| `GRAPHSERVICEACTOR_DEPRECATION_TEMPLATES.md` | `reports/graphserviceactor-deprecation-templates.md` | Templates |
| `GRAPHSERVICEACTOR_IMPLEMENTATION_PLAN.md` | `reports/graphserviceactor-implementation-plan.md` | Plan |
| `GRAPHSERVICEACTOR_SEARCH_INDEX.md` | `reports/graphserviceactor-search-index.md` | Index |
| `LINK_VALIDATION_REPORT.md` | `reports/link-validation-report-2025-11-04.md` | Validation report |

**Action 3.1.1: Create Reports Directory Structure**
```bash
mkdir -p /home/devuser/workspace/project/docs/reports/{deprecation,audits}
```

**Action 3.1.2: Move and Rename Files**
```bash
# Move deprecation reports
mv GRAPHSERVICEACTOR_DEPRECATION_*.md reports/deprecation/
# Rename in reports directory
cd reports/deprecation
for f in *.md; do
  new_name=$(echo "$f" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  mv "$f" "$new_name"
done

# Move audit reports
mv DOCUMENTATION_AUDIT_*.md reports/audits/
mv LINK_VALIDATION_*.md reports/audits/
mv ALIGNMENT_REPORT.md reports/audits/
# Rename
cd reports/audits
for f in *.md; do
  new_name=$(echo "$f" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  mv "$f" "$new_name"
done
```

---

#### Category B: Neo4j Migration Document

| Current Path | Target Path | Reason |
|--------------|-------------|--------|
| `NEO4J_SETTINGS_MIGRATION_DOCUMENTATION_REPORT.md` | `guides/migration/neo4j-settings-migration.md` | Migration guide, not report |

**Action 3.1.3: Move to Migration Guides**
```bash
mv NEO4J_SETTINGS_MIGRATION_DOCUMENTATION_REPORT.md \
   guides/migration/neo4j-settings-migration.md
```

---

### 3.2 Architecture SCREAMING_SNAKE_CASE (6 files)

| Current Path | Target Path | Action |
|--------------|-------------|--------|
| `concepts/architecture/00-ARCHITECTURE-OVERVIEW.md` | `concepts/architecture/00-architecture-overview.md` | RENAME (keep 00 prefix) |
| `concepts/architecture/CQRS_DIRECTIVE_TEMPLATE.md` | `concepts/architecture/cqrs-directive-template.md` | RENAME |
| `concepts/architecture/PIPELINE_INTEGRATION.md` | `concepts/architecture/pipeline-integration.md` | RENAME |
| `concepts/architecture/PIPELINE_SEQUENCE_DIAGRAMS.md` | `concepts/architecture/pipeline-sequence-diagrams.md` | RENAME |
| `concepts/architecture/QUICK_REFERENCE.md` | `concepts/architecture/quick-reference.md` | RENAME |

**Action 3.2.1: Batch Rename Architecture Files**
```bash
cd /home/devuser/workspace/project/docs/concepts/architecture
for f in *[A-Z_]*.md; do
  if [ -f "$f" ]; then
    new_name=$(echo "$f" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    echo "Renaming: $f → $new_name"
    mv "$f" "$new_name"
  fi
done
```

---

### 3.3 Other Directory SCREAMING_SNAKE_CASE (5 files)

| Current Path | Target Path | Action |
|--------------|-------------|--------|
| `guides/operations/PIPELINE_OPERATOR_RUNBOOK.md` | `guides/operations/pipeline-operator-runbook.md` | RENAME |
| `guides/CONTRIBUTING.md` | (KEEP - Standard convention) | NO CHANGE |
| `implementation/STRESS_MAJORIZATION_IMPLEMENTATION.md` | `implementation/stress-majorization-implementation.md` | RENAME |
| `multi-agent-docker/ARCHITECTURE.md` | `multi-agent-docker/architecture.md` | RENAME |
| `multi-agent-docker/DOCKER-ENVIRONMENT.md` | `multi-agent-docker/docker-environment.md` | RENAME |
| `multi-agent-docker/GOALIE-INTEGRATION.md` | `multi-agent-docker/goalie-integration.md` | RENAME |
| `multi-agent-docker/PORT-CONFIGURATION.md` | `multi-agent-docker/port-configuration.md` | RENAME |
| `multi-agent-docker/TOOLS.md` | `multi-agent-docker/tools.md` | RENAME |
| `multi-agent-docker/TROUBLESHOOTING.md` | `multi-agent-docker/troubleshooting.md` | RENAME |

**Note:** Keep `README.md` and `CONTRIBUTING.md` in UPPERCASE (standard convention)

**Action 3.3.1: Batch Rename Mixed Directories**
```bash
# Multi-agent-docker
cd /home/devuser/workspace/project/docs/multi-agent-docker
for f in [A-Z]*.md; do
  if [ "$f" != "README.md" ]; then
    new_name=$(echo "$f" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    echo "Renaming: $f → $new_name"
    mv "$f" "$new_name"
  fi
done

# Operations
cd /home/devuser/workspace/project/docs/guides/operations
mv PIPELINE_OPERATOR_RUNBOOK.md pipeline-operator-runbook.md

# Implementation
cd /home/devuser/workspace/project/docs/implementation
mv STRESS_MAJORIZATION_IMPLEMENTATION.md stress-majorization-implementation.md
```

---

### Phase 3 Reference Update Strategy

**Total Files Requiring Reference Updates:** ~50-70 files (estimated)

**Update Script Approach:**
```bash
#!/bin/bash
# phase3-update-references.sh

# Define old→new mappings
declare -A renames=(
  ["ALIGNMENT_REPORT.md"]="reports/audits/alignment-report-2025-11-04.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_ANALYSIS.md"]="reports/deprecation/graphserviceactor-deprecation-analysis.md"
  # ... (add all 32 renames)
)

# Update references
for old_path in "${!renames[@]}"; do
  new_path="${renames[$old_path]}"
  echo "Updating references: $old_path → $new_path"

  find /home/devuser/workspace/project/docs -type f -name "*.md" -exec \
    sed -i "s|$old_path|$new_path|g" {} +
done
```

---

### Phase 3 Summary

| Category | Files | Action |
|----------|-------|--------|
| Root reports → /reports/ | 11 | MOVE & RENAME |
| Neo4j migration → /guides/migration/ | 1 | MOVE & RENAME |
| Architecture case normalization | 5 | RENAME |
| Multi-agent-docker case normalization | 6 | RENAME |
| Other directories case normalization | 3 | RENAME |
| **Total** | **26** | |

**Phase 3 Deliverables:**
- New `/docs/reports/` directory structure created
- All SCREAMING_SNAKE_CASE files converted to kebab-case
- All internal references updated
- Reference update script created and executed
- Link validation passed

---

## Phase 4: Disambiguation (5+ files)

**Priority:** LOW
**Estimated Time:** 1 hour
**Risk Level:** LOW

### 4.1 Semantic Physics Files (3 files)

**Current State:**
```
concepts/architecture/semantic-physics.md
concepts/architecture/semantic-physics-system.md
reference/semantic-physics-implementation.md
```

**Issue:** Similar names, unclear differentiation

**Resolution:**

| Current Path | Target Path | Purpose |
|--------------|-------------|---------|
| `concepts/architecture/semantic-physics.md` | `concepts/architecture/semantic-physics-overview.md` | High-level concept |
| `concepts/architecture/semantic-physics-system.md` | `concepts/architecture/semantic-physics-architecture.md` | System architecture |
| `reference/semantic-physics-implementation.md` | `reference/semantic-physics-api-reference.md` | API/implementation reference |

**Action 4.1.1: Rename with Descriptive Suffixes**
```bash
cd /home/devuser/workspace/project/docs

mv concepts/architecture/semantic-physics.md \
   concepts/architecture/semantic-physics-overview.md

mv concepts/architecture/semantic-physics-system.md \
   concepts/architecture/semantic-physics-architecture.md

mv reference/semantic-physics-implementation.md \
   reference/semantic-physics-api-reference.md
```

**Update Frontmatter:**
```yaml
# semantic-physics-overview.md
---
title: Semantic Physics - Overview
description: High-level concepts and introduction to semantic physics
category: concepts
---

# semantic-physics-architecture.md
---
title: Semantic Physics - System Architecture
description: Technical architecture and component design
category: architecture
---

# semantic-physics-api-reference.md
---
title: Semantic Physics - API Reference
description: Implementation details and API documentation
category: reference
---
```

---

### 4.2 REST API Files (2 files)

**Current State:**
```
reference/api/rest-api-complete.md
reference/api/rest-api-reference.md
```

**Issue:** Both sound like complete references

**Resolution:**

| Current Path | Target Path | Purpose |
|--------------|-------------|---------|
| `reference/api/rest-api-reference.md` | `reference/api/02-rest-api.md` | Main REST API documentation (numbered sequence) |
| `reference/api/rest-api-complete.md` | `reference/api/rest-api-detailed-spec.md` | Detailed OpenAPI specification |

**Action 4.2.1: Disambiguate REST Documentation**
```bash
cd /home/devuser/workspace/project/docs/reference/api

# Main API doc becomes part of numbered sequence
mv rest-api-reference.md 02-rest-api.md

# Complete spec gets more descriptive name
mv rest-api-complete.md rest-api-detailed-spec.md
```

**Update README.md:**
```markdown
## API Reference Structure

1. [Authentication](01-authentication.md) - Auth mechanisms
2. [REST API](02-rest-api.md) - Core REST endpoints
3. [WebSocket API](03-websocket.md) - Real-time communication
4. [REST API Detailed Spec](rest-api-detailed-spec.md) - Complete OpenAPI specification
5. [Neo4j Quick Start](neo4j-quick-start.md) - Database quick reference
```

---

### 4.3 Additional Disambiguation Candidates

#### 4.3.1 Reasoning Files
**Current:**
```
concepts/architecture/reasoning-data-flow.md
concepts/architecture/reasoning-tests-summary.md
concepts/ontology-reasoning.md
guides/ontology-reasoning-integration.md
```

**Proposed:**
- `reasoning-data-flow.md` → OK (specific)
- `reasoning-tests-summary.md` → `reasoning-test-results.md` (more accurate)
- `ontology-reasoning.md` → `ontology-reasoning-concepts.md` (clarify it's conceptual)
- `ontology-reasoning-integration.md` → OK (already specific)

---

#### 4.3.2 Stress Majorization Files
**Current:**
```
concepts/architecture/stress-majorization.md
guides/stress-majorization-guide.md
implementation/stress-majorization-implementation.md
```

**Analysis:** Already well-differentiated by directory structure. No changes needed.

---

### Phase 4 Summary

| File Operation | Source | Destination | Reason |
|----------------|--------|-------------|--------|
| 4.1.1 | `semantic-physics.md` | `semantic-physics-overview.md` | Clarify scope |
| 4.1.2 | `semantic-physics-system.md` | `semantic-physics-architecture.md` | Clarify content |
| 4.1.3 | `semantic-physics-implementation.md` | `semantic-physics-api-reference.md` | Clarify purpose |
| 4.2.1 | `rest-api-reference.md` | `02-rest-api.md` | Integrate into sequence |
| 4.2.2 | `rest-api-complete.md` | `rest-api-detailed-spec.md` | Distinguish from main |
| 4.3.1 | `reasoning-tests-summary.md` | `reasoning-test-results.md` | More accurate name |
| 4.3.2 | `ontology-reasoning.md` | `ontology-reasoning-concepts.md` | Distinguish from guide |

**Phase 4 Deliverables:**
- 7 files renamed with descriptive suffixes
- Frontmatter updated to reflect new purposes
- README files updated with structure explanations
- All cross-references updated

---

## Cross-Cutting Concerns

### Dependency Order

**Execution must follow this order:**
1. **Phase 1** (duplicates) → Must complete first to avoid conflicts
2. **Phase 2** (numbering) → Depends on Phase 1 deletions
3. **Phase 3** (case) → Can run parallel with Phase 2
4. **Phase 4** (disambiguation) → Should run last

### Git Strategy

**Recommended Approach:**
```bash
# Create feature branch
git checkout -b docs/filename-standardization

# Phase 1 - One commit per action
git add -A && git commit -m "docs: merge development-setup duplicates (Phase 1.1.1)"
git add -A && git commit -m "docs: merge adding-features duplicates (Phase 1.1.2)"
git add -A && git commit -m "docs: consolidate testing guides (Phase 1.1.3)"
# ... etc

# Phase 2 - One commit per phase
git add -A && git commit -m "docs: fix developer guide numbering (Phase 2.1)"

# Phase 3 - One commit for moves, one for renames
git add -A && git commit -m "docs: move reports to /reports/ directory (Phase 3.1)"
git add -A && git commit -m "docs: normalize SCREAMING_SNAKE_CASE to kebab-case (Phase 3.2-3.3)"

# Phase 4 - One commit
git add -A && git commit -m "docs: disambiguate similar filenames (Phase 4)"

# Final validation commit
git add -A && git commit -m "docs: update all cross-references and validate links"
```

---

## Validation Steps

### Post-Phase Validation

**After Each Phase:**
1. Run link checker:
```bash
npx markdown-link-check /home/devuser/workspace/project/docs/**/*.md
```

2. Verify no broken references:
```bash
# Check for common patterns that might be broken
grep -r "\](\.\./" /home/devuser/workspace/project/docs --include="*.md" | \
  while read line; do
    echo "Checking: $line"
    # Extract and verify file exists
  done
```

3. Check for orphaned files:
```bash
# Find files not referenced anywhere
find /home/devuser/workspace/project/docs -name "*.md" | while read file; do
  refs=$(grep -r "$(basename "$file")" /home/devuser/workspace/project/docs --include="*.md" | wc -l)
  if [ $refs -eq 1 ]; then
    echo "Orphaned: $file (only self-reference)"
  fi
done
```

---

### Final Validation Checklist

- [ ] All 30 files processed according to plan
- [ ] Zero broken internal links (run link checker)
- [ ] All SCREAMING_SNAKE_CASE files converted
- [ ] All numbering sequences validated (01, 02, 03, etc.)
- [ ] All cross-references updated
- [ ] Git history is clean and well-documented
- [ ] README files updated to reflect new structure
- [ ] No orphaned files (files with no references)
- [ ] All frontmatter metadata updated
- [ ] Documentation index regenerated (if exists)

---

## Rollback Strategy

### Per-Phase Rollback

**If issues discovered during a phase:**
```bash
# Rollback to before phase started
git log --oneline  # Find commit hash before phase
git reset --hard <commit-hash>

# Alternative: Revert specific commits
git revert <commit-hash>
```

### Full Rollback

**If critical issues require complete rollback:**
```bash
# Return to main branch
git checkout main

# Delete feature branch
git branch -D docs/filename-standardization

# Start over with lessons learned
git checkout -b docs/filename-standardization-v2
```

### Backup Strategy

**Before starting:**
```bash
# Create backup branch
git checkout -b docs/filename-standardization-backup

# Create tarball backup
tar -czf docs-backup-$(date +%Y%m%d).tar.gz \
  /home/devuser/workspace/project/docs
```

---

## Time Estimates

| Phase | Optimistic | Realistic | Pessimistic |
|-------|-----------|-----------|-------------|
| Phase 1 (Duplicates) | 1.5 hrs | 2.5 hrs | 4 hrs |
| Phase 2 (Numbering) | 20 min | 30 min | 1 hr |
| Phase 3 (Case Norm) | 45 min | 1.5 hrs | 3 hrs |
| Phase 4 (Disambig) | 30 min | 1 hr | 2 hrs |
| Validation | 30 min | 1 hr | 2 hrs |
| **Total** | **4 hrs** | **6.5 hrs** | **12 hrs** |

**Recommended Schedule:**
- **Day 1 (3-4 hours):** Phase 1 - Critical duplicates
- **Day 2 (2-3 hours):** Phase 2 & 3 - Numbering and case normalization
- **Day 3 (1-2 hours):** Phase 4 & final validation

---

## Reference Update Automation

### Script: update-all-references.sh

```bash
#!/bin/bash
# Location: /home/devuser/workspace/project/docs/scripts/update-all-references.sh

set -e

DOCS_ROOT="/home/devuser/workspace/project/docs"

# Phase 1 reference updates
declare -A phase1_renames=(
  ["guides/developer/development-setup.md"]="guides/developer/01-development-setup.md"
  ["guides/developer/adding-a-feature.md"]="guides/developer/04-adding-features.md"
  ["guides/developer/testing-guide.md"]="guides/developer/05-testing-guide.md"
  ["guides/testing-guide.md"]="guides/developer/05-testing-guide.md"
  ["guides/developer/05-testing.md"]="guides/developer/05-testing-guide.md"
)

# Phase 3 reference updates (SCREAMING_SNAKE_CASE)
declare -A phase3_renames=(
  ["ALIGNMENT_REPORT.md"]="reports/audits/alignment-report-2025-11-04.md"
  ["DEPRECATION_STRATEGY_INDEX.md"]="reports/deprecation-strategy-index.md"
  ["DOCUMENTATION_AUDIT_COMPLETION_REPORT.md"]="reports/audits/documentation-audit-completion-2025-11-04.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_ANALYSIS.md"]="reports/deprecation/graphserviceactor-deprecation-analysis.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_DELIVERY.md"]="reports/deprecation/graphserviceactor-deprecation-delivery.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_RESEARCH.md"]="reports/deprecation/graphserviceactor-deprecation-research.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_SUMMARY.md"]="reports/deprecation/graphserviceactor-deprecation-summary.md"
  ["GRAPHSERVICEACTOR_DEPRECATION_TEMPLATES.md"]="reports/deprecation/graphserviceactor-deprecation-templates.md"
  ["GRAPHSERVICEACTOR_IMPLEMENTATION_PLAN.md"]="reports/deprecation/graphserviceactor-implementation-plan.md"
  ["GRAPHSERVICEACTOR_SEARCH_INDEX.md"]="reports/deprecation/graphserviceactor-search-index.md"
  ["LINK_VALIDATION_REPORT.md"]="reports/audits/link-validation-report-2025-11-04.md"
  ["NEO4J_SETTINGS_MIGRATION_DOCUMENTATION_REPORT.md"]="guides/migration/neo4j-settings-migration.md"
  ["concepts/architecture/00-ARCHITECTURE-OVERVIEW.md"]="concepts/architecture/00-architecture-overview.md"
  ["concepts/architecture/CQRS_DIRECTIVE_TEMPLATE.md"]="concepts/architecture/cqrs-directive-template.md"
  ["concepts/architecture/PIPELINE_INTEGRATION.md"]="concepts/architecture/pipeline-integration.md"
  ["concepts/architecture/PIPELINE_SEQUENCE_DIAGRAMS.md"]="concepts/architecture/pipeline-sequence-diagrams.md"
  ["concepts/architecture/QUICK_REFERENCE.md"]="concepts/architecture/quick-reference.md"
  ["guides/operations/PIPELINE_OPERATOR_RUNBOOK.md"]="guides/operations/pipeline-operator-runbook.md"
  ["implementation/STRESS_MAJORIZATION_IMPLEMENTATION.md"]="implementation/stress-majorization-implementation.md"
  ["multi-agent-docker/ARCHITECTURE.md"]="multi-agent-docker/architecture.md"
  ["multi-agent-docker/DOCKER-ENVIRONMENT.md"]="multi-agent-docker/docker-environment.md"
  ["multi-agent-docker/GOALIE-INTEGRATION.md"]="multi-agent-docker/goalie-integration.md"
  ["multi-agent-docker/PORT-CONFIGURATION.md"]="multi-agent-docker/port-configuration.md"
  ["multi-agent-docker/TOOLS.md"]="multi-agent-docker/tools.md"
  ["multi-agent-docker/TROUBLESHOOTING.md"]="multi-agent-docker/troubleshooting.md"
)

# Phase 4 reference updates (disambiguation)
declare -A phase4_renames=(
  ["concepts/architecture/semantic-physics.md"]="concepts/architecture/semantic-physics-overview.md"
  ["concepts/architecture/semantic-physics-system.md"]="concepts/architecture/semantic-physics-architecture.md"
  ["reference/semantic-physics-implementation.md"]="reference/semantic-physics-api-reference.md"
  ["reference/api/rest-api-reference.md"]="reference/api/02-rest-api.md"
  ["reference/api/rest-api-complete.md"]="reference/api/rest-api-detailed-spec.md"
  ["concepts/architecture/reasoning-tests-summary.md"]="concepts/architecture/reasoning-test-results.md"
  ["concepts/ontology-reasoning.md"]="concepts/ontology-reasoning-concepts.md"
)

update_references() {
  local phase=$1
  declare -n renames=$2

  echo "=== Updating references for $phase ==="

  for old_path in "${!renames[@]}"; do
    new_path="${renames[$old_path]}"
    echo "  $old_path → $new_path"

    # Update markdown links [text](path)
    find "$DOCS_ROOT" -type f -name "*.md" -exec \
      sed -i "s|]($old_path)|]($new_path)|g" {} +

    # Update markdown links with /docs/ prefix
    find "$DOCS_ROOT" -type f -name "*.md" -exec \
      sed -i "s|](/docs/$old_path)|](/docs/$new_path)|g" {} +

    # Update relative links
    find "$DOCS_ROOT" -type f -name "*.md" -exec \
      sed -i "s|](\.\./$old_path)|](../$new_path)|g" {} +
  done
}

# Main execution
echo "Starting reference updates..."

case "${1:-all}" in
  phase1)
    update_references "Phase 1" phase1_renames
    ;;
  phase3)
    update_references "Phase 3" phase3_renames
    ;;
  phase4)
    update_references "Phase 4" phase4_renames
    ;;
  all)
    update_references "Phase 1" phase1_renames
    update_references "Phase 3" phase3_renames
    update_references "Phase 4" phase4_renames
    ;;
  *)
    echo "Usage: $0 {phase1|phase3|phase4|all}"
    exit 1
    ;;
esac

echo "Reference updates complete!"
```

**Usage:**
```bash
chmod +x /home/devuser/workspace/project/docs/scripts/update-all-references.sh

# Update specific phase
./scripts/update-all-references.sh phase1

# Update all phases
./scripts/update-all-references.sh all
```

---

## Success Criteria

### Quantitative Metrics
- [ ] 30 files processed (7 duplicates, 2 numbering conflicts, 26 case normalizations, 7 disambiguations)
- [ ] 0 broken internal links
- [ ] 100% of references updated
- [ ] 0 SCREAMING_SNAKE_CASE files remaining (except README, CONTRIBUTING)
- [ ] All numbering sequences valid (no gaps, no duplicates)

### Qualitative Metrics
- [ ] File purposes clear from names
- [ ] Directory structure logical and navigable
- [ ] Documentation findable via intuitive naming
- [ ] No user-facing disruption (redirects if needed)
- [ ] Git history tells clear story of changes

---

## Risk Mitigation

### High-Risk Operations

| Operation | Risk | Mitigation |
|-----------|------|------------|
| Merging duplicate content | Loss of unique content | Manual review before merge, keep backups |
| Mass reference updates | Breaking links | Automated script with validation, test on subset first |
| Moving report files | External links breaking | Create redirect pages, document in changelog |
| Renaming numbered sequences | Breaking navigation | Update all README files with new structure |

### Contingency Plans

**If link checker fails:**
1. Revert last commit
2. Review broken links manually
3. Fix issues individually
4. Re-run validation
5. Proceed only when green

**If merge conflicts occur:**
1. Pause operation
2. Document conflict context
3. Resolve manually with senior reviewer
4. Update plan if patterns emerge
5. Continue with adjusted approach

---

## Post-Completion Tasks

### Documentation Updates
- [ ] Update main README.md with new structure
- [ ] Update CONTRIBUTING.md with naming conventions
- [ ] Create style guide for future documentation
- [ ] Document lessons learned
- [ ] Update onboarding materials

### Communication
- [ ] Notify team of changes
- [ ] Update documentation wiki (if exists)
- [ ] Create migration guide for external references
- [ ] Announce in team chat/email
- [ ] Update project website (if applicable)

### Maintenance
- [ ] Add pre-commit hook for filename validation
- [ ] Create documentation linter config
- [ ] Set up automated link checking in CI
- [ ] Schedule quarterly documentation audits
- [ ] Create template for new documentation files

---

## Appendix A: Complete File Inventory

### All 30+ Files Affected

#### Phase 1: Duplicates (7 files)
1. `/docs/guides/developer/development-setup.md` → DELETE
2. `/docs/guides/developer/adding-a-feature.md` → DELETE
3. `/docs/guides/developer/testing-guide.md` → DELETE
4. `/docs/guides/testing-guide.md` → DELETE
5. `/docs/guides/developer/05-testing.md` → RENAME to 05-testing-guide.md
6. `/docs/guides/xr-setup.md` → DIFFERENTIATE
7. `/docs/guides/user/xr-setup.md` → DIFFERENTIATE

#### Phase 2: Numbering (2 files)
8. `/docs/guides/developer/04-testing-status.md` → ANALYZE & RESOLVE
9. `/docs/reference/api/02-[missing].md` → CREATE

#### Phase 3: Case Normalization (26 files)
10-20. Root level reports (11 files) → MOVE & RENAME
21. Neo4j migration doc → MOVE & RENAME
22-26. Architecture docs (5 files) → RENAME
27-32. Multi-agent-docker docs (6 files) → RENAME
33-35. Other directories (3 files) → RENAME

#### Phase 4: Disambiguation (7 files)
36. `semantic-physics.md` → RENAME with suffix
37. `semantic-physics-system.md` → RENAME with suffix
38. `semantic-physics-implementation.md` → RENAME with suffix
39. `rest-api-reference.md` → RENAME to numbered sequence
40. `rest-api-complete.md` → RENAME with suffix
41. `reasoning-tests-summary.md` → RENAME more accurately
42. `ontology-reasoning.md` → RENAME with suffix

---

## Appendix B: Naming Conventions

### Standard Conventions

**File Naming:**
- Use `kebab-case` for all documentation files
- Number sequences: `01-`, `02-`, `03-` (two digits)
- Descriptive suffixes: `-guide`, `-reference`, `-overview`, `-api`
- Date suffixes for reports: `-2025-11-04`

**Directory Structure:**
```
docs/
├── getting-started/          # User onboarding
├── guides/                   # How-to guides
│   ├── developer/            # Developer-specific (numbered)
│   ├── user/                 # End-user guides
│   ├── operations/           # Operations/deployment
│   └── migration/            # Migration guides
├── concepts/                 # Conceptual documentation
│   └── architecture/         # Architecture docs
├── reference/                # API and technical reference
│   └── api/                  # API documentation (numbered)
├── reports/                  # Status reports and audits
│   ├── audits/               # Audit reports
│   └── deprecation/          # Deprecation tracking
└── implementation/           # Implementation details
```

**Exceptions:**
- `README.md` - Always uppercase (convention)
- `CONTRIBUTING.md` - Always uppercase (GitHub convention)
- `LICENSE.md` - Always uppercase (legal convention)

---

## Appendix C: Link Validation Script

```bash
#!/bin/bash
# validate-links.sh

DOCS_ROOT="/home/devuser/workspace/project/docs"
FAILED=0

echo "Validating all internal documentation links..."

find "$DOCS_ROOT" -name "*.md" -type f | while read -r file; do
  echo "Checking: $file"

  # Extract markdown links
  grep -oP '\]\(\K[^)]+' "$file" | grep -v '^http' | while read -r link; do
    # Resolve relative path
    dir=$(dirname "$file")
    target="$dir/$link"

    # Normalize path
    target=$(realpath -m "$target" 2>/dev/null || echo "$target")

    # Check if file exists
    if [ ! -f "$target" ] && [ ! -d "$target" ]; then
      echo "  ✗ BROKEN: $link"
      FAILED=$((FAILED + 1))
    fi
  done
done

if [ $FAILED -eq 0 ]; then
  echo "✓ All links valid!"
  exit 0
else
  echo "✗ $FAILED broken links found"
  exit 1
fi
```

---

## Appendix D: Quick Reference Commands

```bash
# Find all duplicates by filename
find docs -type f -name "*.md" -exec basename {} \; | sort | uniq -d

# Find SCREAMING_SNAKE_CASE files
find docs -type f -name "*[A-Z_][A-Z_]*.md" | grep -v "README\|CONTRIBUTING"

# Count files by directory
find docs -type f -name "*.md" | xargs dirname | sort | uniq -c

# Find orphaned files (no references)
for file in docs/**/*.md; do
  refs=$(grep -r "$(basename "$file")" docs --include="*.md" | wc -l)
  [ $refs -eq 1 ] && echo "Orphaned: $file"
done

# Validate all links
npx markdown-link-check docs/**/*.md

# Check for broken relative links
find docs -name "*.md" -exec grep -H '\](\.\./' {} \; | \
  awk -F: '{print $1}' | sort -u
```

---

**END OF EXECUTION PLAN**

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-04 | System Architect | Initial comprehensive plan |

**Approval Status:** READY FOR EXECUTION
**Next Review:** After Phase 1 completion

**Questions or Issues:** Document in `/docs/reports/filename-standardization-issues.md`
