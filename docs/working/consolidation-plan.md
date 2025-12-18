# Documentation Consolidation Plan

**Analysis Date:** 2025-12-18
**Total Files:** 288 markdown files
**Duplicate Groups:** 14
**Consolidation Opportunities:** 47 high-value consolidations

## Executive Summary

The `/docs` directory contains significant redundancy and organizational challenges:

- **14 duplicate filename groups** with 2-15 files each
- **20 content similarity groups** across major topics
- **Inconsistent directory structure** mixing concepts/explanations, scattering guides
- **Large working/ directory** with 18 files that should be archived or integrated
- **Archive confusion** with some current content misfiled as archived

### Impact Metrics

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Duplicate/redundant docs | ~65 files | ~20 files | 69% reduction |
| README files | 15 | 10 focused indices | Better navigation |
| Architecture docs | 12+ scattered | Organized hierarchy | Clear structure |
| Working docs | 18 active | 2-3 temporary | 83% reduction |
| Directory depth | Inconsistent | 3 levels max | Easier browsing |

## Priority-Based Consolidation

### Phase 1: Quick Wins (High Impact, Low Effort)

**Timeline:** 1-2 days
**Impact:** Immediate 30% reduction in redundancy

#### 1.1 Remove Duplicate Directory Structure

**Action:** Delete `concepts/` directory entirely

- `concepts/architecture/core/client.md` → Already in `explanations/architecture/core/client.md`
- `concepts/architecture/core/server.md` → Already in `explanations/architecture/core/server.md`

**Files affected:** 4 duplicates
**Effort:** 30 minutes
**Commands:**
```bash
# Verify content match
diff concepts/architecture/core/client.md explanations/architecture/core/client.md
diff concepts/architecture/core/server.md explanations/architecture/core/server.md

# Remove if identical
rm -rf docs/concepts/
```

#### 1.2 Consolidate README Files

**Current state:** 15 README.md/readme.md files
**Target:** 10 focused directory indices

**Actions:**

| File | Action | Rationale |
|------|--------|-----------|
| `README.md` | Keep | Main entry point |
| `guides/readme.md` | Rename to index | Directory index |
| `guides/infrastructure/readme.md` | Merge up | Redundant |
| `guides/developer/readme.md` | Merge up | Redundant |
| `explanations/architecture/gpu/readme.md` | Merge up | Redundant |
| `reference/api/readme.md` | Merge up | Redundant |
| `archive/*/README.md` | Keep minimal | Archive navigation |

**Effort:** 2 hours
**Script:**
```bash
# Standardize to README.md (uppercase)
for dir in guides guides/infrastructure guides/developer; do
  if [ -f "$dir/readme.md" ]; then
    mv "$dir/readme.md" "$dir/README.md"
  fi
done
```

#### 1.3 Archive Completed Working Documents

**Action:** Move completed analyses from `working/` to appropriate locations

| Document | Current Location | Target Location | Reason |
|----------|------------------|-----------------|--------|
| `CLIENT_ARCHITECTURE_ANALYSIS.md` | `working/` | `archive/analysis/` | Analysis complete, findings incorporated |
| Link validation reports | `working/` | `archive/reports/` | Historical data |
| Completed tasks | `working/` | `archive/implementation-logs/` | Implementation complete |

**Files affected:** 10-12 working documents
**Effort:** 1 hour

#### 1.4 Remove Archive Data Duplicates

**Action:** Delete `archive/data/pages/` directory (exact duplicate of `archive/data/markdown/`)

```bash
# Verify exact duplicates
for file in archive/data/pages/*.md; do
  basename=$(basename "$file")
  diff "$file" "archive/data/markdown/$basename" || echo "DIFF: $basename"
done

# Remove if all identical
rm -rf archive/data/pages/
```

**Files affected:** 3 duplicate files
**Effort:** 15 minutes

---

### Phase 2: Medium Priority (Medium Impact, Medium Effort)

**Timeline:** 3-5 days
**Impact:** 40% reduction in redundancy, improved navigation

#### 2.1 Consolidate API Reference Documentation

**Current state:** 6 API reference documents with overlapping content

**Target structure:**
```
reference/api/
├── rest-api-complete.md          # PRIMARY: Consolidated REST API reference
├── graphql-api.md                # GraphQL specific (if exists)
└── websocket-api.md              # WebSocket API endpoints

explanations/architecture/
└── api-handlers-reference.md     # KEEP: Implementation details

diagrams/server/api/
└── rest-api-architecture.md      # KEEP: Visual architecture
```

**Consolidation actions:**

1. **Merge into `reference/api/rest-api-complete.md`:**
   - `reference/API_REFERENCE.md` (17KB)
   - `reference/api-complete-reference.md` (25KB)
   - `reference/api/rest-api-reference.md` (13KB)

2. **Keep separate (different purposes):**
   - `explanations/architecture/api-handlers-reference.md` - Implementation guide
   - `diagrams/server/api/rest-api-architecture.md` - Visual documentation

3. **Update cross-references**

**Effort:** 4 hours
**Benefits:**
- Single source of truth for REST API reference
- Clear separation: Reference vs Implementation vs Diagrams
- Easier to maintain and update

#### 2.2 Organize Guides Directory

**Current state:** 31 guides in flat structure
**Target:** Organized by topic in subdirectories

**New structure:**
```
guides/
├── README.md                        # Navigation index
├── getting-started/
│   ├── installation.md
│   ├── first-steps.md
│   └── development-workflow.md
├── features/
│   ├── semantic-forces.md
│   ├── physics-simulation.md
│   └── xr-immersive.md
├── ontology/
│   ├── neo4j-setup.md              # Moved from guides/neo4j-integration.md
│   ├── semantic-forces.md          # Ontology-specific guide
│   └── reasoning-integration.md
├── infrastructure/
│   ├── docker-environment.md
│   ├── deployment.md
│   └── troubleshooting.md          # Consolidated
├── developer/
│   ├── websocket-best-practices.md
│   ├── testing-guide.md            # Consolidated
│   └── telemetry-logging.md
└── ai-models/
    ├── comfyui-integration.md
    ├── blender-mcp.md
    └── model-deployment.md
```

**Move operations:**

```bash
# Create directories
mkdir -p guides/{getting-started,features,ontology,developer,infrastructure,ai-models}

# Move files (examples)
mv guides/development-workflow.md guides/getting-started/
mv guides/neo4j-integration.md guides/ontology/neo4j-setup.md
mv guides/ontology-*.md guides/ontology/

# Update README.md with navigation
```

**Effort:** 6 hours
**Benefits:**
- Logical grouping by topic
- Easier to find relevant guides
- Scalable structure for new guides

#### 2.3 Consolidate WebSocket Protocol Documentation

**Current state:** 7 WebSocket-related documents

**Target structure:**
```
reference/protocols/
└── binary-websocket.md             # PRIMARY: Protocol specification
                                     # Merge: reference/websocket-protocol.md
                                     # Merge: reference/api/03-websocket.md

explanations/architecture/components/
└── websocket-protocol.md           # KEEP: Architecture explanation

diagrams/infrastructure/websocket/
└── binary-protocol-complete.md     # KEEP: Diagrams

guides/developer/
└── websocket-best-practices.md     # KEEP: Developer guide

guides/migration/
└── json-to-binary-protocol.md      # KEEP: Migration guide
```

**Actions:**
1. Merge 3 reference docs into single `reference/protocols/binary-websocket.md`
2. Keep distinct: specification, architecture, guide, migration
3. Add cross-references between documents

**Effort:** 3 hours

#### 2.4 Merge Testing Documentation

**Current state:** 6 testing documents (2 archived)

**Target structure:**
```
guides/developer/
└── testing-guide.md                # PRIMARY: Consolidated testing guide
                                     # Merge: guides/developer/test-execution.md

diagrams/infrastructure/testing/
└── test-architecture.md            # KEEP: Architecture diagrams

explanations/architecture/
└── reasoning-tests-summary.md      # KEEP: Specialized reasoning tests
```

**Actions:**
1. Merge `guides/developer/test-execution.md` into `guides/testing-guide.md`
2. Organize as:
   - Getting Started with Testing
   - Unit Testing
   - Integration Testing
   - Test Execution and CI
   - Architecture (link to diagrams)
3. Archive old versions from `archive/`

**Effort:** 2 hours

#### 2.5 Consolidate Troubleshooting Documentation

**Current state:** 2 troubleshooting documents

**Action:** Merge `guides/infrastructure/troubleshooting.md` into `guides/troubleshooting.md`

**New structure:**
```markdown
# Troubleshooting Guide

## General Issues
- Common errors
- Environment problems

## Infrastructure Issues
(Content from guides/infrastructure/troubleshooting.md)
- Docker issues
- Deployment problems
- Service failures

## Development Issues
- Build errors
- Test failures

## XR and Immersive Issues
- VR setup problems
- Performance issues
```

**Effort:** 1 hour

---

### Phase 3: Major Reorganization (High Impact, High Effort)

**Timeline:** 1-2 weeks
**Impact:** Clear, maintainable documentation structure

#### 3.1 Consolidate Architecture Documentation

**Current state:** 12+ architecture documents scattered across directories

**Target structure:**
```
ARCHITECTURE_OVERVIEW.md            # PRIMARY: High-level overview with links
                                     # Merge: ARCHITECTURE_COMPLETE.md

explanations/architecture/
├── README.md                        # Architecture navigation index
├── overview.md                      # Detailed overview
├── hexagonal-cqrs.md               # Pattern explanation
├── adapter-patterns.md             # Adapter details
├── services-architecture.md        # Services layer
├── integration-patterns.md         # Integration layer
├── core/
│   ├── client.md                   # Client architecture
│   ├── server.md                   # Server architecture
│   └── database.md                 # Data architecture
├── components/
│   ├── websocket-protocol.md       # Component architectures
│   └── ...
└── subsystems/
    ├── blender-mcp.md              # Moved from architecture/
    ├── ontology-storage.md
    └── analytics.md

diagrams/architecture/
└── (mirrors explanations/architecture/ structure)

architecture/                        # ADRs and major decisions only
├── decisions/
│   ├── ADR-001-hexagonal.md
│   └── ...
└── phase-reports/
    └── phase1-completion.md
```

**Major consolidation:**

1. **Merge into ARCHITECTURE_OVERVIEW.md:**
   - `ARCHITECTURE_COMPLETE.md`

2. **Reorganize explanations/architecture/:**
   - Create clear hierarchy: core → components → subsystems
   - Move subsystem docs from `architecture/` to `explanations/architecture/subsystems/`

3. **Archive working analyses:**
   - `working/CLIENT_ARCHITECTURE_ANALYSIS.md` → `archive/analysis/`
   - After incorporating findings into active docs

4. **Move specialized to active:**
   - `archive/specialized/client-typescript-architecture.md` → `explanations/architecture/core/client-typescript.md`
   - `archive/specialized/client-components-reference.md` → `reference/client/components.md`

**Effort:** 12 hours
**Benefits:**
- Single entry point for architecture
- Clear hierarchy: overview → layer → component
- Separation of explanation vs decisions

#### 3.2 Organize Ontology and Semantic Documentation

**Current state:** 47 ontology-related documents across multiple directories

**Target structure:**
```
guides/ontology/
├── README.md                        # Ontology guides index
├── neo4j-setup.md                  # Moved from guides/neo4j-integration.md
├── semantic-forces-guide.md        # User guide
├── reasoning-integration.md        # Integration guide
└── visualization.md                # Visualization guide

explanations/ontology/
├── README.md                        # Ontology concepts index
├── ontology-overview.md            # High-level concepts
├── neo4j-integration.md            # Architecture (keep separate from guide)
├── ontology-pipeline-integration.md
├── hierarchical-visualization.md   # Merged from explanations/architecture/
├── semantic-forces.md              # Concepts (separate from guide)
└── graph-algorithms.md

reference/ontology/
├── schema.md                        # Ontology schema reference
├── api.md                          # Ontology API reference
└── data-model.md                   # Data model specification

audits/                              # Migration reports
├── neo4j-settings-migration-audit.md
├── neo4j-migration-action-plan.md
└── neo4j-migration-summary.md
```

**Actions:**

1. **Create new directories:**
   ```bash
   mkdir -p guides/ontology reference/ontology
   ```

2. **Move guides:**
   - `guides/neo4j-integration.md` → `guides/ontology/neo4j-setup.md`
   - `guides/ontology-*.md` → `guides/ontology/`

3. **Consolidate visualizations:**
   - Merge `explanations/architecture/hierarchical-visualization.md` into `explanations/ontology/hierarchical-visualization.md`

4. **Create reference docs:**
   - Extract schema/API content into `reference/ontology/`

5. **Keep audits separate:**
   - Migration reports stay in `audits/` as historical records

**Effort:** 8 hours
**Benefits:**
- Clear separation: guides vs explanations vs reference
- All ontology content discoverable from 3 entry points
- Audit trail preserved

#### 3.3 Organize Client-Side Documentation

**Current state:** 8 client architecture documents

**Target structure:**
```
explanations/architecture/core/
├── client.md                        # Core client architecture overview
└── client-typescript.md            # TypeScript architecture specifics
                                     # Content from archive/specialized/

diagrams/client/
├── README.md                        # Client diagrams index
├── state/
│   └── state-management-complete.md
├── rendering/
│   └── threejs-pipeline-complete.md
└── xr/
    └── xr-architecture-complete.md

reference/client/
├── README.md                        # Client reference index
├── components.md                   # Component API reference
│                                    # From archive/specialized/client-components-reference.md
└── state-api.md                    # State management API

guides/client/
├── state-management.md
├── three-js-rendering.md
└── xr-integration.md
```

**Actions:**

1. **Remove duplicate:**
   - Delete `concepts/architecture/core/client.md`

2. **Incorporate archived content:**
   - `archive/specialized/client-typescript-architecture.md` → `explanations/architecture/core/client-typescript.md`
   - `archive/specialized/client-components-reference.md` → `reference/client/components.md`

3. **Archive working analysis:**
   - `working/CLIENT_ARCHITECTURE_ANALYSIS.md` → `archive/analysis/client-architecture-2025-12.md`
   - After incorporating findings

4. **Create reference directory:**
   ```bash
   mkdir -p reference/client
   ```

**Effort:** 6 hours

#### 3.4 Clean Archive Directory

**Goal:** Ensure `archive/` contains only truly outdated content

**Review categories:**

1. **archive/specialized/** - Move active content to main docs
2. **archive/fixes/** - Keep only historical quick-references
3. **archive/docs/guides/** - Delete if content in current guides
4. **archive/working/** - Clean out temporary files
5. **archive/data/** - Remove pages/ duplicate

**Decision criteria:**

| Archive if... | Keep active if... |
|---------------|-------------------|
| Content superseded by newer doc | Still current and relevant |
| Migration/audit completed | Ongoing or future reference |
| Temporary analysis finished | Analysis still being used |
| Old version with current version exists | Only version available |

**Effort:** 4 hours

---

## Detailed File-by-File Actions

### Duplicate Filename Resolution

#### README.md Files (15 total → 10 target)

| File | Action | Notes |
|------|--------|-------|
| `README.md` | **Keep** | Main project README |
| `audits/README.md` | **Keep** | Audits index |
| `archive/README.md` | **Keep** | Archive index |
| `diagrams/README.md` | **Keep** | Diagrams index |
| `explanations/architecture/README.md` | **Keep** | Architecture index |
| `guides/readme.md` | **Standardize to README.md** | Guides index |
| `guides/ai-models/README.md` | **Keep** | AI models index |
| `guides/infrastructure/readme.md` | **Merge up to guides/README.md** | Redundant |
| `guides/developer/readme.md` | **Merge up to guides/README.md** | Redundant |
| `explanations/architecture/gpu/readme.md` | **Merge up** | Redundant |
| `reference/api/readme.md` | **Merge up to reference/README.md** | Redundant |
| `archive/fixes/README.md` | **Keep minimal** | Quick nav only |
| `archive/deprecated-patterns/README.md` | **Keep minimal** | Quick nav only |
| `archive/working/README.md` | **Keep minimal** | Quick nav only |
| `archive/reports/README.md` | **Keep minimal** | Quick nav only |

#### WebSocket Protocol (2 files)

| File | Action |
|------|--------|
| `reference/websocket-protocol.md` | Merge into `reference/protocols/binary-websocket.md` |
| `explanations/architecture/components/websocket-protocol.md` | **Keep** (different purpose: architecture) |

#### Neo4j Integration (2 files)

| File | Action |
|------|--------|
| `guides/neo4j-integration.md` | Rename to `guides/ontology/neo4j-setup.md` |
| `explanations/ontology/neo4j-integration.md` | **Keep** (architecture explanation) |

#### Client/Server Core (4 files)

| File | Action |
|------|--------|
| `concepts/architecture/core/client.md` | **Delete** (duplicate) |
| `explanations/architecture/core/client.md` | **Keep** (primary) |
| `concepts/architecture/core/server.md` | **Delete** (duplicate) |
| `explanations/architecture/core/server.md` | **Keep** (primary) |

#### Troubleshooting (2 files)

| File | Action |
|------|--------|
| `guides/troubleshooting.md` | **Keep** (merge infrastructure into this) |
| `guides/infrastructure/troubleshooting.md` | Merge into `guides/troubleshooting.md` |

#### Semantic Forces (2 files)

| File | Action |
|------|--------|
| `guides/features/semantic-forces.md` | **Keep** (user guide) |
| `explanations/physics/semantic-forces.md` | **Keep** (technical explanation) |
| | Add cross-references |

#### Hierarchical Visualization (2 files)

| File | Action |
|------|--------|
| `explanations/ontology/hierarchical-visualization.md` | **Keep** (primary) |
| `explanations/architecture/hierarchical-visualization.md` | Merge into ontology version |

#### Quick Reference (2 files)

| File | Action |
|------|--------|
| `explanations/architecture/quick-reference.md` | **Keep** (current) |
| `archive/fixes/quick-reference.md` | **Delete** (outdated) |

#### XR Setup (2 archived files)

| File | Action |
|------|--------|
| `archive/docs/guides/xr-setup.md` | Verify content in current guide, then delete |
| `archive/docs/guides/user/xr-setup.md` | Verify content in current guide, then delete |
| Current: `guides/vircadia-xr-complete-guide.md` | Ensure all content from archived versions |

#### Archive Data Duplicates (6 files)

| File | Action |
|------|--------|
| `archive/data/markdown/*.md` | **Keep** |
| `archive/data/pages/*.md` | **Delete** (exact duplicates) |

---

## Implementation Strategy

### Phase 1 Execution (Days 1-2)

**Day 1 Morning: Setup and Quick Deletions**
```bash
# 1. Backup
git checkout -b docs-consolidation-phase1
tar -czf docs-backup-$(date +%Y%m%d).tar.gz docs/

# 2. Remove concepts/ duplicate
rm -rf docs/concepts/

# 3. Remove archive/data/pages/ duplicate
rm -rf docs/archive/data/pages/

# 4. Commit
git add -A
git commit -m "docs: remove duplicate directories (concepts, archive/data/pages)"
```

**Day 1 Afternoon: README Consolidation**
```bash
# Standardize case
mv docs/guides/readme.md docs/guides/README.md
# ... (other case changes)

# Merge redundant READMEs
# (Manual merge of content)

git add -A
git commit -m "docs: consolidate and standardize README files"
```

**Day 2: Archive Working Documents**
```bash
# Move completed analyses
mv docs/working/CLIENT_ARCHITECTURE_ANALYSIS.md docs/archive/analysis/
# ... (other moves)

git add -A
git commit -m "docs: archive completed working documents"
```

### Phase 2 Execution (Days 3-7)

**Day 3-4: API Reference Consolidation**
- Manual merge of API reference files
- Update cross-references
- Test all links

**Day 5: Guides Directory Organization**
- Create subdirectories
- Move files
- Update README navigation

**Day 6-7: WebSocket and Testing Docs**
- Consolidate WebSocket protocol docs
- Merge testing documentation
- Update cross-references

### Phase 3 Execution (Days 8-15)

**Week 2: Architecture Reorganization**
- Consolidate architecture docs
- Organize ontology documentation
- Client-side doc organization
- Archive cleanup

---

## Cross-Reference Update Checklist

After each consolidation:

- [ ] Update all internal links pointing to moved files
- [ ] Update navigation indices (README files)
- [ ] Add breadcrumb navigation where helpful
- [ ] Ensure bidirectional links (reference ↔ explanation ↔ guide)
- [ ] Update search keywords/frontmatter if used
- [ ] Test documentation build (if automated)
- [ ] Update ARCHITECTURE_OVERVIEW.md master index

---

## File Patterns to Establish

### 1. Documentation Types

**Reference Documentation** (`reference/`)
- API specifications
- Protocol specifications
- Schema definitions
- Component APIs
- **Focus:** What it is, how to call it

**Explanation Documentation** (`explanations/`)
- Architecture concepts
- Design patterns
- System components
- Technical concepts
- **Focus:** Why it works this way, how it fits together

**Guide Documentation** (`guides/`)
- How-to guides
- Tutorials
- Setup instructions
- Best practices
- **Focus:** How to accomplish tasks

**Visual Documentation** (`diagrams/`)
- Architecture diagrams
- Data flow diagrams
- Sequence diagrams
- **Focus:** Visual representation

### 2. File Naming Conventions

- Use kebab-case: `rest-api-reference.md`
- Be descriptive: `neo4j-setup.md` not `neo4j.md`
- Include type in name where helpful: `*-reference.md`, `*-guide.md`, `*-architecture.md`

### 3. Directory Structure Rules

- Maximum 3 levels deep (4 for diagrams)
- Each directory has README.md navigation index
- Group by topic, not by file type
- Use consistent naming across parallel structures

### 4. README Index Pattern

Every directory README.md should have:
```markdown
# [Directory Name]

Brief description of directory contents.

## Contents

### [Category 1]
- [Document 1](./document-1.md) - Brief description
- [Document 2](./document-2.md) - Brief description

### [Category 2]
...

## Related Documentation
- [Related topic](../other/related.md)
```

---

## Validation and Testing

### Post-Consolidation Checks

1. **Link Validation**
   ```bash
   # Find all broken links
   find docs -name "*.md" -exec grep -l "\.md)" {} \; | \
     xargs -I {} sh -c 'echo "Checking: {}"; grep -o "\[.*\](.*\.md)" {}'
   ```

2. **Duplicate Content Detection**
   ```bash
   # Find similar files
   fdupes -r docs/
   ```

3. **Structure Validation**
   ```bash
   # Verify no files in wrong locations
   find docs/reference -name "*-guide.md"  # Should be empty
   find docs/guides -name "*-reference.md" # Should be empty
   ```

4. **Navigation Completeness**
   - Every directory has README.md
   - Every major doc is linked from a README
   - Main README.md links to all top-level docs

---

## Success Metrics

### Quantitative

- [ ] Reduce total markdown files from 288 to ~220 (23% reduction)
- [ ] Reduce README files from 15 to 10
- [ ] Reduce architecture docs from 12 to 6 + organized hierarchy
- [ ] Reduce working/ docs from 18 to <5
- [ ] Eliminate all exact duplicates (concepts/, archive/data/pages/)

### Qualitative

- [ ] Every doc has clear purpose and location
- [ ] No confusion between reference/explanation/guide
- [ ] Easy to find documentation on any topic
- [ ] Clear navigation from main README
- [ ] Consistent structure across topics
- [ ] Archive contains only historical content

---

## Maintenance Guidelines

### Preventing Future Redundancy

1. **Before creating new doc:**
   - Check if topic already documented
   - Determine correct type (reference/explanation/guide)
   - Place in appropriate directory
   - Update directory README

2. **When updating docs:**
   - Update all related docs (ref + explanation + guide)
   - Update cross-references
   - Archive old versions if creating new

3. **Regular reviews:**
   - Quarterly review of working/ directory
   - Annual review of archive/ for outdated content
   - Check for new duplicates

4. **Documentation decision matrix:**

   | If creating... | Location | Example |
   |----------------|----------|---------|
   | API specification | `reference/api/` | REST endpoint details |
   | Protocol spec | `reference/protocols/` | WebSocket binary protocol |
   | Architecture explanation | `explanations/architecture/` | Hexagonal CQRS pattern |
   | Concept explanation | `explanations/[topic]/` | Semantic forces theory |
   | Setup guide | `guides/getting-started/` | Installation steps |
   | Feature guide | `guides/features/` | How to use XR features |
   | Developer guide | `guides/developer/` | Testing best practices |
   | Architecture diagram | `diagrams/architecture/` | System overview diagram |

---

## Risk Mitigation

### Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Broken links after moves | High | Automated link checking before commit |
| Lost content during merge | High | Diff review, backup before consolidation |
| Confusion during transition | Medium | Phase-by-phase approach, clear commits |
| Incorrect categorization | Medium | Peer review of structure changes |
| Regression to old patterns | Low | Documentation guidelines, template files |

### Rollback Plan

Each phase committed separately:
```bash
# If Phase 2 has issues, rollback just that phase
git log --oneline --grep="phase2"
git revert <commit-hash>
```

Full backup created before Phase 1:
```bash
tar -xzf docs-backup-YYYYMMDD.tar.gz
```

---

## Appendix: Complete File Inventory

### Files to Delete (37 total)

**Exact Duplicates (4):**
- `docs/concepts/architecture/core/client.md`
- `docs/concepts/architecture/core/server.md`
- `docs/archive/data/pages/*.md` (3 files)

**Merge into Other Docs (15):**
- `docs/ARCHITECTURE_COMPLETE.md` → merge to ARCHITECTURE_OVERVIEW.md
- `docs/reference/API_REFERENCE.md` → merge to reference/api/rest-api-complete.md
- `docs/reference/api-complete-reference.md` → merge to reference/api/rest-api-complete.md
- `docs/reference/api/rest-api-reference.md` → merge to reference/api/rest-api-complete.md
- `docs/reference/websocket-protocol.md` → merge to reference/protocols/binary-websocket.md
- `docs/reference/api/03-websocket.md` → merge to reference/protocols/binary-websocket.md
- `docs/guides/infrastructure/troubleshooting.md` → merge to guides/troubleshooting.md
- `docs/guides/developer/test-execution.md` → merge to guides/testing-guide.md
- `docs/explanations/architecture/hierarchical-visualization.md` → merge to explanations/ontology/hierarchical-visualization.md
- `docs/guides/infrastructure/readme.md` → merge up
- `docs/guides/developer/readme.md` → merge up
- `docs/explanations/architecture/gpu/readme.md` → merge up
- `docs/reference/api/readme.md` → merge up

**Archive After Content Verification (8):**
- `docs/archive/fixes/quick-reference.md`
- `docs/archive/docs/guides/xr-setup.md`
- `docs/archive/docs/guides/user/xr-setup.md`
- `docs/archive/tests/test_README.md`
- `docs/archive/docs/guides/developer/05-testing-guide.md`
- `docs/working/CLIENT_ARCHITECTURE_ANALYSIS.md` (move to archive/analysis/)
- Various completed working/ documents

**Archive Directory Cleanup (10+):**
- Review and remove outdated files after content migration

### Files to Move (45+ files)

See Phase 2 and Phase 3 sections for detailed move operations.

### Files to Create (15+ new files)

**Directory Indices:**
- `guides/getting-started/README.md`
- `guides/features/README.md`
- `guides/ontology/README.md`
- `reference/client/README.md`
- `reference/server/README.md`
- `reference/ontology/README.md`

**Reorganized Content:**
- `explanations/architecture/core/client-typescript.md` (from archive)
- `reference/client/components.md` (from archive)
- Various consolidated documents

---

## Timeline Summary

| Phase | Duration | Effort | Files Affected | Key Outcomes |
|-------|----------|--------|----------------|--------------|
| **Phase 1** | 1-2 days | 4 hours | 20-25 files | Remove duplicates, standardize READMEs |
| **Phase 2** | 3-5 days | 16 hours | 35-40 files | Consolidate references, organize guides |
| **Phase 3** | 1-2 weeks | 30 hours | 60+ files | Major reorganization, clear structure |
| **Total** | 2-3 weeks | 50 hours | 115+ files | Clean, maintainable documentation |

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Create backup** of current docs/
3. **Execute Phase 1** (quick wins)
4. **Validate Phase 1** results
5. **Proceed to Phase 2** if Phase 1 successful
6. **Continuous validation** throughout process
7. **Document lessons learned** for future maintenance

---

**Plan Status:** DRAFT
**Last Updated:** 2025-12-18
**Next Review:** After Phase 1 completion
