# VisionFlow Documentation Refactoring - Execution Plan

**Chief Documentation Architect**: Documentation Coordination Swarm
**Last Updated**: 2025-10-08T19:26:00Z
**Session ID**: swarm-docs-refactor
**Total Files**: 213 markdown documents
**Estimated Duration**: 11 hours (across 4 phases)

---

## Executive Summary

This execution plan orchestrates a comprehensive documentation refactoring following the **Diátaxis framework** for the VisionFlow AR-AI Knowledge Graph project. The refactoring addresses structural issues, content duplication, and improves navigation through a 4-phase approach executed by specialized agent swarms.

### Key Objectives

1. **Structural Clarity**: Implement Diátaxis-aligned directory structure
2. **Content Consolidation**: Eliminate duplication while preserving all information
3. **Navigation Excellence**: Create multiple navigation paths for different user journeys
4. **Historical Preservation**: Archive obsolete content with proper context
5. **Quality Assurance**: Validate all links, references, and technical accuracy

---

## Current State Analysis

### Documentation Inventory

- **Total Files**: 213 markdown documents
- **Current Structure**: 15+ directories with some overlap
- **Key Issues**:
  - 3 competing top-level navigation files
  - Content duplication (WebSocket docs in 3 locations)
  - Unclear archive structure
  - Mixed audience targeting

### Existing Assets

**Strong Foundation**:
- `00-INDEX.md` (20,723 bytes) - Excellent cross-referencing, comprehensive navigation
- `README.md` (10,939 bytes) - Current status with metrics
- `index.md` (5,963 bytes) - Diátaxis-aligned navigation hub
- `reference/agents/` - 50+ well-organized agent specifications

**Needs Consolidation**:
- `websocket-consolidation-plan.md` - Technical guide in wrong location
- Multiple archive directories without clear dating
- `guides/` and `development/` overlap

---

## Architectural Decisions

### AD001: Canonical Project Overview
**Decision**: Use `docs/README.md` as the canonical project overview
**Rationale**: Most comprehensive, includes current status and metrics
**Action**: Keep README.md, refactor others as navigation aids

### AD002: Master Table of Contents
**Decision**: Preserve `00-INDEX.md` as master navigation with cross-references
**Rationale**: Excellent cross-linking proven valuable for complex documentation
**Action**: Update to reflect new structure, maintain all cross-references

### AD003: Diátaxis Navigation Hub
**Decision**: Use `index.md` as Diátaxis-aligned quick start hub
**Rationale**: Provides learning-path oriented navigation
**Action**: Simplify to 4 clear paths: tutorials/how-to/explanation/reference

### AD004: Archive Structure
**Decision**: Consolidate all archives under `docs/_archive/YYYY-MM/`
**Rationale**: Single archive location with clear dating prevents confusion
**Action**: Move all archived content to dated directories with context README.md

### AD005: Guides vs Development
**Decision**: `guides/` = user tasks, `development/` = contributor workflows
**Rationale**: Clear audience separation improves discoverability
**Action**: Move developer-specific content to development/

### AD006: Concepts vs Architecture
**Decision**: `concepts/` = understanding principles, `architecture/` = system design
**Rationale**: Diátaxis separation: explanation vs technical specification
**Action**: High-level explanations → concepts/, detailed designs → architecture/

*Full decision log: `_consolidated/architectural-decisions.json`*

---

## Target Structure

```
docs/
├── README.md                      # Canonical project overview (current status, metrics)
├── index.md                       # Diátaxis quick navigation hub (4 learning paths)
├── 00-INDEX.md                    # Master TOC with cross-references
├── contributing.md                # Expanded contribution guide
│
├── getting-started/               # TUTORIALS (learning-oriented)
│   ├── 00-index.md
│   ├── 01-installation.md
│   ├── 02-quick-start.md
│   └── 03-basic-operations.md
│
├── guides/                        # HOW-TO (task-oriented)
│   ├── user/                      # End-user operational guides
│   ├── developer/                 # Development task guides
│   ├── technical/                 # Technical consolidation guides
│   │   └── websocket-consolidation.md
│   └── deployment/                # Deployment and operations
│
├── concepts/                      # EXPLANATION (understanding-oriented)
│   ├── 00-index.md
│   ├── 01-system-overview.md
│   ├── 02-agentic-workers.md
│   ├── 03-gpu-compute.md
│   ├── 04-networking.md
│   └── 05-security.md
│
├── reference/                     # REFERENCE (information-oriented)
│   ├── api/                       # REST, WebSocket, GPU, Voice APIs
│   │   ├── websocket-api.md      # Consolidated WebSocket reference
│   │   ├── binary-protocol.md
│   │   ├── rest-api.md
│   │   ├── gpu-algorithms.md
│   │   └── voice-api.md
│   ├── agents/                    # Agent specifications by category
│   │   ├── README.md             # Enhanced with capability matrix
│   │   ├── core/
│   │   ├── github/
│   │   ├── optimization/
│   │   ├── swarm/
│   │   ├── consensus/
│   │   └── [...]/
│   ├── config/                    # Configuration reference
│   └── glossary.md
│
├── architecture/                  # Technical specifications
│   ├── hybrid_docker_mcp_architecture.md
│   ├── xr-immersive-system.md
│   ├── vircadia-react-xr-integration.md
│   └── components/
│       └── websocket-protocol.md # Component implementation
│
├── deployment/                    # Operations reference
│   ├── vircadia-docker-deployment.md
│   └── multi-agent-docker/
│
├── development/                   # Contributor workflows
│   ├── setup.md
│   ├── workflow.md
│   ├── testing.md
│   └── debugging.md
│
├── research/                      # Technical research
│   └── owl_rdf_ontology_integration_research.md
│
└── _archive/                      # Historical documentation
    ├── 2025-10/
    │   ├── README.md             # Archival context
    │   └── [archived files]
    └── legacy-docs/
        ├── README.md
        └── [legacy content]
```

---

## Phase 1: Documentation Audit & Classification

**Duration**: 2 hours
**Agents**: Audit Specialist, Content Analyst

### Agent: Audit Specialist

**Mission**: Classify all 213 markdown files by Diátaxis category

**Tasks**:
1. Scan every markdown file in `/workspace/ext/docs`
2. Classify each file:
   - **Tutorial**: Step-by-step learning (getting-started)
   - **How-to**: Task-oriented solutions (guides)
   - **Explanation**: Conceptual understanding (concepts)
   - **Reference**: Technical specifications (reference, architecture)
   - **Archive**: Obsolete or superseded content
3. Flag duplicate or overlapping content
4. Identify outdated technical information

**Deliverables**:
- `_consolidated/audit-report.md` - Complete file classification matrix
- `_consolidated/obsolete-files.md` - Files to archive with reasons
- Store classification in memory: `swarm/audit/classification`

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Documentation audit and classification"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks post-edit --memory-key "swarm/audit/classification" --file "audit-report.md"
npx claude-flow@alpha hooks notify --message "Audit complete: 213 files classified"
```

---

### Agent: Content Analyst

**Mission**: Identify content overlap and generate merge recommendations

**Tasks**:
1. Analyze overlapping content areas:
   - WebSocket documentation (3 locations)
   - Project overview files (README.md, index.md, 00-INDEX.md)
   - Agent reference organization
2. Map document dependencies and cross-references
3. Determine canonical sources for merged content
4. Generate specific merge instructions

**Deliverables**:
- `_consolidated/overlap-analysis.md` - Content overlap matrix
- `_consolidated/merge-plan.md` - Specific merge recommendations with priority
- Store in memory: `swarm/audit/overlap`

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Content overlap analysis"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks post-edit --memory-key "swarm/audit/overlap" --file "overlap-analysis.md"
npx claude-flow@alpha hooks notify --message "Overlap analysis complete with merge plan"
```

---

## Phase 2: Structure Implementation & Content Migration

**Duration**: 4 hours
**Agents**: Structure Engineer, Migration Specialist, Archive Manager

### Agent: Structure Engineer

**Mission**: Implement target directory structure

**Tasks**:
1. Create all target directories per structure diagram
2. Set up guides/ subdirectories: user/, developer/, technical/, deployment/
3. Create _archive/YYYY-MM/ structure with dated directories
4. Create index.md scaffolds for each major directory

**Deliverables**:
- Complete target directory tree
- index.md templates in each directory
- `_consolidated/structure-log.md` - Directory creation log

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Directory structure implementation"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks notify --message "Target structure created"
```

---

### Agent: Migration Specialist

**Mission**: Move files to appropriate directories preserving git history

**Tasks**:
1. Read classification from memory: `swarm/audit/classification`
2. Use `git mv` for all file moves (preserves history)
3. Update internal links during migration:
   - Relative paths adjusted for new locations
   - Cross-references updated
4. Handle file renames for clarity (document all renames)
5. Track every move in migration log

**Deliverables**:
- All files moved to target locations
- `_consolidated/migration-log.md` - Complete move record with old→new paths
- Store in memory: `swarm/structure/migration`

**Critical Rules**:
- **ALWAYS use `git mv`** for moves (preserves history)
- Update all relative links when moving files
- Document every rename with rationale
- Verify no broken links after each batch

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "File migration with history preservation"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# Check memory for classification
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work with git mv]
npx claude-flow@alpha hooks post-edit --memory-key "swarm/structure/migration" --file "migration-log.md"
npx claude-flow@alpha hooks notify --message "Migration complete: 213 files moved"
```

---

### Agent: Archive Manager

**Mission**: Organize obsolete content in dated archive structure

**Tasks**:
1. Read obsolete file list from memory: `swarm/audit/classification`
2. Create `_archive/2025-10/` directory
3. Move obsolete content to dated archive
4. Create `_archive/2025-10/README.md` explaining:
   - What was archived and why
   - Historical context and significance
   - Where superseding documentation is located
5. Update references to archived docs (redirect to new locations)

**Deliverables**:
- Organized `_archive/` with dated directories
- Context README.md in each archive directory
- `_consolidated/archive-log.md` - Archival decisions log

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Archive organization"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks notify --message "Archive organized with context"
```

---

## Phase 3: Content Consolidation & Deduplication

**Duration**: 3 hours
**Agents**: Content Merger, Quality Assurance, Index Builder

### Agent: Content Merger

**Mission**: Consolidate overlapping content without information loss

**Tasks**:
1. Read merge plan from memory: `swarm/audit/overlap`
2. Execute specific merges:
   - **WebSocket docs**: Consolidate to `reference/api/websocket-api.md`
   - Keep `architecture/components/websocket-protocol.md` for implementation
   - Move `websocket-consolidation-plan.md` to `guides/technical/`
3. **Agent reference**: Enhance `reference/agents/README.md` with capability matrix
4. Document every merge decision

**Deliverables**:
- Consolidated documents (no duplication)
- `_consolidated/consolidation-report.md` - Merge summary with decisions
- Store in memory: `swarm/consolidation/merges`

**Merge Rules**:
- Preserve all unique information
- Cite original sources in merged docs
- Mark superseded content clearly
- Update cross-references to merged locations

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Content consolidation"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks post-edit --memory-key "swarm/consolidation/merges" --file "consolidation-report.md"
npx claude-flow@alpha hooks notify --message "Content consolidated with zero information loss"
```

---

### Agent: Quality Assurance

**Mission**: Verify no information loss and validate technical accuracy

**Tasks**:
1. Review all merged documents
2. Cross-reference with original content to verify completeness
3. Validate all code examples are current
4. Check technical accuracy of descriptions
5. Test key navigation paths

**Deliverables**:
- `_consolidated/qa-report.md` - Quality verification report
- List of any issues requiring attention

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Quality assurance review"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks notify --message "QA complete: zero information loss verified"
```

---

### Agent: Index Builder

**Mission**: Rebuild all navigation indices

**Tasks**:
1. **Update `README.md`**:
   - Canonical project overview
   - Current status and metrics
   - Link to navigation hubs
2. **Rebuild `index.md`**:
   - Diátaxis 4-path navigation (tutorials/how-to/explanation/reference)
   - Quick links by user journey
3. **Update `00-INDEX.md`**:
   - Master TOC with new structure
   - Maintain all cross-references
   - Update document relationship diagrams
4. **Create directory index.md files**:
   - Local navigation for each major directory

**Deliverables**:
- Updated root navigation files (README.md, index.md, 00-INDEX.md)
- index.md in every major directory
- `_consolidated/navigation-rebuild.md` - Index update log

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Navigation index rebuild"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks notify --message "Navigation indices rebuilt"
```

---

## Phase 4: Validation & Quality Assurance

**Duration**: 2 hours
**Agents**: Link Validator, Content Reviewer, Completeness Auditor

### Agent: Link Validator

**Mission**: Validate all internal links and cross-references

**Tasks**:
1. Scan all markdown files for internal links
2. Verify every link target exists
3. Check cross-reference accuracy (section anchors)
4. Test navigation paths from root to leaves
5. Validate mermaid diagram syntax

**Deliverables**:
- `_consolidated/link-validation.md` - Complete link validation report
- `_consolidated/broken-links.md` - Any broken links found (should be zero)

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Link validation"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks notify --message "Link validation complete: all links valid"
```

---

### Agent: Content Reviewer

**Mission**: Review content quality and Diátaxis alignment

**Tasks**:
1. Verify Diátaxis classification is correct:
   - Tutorials are learning-oriented
   - How-to guides are task-oriented
   - Concepts are understanding-oriented
   - Reference is information-oriented
2. Check UK English spelling consistency
3. Validate code examples reflect current implementation
4. Review technical descriptions for accuracy
5. Verify contribution guide is complete

**Deliverables**:
- `_consolidated/content-review.md` - Content quality report
- List of any corrections needed

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Content quality review"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks notify --message "Content review complete: Diátaxis aligned"
```

---

### Agent: Completeness Auditor

**Mission**: Verify all files accounted for and generate final validation

**Tasks**:
1. Count all files in new structure
2. Cross-reference with original 213 file count
3. Verify no orphaned files exist
4. Check git history preservation (sample files)
5. Generate final validation report with metrics:
   - Files moved/merged/archived breakdown
   - Link validation summary
   - Content quality summary
   - Success criteria checklist

**Deliverables**:
- `_consolidated/FINAL-VALIDATION-REPORT.md` - Complete validation with metrics
- Sign-off for production deployment

**Coordination**:
```bash
npx claude-flow@alpha hooks pre-task --description "Completeness audit"
npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
# [work]
npx claude-flow@alpha hooks post-edit --memory-key "swarm/validation/results" --file "FINAL-VALIDATION-REPORT.md"
npx claude-flow@alpha hooks notify --message "Final validation complete: READY FOR DEPLOYMENT"
npx claude-flow@alpha hooks session-end --export-metrics true
```

---

## Coordination Protocol

### Swarm Topology
**Type**: Hierarchical
**Coordinator**: Chief Documentation Architect
**Execution**: Claude Code Task tool spawns agents concurrently

### Memory-Based Coordination

**Memory Keys**:
- `swarm/architect/decisions` - Architectural decisions (JSON)
- `swarm/audit/classification` - Phase 1 file classification
- `swarm/audit/overlap` - Phase 1 overlap analysis
- `swarm/structure/migration` - Phase 2 migration log
- `swarm/consolidation/merges` - Phase 3 merge summary
- `swarm/validation/results` - Phase 4 validation report

### Hook Integration

**Every Agent MUST**:
1. **Before work**:
   ```bash
   npx claude-flow@alpha hooks pre-task --description "[task]"
   npx claude-flow@alpha hooks session-restore --session-id "swarm-docs-refactor"
   ```

2. **During work**:
   ```bash
   npx claude-flow@alpha hooks post-edit --memory-key "swarm/[phase]/[key]" --file "[file]"
   ```

3. **After work**:
   ```bash
   npx claude-flow@alpha hooks notify --message "[completion message]"
   npx claude-flow@alpha hooks post-task --task-id "[task-id]"
   ```

4. **Phase completion**:
   ```bash
   npx claude-flow@alpha hooks session-end --export-metrics true
   ```

---

## Risk Mitigation

### High Risk: Information Loss
**Mitigation**:
- Git history preservation via `git mv`
- QA agent verifies no content loss
- Archive with context rather than delete
- Completeness audit validates 100% coverage

### Medium Risk: Broken Links
**Mitigation**:
- Automated link validation in Phase 4
- Update cross-references during migration
- Test all navigation paths
- Document intentional external links

### Medium Risk: Miscategorization
**Mitigation**:
- Use Diátaxis framework for categorization
- Content Analyst validates classification
- Allow hybrid docs with clear primary category
- Document edge cases

### Low Risk: Duplicate Work
**Mitigation**:
- Parallel agent execution via Task tool
- Memory-based coordination prevents conflicts
- Clear phase boundaries with deliverables
- Chief Architect reviews before proceeding

---

## Success Criteria

### Structure (Must Have)
- [ ] All 213 files accounted for (moved/merged/archived)
- [ ] Target directory structure fully implemented
- [ ] Archive structure with dated directories and context
- [ ] No orphaned files or broken directory references

### Navigation (Must Have)
- [ ] README.md is canonical project overview
- [ ] index.md provides Diátaxis-aligned navigation
- [ ] 00-INDEX.md maintains comprehensive cross-references
- [ ] Each directory has index.md with local navigation

### Quality (Must Have)
- [ ] Zero broken internal links
- [ ] UK English spelling consistent
- [ ] All code examples validated
- [ ] Mermaid diagrams render correctly

### Diátaxis Alignment (Must Have)
- [ ] getting-started/ contains only tutorials
- [ ] guides/ contains only how-to content
- [ ] concepts/ contains only explanations
- [ ] reference/ contains only specifications
- [ ] Clear categorization with minimal hybrid docs

### Completeness (Must Have)
- [ ] No information lost during consolidation
- [ ] Git history preserved for all moves
- [ ] All architectural decisions documented
- [ ] Validation report confirms 100% coverage

---

## Execution Timeline

| Phase | Duration | Agents | Key Deliverables |
|-------|----------|--------|------------------|
| **Phase 1: Audit** | 2 hours | 2 agents | Classification matrix, merge plan |
| **Phase 2: Restructure** | 4 hours | 3 agents | New structure, migrated files, organized archive |
| **Phase 3: Consolidation** | 3 hours | 3 agents | Consolidated docs, rebuilt indices |
| **Phase 4: Validation** | 2 hours | 3 agents | Final validation, sign-off |
| **Total** | **11 hours** | **11 agents** | Complete refactored documentation |

---

## Next Steps

### Immediate Actions
1. ✅ Store architectural decisions in memory
2. ✅ Create this execution plan
3. ✅ Set up todo tracking
4. ⏭️ Review plan with stakeholders
5. ⏭️ Spawn Phase 1 agents via Claude Code Task tool

### Phase Progression
- **Phase 1**: Execute after plan approval
- **Phase 2**: Execute after Phase 1 deliverables reviewed
- **Phase 3**: Execute after Phase 2 structure verified
- **Phase 4**: Execute after Phase 3 consolidation complete

### Deployment
- **Review**: Chief Architect reviews final validation
- **Approve**: Stakeholder sign-off
- **Deploy**: Commit restructured documentation
- **Communicate**: Announce changes to team
- **Monitor**: Track navigation usage and feedback

---

## Deliverables Summary

All deliverables stored in `_consolidated/`:

### Phase 1
- `audit-report.md` - File classification matrix
- `obsolete-files.md` - Files to archive
- `overlap-analysis.md` - Content overlap matrix
- `merge-plan.md` - Specific merge recommendations

### Phase 2
- `structure-log.md` - Directory creation log
- `migration-log.md` - Complete file move record
- `archive-log.md` - Archival decisions

### Phase 3
- `consolidation-report.md` - Merge summary
- `qa-report.md` - Quality verification
- `navigation-rebuild.md` - Index update log

### Phase 4
- `link-validation.md` - Link validation report
- `broken-links.md` - Issues found (should be empty)
- `content-review.md` - Content quality report
- `FINAL-VALIDATION-REPORT.md` - Complete validation with sign-off

### Meta
- `architectural-decisions.json` - All architectural decisions
- `EXECUTION_PLAN.md` - This document

---

**Status**: READY FOR PHASE 1 EXECUTION
**Approval Required**: Chief Architect + Stakeholders
**Estimated Completion**: 2025-10-09 (11 hours from approval)

---

*This execution plan follows VisionFlow documentation standards: UK English, Diátaxis framework, comprehensive cross-referencing, and parallel agent execution via Claude Code Task tool.*
