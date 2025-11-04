# Documentation Deprecation Strategy - Master Index

**Last Updated**: November 4, 2025
**Purpose**: Central navigation for all deprecation documentation strategies

---

## Overview

This index provides quick access to deprecation templates, implementation plans, and completed migration reports for architectural changes in the VisionFlow project.

---

## Active Deprecations

### 1. GraphServiceActor Deprecation (November 2025)

**Status**: 🔄 **TEMPLATES READY - AWAITING IMPLEMENTATION**
**Impact**: 8 files, 38 references
**Effort**: ~4.5 hours estimated

#### Quick Links
- **📋 Summary**: [`graphserviceactor-deprecation-summary.md`](./graphserviceactor-deprecation-summary.md) - Start here
- **📝 Templates**: [`graphserviceactor-deprecation-templates.md`](./graphserviceactor-deprecation-templates.md) - 5 reusable templates
- **🎯 Implementation Plan**: [`graphserviceactor-implementation-plan.md`](./graphserviceactor-implementation-plan.md) - File-by-file instructions

#### Context
- **What**: Monolithic GraphServiceActor (48,000+ tokens) → Hexagonal CQRS architecture
- **Why**: Stale cache issues, tight coupling, difficult testing
- **When**: Migration started September 2025, CQRS complete November 2025, actor removal planned December 2025
- **Current State**: TransitionalGraphSupervisor provides bridge pattern

#### Files Affected
1. `/docs/concepts/architecture/hexagonal-cqrs-architecture.md` (9 refs) 🔴 Priority
2. `/docs/concepts/architecture/gpu/communication-flow.md` (16 refs) 🔴 Priority
3. `/docs/concepts/architecture/core/server.md` (5 refs) 🟡 High
4. `/docs/concepts/architecture/quick-reference.md` (2 refs) 🟡 High
5. `/docs/concepts/ontology-pipeline-integration.md` (1 ref) 🟢 Medium
6. `/docs/guides/pipeline-admin-api.md` (1 ref) 🟢 Medium
7. `/docs/concepts/architecture/core/client.md` (1 ref) 🟢 Medium
8. `/docs/concepts/architecture/gpu/optimizations.md` (1 ref) 🟢 Low

---

## Completed Deprecations

### 1. SQLite Settings Repository Migration (November 2025)

**Status**: ✅ **COMPLETE**
**Impact**: 5 files updated, zero documentation debt
**Effort**: 4 hours (actual)

#### Quick Links
- **📊 Completion Report**: [`NEO4j-settings-migration-documentation-report.md`](./NEO4j-settings-migration-documentation-report.md)

#### Context
- **What**: SQLite settings repository → Neo4j settings repository
- **Why**: Graph database better suited for hierarchical settings, improved performance (~90x speedup with cache)
- **When**: Migration completed November 2025
- **Current State**: Neo4jSettingsRepository fully operational in production

#### Files Updated
1. `/docs/concepts/architecture/ports/02-settings-repository.md` ✅
2. `/docs/concepts/architecture/00-ARCHITECTURE-overview.md` ✅
3. `/docs/concepts/architecture/ports/01-overview.md` ✅
4. `/docs/guides/neo4j-migration.md` ✅
5. `/docs/alignment-report.md` ✅

#### Success Metrics
- ✅ 100% file completion rate (5/5)
- ✅ Zero documentation discrepancies
- ✅ Production code alignment verified
- ✅ Developer-ready documentation with working examples

---

## Deprecation Template Library

### Standard Templates (Reusable Across Migrations)

All templates are based on the proven Neo4j migration pattern (100% success rate).

#### 1. Top-of-File Banner Templates

**Standard Version**:
```markdown
> ⚠️ **ARCHITECTURAL MIGRATION NOTICE (November 2025)**
> This document references the legacy **[COMPONENT]** pattern, which has been replaced by [NEW PATTERN].
> See `/docs/[PATH-TO-CURRENT-ARCH]` for the current architecture.
```

**Detailed Version** (for comprehensive architectural docs):
```markdown
> ⚠️ **ARCHITECTURAL MIGRATION NOTICE (November 2025)**
>
> **Legacy Pattern**: [OLD COMPONENT]
> **Current Architecture**: [NEW PATTERN]
> **Migration Status**: ✅ [NEW] implementation complete
> **Documentation Status**: ⚠️ This document contains historical references
>
> **What Changed**:
> - ❌ [Old behavior] → ✅ [New behavior]
>
> **Migration Guide**: `/docs/guides/[migration-guide].md`
> **Current Architecture**: `/docs/concepts/architecture/[current-arch].md`
```

**Transitional Version** (for in-progress migrations):
```markdown
> ⚠️ **TRANSITIONAL ARCHITECTURE NOTICE (November 2025)**
> This document describes the **transitional state** during [COMPONENT] deprecation.
> [NEW PATTERN] is operational, [BRIDGE PATTERN] provides backward compatibility.
> **Current Status**: Phase [N] of [M] ([description])
```

#### 2. Inline Deprecation Templates

**Code Example Warning**:
```markdown
**Legacy Pattern** ❌ **DEPRECATED (November 2025)**

[Old code example]

**Current Pattern** ✅ **PRODUCTION (November 2025)**

[New code example]

**Migration Path**: See `/docs/guides/[migration-guide].md` section X.Y
```

**Architecture Diagram Warning**:
```markdown
**Historical Architecture Diagram** ⚠️ **FOR REFERENCE ONLY**

The diagram below shows the **legacy pattern** (pre-November 2025).
For the **current architecture**, see `/docs/concepts/architecture/[current].md`.

[Diagram]

**Current Architecture**: See section X.Y
```

**Feature Description Warning**:
```markdown
### [Feature Name] ⚠️ **ARCHITECTURE CHANGED**

> **Historical Context**: Previously [old approach]
> **Current Implementation**: [new approach]
> **Migration Date**: November 2025

**Legacy Approach** (for reference only):
- [Old behavior]

**Current Approach**:
- [New behavior]
```

#### 3. Migration Path Template

```markdown
## Migration Example: [Operation Name]

### Before: [Legacy Pattern] ❌

```[language]
// DEPRECATED: [explanation]
[old code]
```

### After: [Current Pattern] ✅

```[language]
// CURRENT: [explanation]
[new code]
```

**Key Improvements**:
- ✅ [Benefit 1]
- ✅ [Benefit 2]
- ✅ [Benefit 3]
```

#### 4. Timeline Template

```markdown
## [Component] Deprecation Timeline

| Phase | Status | Date | Description |
|-------|--------|------|-------------|
| **Phase 1: Analysis** | ✅ Complete | [Date] | [Description] |
| **Phase 2: Design** | ✅ Complete | [Date] | [Description] |
| **Phase 3: Implementation** | ✅ Complete | [Date] | [Description] |
| **Phase 4: Transition** | 🔄 In Progress | [Date] | [Description] |
| **Phase 5: Removal** | ⏳ Planned | [Date] | [Description] |

**Current Status**: Phase [N] - [Description]
**Production Status**: ✅ [New component] fully operational
**Legacy Code**: ⚠️ [Old component] [current state]
```

---

## Status Icon Guide

Use these consistently across all deprecation documentation:

| Icon | Meaning | Usage |
|------|---------|-------|
| ✅ | **Complete / Production / Active** | Fully operational, current implementation |
| 🔄 | **In Progress / Transitional** | Work underway, hybrid state |
| ⚠️ | **Deprecated / Legacy / Warning** | Old pattern, avoid new usage |
| ❌ | **Removed / Obsolete** | No longer available |
| ⏳ | **Planned** | Future work scheduled |
| 🔴 | **Critical Priority** | High impact, urgent attention |
| 🟡 | **High Priority** | Important, schedule soon |
| 🟢 | **Medium/Low Priority** | Can be scheduled flexibly |

---

## Deprecation Workflow Template

For any future architectural deprecation, follow this proven workflow:

### Phase 1: Analysis (1-2 hours)
1. Search codebase for component references (`rg "ComponentName"`)
2. Categorize files by impact (architectural deep-dives, guides, API docs)
3. Count total references and estimate effort
4. Identify current implementation to reference

### Phase 2: Template Creation (1-2 hours)
1. Review this index and Neo4j migration report
2. Adapt templates to specific component deprecation
3. Create component-specific implementation plan
4. Define success metrics and quality checklist

### Phase 3: Documentation Updates (2-4 hours)
1. Update files in priority order (critical → high → medium → low)
2. Apply templates consistently (use decision tree)
3. Create master migration guide
4. Update cross-reference documents

### Phase 4: Verification (15-30 minutes)
1. Run quality checklist on each file
2. Verify all internal links work
3. Check consistency of status indicators
4. Search for component name - ensure all have deprecation context

### Phase 5: Completion Report (30 minutes)
1. Document effort metrics (estimated vs actual)
2. Capture lessons learned
3. Update this index with completed migration
4. Mark tasks complete in audit reports

---

## Quality Standards

### Every Deprecation Documentation Update Must Include

✅ **Clear deprecation notice** (banner or inline as appropriate)
✅ **Migration date** (when replacement became available)
✅ **Status indicators** (✅/❌/⚠️/🔄/⏳)
✅ **Links to migration guide** and current architecture
✅ **Code examples** marked legacy vs current (if applicable)
✅ **Historical context** preserved (for educational value)

### Consistency Rules (Across All Deprecations)

- **Icons**: Use standard set (✅/❌/⚠️/🔄/⏳) - no custom variations
- **Dates**: "November 2025" format (not "Nov 2025" or "11/2025")
- **Paths**: Absolute from project root (`/docs/...`)
- **Status**: "DEPRECATED" (consistent capitalization)
- **Code blocks**: Always show "Before/After" or "Legacy/Current"
- **Links**: Bidirectional (guide → docs, docs → guide)

---

## Reference Documentation

### Primary Guides
- **Architecture Overview**: `/docs/concepts/architecture/00-ARCHITECTURE-overview.md`
- **Alignment Report**: `/docs/alignment-report.md`
- **Audit Report**: `/docs/documentation-audit-completion-report.md`

### Migration Guides
- **Neo4j Settings**: `/docs/guides/neo4j-migration.md`
- **GraphServiceActor**: `/docs/guides/graphserviceactor-migration.md` (to be created)

### Port Documentation
- **Settings Repository**: `/docs/concepts/architecture/ports/02-settings-repository.md`
- **Knowledge Graph**: `/docs/concepts/architecture/ports/03-knowledge-graph-repository.md`
- **Ports Overview**: `/docs/concepts/architecture/ports/01-overview.md`

---

## Future Deprecations (Planned)

### Potential Future Migrations

Track these for future deprecation documentation needs:

1. **Three-Database Legacy References** ⏳
   - **What**: Legacy references to knowledge.db, ontology.db, graph.db (pre-unification)
   - **Current**: unified.db with UnifiedGraphRepository and UnifiedOntologyRepository
   - **Files Affected**: ~15 files with "three-database" references
   - **Priority**: 🟢 Low (mostly marked with historical notices already)

2. **Legacy Actor References** ⏳
   - **What**: References to deprecated actor patterns beyond GraphServiceActor
   - **Current**: Various (case-by-case)
   - **Files Affected**: TBD
   - **Priority**: 🟢 Low (address as needed)

3. **API Endpoint Deprecations** ⏳
   - **What**: REST API endpoints that may be deprecated in favor of GraphQL
   - **Current**: REST API operational
   - **Files Affected**: `/docs/reference/rest-api-reference.md` and related
   - **Priority**: 🟢 Low (no immediate plans)

---

## Metrics and Success Tracking

### Completed Migrations

| Migration | Files Updated | References Addressed | Effort (Actual) | Completion Date | Success Rate |
|-----------|---------------|---------------------|-----------------|-----------------|--------------|
| Neo4j Settings | 5 | 22+ | 4 hours | Nov 4, 2025 | ✅ 100% |

### In-Progress Migrations

| Migration | Files to Update | References to Address | Effort (Est.) | Target Date | Status |
|-----------|-----------------|----------------------|---------------|-------------|--------|
| GraphServiceActor | 8 | 38 | 4.5 hours | TBD | 🔄 Templates Ready |

### Overall Documentation Health

- **Total Deprecation Notices Added**: 5 files (Neo4j complete)
- **Total Documentation Debt Resolved**: Neo4j settings (100%)
- **Pending Documentation Debt**: GraphServiceActor (38 references)
- **Consistency Score**: 100% (Neo4j migration)
- **Developer Feedback**: Positive (clear migration paths)

---

## Contributing to Deprecation Documentation

### When Adding New Deprecation Notices

1. **Use This Index**: Start here, review completed migrations for patterns
2. **Follow Templates**: Use template library - don't reinvent
3. **Maintain Consistency**: Use standard icons, date formats, and structure
4. **Update This Index**: Add new deprecation to "Active" or "Completed" section
5. **Link Bidirectionally**: Ensure migration guide ↔ documentation cross-references
6. **Run Quality Checklist**: Verify all standards met before considering complete

### When Completing a Deprecation

1. Move entry from "Active" to "Completed" section
2. Add completion report link
3. Document actual effort vs. estimate
4. Capture lessons learned for future migrations
5. Update overall metrics table

---

## Search Commands

### Find All Deprecation Notices
```bash
# Search for deprecation banners
rg "ARCHITECTURAL MIGRATION NOTICE|DEPRECATION NOTICE|TRANSITIONAL ARCHITECTURE" /home/devuser/workspace/project/docs/

# Find status indicators
rg "(✅|❌|⚠️|🔄|⏳)" /home/devuser/workspace/project/docs/

# Search for specific component deprecations
rg "GraphServiceActor.*DEPRECATED|Neo4j.*MIGRATION" /home/devuser/workspace/project/docs/
```

### Verify Consistency
```bash
# Check date format consistency
rg "November 2025|December 2025|Sept 2025" /home/devuser/workspace/project/docs/

# Find TODO markers (should be none in completed deprecations)
rg "(TODO|FIXME|TBD)" /home/devuser/workspace/project/docs/

# Check internal links
rg "\[.*\]\(/docs/.*\)" /home/devuser/workspace/project/docs/ -o
```

---

## Contact and Support

### Questions About Deprecation Strategy

- **Template Usage**: See template decision tree in `graphserviceactor-deprecation-templates.md`
- **Implementation Help**: Review completed Neo4j migration report for proven patterns
- **Quality Standards**: Refer to "Quality Standards" section above

### Suggesting Improvements

If you identify ways to improve the deprecation documentation strategy:
1. Review completed migrations for context
2. Propose changes that maintain consistency
3. Update this index with rationale
4. Consider impact on existing deprecation notices

---

**Index Version**: 1.0.0
**Last Updated**: November 4, 2025
**Maintained By**: Documentation Team
**Review Cycle**: Monthly (or after each major migration)

---

## Quick Navigation

**Jump to**:
- [Active Deprecations](#active-deprecations) - Current work
- [Completed Deprecations](#completed-deprecations) - Success stories
- [Template Library](#deprecation-template-library) - Reusable patterns
- [Status Icons](#status-icon-guide) - Consistent usage
- [Workflow](#deprecation-workflow-template) - Process guide
- [Quality Standards](#quality-standards) - Success criteria
- [Future Deprecations](#future-deprecations-planned) - Pipeline
