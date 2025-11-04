# GraphServiceActor Documentation Search - Quick Index

**Search Date**: November 4, 2025
**Status**: COMPLETE - All references identified and catalogued

---

## Quick Stats

- **Total Files Analyzed**: 11 markdown files
- **Total References Found**: 38 mentions
- **Files with References**: 11 (100% of scanned files)
- **Estimated Deprecation Effort**: 2-3 hours (add notices)
- **Migration Status**: Phase 2 IN PROGRESS (~15% complete)

---

## Files with GraphServiceActor References

### Tier 1: CRITICAL (Must Review)

1. **`/docs/concepts/architecture/hexagonal-cqrs-architecture.md`** (8 references)
   - Status: AUTHORITATIVE SOURCE
   - Contains: Complete migration plan (4 phases), problem analysis, solution architecture
   - Action: Ensure marked as current production implementation
   - Lines: 39-41, 123, 126, 909, 955, 1168, 1274, 1845

2. **`/docs/alignment-report.md`** (9 references)
   - Status: RECENT STRATEGIC AUDIT
   - Contains: 38-reference summary, deprecation task planning
   - Action: Already comprehensive, use as reference
   - Lines: 19, 76, 233, 235, 246, 281-282, 458, 531, 537

### Tier 2: HIGH (Add Deprecation Notices)

3. **`/docs/concepts/architecture/gpu/communication-flow.md`** (7 references)
   - Status: Current GPU subsystem documentation
   - Contains: Actor message passing patterns to GPUManagerActor
   - Action: Add deprecation notice for GPU comm patterns
   - Lines: 5, 10, 16, 17, 24, 40, 41, 71

4. **`/docs/concepts/architecture/core/server.md`** (6 references)
   - Status: Current transitional architecture
   - Contains: TransitionalGraphSupervisor wrapper description
   - Action: Update to reflect Phase 2 progress
   - Lines: 33, 154, 168, 233-234

5. **`/docs/concepts/architecture/quick-reference.md`** (2 references)
   - Status: Migration planning document
   - Contains: Size metrics (156KB, 4614 lines), target deletion
   - Action: Update metrics as migration progresses
   - Lines: 7, 191

### Tier 3: MEDIUM (Update When Phases Complete)

6. **`/docs/guides/pipeline-admin-api.md`** (1 reference)
   - Status: API documentation (wiring needed)
   - Contains: Required actor addresses listing
   - Action: Update when Phase 2 command handlers complete
   - Line: 316

7. **`/docs/concepts/ontology-pipeline-integration.md`** (1 reference)
   - Status: Current integration guide
   - Contains: GraphServiceActor broadcasting pattern
   - Action: Update broadcasting pattern when event bus implemented (Phase 3)
   - Line: 201

8. **`/docs/NEO4j-settings-migration-documentation-report.md`** (2 references)
   - Status: Related database migration (complete)
   - Contains: GraphServiceActor deprecation as next priority
   - Action: Reference in deprecation task documentation
   - Lines: 381, 430

### Tier 4: LOW (Reference/Audit Only)

9. **`/docs/documentation-audit-completion-report.md`** (1 reference)
   - Status: Audit findings
   - Contains: Task tracking for deprecation notices
   - Action: No changes needed, already identified
   - Line: 134

10. **`/docs/concepts/architecture/gpu/optimizations.md`** (1 reference)
    - Status: Historical optimization effort
    - Contains: Performance improvements on monolithic actor
    - Action: Mark as historical (temporary measure)
    - Line: 5

11. **`/docs/concepts/architecture/core/client.md`** (1 reference)
    - Status: Minimal reference in client docs
    - Contains: Brief mention in actor communication
    - Action: No action needed
    - Line: 1037

---

## Reference Summary Table

| File | Refs | Type | Tier | Action |
|------|------|------|------|--------|
| hexagonal-cqrs-architecture.md | 8 | Design | 1 | Review authority status |
| alignment-report.md | 9 | Strategic | 1 | Use as reference |
| gpu/communication-flow.md | 7 | Architecture | 2 | Add deprecation notice |
| server.md | 6 | Current State | 2 | Update Phase 2 progress |
| quick-reference.md | 2 | Reference | 2 | Update metrics |
| NEO4j-settings-migration-documentation-report.md | 2 | Migration | 3 | Reference in task docs |
| pipeline-admin-api.md | 1 | API | 3 | Update Phase 2 complete |
| ontology-pipeline-integration.md | 1 | Integration | 3 | Update Phase 3 complete |
| documentation-audit-completion-report.md | 1 | Audit | 4 | Already documented |
| gpu/optimizations.md | 1 | Performance | 4 | Mark as historical |
| client.md | 1 | Client | 4 | No action |
| **TOTAL** | **38** | — | — | **~2-3 hours effort** |

---

## The Core Problem

**GraphServiceActor** is a monolithic 48,000+ token actor with:
- 4,614 lines of code
- 46 message handlers
- 129 message types
- Mixed responsibilities: cache, physics, WebSocket, semantics, settings, GitHub sync

**Critical Bug**: In-memory cache becomes STALE after database writes
- Example: GitHub sync writes 316 nodes → cache still shows 63 nodes
- Cause: No cache invalidation mechanism
- Impact: API returns stale data to clients

---

## The Solution: 4-Phase Migration

| Phase | Focus | Status | Timeline |
|-------|-------|--------|----------|
| 1 | Query handlers (reads) | ✅ DONE | 1 week |
| 2 | Command handlers (writes) | 🔄 IN PROGRESS | 1-2 weeks |
| 3 | Event bus (cache invalidation) | 🔴 NOT STARTED | 1-2 weeks |
| 4 | Legacy removal (delete actor) | 🔴 NOT STARTED | 1-2 weeks |

**Current Progress**: ~15% (Phase 1 only)

---

## Recommended Next Steps

### IMMEDIATE (This Week)
1. Add deprecation notices to all 8 files (use template)
2. Mark sections as "PARTIALLY HISTORICAL"
3. Link to hexagonal-cqrs-architecture.md
4. Update Phase 2 progress status

### THIS SPRINT (1-2 Weeks)
1. Complete Phase 2: Command handlers for writes
2. Create event bus infrastructure
3. Update examples with CQRS patterns
4. Add comprehensive tests

### NEXT SPRINT (Weeks 3-4)
1. Implement event subscribers (cache invalidation, WebSocket broadcast)
2. Update GitHub sync to emit events
3. Remove actor message usage from handlers
4. Final audit for zero actor references

---

## Deprecation Notice Template

Copy this template to GraphServiceActor sections:

```markdown
> ⚠️ **DEPRECATION NOTICE - GraphServiceActor**
> 
> This component is being replaced by hexagonal CQRS architecture.
> 
> **Current Status**: Phase 2/4 (Command Handlers - IN PROGRESS)
> **Target Completion**: ~4 weeks (Phases 2-4)
> **Impact**: Cache coherency issues (stale data after GitHub sync)
> 
> **Replacement Pattern**: Use CQRS query/command handlers instead of actor messages
> 
> **Migration Details**:
> - See `/docs/concepts/architecture/hexagonal-cqrs-architecture.md` for complete plan
> - See `/docs/concepts/architecture/quick-reference.md` for implementation checklist
> 
> **What's Changing**:
> - Before: Send actor messages with mixed responsibilities
> - After: Use command/query handlers with event-driven side effects
> 
> **This Document**: PARTIALLY HISTORICAL - contains current architecture but marked for replacement
```

---

## Key Documents

### Essential Reading
1. **Hexagonal CQRS Architecture** (most important)
   - `/docs/concepts/architecture/hexagonal-cqrs-architecture.md`
   - Contains: Complete problem, solution, 4-phase plan, code examples

2. **Quick Reference** (implementation guide)
   - `/docs/concepts/architecture/quick-reference.md`
   - Contains: Migration checklist, common patterns, next steps

3. **Alignment Report** (strategic overview)
   - `/docs/alignment-report.md`
   - Contains: 38-reference summary, deprecation task planning

### Supporting Documentation
- **Server Architecture**: `/docs/concepts/architecture/core/server.md`
- **GPU Communication**: `/docs/concepts/architecture/gpu/communication-flow.md`
- **Ontology Pipeline**: `/docs/concepts/ontology-pipeline-integration.md`
- **Pipeline Admin API**: `/docs/guides/pipeline-admin-api.md`

---

## Complete Analysis Report

For detailed line-by-line analysis of all references:

📄 **Full Report**: `/docs/graphserviceactor-deprecation-analysis.md` (380 lines)

Contains:
- All 11 files with specific line numbers
- Complete context for each reference
- Classification by type and status
- Critical findings and metrics
- Replacement code patterns
- File-by-file action items
- References for further reading

---

## Quick Navigation

**Find references by file**:
- Architecture files (5): hexagonal-cqrs, server, gpu/communication-flow, gpu/optimizations, ontology-pipeline
- Strategic planning (3): ALIGNMENT-REPORT, NEO4J-MIGRATION, DOCUMENTATION-AUDIT
- API/Reference (2): pipeline-admin-api, QUICK-REFERENCE
- Client (1): client.md

**Find references by type**:
- Problem description: 5 refs (why it's being deprecated)
- Migration plan: 3 refs (4-phase roadmap)
- Strategic/planning: 9 refs (task tracking and planning)
- Architecture design: 8 refs (pattern examples and diagrams)
- Implementation details: 7 refs (actor message passing)
- Current state: 6 refs (transitional architecture)
- Integration points: 2 refs (API wiring)
- Performance: 1 ref (optimization history)

---

## Status Summary

✅ **Search Complete**: All 11 files analyzed, 38 references identified
✅ **Analysis Complete**: References categorized and contextualized
✅ **Priority Set**: Tier-1/2 files identified for immediate action
⚠️ **Documentation Debt**: 38 refs, 0 deprecation notices (HIGH PRIORITY)
🔄 **Migration Status**: Phase 2 IN PROGRESS, ~15% complete
🔴 **Remaining Work**: Phases 2-4 (3-4 weeks estimated)

---

**Generated**: November 4, 2025
**Scope**: Complete documentation corpus search
**Result**: COMPREHENSIVE INDEX WITH ACTIONABLE ITEMS
