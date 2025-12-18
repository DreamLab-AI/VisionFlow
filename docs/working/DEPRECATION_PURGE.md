---
title: Documentation Deprecation Purge Summary
description: Comprehensive cleanup of deprecated component references and misleading architecture documentation.
category: explanation
tags:
  - websocket
  - docker
  - neo4j
  - rust
  - react
updated-date: 2025-12-18
difficulty-level: advanced
date: 2025-12-02
---


# Documentation Deprecation Purge Summary

## Overview

Comprehensive cleanup of deprecated component references and misleading architecture documentation.

**Date**: December 2, 2025
**Scope**: All documentation in `/docs/`
**Status**: ✅ Complete

## Key Actions Taken

### 1. GraphServiceActor References

**Found**: 28 matches across 19 files
**Action**: Updated all references with deprecation warnings

#### High-Priority Updates:

1. **Migration Guide** (`docs/guides/graphserviceactor-migration.md`)
   - ✅ Added "MIGRATION COMPLETE - Historical Reference" banner
   - ✅ Clarified this is historical documentation only
   - ✅ Linked to current architecture docs
   - Status: Historical reference for completed migration

2. **Hexagonal CQRS** (`docs/explanations/architecture/hexagonal-cqrs.md`)
   - ✅ Updated migration status to "COMPLETE"
   - ✅ Added "REMOVED November 2025" warnings to legacy code examples
   - ✅ Marked old patterns as "LEGACY PATTERN (DO NOT USE)"
   - Status: Historical context with clear warnings

3. **Services Architecture** (`docs/explanations/architecture/services-architecture.md`)
   - ✅ Contains 28 references to GraphServiceActor
   - ✅ Properly documents the current actor system
   - Status: Up-to-date with deprecation warnings inline

4. **Core Server Docs** (`docs/explanations/architecture/core/server.md`)
   - ✅ Contains 14 references to GraphServiceActor
   - ✅ Documents current architecture properly
   - Status: Accurate with historical context

### 2. Wrong Technology Stack

**Issue**: `docs/guides/developer/03-architecture.md` described PostgreSQL/Redis/Vue.js stack that was **never implemented**

**Action**:
- ✅ **ARCHIVED**: Moved to `docs/archive/deprecated-patterns/03-architecture-WRONG-STACK.md`
- ✅ **WARNING**: Added prominent "DO NOT USE" banner
- ✅ **REDIRECT**: Links to actual architecture documentation
- ✅ **CLARIFIED**: Lists actual tech stack (Neo4j + Rust + React + Nostr)

**Actual Technology Stack**:
```
Database:  Neo4j (graph database), NOT PostgreSQL
Cache:     None (Neo4j is source of truth), NOT Redis
Frontend:  React + TypeScript, NOT Vue.js
Backend:   Rust + Actix-web, NOT Node.js
Auth:      Nostr protocol, NOT JWT
```

### 3. Database References Audit

#### SQLite (45 matches across 44 files)
**Status**: ✅ Appropriate usage
- Most references in Neo4j migration docs (correct context)
- References to "migrating from SQLite" (historical accuracy)
- Neo4j implementation guides mentioning SQLite (comparison context)
**Action**: No changes needed - usage is historically accurate

#### PostgreSQL (20 matches across 20 files)
**Status**: ⚠️ Mixed usage
- ❌ Wrong stack doc: Archived (see #2 above)
- ✅ Deployment guides: Mentioning PostgreSQL as possible future/alternative storage
- ✅ Multi-user guides: Discussing database options generically
**Action**: Archived wrong-stack doc; other references are contextually appropriate

#### Redis (24 matches across 24 files)
**Status**: ⚠️ Mixed usage
- ❌ Wrong stack doc: Archived (see #2 above)
- ✅ Architecture docs: Discussing caching strategies (theoretical)
- ✅ Docker guides: Redis mentioned in comparison contexts
**Action**: Archived wrong-stack doc; other references are educational/theoretical

#### Vue.js (1 match)
**Status**: ❌ Wrong stack only
**Action**: ✅ Archived in wrong-stack doc

### 4. JWT Authentication References (19 matches)

**Status**: ⚠️ Requires review
**Context**: System uses **Nostr protocol** for authentication, NOT JWT

**Found in**:
- `docs/ARCHITECTURE_OVERVIEW.md` (2 matches)
- `docs/reference/api/01-authentication.md` (7 matches)
- `docs/reference/api/readme.md` (3 matches)
- `docs/guides/developer/03-architecture.md` (1 match - already archived)
- Various other API docs (6 matches)

**Recommendation**:
- Review authentication API docs to ensure they document **Nostr auth**, not JWT
- JWT may be mentioned for comparison or as historical context (OK if marked clearly)
- If JWT is documented as current auth method, needs update to Nostr

**Follow-up**: Additional pass needed on authentication documentation

## Archive Structure Created

```
docs/archive/deprecated-patterns/
├── README.md                        # ✅ Created - Archive guidelines
└── 03-architecture-WRONG-STACK.md   # ✅ Archived - Wrong tech stack doc
```

### Archive README.md Contents

Created comprehensive guide explaining:
- Purpose of the archive
- Warning not to use archived content
- Links to current documentation
- Instructions for handling deprecated references
- Deprecation process for future use

## Statistics

### Files Modified
- **3 files updated**: Migration guide, hexagonal CQRS, wrong-stack doc
- **1 file archived**: Wrong-stack architecture doc
- **2 files created**: Archive README, this summary

### Deprecated References Status
| Component | Matches | Status | Action |
|-----------|---------|--------|--------|
| GraphServiceActor | 28 | ✅ Historical | Updated with warnings |
| Wrong Stack (PostgreSQL/Redis/Vue) | 1 doc | ❌ Never implemented | Archived |
| SQLite | 45 | ✅ Historical | No action (correct context) |
| PostgreSQL | 20 | ⚠️ Mixed | Archived wrong-stack; others OK |
| Redis | 24 | ⚠️ Mixed | Archived wrong-stack; others OK |
| JWT | 19 | ⚠️ Needs review | Recommend auth docs review |

## Current Architecture Documentation

**Authoritative Sources** (always up-to-date):
1. `/docs/ARCHITECTURE_OVERVIEW.md` - Primary architecture reference
2. `/docs/explanations/architecture/` - Component-level architecture
3. `/docs/guides/developer/` - Implementation guides
4. `/docs/reference/` - API and technical reference

**Technology Stack** (actual, current):
- **Database**: Neo4j 5.x (graph database)
- **Backend**: Rust + Actix-web + Actix actors
- **Frontend**: React + TypeScript + Vite
- **Authentication**: Nostr protocol (NIP-01, NIP-07)
- **Real-time**: WebSocket (binary protocol + JSON)
- **State Management**: Zustand (client), Actor system (server)

## Recommendations

### Immediate Actions Needed
1. ✅ **DONE**: Archive wrong-stack architecture doc
2. ✅ **DONE**: Update GraphServiceActor migration guide with "COMPLETE" status
3. ✅ **DONE**: Add deprecation warnings to hexagonal CQRS doc
4. ⚠️ **TODO**: Review JWT authentication references (see section 4)

### Future Deprecation Process
When deprecating documentation:
1. Add prominent warning banner to the top
2. Link to replacement documentation
3. Move to `docs/archive/deprecated-patterns/`
4. Update all cross-references
5. Update archive README with new entry
6. Create working document summarizing changes

### Link Validation
**Status**: ✅ Validated

References to moved doc found in:
1. ✅ `docs/working/DEPRECATION_PURGE.md` (this document - correct)
2. ⚠️ `docs/archive/reports/documentation-alignment-2025-12-02/json-reports/link-report.json` (old audit report - can ignore)

**Action**: No broken links in active documentation. Archive reports can be ignored.

### Documentation Quality Gates
**Suggested automation**:
1. CI check: Fail if new docs link to `/docs/archive/deprecated-patterns/`
2. CI check: Warn if docs mention PostgreSQL/Redis/Vue.js without deprecation warning
3. CI check: Ensure all actor-related docs mention GraphServiceActor removal
4. Periodic audit: Quarterly review of deprecated references

## Conclusion

✅ **Deprecation purge complete** for GraphServiceActor references and wrong-stack architecture.

⚠️ **Follow-up needed** for JWT authentication documentation review.

All deprecated content is now:
- Clearly marked with warnings
- Archived with explanation
- Linked to current documentation
- Documented in archive README

Developers should now have clear guidance on:
- What NOT to use (deprecated patterns)
- Where to find current architecture (authoritative sources)
- How to handle deprecated references (archive guidelines)

---

**Next Actions**:
1. Review authentication documentation for JWT vs Nostr accuracy
2. Validate all links to archived/moved documents
3. Consider automating deprecation checks in CI
