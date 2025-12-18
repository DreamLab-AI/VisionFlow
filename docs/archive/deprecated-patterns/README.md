---
title: Deprecated Patterns Archive
description: Historical reference for deprecated architectural patterns and implementation approaches
category: explanation
tags:
  - neo4j
  - rust
  - react
updated-date: 2025-12-18
difficulty-level: advanced
---


# Deprecated Patterns Archive

## Purpose

This directory contains **deprecated documentation** that describes architectural patterns, components, or implementation approaches that were **planned but never implemented** or **replaced with better alternatives**.

## ⚠️ WARNING: DO NOT USE

**All content in this directory is OBSOLETE and should NOT be used for:**
- New feature development
- Architecture decisions
- Learning current system patterns
- Implementation guidance

**Use this directory ONLY for:**
- Historical reference
- Understanding past architectural decisions
- Migration context when updating old code

## Current Architecture

For current, accurate architecture documentation, see:
- `/docs/ARCHITECTURE_OVERVIEW.md` - Current system architecture
- `/docs/explanations/architecture/` - Component-level architecture docs
- `/docs/guides/developer/` - Developer implementation guides

## Archived Patterns

### 1. Wrong Technology Stack (docs/archive/deprecated-patterns/03-architecture-WRONG-STACK.md)

**Status**: ❌ NEVER IMPLEMENTED
**Issue**: Described PostgreSQL + Redis + Vue.js stack that was never built
**Reality**: System uses Neo4j + Rust + React
**Archived**: December 2025

### 2. GraphServiceActor Monolith (see migration guide)

**Status**: ✅ REPLACED (November 2025)
**Issue**: 48K+ token monolithic actor with stale cache issues
**Reality**: Replaced with modular actor architecture + Neo4j
**Migration Guide**: `/docs/guides/graphserviceactor-migration.md`

## How to Use This Archive

### If You Find a Link to Archived Content:

1. **Don't follow the archived pattern** - It's deprecated for a reason
2. **Check the migration guide** - See what replaced it
3. **Update the link** - Point to current documentation
4. **Report broken docs** - File an issue if docs are still pointing here

### If You're Migrating Old Code:

1. **Read the migration guide first** - Each archived pattern should reference its replacement
2. **Use current architecture docs** - Don't try to understand old patterns
3. **Ask for help** - If migration path isn't clear, ask the team

### If You're Writing New Docs:

1. **Never link to archived content** - Always link to current docs
2. **Add deprecation warnings** - If referencing old patterns, mark them clearly
3. **Keep archives minimal** - Don't expand on deprecated patterns

## Deprecation Process

When deprecating documentation:

1. **Add prominent warning banner** to the top of the doc
2. **Link to replacement documentation**
3. **Move to this archive directory**
4. **Update cross-references** - Fix all links pointing to the doc
5. **Update this README** - Add entry explaining what was deprecated

---

---

## Related Documentation

- [Architecture Overview (OBSOLETE - WRONG STACK)](03-architecture-WRONG-STACK.md)
- [Settings API Authentication](../../guides/features/settings-authentication.md)
- [Hexagonal Architecture Ports - Overview](../../explanations/architecture/ports/01-overview.md)
- [User Settings Implementation Summary](../reports/2025-12-02-user-settings-summary.md)
- [KnowledgeGraphRepository Port](../../explanations/architecture/ports/03-knowledge-graph-repository.md)

## Questions?

If you're unsure whether content is current or deprecated:
1. Check the `status` field in the YAML frontmatter
2. Look for deprecation warnings at the top of the doc
3. Check the last updated date
4. Ask the team if still unclear

---

**Remember**: This is a historical archive, not a source of truth for current architecture!
