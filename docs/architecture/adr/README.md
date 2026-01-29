---
title: Architectural Decision Records
description: Index of all architectural decision records for VisionFlow
category: reference
diataxis: reference
tags:
  - adr
  - architecture
  - decisions
updated-date: 2025-01-29
---

# Architectural Decision Records (ADRs)

This directory contains all architectural decision records for VisionFlow.

## What is an ADR?

An Architectural Decision Record captures an important architectural decision made along with its context and consequences.

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](ADR-0001-neo4j-persistent-with-filesystem-sync.md) | Neo4j Persistent with Filesystem Sync | Accepted | 2025-11-05 |

## Status Legend

- **Proposed** - Under discussion
- **Accepted** - Approved and being implemented
- **Deprecated** - No longer recommended
- **Superseded** - Replaced by another ADR

## Creating a New ADR

1. Copy the template from `_template.md`
2. Name it `ADR-NNNN-title-in-kebab-case.md`
3. Fill in all sections
4. Submit for review

## ADR Template

```markdown
---
title: ADR-NNNN: Title
description: Brief description
category: reference
diataxis: reference
tags:
  - adr
  - architecture
updated-date: YYYY-MM-DD
---

# ADR-NNNN: Title

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Deprecated | Superseded
**Decision Maker:** Team/Person

## Context

What is the issue that we're seeing that is motivating this decision?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?

### Positive

### Negative

## Implementation

How will this be implemented?

## References

Links to related documents, issues, or discussions.
```

## Related

- [Architecture Overview](../README.md)
- [System Diagrams](../diagrams/README.md)
