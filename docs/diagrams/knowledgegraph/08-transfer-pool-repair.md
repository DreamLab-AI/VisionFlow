---
id: KG-08
title: Retained explorer transferable-buffer repair
area: knowledgegraph
governing: []
adrs: [ADR-2002]
sources:
  - ../knowledgeGraph/explorer/modern/src/workers/positionTransport.ts
  - ../knowledgeGraph/explorer/modern/src/workers/physics.worker.ts
verified_commit: 4809ea9772e61766058cf20d38e7527082e78fed
---

## For developers

This scoped successor covers KG-04.2 in the retained modern explorer. It does not declare this explorer the canonical publisher; visionGraph retains that role.

## For the business

The retained explorer can reclaim buffers returned through a real worker transfer instead of exhausting its two-buffer pool.

## KG-08.1 Count transfers rather than retaining object identity

```mermaid
flowchart TB
    TAKE["acquire buffer<br/>positionTransport.ts:74"] --> COUNT["count outstanding transfers<br/>positionTransport.ts:55"]
    RETURN["returned transferable wrapper"] --> GEN["current generation only<br/>physics.worker.ts:289"]
    GEN --> RELEASE["consume outstanding count<br/>positionTransport.ts:84<br/>positionTransport.ts:86"]
    COUNT --> RELEASE
```

**What it shows:** transferred buffers return as new JavaScript wrappers, so pool accounting uses the number of outstanding transfers. **Why it is this way:** object identity does not survive structured cloning. The existing worker generation guard continues to reject returns from an earlier graph. **Invariant:** a return requires an outstanding transfer (`positionTransport.ts:84`).
