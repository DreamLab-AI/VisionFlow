---
id: KG-04
title: Retained explorer — scope limits, worker transfers and publication boundary
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/adr/ADR-2001-corpus-consolidation.md
adrs: [ADR-2001]
sources:
  - ../knowledgeGraph/docs/adr/ADR-2001-corpus-consolidation.md
  - ../knowledgeGraph/explorer/modern/package.json
  - ../knowledgeGraph/explorer/modern/src/router.tsx
  - ../knowledgeGraph/explorer/modern/src/types/scope.ts
  - ../knowledgeGraph/explorer/modern/src/stores/scopeStore.ts
  - ../knowledgeGraph/explorer/modern/src/workers/protocol.ts
  - ../knowledgeGraph/explorer/modern/src/workers/positionTransport.ts
  - ../knowledgeGraph/explorer/modern/src/workers/physics.worker.ts
  - ../knowledgeGraph/explorer/modern/src/hooks/useSimulation.ts
  - ../knowledgeGraph/explorer/modern/src/api/pageService.ts
  - ../knowledgeGraph/explorer/modern/src/components/Canvas/GraphCanvas.tsx
  - ../knowledgeGraph/explorer/FORMAT-NGG1.md
  - ../knowledgeGraph/docs/architecture/explorer.md
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/types/scope.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/positionTransport.ts
  - ../visionGraph/.github/workflows/publish.yml
  - ../knowledgeGraph/archive/github-workflows/build.yml
verified_commit: {knowledgegraph: 5294e88b6602a63b5afe96c79f2779d00274a05e, visiongraph: d8277151a1404b456b5c7206187f7e4b50f25be1}
---

**Repair note — 2026-11-03:** selected counterexamples below have a [scoped successor verification](08-transfer-pool-repair.md). This historical topic retains its declared source revision; the successor states the repaired contracts and acceptance boundaries.

## For developers

This topic distinguishes the retained knowledgeGraph viewer from visionGraph's configured publication copy. Follow the scope, transfer and page-content boundaries before reusing it; [the dated assessment](../../estate-review/2026-11-01-knowledgegraph-explorer-contracts.md) records probes and the knowledgeGraph ADR-2001 section dispositions.

## For the business

A successful viewer build does not establish a working browsing journey. The retained viewer can stop receiving positions and can describe an unavailable page as an unfinished concept; the configured publishing copy has a different buffer pool, so those results cannot be generalised to the public site.

## KG-04.1 Retained viewer topology and legacy routing

```mermaid
flowchart TB
    ROUTER["Lazy routes: home, page, graph, search, data, about<br/>router.tsx:35-41"]
    GP["GraphPage imports the graph route<br/>router.tsx:37"]
    WORKER["Worker creates NggExplorer<br/>physics.worker.ts:235-242"]
    WASM["External vowl-wasm v0.1.1 release tarball<br/>explorer/modern/package.json:20"]
    OLD["Legacy ontology path redirects to graph<br/>query string preserved; path nodeId dropped<br/>router.tsx:48-50,87-88"]
    ROUTER --> GP --> WORKER --> WASM
    ROUTER --> OLD
    DRIFT["DRIFT: architecture still names an in-tree Rust crate<br/>docs/architecture/explorer.md:10<br/>ADR-2001 retains historical-absent decision lineage<br/>docs/adr/ADR-2001-corpus-consolidation.md:112"]
    WASM -.-> DRIFT
```

**What it shows:** The retained SPA loads the graph through a lazy route and an external WASM package. The legacy redirect preserves query parameters but does not translate the path's node identity.

**Why it is this way:** knowledgeGraph ADR-2001 preserves missing historical decisions as absent rather than inventing authority; the package pin is the current dependency evidence.

## KG-04.2 Tier and transfer flow — overview is baked

```mermaid
sequenceDiagram
    participant S as scopeStore
    participant H as useSimulation
    participant W as physics worker
    participant P as DoubleBuffer
    S->>S: overview has no worker buffer<br/>scopeStore.ts:650-658
    S->>H: domain tier buffer or minted focus buffer<br/>scopeStore.ts:650-658
    H->>W: transfer a copy of the tier buffer<br/>useSimulation.ts:227-231
    W->>W: loadCsr and publish seed<br/>physics.worker.ts:246-266
    W->>P: acquire buffer, copy positions<br/>physics.worker.ts:166-183
    W->>H: transfer positions buffer<br/>physics.worker.ts:183
    H->>H: copy into stable positions view<br/>useSimulation.ts:145-155
    H->>W: transfer returned buffer<br/>useSimulation.ts:159-160
    W->>P: same generation: release<br/>physics.worker.ts:287-289
    Note over W,P: DEBT: release requires original object identity<br/>explorer/modern/src/workers/positionTransport.ts:76-82<br/>A real transferred return is a new object, two sends exhaust pool
    Note over S,W: Shared transport is disabled by default<br/>protocol.ts:153-155, overview has no simulation<br/>Rendered canvas uses demand frames: GraphCanvas.tsx:298
```

**What it shows:** Worker positions cross a transfer boundary; the retained pool rejects the returned objects. A Node MessageChannel probe reproduces exhaustion, while the visionGraph counterpart reclaims four successive transfers.

**Why it is this way:** The worker uses two reusable buffers to bound allocation; its object-identity ownership check conflicts with actual transfer semantics. This is a source and transport finding, not a deployed-browser measurement.

## KG-04.3 Scope contract — count ceilings after selection

```mermaid
flowchart TB
    DOMAIN["Domain: whole at 300 or fewer nodes;<br/>otherwise collapsed categories<br/>scopeStore.ts:215-224"]
    CUT["Expanded members are budget-truncated<br/>scopeStore.ts:303<br/>CSR edges stop at MAX_EDGES<br/>scopeStore.ts:172"]
    FOCUS["Focus: bounded ego selection then serialisation<br/>scopeStore.ts:466-508"]
    OVER["Overview arrays supplied by caller<br/>scopeStore.ts:512-543"]
    ASSERT["assertScope checks upper count bounds<br/>1500 nodes; 4000 edges; focus 300<br/>explorer/modern/src/types/scope.ts:144-157"]
    DOMAIN --> CUT --> ASSERT
    FOCUS --> ASSERT
    OVER --> ASSERT
    GAP["DEBT: no finite-integer or non-negative check;<br/>no array-length or endpoint validation here<br/>explorer/modern/src/types/scope.ts:144-157<br/>Inconsistent overview arrays can be published"]
    ASSERT --> GAP
    URL["Unknown domain decodes to overview<br/>explorer/modern/src/types/scope.ts:206-218"]
```

**What it shows:** Selection can truncate before the guard runs. Upper count limits are enforced, but the guard alone does not establish a structurally valid render model.

**Why it is this way:** The builders bound displayed detail to keep scopes manageable; stronger data-shape guarantees must come from another boundary and are not supplied by this assertion.

## KG-04.4 Page content — failure, fallback and cache boundaries

```mermaid
sequenceDiagram
    participant P as pageService
    participant I as search index
    participant A as page API
    participant M as title Markdown
    participant G as GitHub raw
    P->>I: fetch and resolve document<br/>pageService.ts:158-220
    alt document matched
        P->>A: slug JSON, awaited first<br/>pageService.ts:224
        P->>M: title-form Markdown<br/>pageService.ts:129-133,225
        alt non-OK, network error or invalid HTML-like body
            P->>G: same title on gh-pages<br/>pageService.ts:134-151
        end
        P->>P: Markdown else definition else no-detail message<br/>pageService.ts:233-243
    else no match or unavailable index
        P->>P: return concept-exists stub, no page API attempt<br/>pageService.ts:267-274
    end
    Note over P,G: DEBT: API body is not a content fallback,<br/>successful caches have no generation expiry<br/>pageService.ts:58-59,127,158-173,233-243
    Note over P,I: DEBT: malformed search document can throw during lookup<br/>pageService.ts:184-216<br/>Index outage is not distinguished from absent content
```

**What it shows:** Page metadata and prose are fetched sequentially after index resolution. External fallback applies beyond 404, and successful responses remain cached for the module lifetime.

**Why it is this way:** The reader combines structured metadata with cleaned Markdown. Its fallback paths favour displaying something, but do not preserve the distinction between missing content and failed retrieval.

## KG-04.6 Configured publisher uses a different viewer tree

```mermaid
flowchart TB
    KG["Retained knowledgeGraph viewer<br/>vowl-wasm v0.1.1<br/>explorer/modern/package.json:20<br/>Six domains: explorer/modern/src/types/scope.ts:73-80"]
    ARCH["Historical workflow is under archive<br/>archive/github-workflows/build.yml:1"]
    VG["visionGraph publication copy<br/>vowl-wasm v0.2.3 and ontomark-graph v0.2.0<br/>WasmVOWL/modern/package.json:24-25<br/>Nine domains: WasmVOWL/modern/src/types/scope.ts:76-86"]
    BUILD["Configured workflow builds visionGraph copy<br/>publish.yml:430-449"]
    SITE["Configured destination: knowledgeGraph gh-pages<br/>narrativegoldmine.com<br/>publish.yml:537-542"]
    POOL["Different pool reclaims transferred objects<br/>WasmVOWL/modern/src/workers/positionTransport.ts:49-94"]
    KG -.-> ARCH
    VG --> BUILD --> SITE
    VG --> POOL
    NOTE["INVARIANT: implementation findings belong to a source revision<br/>OPEN: workflow configuration does not prove a live deployment<br/>publish.yml:537-542"]
    SITE -.-> NOTE
```

**What it shows:** The inspected publisher chooses the visionGraph tree, whose dependencies, domains and pool differ from this retained viewer. The diagram establishes configured provenance, not the currently served site revision.

**Why it is this way:** knowledgeGraph ADR-2001 requires an identified consumer build. Repository names and a shared public destination do not make the two source trees interchangeable.
