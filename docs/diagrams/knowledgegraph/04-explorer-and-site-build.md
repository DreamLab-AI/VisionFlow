---
id: KG-04
title: Retired explorer — what left main, what stayed, and the one live viewer
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/adr/ADR-2001-corpus-consolidation.md
adrs: [ADR-2001, ADR-2002]
sources:
  - ../knowledgeGraph/docs/adr/ADR-2001-corpus-consolidation.md
  - ../knowledgeGraph/docs/adr/ADR-2002-corpus-is-the-ontology.md
  - ../knowledgeGraph/README.md
  - ../knowledgeGraph/LICENSING.md
  - ../knowledgeGraph/docs/architecture/explorer.md
  - ../knowledgeGraph/docs/reference/FORMAT-NGG1.md
  - ../knowledgeGraph/archive/github-workflows/build.yml
  - ../knowledgeGraph/.github/workflows/adr-ledger.yml
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/types/scope.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/stores/scopeStore.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/positionTransport.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/fetchArtefact.ts
  - ../visionGraph/.github/workflows/publish.yml
verified_commit: {knowledgegraph: 22c9bb8f54eafbc68faa3c0f2186bbae40364a93, visiongraph: dbc50f2f1075cab9fe108d4dcf917c4917d6f454}
---

**Retirement note — 2026-11-23:** knowledgeGraph PR 6 (`d77a41c`) removed the retained `explorer/` copy from main. Every panel below is redrawn at `22c9bb8`; the retired tree's findings stay readable at `7815bbc` in this topic's git history and in [KG-08](08-transfer-pool-repair.md).

## For developers

Nothing in knowledgeGraph builds a viewer any more. What stayed is the committed NGG1 tier data and its contract, now at docs/reference/FORMAT-NGG1.md; the only explorer source is visionGraph's publishing copy (VG-06), which this topic follows for scope, transfer and page boundaries.

## For the business

The public graph at narrativegoldmine.com/graph/ is built from one source tree only, so there is no second, stale copy to confuse with it. Findings about the old copy no longer describe anything on main.

## KG-04.1 What PR 6 removed, moved and kept

```mermaid
flowchart TB
    OLD["CHANGED at d77a41c (PR 6): explorer/ removed from main<br/>stale WasmVOWL copy on vowl-wasm v0.1.1, nothing built it<br/>README.md:95"]
    LAST["Retired tree readable at 7815bbc<br/>README.md:95, docs/architecture/explorer.md:3-11"]
    MOVE["Moved verbatim, still MIT<br/>docs/reference/FORMAT-NGG1.md and assets/images/ngm-hero.webp<br/>LICENSING.md:29, README.md:68-69"]
    WF["Only workflow left is the ADR ledger gate<br/>.github/workflows/adr-ledger.yml:1-4<br/>historical build is archived, archive/github-workflows/build.yml:1"]
    OLD --> LAST
    OLD --> MOVE
    OLD --> WF
    DRIFT["DOC-DRIFT: the moved contract still names explorer-relative consumers<br/>rust-wasm crate and modern/src/lib/ngg1.ts, docs/reference/FORMAT-NGG1.md:4<br/>its ADR-NG-001 link targets a historical-absent record<br/>docs/reference/FORMAT-NGG1.md:3, docs/adr/ADR-2001-corpus-consolidation.md:110-113"]
    MOVE --- DRIFT
```

**What it shows:** the retirement is a removal plus two moves; the architecture document keeps its old body behind a dated banner rather than being rewritten.
**Why it is this way:** the PR's stated reason is that nothing here built or deployed the copy (`README.md:95`); historical records keep their `explorer/` paths and resolve at `7815bbc`.

## KG-04.2 Tier and transfer flow — now only in the visionGraph copy

```mermaid
sequenceDiagram
    participant S as scopeStore
    participant W as physics worker
    participant P as DoubleBuffer
    S->>S: overview mints a worker buffer from overview.json<br/>scopeStore.ts:560
    S->>W: worker buffer, else the loaded domain tier<br/>scopeStore.ts:690
    W->>W: loadScope then loadCsr<br/>physics.worker.ts:282,290
    W->>P: acquire a free buffer<br/>physics.worker.ts:200
    W->>S: transfer positions<br/>physics.worker.ts:204
    S->>W: return the buffer<br/>physics.worker.ts:339
    W->>P: same generation only: release<br/>physics.worker.ts:341
    Note over W,P: INVARIANT: a return needs an outstanding transfer<br/>positionTransport.ts:84, counted, not identity-checked
    Note over S,W: CHANGED at visionGraph 2cf4a52: the overview is no longer baked,<br/>see VG-06.7. The retired copy's identical pool repair is KG-08
```

**What it shows:** the transfer contract this topic used to trace in the retained copy now exists only in the visionGraph tree, and the overview joins it.
**Why it is this way:** with the copy gone, the pool and overview behaviour that the public site depends on are the visionGraph ones; KG-08 keeps the retired repair at its own stamp.

## KG-04.3 Scope contract — count ceilings after selection

```mermaid
flowchart TB
    OVER["Overview arrays supplied by caller<br/>scopeStore.ts:522"]
    ASSERT["assertScope checks upper count bounds<br/>1500 nodes, 4000 edges, focus 300<br/>scope.ts:33-37,154-167"]
    OVER --> ASSERT
    GAP["DEBT: no finite-integer or non-negative check<br/>no array-length or endpoint validation here<br/>scope.ts:154-167, same gap as VG-06.5"]
    ASSERT --> GAP
    URL["Unknown domain decodes to overview<br/>scope.ts:216-229"]
    NOTE["CHANGED at d77a41c: the retained copy's six-domain guard<br/>left main with it, nine domains remain, scope.ts:77-87"]
    URL --- NOTE
```

**What it shows:** the guard that survives is the visionGraph one; it carries the same upper-bound-only limits the retained copy had.
**Why it is this way:** builders bound displayed detail; stronger data-shape guarantees still have to come from another boundary.

## KG-04.4 Page content — the surviving fetch path

```mermaid
flowchart TB
    IDX["Resolve the page through the search index<br/>pageService.ts:163"]
    GZ["Try gzip then raw representation<br/>fetchArtefact.ts:22"]
    IDX --> GZ
    REF["Failure, fallback and cache boundaries<br/>see VG-06.6"]
    GZ --> REF
    GONE["CHANGED at d77a41c: the retained copy's<br/>GitHub-raw fallback chain left main with explorer/<br/>README.md:95"]
    REF --- GONE
```

**What it shows:** page prose reaches readers only through the visionGraph reader; its debts are recorded once, in VG-06.6.
**Why it is this way:** keeping one reader removes the duplicated fallback rules this panel used to contrast.

## KG-04.6 Configured publisher builds the only viewer tree

```mermaid
flowchart TB
    VG["visionGraph publication copy<br/>vowl-wasm v0.4.0 and ontomark-graph v0.2.0<br/>WasmVOWL/modern/package.json:24-25<br/>CHANGED at 7808310: was vowl-wasm v0.2.3"]
    BUILD["Configured workflow builds it<br/>publish.yml:441-458"]
    SITE["Configured destination: knowledgeGraph gh-pages<br/>narrativegoldmine.com<br/>publish.yml:546-551"]
    KG["knowledgeGraph main holds data and contract, no viewer<br/>README.md:95"]
    VG --> BUILD --> SITE
    KG -.->|"gh-pages target only"| SITE
    NOTE["INVARIANT: implementation findings belong to a source revision<br/>OPEN: workflow configuration does not prove a live deployment<br/>publish.yml:546-551"]
    SITE -.-> NOTE
```

**What it shows:** the inspected publisher builds the visionGraph tree into knowledgeGraph's gh-pages; there is no longer a second tree to distinguish.
**Why it is this way:** knowledgeGraph ADR-2001 requires an identified consumer build; the retirement makes that identity unambiguous without proving which revision is served.
