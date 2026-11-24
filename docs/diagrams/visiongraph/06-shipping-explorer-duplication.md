---
id: VG-06
title: Publishing explorer — shared renderer, corpus adapters, layout settings and the retired copy
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-007]
sources:
  - ../visionGraph/publishing-tools/WasmVOWL/CLAUDE.md
  - ../visionGraph/publishing-tools/WasmVOWL/CAPABILITIES.md
  - ../visionGraph/publishing-tools/WasmVOWL/Dockerfile
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/lib/nggGraph.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/router.tsx
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/site/mesh.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/types/scope.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/site/domains.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/fetchArtefact.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/positionTransport.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/stores/scopeStore.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/build/runtimeNotices.ts
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/docs/adr/ADR-VG-007-shared-ontology-workspace-and-profile.md
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/protocol.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/pages/GraphPage.tsx
  - ../knowledgeGraph/README.md
  - ../knowledgeGraph/LICENSING.md
verified_commit: {visiongraph: dbc50f2f1075cab9fe108d4dcf917c4917d6f454, knowledgegraph: 22c9bb8f54eafbc68faa3c0f2186bbae40364a93}
---

## For developers

The configured publisher builds this host, whose renderer and codec come from pinned releases. Since knowledgeGraph retired its stale explorer/ copy (PR 6, 2026-11-23) this is the only explorer source in the estate; the duplication this topic was named for is gone.

## For the business

Readers can explore relationships and open corpus prose, but a successful graph build does not establish that every page is current or available. The dated audit separates passing interaction tests from unresolved content and deployment acceptance.

## VG-06.1 Configured publisher and retained sibling

```mermaid
flowchart TB
    %% Source: ../visionGraph/.github/workflows/publish.yml:441
    A["visionGraph publisher builds<br/>publishing-tools/WasmVOWL/modern"]
    %% Source: ../visionGraph/.github/workflows/publish.yml:460
    A --> B["Copy built shell/assets into www<br/>preserve authoritative machine trees"]
    %% Source: ../visionGraph/.github/workflows/publish.yml:544
    B --> C["Deploy assembled site to<br/>knowledgeGraph gh-pages"]
    %% Source: ../knowledgeGraph/README.md:95
    %% Source: ../knowledgeGraph/LICENSING.md:28-29
    D["CHANGED at knowledgeGraph d77a41c (PR 6):<br/>explorer/ copy on vowl-wasm 0.1.1 removed from main<br/>last present at 7815bbc, FORMAT-NGG1.md and hero moved out"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/package.json:24-25
    A --> E["Configured host uses shared graph 0.2.0<br/>and vowl-wasm 0.4.0<br/>CHANGED at 7808310: was vowl-wasm 0.2.3"]
    D -.->|"only explorer source left"| A
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/Dockerfile:6
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/Dockerfile:13
    F["Legacy Dockerfile downloads WebVOWL WAR<br/>Tomcat path is separate from this publisher"]
    %% Source: ../visionGraph/.github/workflows/publish.yml:57
    %% Source: ../visionGraph/.github/workflows/publish.yml:335
    H["Publisher runner pinned to Ubuntu 24.04<br/>explorer dependencies and Chromium installed<br/>before the long Quartz render"]
    A --> H
```

**What it shows:** The workflow selects a concrete source tree and distribution repository. It does not identify the revision currently served by the public site.
**Why it is this way:** ADR-VG-007 keeps the site host and deployment here while extracting reusable graph software. KG-04 records the sibling's retirement. The runner pin and early browser install follow a failed run recorded in ADR-VG-007 (`docs/adr/ADR-VG-007-shared-ontology-workspace-and-profile.md:188-195`).

## VG-06.2 Shared renderer and codec, with host-owned interaction

```mermaid
flowchart TB
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/lib/nggGraph.ts:7
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/lib/nggGraph.ts:49
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/lib/nggGraph.ts:94
    A["Released Rust/WASM validates NGG1<br/>adapter copies metadata and frees graph"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:55
    A --> B["Host maps scope nodes, colours and edges<br/>into shared GraphData"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:124
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:139
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:134
    B --> C["mountGraph and setGraph<br/>same-generation filters preserve viewport<br/>destroy view on unmount"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:106
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:119-121
    C --> D["Selection and activation update host store<br/>drag mutates scope positions<br/>simulated tiers pin and unpin nodes"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:47
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:161
    E["Any scope with a worker buffer simulates<br/>finite bounded snapshots update view<br/>CHANGED at 2cf4a52: overview included, see VG-06.7"] --> C
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/positionTransport.ts:82
    E --> F["Transfer pool counts in-flight ownership<br/>does not require returned object identity"]
```

**What it shows:** The publishing host has adopted shared drawing and Rust binary ownership. Local state, physics and content integration remain host responsibilities.
**Why it is this way:** ADR-VG-007 separates shared graph ownership from application policy. Current tests exercise real shared modules with DOM/canvas stubs, not a deployed browser.

## VG-06.3 Historical capability measurements and current build boundary

```mermaid
flowchart TB
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/CAPABILITIES.md:24
    A["Historical capability table<br/>Python pipeline, 7444 pages<br/>six domains and July measurements"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/CAPABILITIES.md:50
    A --> B["Historical claim: focus does not simulate"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:47
    B -.-> C["DRIFT: current host simulates every scope<br/>with a worker buffer, overview included"]
    %% Source: ../visionGraph/.github/workflows/publish.yml:48
    %% Source: ../visionGraph/.github/workflows/publish.yml:175
    D["Current publisher uses ontomark v0.9.0<br/>all-vault public build"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/package.json:8
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/package.json:24-25
    D --> E["npm installs released graph and WASM<br/>local Rust builds runtime-notices tool"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/build/runtimeNotices.ts:12
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/build/runtimeNotices.ts:22
    E --> F["Rollup collects rendered browser/worker modules<br/>Rust emits runtime inventory and licence text"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/CLAUDE.md:8
    G["DRIFT: guide still claims React Three Fiber<br/>feature completion and historical FPS"]
```

**What it shows:** The old measured capability table and broad guide claims do not describe the current producer or renderer. Local Rust still builds the notice tool, not the published engine.
**Why it is this way:** The extraction decision changes component ownership. Historical timings and vulnerability counts need dated, scope-specific evidence rather than promotion into current guarantees.

## VG-06.4 Current links and route behaviour

```mermaid
flowchart TB
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/site/mesh.ts:15
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/site/mesh.ts:22
    A["Publishing host: nine MESH_REPOS<br/>Loom and sidestr-rs included"]
    %% Source: ../knowledgeGraph/README.md:95
    B["CHANGED at knowledgeGraph d77a41c:<br/>the sibling's seven-entry link table<br/>left main with explorer/"]
    A -.->|"one link table now"| B
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/router.tsx:64
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/router.tsx:135
    C["Publishing router uses Vite BASE_URL<br/>trimmed to browser-router basename"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts:169
    C --> D["Data requests remain root-absolute<br/>router basename is not data relocation"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/router.tsx:48
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/router.tsx:108
    C --> E["Legacy ontology node route redirects<br/>query retained, path node identity omitted"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/router.tsx:19
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/router.tsx:32
    C --> F["DEBT: chunk reload flag cleared immediately<br/>and at module load, not a durable retry budget"]
```

**What it shows:** The earlier missing-Loom statement is obsolete. Sub-path routing, root data paths and legacy identity handling are separate contracts.
**Why it is this way:** Site chrome and routes remain local under ADR-VG-007; a mesh hyperlink is not evidence of live Loom integration or consumer convergence.

## VG-06.5 Nine domain IDs and bounded scope validation

```mermaid
flowchart TB
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/types/scope.ts:77-87
    A["Publishing host: nine ordered domain slugs<br/>index is NGG1 domain ID"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/site/domains.ts:60
    A --> B["Descriptor table derives IDs, labels<br/>icons and tints from that order"]
    %% Source: ../knowledgeGraph/README.md:95
    C["Removed sibling held six slugs, same first six IDs<br/>no second domain table on main now"]
    C -.->|"retired d77a41c"| A
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/stores/scopeStore.ts:522
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/types/scope.ts:154
    D["Overview builder copies supplied arrays<br/>assertScope enforces count ceilings"]
    D --> E["DEBT: count ceilings do not establish<br/>finite integer counts, array lengths<br/>or valid overview edge endpoints"]
```

**What it shows:** Domain order remains a public identity contract. Binary codec validation and the JSON overview builder have different validation boundaries.
**Why it is this way:** The profile and consumer taxonomy must agree without renumbering existing IDs. Passing domain presence gates does not establish a structurally valid graph.

## VG-06.6 Prose lookup, compression and freshness limits

```mermaid
flowchart TB
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts:163
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts:189
    A["Resolve requested page through search index<br/>successful index is cached in module"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts:229
    A --> B["Fetch page JSON then title-based Markdown<br/>successful responses have module caches"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/fetchArtefact.ts:22
    B --> C["Try gzip then raw representation<br/>gzip magic selects decompression stream"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/fetchArtefact.ts:29
    C --> D["DEBT: corrupt stream can fail on body read<br/>after fetch helper has returned<br/>raw fallback is then not attempted"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts:238
    B --> E["Use mirror prose, else definition<br/>and a link to Quartz full page"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts:276
    A --> F["DEBT: unavailable or unmatched index<br/>returns concept-exists stub<br/>without a page lookup"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts:61
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/api/pageService.ts:113
    B --> G["DEBT: caches have no generation identity<br/>long-lived session can retain old prose"]
```

**What it shows:** The current host removes the external raw-GitHub fallback and adds gzip support, but still conflates some outages with missing content and retains successful responses indefinitely.
**Why it is this way:** These are current consumer boundaries reproduced with injected responses. They do not establish a live-site outage or override the separate publication identity findings.

## VG-06.7 Explorer layout settings and the settling overview

```mermaid
flowchart TB
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/protocol.ts:125-130
    L["EXPLORER_LAYOUT: collideRadius 21, collideStrength 0.7<br/>collideIterations 3, componentGravity 0.1<br/>CHANGED at 7808310: vowl-wasm 0.4.0 keys"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts:166
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts:170
    L --> W["Worker holds sticky params, seeded from EXPLORER_LAYOUT<br/>applyLayoutParams pushes each via setParam"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts:290-291
    W --> R["Re-applied after every loadCsr<br/>so each scope gets the same layout"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/workers/physics.worker.ts:320
    W --> S["A later setParam updates the sticky map<br/>reheat is one-shot and never stored"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/stores/scopeStore.ts:560
    O["CHANGED at 2cf4a52 (PR 5): buildOverviewScope mints<br/>an NGG1 worker buffer from overview.json<br/>baked positions are only the warm-start seed"] --> R
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:164-165
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:126-129
    R --> C["CHANGED at be8f9de (PR 5): overview camera fits every frame<br/>until wheel, pointerdown or keydown in that generation<br/>other tiers fit on the first frame only"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/pages/GraphPage.tsx:5
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/CAPABILITIES.md:22
    C --- N["GraphPage route comment now reads overview T0 worker-settled<br/>DOC-DRIFT: CAPABILITIES.md still says the worker drives domain scopes"]
```

**What it shows:** every tier, the overview included, now settles in the worker under one sticky set of vowl-wasm 0.4.0 collision and component-gravity settings; the overview camera follows the contraction until the user takes the view.
**Why it is this way:** ADR-VG-007's 2026-11-23 notes record that the baked overview fitted to a laptop viewport overlapped its hub discs; radius 21 matches ontomark-graph's largest 12 px disc (`protocol.ts:113-124`).
