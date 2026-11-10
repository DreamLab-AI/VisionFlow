---
id: VG-06
title: Publishing explorer — shared renderer, corpus adapters and retained-copy drift
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
  - ../knowledgeGraph/explorer/modern/package.json
  - ../knowledgeGraph/explorer/modern/src/site/mesh.ts
  - ../knowledgeGraph/explorer/modern/src/types/scope.ts
verified_commit: {visiongraph: d8277151a1404b456b5c7206187f7e4b50f25be1, knowledgegraph: 5294e88b6602a63b5afe96c79f2779d00274a05e}
---

## For developers

The configured publisher builds this host, whose renderer and codec come from pinned releases. Read the adapters and content path separately from the retained knowledgeGraph copy; their dependency, domain and transport contracts differ.

## For the business

Readers can explore relationships and open corpus prose, but a successful graph build does not establish that every page is current or available. The dated audit separates passing interaction tests from unresolved content and deployment acceptance.

## VG-06.1 Configured publisher and retained sibling

```mermaid
flowchart TB
    %% Source: ../visionGraph/.github/workflows/publish.yml:430
    A["visionGraph publisher builds<br/>publishing-tools/WasmVOWL/modern"]
    %% Source: ../visionGraph/.github/workflows/publish.yml:450
    A --> B["Copy built shell/assets into www<br/>preserve authoritative machine trees"]
    %% Source: ../visionGraph/.github/workflows/publish.yml:535
    B --> C["Deploy assembled site to<br/>knowledgeGraph gh-pages"]
    %% Source: ../knowledgeGraph/explorer/modern/package.json:20
    D["Retained knowledgeGraph/explorer<br/>vowl-wasm 0.1.1 and React Three Fiber"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/package.json:24
    A --> E["Configured host uses shared graph 0.2.0<br/>and vowl-wasm 0.2.3"]
    D -.->|"different source and package contracts"| E
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/Dockerfile:6
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/Dockerfile:13
    F["Legacy Dockerfile downloads WebVOWL WAR<br/>Tomcat path is separate from this publisher"]
```

**What it shows:** The workflow selects a concrete source tree and distribution repository. It does not identify the revision currently served by the public site.
**Why it is this way:** ADR-VG-007 keeps the site host and deployment here while extracting reusable graph software. The retained sibling is assessed separately in KG-04.

## VG-06.2 Shared renderer and codec, with host-owned interaction

```mermaid
flowchart TB
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/lib/nggGraph.ts:7
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/lib/nggGraph.ts:49
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/lib/nggGraph.ts:94
    A["Released Rust/WASM validates NGG1<br/>adapter copies metadata and frees graph"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:52
    A --> B["Host maps scope nodes, colours and edges<br/>into shared GraphData"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:121
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:126
    B --> C["mountGraph and setGraph<br/>same-generation filters preserve viewport<br/>destroy view on unmount"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:103
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:116
    C --> D["Selection and activation update host store<br/>drag mutates scope positions<br/>simulated tiers pin and unpin nodes"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:44
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:148
    E["Domain and focus may simulate<br/>finite bounded snapshots update view"] --> C
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
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx:44
    B -.-> C["DRIFT: current host simulates<br/>domain and focus scopes"]
    %% Source: ../visionGraph/.github/workflows/publish.yml:48
    %% Source: ../visionGraph/.github/workflows/publish.yml:174
    D["Current publisher uses ontomark v0.8.0<br/>all-vault public build"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/package.json:8
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/package.json:24
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
    %% Source: ../knowledgeGraph/explorer/modern/src/site/mesh.ts:15
    %% Source: ../knowledgeGraph/explorer/modern/src/site/mesh.ts:38
    B["Retained sibling: seven MESH_REPOS<br/>separate LOOM export"]
    A -.->|"different link tables"| B
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
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/types/scope.ts:75
    A["Publishing host: nine ordered domain slugs<br/>index is NGG1 domain ID"]
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/site/domains.ts:60
    A --> B["Descriptor table derives IDs, labels<br/>icons and tints from that order"]
    %% Source: ../knowledgeGraph/explorer/modern/src/types/scope.ts:73
    C["Retained sibling: six ordered slugs<br/>same first six IDs"]
    C -.->|"three extra domains"| A
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/stores/scopeStore.ts:512
    %% Source: ../visionGraph/publishing-tools/WasmVOWL/modern/src/types/scope.ts:153
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
