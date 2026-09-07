---
id: KG-04
title: Explorer — NGG1 tier reader, worker transport, React SPA, and the externalised WASM crate
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
adrs: [ADR-2001]
sources:
  - ../knowledgeGraph/explorer/modern/package.json
  - ../knowledgeGraph/explorer/modern/src/router.tsx
  - ../knowledgeGraph/explorer/modern/src/types/scope.ts
  - ../knowledgeGraph/explorer/modern/src/workers/protocol.ts
  - ../knowledgeGraph/explorer/modern/src/api/pageService.ts
  - ../knowledgeGraph/explorer/FORMAT-NGG1.md
  - ../knowledgeGraph/explorer/license.txt
  - ../knowledgeGraph/docs/architecture/explorer.md
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
  - ../knowledgeGraph/docs/ecosystem.md
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../visionGraph/.github/workflows/publish.yml
verified_commit: 2791111fc
---

## KG-04.1 Explorer topology — MIT viewer over AGPL-built, ODbL-licensed data

```mermaid
flowchart TB
    subgraph MODERN["explorer/modern — React 19 + Vite 6 + R3F 9<br/>package.json: webvowl-modern"]
        ROUTER["router.tsx — routes: / /page/:id /graph<br/>/search /data /about; /ontology* redirects to /graph<br/>router.tsx:35-40"]
        GP["GraphPage — the ONLY route that pulls<br/>WASM + worker + renderer chunks<br/>router.tsx:9,37"]
        WORKER["physics.worker.ts — drives NggExplorer"]
    end
    subgraph WASM["@dreamlab-ai/vowl-wasm — EXTERNAL package<br/>explorer/modern/package.json:20"]
        NGGEXP["NggExplorer wasm_bindgen class<br/>loadCsr/tick/isFinished/positionsPtr — see VW-*"]
    end
    ROUTER --> GP --> WORKER --> NGGEXP
    note1["DOC-DRIFT: BASELINE-narrativegoldmine.md:14 and architecture/explorer.md:10<br/>both cite explorer/rust-wasm/src/ngg1.rs as an IN-TREE crate. At this commit<br/>there is no rust-wasm/ directory in explorer/ at all — explorer/modern/package.json:20<br/>fetches '@dreamlab-ai/vowl-wasm' as a GitHub release TARBALL v0.1.1<br/>#40;GitHub release, not npm#41;. The reader now lives in the sibling vowl-wasm repo —<br/>EXTERNAL: see VW-* for the crate itself. DIVERGENCE: the copy that actually SHIPS<br/>to narrativegoldmine.com is a DIFFERENT tree — visionGraph's publishing-tools/<br/>WasmVOWL/modern/package.json:20 pins the SAME package as npm '0.1.2', not this<br/>tarball's v0.1.1 — see KG-04.6 and VG-06"]
```

## KG-04.2 NGG1 tier data flow — fetch → parse → worker tick → instanced render

```mermaid
sequenceDiagram
    autonumber
    participant GP as GraphPage
    participant SCOPE as scopeStore<br/>assertScope on every construction
    participant WK as physics.worker.ts
    participant NGG as NggExplorer #40;@dreamlab-ai/vowl-wasm#41;
    participant CANVAS as GraphCanvas — orthographic, frameloop='demand'

    GP->>GP: fetch tier for current URL<br/>T0 overview.json · T1 domain-*.bin · T2 ego #40;client-derived#41;
    GP->>SCOPE: build GraphScope + RenderModel
    SCOPE->>SCOPE: assertScope#40;#41; — throws RangeError over MAX_NODES/MAX_EDGES/FOCUS_MAX<br/>scope.ts:144-155
    SCOPE->>WK: mint NGG1 buffer #40;T0/T2 serialised client-side#41;
    WK->>NGG: loadCsr#40;buffer#41;, then tick loop #40;16ms#41;
    NGG-->>WK: positions #40;Float32Array over WASM linear memory#41;
    WK->>CANVAS: transferable ping-pong #40;positionTransport.ts#41;
    CANVAS->>CANVAS: read live buffer per frame, write instance matrices<br/>zero allocation in useFrame
    Note over WK,CANVAS: DIVERGENCE: SharedArrayBuffer transport exists on both sides but<br/>is HARD-DISABLED — canUseSharedMemory#40;#41; unconditionally returns false<br/>#40;protocol.ts:153#41; — a half-written frame amplified to ~1e20 and blanked<br/>the view. Re-enable only behind a double-buffered SAB + Atomics flip.
```

## KG-04.3 Scope contract — the caps assertScope enforces

```mermaid
flowchart LR
    INV["INVARIANTS — scope.ts:30-36"]
    MN["MAX_NODES: 1500"]
    ME["MAX_EDGES: 4000"]
    FM["FOCUS_MAX: 300"]
    INV --> MN & ME & FM
    ASSERT["assertScope#40;#41; — scope.ts:144<br/>throws RangeError on any over-budget scope"]
    MN --> ASSERT
    ME --> ASSERT
    FM --> ASSERT
    note1["INVARIANT: caps are named CONTRACT, not tuning — construction of a<br/>GraphScope above them must be rejected, never silently truncated<br/>(scopeStore.ts is documented as the only place a GraphScope is built)"]
```

## KG-04.4 Page content — fetched by slug (structured) and by title (prose), independently

```mermaid
sequenceDiagram
    autonumber
    participant PS as pageService.ts
    participant IDX as /api/search-index.json — 6 resolution strategies
    participant PAGEAPI as api/pages/SLUG.json
    participant MDAPI as api/markdown/TITLE.md
    participant RAW as raw.githubusercontent.com fallback

    PS->>IDX: resolve page name — exact id, ci title, labels[],<br/>IRI/suffix, camelCase, hyphen-form
    IDX-->>PS: slug + title
    PS->>PAGEAPI: fetch#40;/api/pages/${slug}.json#41; — pageService.ts:108
    PS->>MDAPI: fetch#40;/api/markdown/${title}.md#41; — pageService.ts:133 TITLE-form
    alt primary 404s
        PS->>RAW: fallback to gh-pages raw markdown — pageService.ts:134
    end
    Note over PS: isValidMarkdown rejects HTML content-type, sessionStorage.setItem<br/>bodies, or a DOCTYPE — detects a static host 404-ing to index.html
    Note over PS: DIVERGENCE: jsonld_to_page_api.py writes a SLUG-form mirror<br/>#40;api/markdown/SLUG.md#41; that pageService.ts never requests — see KG-05.4
```

## KG-04.5 Licence — MIT viewer, deliberately not AGPL

```mermaid
flowchart TB
    UPSTREAM["VisualDataWeb/WebVOWL<br/>Copyright 2014-2019 Link/Lohmann/Marbach/Negru/Wiens"]
    WASMVOWL["DreamLab-AI/WasmVOWL — MIT fork<br/>ecosystem.md:134-149"]
    HERE["explorer/ in THIS repo<br/>license.txt — identical MIT text"]
    VW["vowl-wasm repo — the NGG1 reader crate<br/>published as @dreamlab-ai/vowl-wasm — EXTERNAL: see VW-*"]
    UPSTREAM --> WASMVOWL --> HERE
    WASMVOWL -.->|"same MIT-derivative lineage"| VW
    note1["INVARIANT: explorer/ stays MIT deliberately — AGPL-ing it would be<br/>hollow while the identical WebVOWL-derived code is MIT one repo away<br/>(architecture/explorer.md:302-308)"]
```

## KG-04.6 This copy never ships — the deployed explorer is a DIFFERENT, diverged tree

```mermaid
flowchart TB
    subgraph HERE["knowledgeGraph/explorer/modern — THIS diagram's subject"]
        HPKG["explorer/modern/package.json:20<br/>@dreamlab-ai/vowl-wasm — GitHub RELEASE TARBALL<br/>v0.1.1-pkg.tgz"]
    end
    subgraph SHIP["visionGraph/publishing-tools/WasmVOWL/modern — the SHIPPING tree<br/>EXTERNAL: see VG-06"]
        SPKG["WasmVOWL/modern/package.json:20<br/>@dreamlab-ai/vowl-wasm — npm registry<br/>'0.1.2'"]
    end
    BUILDYML["this repo's build.yml<br/>builds pipeline/ only — NO wasm-pack, NO vite,<br/>NO deploy step (KG-05.1)"]
    PUBYML["visionGraph publish.yml:151<br/>builds THIS SPA — EXTERNAL VG-03.3"]
    SITE["narrativegoldmine.com<br/>external_repository DreamLab-AI/knowledgeGraph gh-pages<br/>EXTERNAL publish.yml:279"]
    HERE -.->|"built by NOTHING in this repo"| BUILDYML
    SHIP --> PUBYML --> SITE
    note1["DOC-DRIFT: KG-04.1 through KG-04.5 document explorer/modern as THE explorer —<br/>accurate for what the source code says, but this copy is never built or<br/>deployed by anything in this repository. The tree that actually ships is<br/>visionGraph's publishing-tools/WasmVOWL/modern #40;180 files#41; — see VG-06 for<br/>its own topic. The two trees have DIVERGED on the one dependency both pin:<br/>HPKG v0.1.1 #40;tarball#41; vs SPKG '0.1.2' #40;npm#41;"]
```
