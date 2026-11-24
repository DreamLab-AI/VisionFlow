---
id: WV-03
title: Ontology input, the owl2vowl normaliser and the NggExplorer layout boundary
area: wasmvowl
governing:
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
adrs: [WasmVOWL:ADR-001]
sources:
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
  - ../WasmVOWL/modern/package.json
  - ../WasmVOWL/modern/package-lock.json
  - ../WasmVOWL/modern/vite.config.ts
  - ../WasmVOWL/modern/src/components/UI/FileDropZone.tsx
  - ../WasmVOWL/modern/src/types/ontology.ts
  - ../WasmVOWL/modern/src/stores/useGraphStore.ts
  - ../WasmVOWL/modern/src/hooks/useWasmSimulation.ts
  - ../WasmVOWL/modern/src/lib/owl2vowl.ts
  - ../WasmVOWL/modern/src/lib/layoutGraph.ts
  - ../WasmVOWL/modern/src/lib/debugHandle.ts
  - ../WasmVOWL/modern/src/stores/useUIStore.ts
  - ../WasmVOWL/modern/src/types/ui.ts
  - ../ontomark/packages/graph/src/layout.js
  - ../WasmVOWL/modern/public/ontologies/minimal.json
  - ../WasmVOWL/legacy/src/app/data/foaf.json
  - ../WasmVOWL/legacy/src/webvowl/js/parser.js
verified_commit: {wasmvowl: 2bb54b3ee2dc3a7eda34b60d9f08fdb0fb535aab, ontomark: 8222170c5243f58f1f9a93ac800660dfe0197ac4}
---

## WV-03.1 Two JSON dialects, one normaliser

```mermaid
flowchart TB
    subgraph O2V["owl2vowl JSON, as the legacy app ships it"]
        LC["class entries carry only id and type<br/>legacy/src/app/data/foaf.json:47-51"]
        LCA["labels and IRIs live in classAttribute<br/>legacy/src/app/data/foaf.json:273-280"]
        LPA["domain and range live in propertyAttribute, as scalar ids<br/>legacy/src/app/data/foaf.json:1423,1427,1450"]
    end
    subgraph HW["hand-written samples under modern/public"]
        MC["class entries with inline label and iri<br/>modern/public/ontologies/minimal.json:18-30"]
        MP["properties with inline domain and range arrays<br/>modern/public/ontologies/minimal.json:59-72"]
    end
    subgraph NRM["normaliseOwl2Vowl, owl2vowl.ts:289"]
        GATE["only .json names reach it, FileDropZone.tsx:63-64<br/>a non-array class or property field throws, owl2vowl.ts:239-245"]
        MRG["merge base and attribute records, base fields win<br/>owl2vowl.ts:249-256,303-306,355"]
        LBL["label from the document, then the IRI fragment, then a type name, then the id<br/>owl2vowl.ts:324-330"]
        END["missing ends from the inverse, else a shared owl:Thing node<br/>owl2vowl.ts:392-408"]
        KIND["edge kind by lower-cased type, subclass, disjoint and set operators kept<br/>owl2vowl.ts:177-191,385"]
        DROP["edges naming no node are dropped into report.dropped<br/>owl2vowl.ts:511-513"]
    end
    O2V --> GATE
    HW --> GATE
    GATE --> MRG --> LBL --> END --> KIND --> DROP
    T1["CHANGED at 2bb54b3: classAttribute and propertyAttribute are now merged,<br/>so owl2vowl files load with labels and edges. The legacy parser's merge,<br/>parser.js:141-149, is the model the header names, owl2vowl.ts:16-17"]
    T2["DOC-DRIFT: the component header still promises JSON/RDF files, FileDropZone.tsx:3"]
    DROP --- T1
    GATE --- T2
```

owl2vowl's `owl:objectProperty` spelling (`legacy/src/app/data/foaf.json:1128`) now maps to `objectProperty`, because the lookup lower-cases the type (`owl2vowl.ts:385`).

## WV-03.2 One normalised object feeds the store and the layout

```mermaid
sequenceDiagram
    autonumber
    participant FDZ as ingest<br/>FileDropZone.tsx:29
    participant GS as loadOntology<br/>useGraphStore.ts:216
    participant H as useWasmSimulation<br/>useWasmSimulation.ts:50
    participant LG as toLayoutGraph<br/>layoutGraph.ts:31
    participant D as createLayout driver<br/>ontomark-graph 0.4.0
    FDZ->>GS: normalised ontology, FileDropZone.tsx:30-31
    GS->>GS: Maps built from it, the object kept whole, graphVersion plus one, useGraphStore.ts:232-283
    GS-->>H: graphVersion changed, useWasmSimulation.ts:60
    H->>LG: the store's ontology, useWasmSimulation.ts:147,154
    LG-->>H: nodes in store order, every edge as source to target, ids for snapshots, layoutGraph.ts:32-39
    H->>D: setGraph(graph), useWasmSimulation.ts:158
    Note over D: EXTERNAL: vowlEngine sends every edge as one undirected spring,<br/>drops self-loops and merges repeated pairs, layoutGraph.ts:13-17
    D-->>H: onPositions xy, index i is ids i, useWasmSimulation.ts:82-85
    H->>GS: updateNodePositions(ids, xy), one store update per snapshot, useGraphStore.ts:331-339
    Note over GS,LG: CHANGED at 2bb54b3: the WebVowl JSON re-serialisation is gone, store and layout share one object<br/>ADR-001 records the move, ADR-001-explorer-consumer-contract-closeout.md:174-176
```

The engine types now come from the package (`useWasmSimulation.ts:15-21`), so a drift in the driver's API is a `tsc` error, not a silent mismatch.

## WV-03.3 Hook lifecycle: init, one cold load per file, live settings

```mermaid
flowchart TB
    V["Vite excludes the package from dependency pre-bundling, wasm and top-level-await plugins<br/>vite.config.ts:8-20"] --> I1
    I1["mount: dynamic import of @dreamlab-ai/vowl-wasm, await default()<br/>useWasmSimulation.ts:71-72"] --> I2["one driver: createLayout(vowlEngine(mod)) with settings.layout as params<br/>useWasmSimulation.ts:75-106"]
    I2 --> L1["per graphVersion: setGraph of an empty graph first, so each file starts cold<br/>useWasmSimulation.ts:134-145"]
    L1 --> L2["then setGraph of the new graph, isRunning true<br/>useWasmSimulation.ts:154-158"]
    L2 --> F1["driver ticks in time-boxed frame slices and reports snapshots<br/>onSettled ends isRunning, useWasmSimulation.ts:86-89"]
    I2 --> S1["settings effect sends only the changed keys with setParams<br/>useWasmSimulation.ts:172-184"]
    I2 --> ERR["onError and onIgnoredParams become notifications<br/>useWasmSimulation.ts:90-105"]
    I2 --> DBG["?debug handle, development builds only<br/>useWasmSimulation.ts:123, debugHandle.ts:28-42"]
    INV["INVARIANT: the engine reloads only on a new graphVersion, which load and clear alone bump<br/>useWasmSimulation.ts:169, useGraphStore.ts:283,303"]
    L2 --- INV
    C1["CHANGED at 2bb54b3: the old per-frame reload risk is gone, positions no longer<br/>re-trigger the load effect. useFrame no longer drives the layout"]
    F1 --- C1
    END["unmount: layout.destroy releases the explorer<br/>useWasmSimulation.ts:125-130"]
    F1 --> END
```

## WV-03.4 What the standalone takes from the engine, and which release

```mermaid
flowchart TB
    PIN["vowl-wasm 0.4.0 exact from npm, ontomark-graph 0.4.0 as a release tarball URL<br/>modern/package.json:19-20"] --> LOCK["lock: both at 0.4.0 with sha512 integrity<br/>modern/package-lock.json:551-561"]
    LOCK --> PAR["defaults: charge 2000, linkDistance 60, RECOMMENDED_LAYOUT_PARAMS,<br/>collideRadius 23 in place of the recommended 21<br/>useUIStore.ts:41-52, layout.js:84"]
    PAR --> GATE["collision and componentGravity keys are gated on vowl-wasm 0.4.0<br/>layout.js:86-92, ignored keys reach onIgnoredParams, useWasmSimulation.ts:99-105"]
    PAR --> KEYS["LayoutSettings: charge, linkDistance, collideRadius, collideStrength,<br/>collideIterations, componentGravity, types/ui.ts:41-55"]
    GATE --> USED["used: NggExplorer through vowlEngine only, useWasmSimulation.ts:77<br/>CHANGED at 2bb54b3: the WebVowl simulator is no longer called here"]
    USED --> OTHER["EXTERNAL: the visionGraph copy runs NggExplorer on vowl-wasm 0.4.0 too,<br/>see VW-04.1 and VG-06.1"]
    PAR --> D1["DEBT: one collision radius for every node, ontovibe's instance-scaled node<br/>still overlaps, ADR-001-explorer-consumer-contract-closeout.md:265,273"]
```
