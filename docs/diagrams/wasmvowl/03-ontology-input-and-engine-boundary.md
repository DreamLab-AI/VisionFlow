---
id: WV-03
title: Ontology input formats and the vowl-wasm engine boundary
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
  - ../WasmVOWL/modern/public/ontologies/minimal.json
  - ../WasmVOWL/legacy/src/app/data/foaf.json
  - ../WasmVOWL/legacy/src/webvowl/js/parser.js
verified_commit: {wasmvowl: af25086a605482c8234580ccd1062cf506e0b181}
---

## WV-03.1 Two JSON dialects, one loader

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
    subgraph LD["modern loader"]
        GATE["only .json names, picker accepts .json<br/>FileDropZone.tsx:59-60,205"]
        REQ["a class array is the only structural check<br/>FileDropZone.tsx:30-31"]
        MAP["label = label.en, label.default or id, useGraphStore.ts:192<br/>source = domain[0], target = range[0], useGraphStore.ts:209-210"]
        DROP["an edge without both ends is discarded<br/>useGraphStore.ts:219-222"]
    end
    O2V --> GATE
    HW --> GATE
    GATE --> REQ --> MAP --> DROP
    T1["TENSION: classAttribute and propertyAttribute are typed, types/ontology.ts:53-54,<br/>but never read, so an owl2vowl file loads with ids as labels and no edges.<br/>The legacy parser merges them, parser.js:141-149, see WV-05.3"]
    T2["DOC-DRIFT: the component header promises JSON/RDF files, FileDropZone.tsx:3"]
    DROP --- T1
    GATE --- T2
```

The edge type test matches only `owl:ObjectProperty` (`useGraphStore.ts:211`); owl2vowl writes `owl:objectProperty` (`legacy/src/app/data/foaf.json:1128`), which falls through to `datatypeProperty`.

## WV-03.2 The store schema is not the schema the engine accepts

```mermaid
sequenceDiagram
    autonumber
    participant GS as useGraphStore<br/>useGraphStore.ts:188
    participant H as useWasmSimulation<br/>useWasmSimulation.ts:36
    participant E as WebVowl.loadOntology<br/>vowl-wasm v0.1.1
    GS-->>H: nodes and edges Maps, built from class and property, useGraphStore.ts:188-223
    H->>H: re-serialise as nodes and edges arrays, useWasmSimulation.ts:90-102
    H->>E: loadOntology(JSON.stringify(graphData)), useWasmSimulation.ts:105
    Note over E: EXTERNAL: the engine parser needs a class array, see VW-02.2 and ES-90.3
    E-->>H: throws, the catch only logs, useWasmSimulation.ts:117-118
    H->>H: initSimulation and isRunning are skipped, useWasmSimulation.ts:106-116
    Note over GS,H: Every node keeps the origin position set at load, useGraphStore.ts:194<br/>TENSION: ADR-001 records this mismatch, ADR-001-explorer-consumer-contract-closeout.md:17,<br/>and still lists it as open after the extraction, ADR-001-explorer-consumer-contract-closeout.md:39
```

The engine type the hook calls is declared by hand (`useWasmSimulation.ts:11-29`), not imported from the package's declarations, so `tsc` cannot catch a drift between the two.

## WV-03.3 Hook lifecycle: init, load, tick

```mermaid
flowchart TB
    I1["mount: dynamic import of @dreamlab-ai/vowl-wasm, await default()<br/>useWasmSimulation.ts:57-58"] --> I2["new WebVowl, setCenter 0,0, linkDistance, chargeStrength<br/>useWasmSimulation.ts:62-68"]
    I2 --> L1["load effect, dependencies nodes, edges, isInitialized, autoStart, iterations<br/>useWasmSimulation.ts:86-120"]
    L1 --> L2["capture getNodeIds once per load, then initSimulation<br/>useWasmSimulation.ts:106-107"]
    L2 --> F1["useFrame: stop when isFinished, else tick and getAlpha<br/>useWasmSimulation.ts:131-148"]
    F1 --> F2["zip ids with getNodePositions, x at 2i, y at 2i+1<br/>one updateNodePosition call per node per frame<br/>useWasmSimulation.ts:151-155"]
    F2 --> F3["ClassNode lerps toward the new store position<br/>see WV-02.3"]
    INV["INVARIANT: index order of ids equals the position buffer order until the next load<br/>useWasmSimulation.ts:21-25,40-42"]
    OPEN1["TENSION: ADR-001 still lists topology reset as open,<br/>ADR-001-explorer-consumer-contract-closeout.md:25,29,<br/>but the code answers it. updateNodePosition is an immer set on the nodes Map,<br/>useGraphStore.ts:260-265, MapSet enabled at useGraphStore.ts:16,<br/>so each frame that writes positions yields a new nodes reference.<br/>The hook subscribes to the whole store, useWasmSimulation.ts:47, the load effect<br/>depends on nodes, useWasmSimulation.ts:120, so every such frame re-runs<br/>loadOntology, getNodeIds and initSimulation, useWasmSimulation.ts:105-107.<br/>Latent today: the engine rejects the payload first, see WV-03.2,<br/>isRunning never turns true and useFrame returns, useWasmSimulation.ts:133"]
    L2 --- INV
    F2 --- OPEN1
    V["Vite excludes the package from dependency pre-bundling, wasm and top-level-await plugins<br/>vite.config.ts:8-20"] --> I1
```

## WV-03.4 What the standalone takes from the engine, and which release

```mermaid
flowchart TB
    PIN["dependency is the v0.1.1 release tarball URL<br/>modern/package.json:19"] --> LOCK["lock: version 0.1.1, resolved tarball, sha512 integrity<br/>modern/package-lock.json:550-554"]
    LOCK --> USED["methods the hook calls: loadOntology, setCenter, setLinkDistance,<br/>setChargeStrength, initSimulation, tick, runSimulation, isFinished,<br/>getAlpha, getNodeIds, getNodePositions<br/>useWasmSimulation.ts:62-68,105-111,137-151"]
    USED --> UNUSED["EXTERNAL: pinning, filtering, hit testing and NggExplorer are on offer<br/>but unused here, see VW-04.1 and VW-01.7"]
    UNUSED --> OTHER["EXTERNAL: the visionGraph copy installs 0.2.3 under the same name,<br/>see VW-04.6, ES-11.3 and VG-06.1"]
```
