---
id: WV-02
title: Modern explorer runtime - load, store, scene and controls
area: wasmvowl
governing:
  - ../WasmVOWL/CLAUDE.md
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
adrs: [WasmVOWL:ADR-001]
sources:
  - ../WasmVOWL/CLAUDE.md
  - ../WasmVOWL/modern/src/App.tsx
  - ../WasmVOWL/modern/src/components/UI/FileDropZone.tsx
  - ../WasmVOWL/modern/src/components/UI/TopMenuBar.tsx
  - ../WasmVOWL/modern/src/components/UI/Sidebar.tsx
  - ../WasmVOWL/modern/src/components/Canvas/GraphCanvas.tsx
  - ../WasmVOWL/modern/src/components/Canvas/GraphScene.tsx
  - ../WasmVOWL/modern/src/components/Canvas/Nodes/ClassNode.tsx
  - ../WasmVOWL/modern/src/components/Canvas/Edges/PropertyEdge.tsx
  - ../WasmVOWL/modern/src/stores/useGraphStore.ts
  - ../WasmVOWL/modern/src/stores/useUIStore.ts
  - ../WasmVOWL/modern/src/hooks/useWasmSimulation.ts
  - ../WasmVOWL/modern/src/utils/export.ts
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
verified_commit: {wasmvowl: af25086a605482c8234580ccd1062cf506e0b181}
---

## WV-02.1 Loading a file or a bundled sample into the graph store

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FDZ as loadFile<br/>FileDropZone.tsx:53
    participant PO as parseOntology<br/>FileDropZone.tsx:27
    participant GS as loadOntology<br/>useGraphStore.ts:180
    participant UI as addNotification<br/>useUIStore.ts:78
    participant APP as App<br/>App.tsx:14
    U->>FDZ: drop or pick a file, FileDropZone.tsx:117-135
    FDZ->>FDZ: reject unless the name ends in .json, FileDropZone.tsx:59-60
    FDZ->>FDZ: file.text then JSON.parse, FileDropZone.tsx:64-65
    FDZ->>PO: raw object
    PO-->>FDZ: OntologyData, or null when there is no class array, FileDropZone.tsx:30-31
    FDZ->>GS: loadOntology(ontology), FileDropZone.tsx:74
    FDZ->>UI: success toast with class and property counts, FileDropZone.tsx:77-81
    FDZ->>APP: onFileLoaded sets hasData, App.tsx:19-21
    Note over U,FDZ: Samples take the same path from fetch of /ontologies/name, FileDropZone.tsx:140-157<br/>three buttons, foaf, sioc and minimal, FileDropZone.tsx:230-248
    Note over FDZ,APP: Errors surface as an inline message and an error toast, FileDropZone.tsx:84-92
```

Input formats and the mapping inside `loadOntology` are WV-03.1 and WV-03.2.

## WV-02.2 The two stores and who reads them

```mermaid
flowchart TB
    subgraph GSB["useGraphStore - zustand plus immer"]
        MS["enableMapSet before the store is created<br/>useGraphStore.ts:16"]
        ST["nodes, edges Maps, filteredNodes, filteredEdges Sets<br/>useGraphStore.ts:20-23"]
        HELP["recomputeFilters and recomputeStatistics edit the caller draft<br/>useGraphStore.ts:78,133"]
        UPD["updateNodePosition writes one node position<br/>useGraphStore.ts:260-265"]
    end
    subgraph USB["useUIStore - plain zustand"]
        SET["settings defaults<br/>useUIStore.ts:31-44"]
        VP["viewport zoom, rotation, target, mode<br/>useUIStore.ts:55-60"]
    end
    MS --> ST --> HELP --> UPD
    UPD --> SET
    SET --> VP
    N1["INVARIANT: helpers are plain functions so a nested action cannot run on stale state<br/>comment at useGraphStore.ts:72-76"]
    N2["INVARIANT: activeFilters is empty whenever loadOntology runs.<br/>loadOntology neither resets nor re-applies filters, useGraphStore.ts:182-185,<br/>but its only caller, the drop zone, renders only at nodeCount 0, App.tsx:23,35-36.<br/>Filters start empty, useGraphStore.ts:176, and New file runs clear,<br/>TopMenuBar.tsx:16, which empties them, useGraphStore.ts:238"]
    N3["DEBT: gravity, friction, maxFPS, lodEnabled, highContrast, reducedMotion<br/>are declared but nothing under src/components or src/hooks reads them<br/>useUIStore.ts:38-43"]
    HELP --- N1
    ST --- N2
    SET --- N3
```

## WV-02.3 Scene graph inside the R3F canvas

```mermaid
flowchart TB
    GC["GraphCanvas: Canvas, PerspectiveCamera, OrbitControls<br/>GraphCanvas.tsx:16-44"] --> MODE["camera at 0,0,500 in 2d, 0,300,500 in 3d, rotate only in 3d<br/>GraphCanvas.tsx:27,37"]
    GC --> GSC["GraphScene runs useWasmSimulation inside the canvas<br/>GraphScene.tsx:13, see WV-03.3"]
    GSC --> VIS["visible = nodes and edges whose id is in the filtered Sets<br/>GraphScene.tsx:16-22"]
    VIS --> PE["PropertyEdge per edge, drawn first<br/>GraphScene.tsx:27-29"]
    VIS --> CN["ClassNode per node<br/>GraphScene.tsx:32-34"]
    CN --> CNL["radius sqrt(instances) times 2 plus 20, times nodeScale<br/>ClassNode.tsx:27-33<br/>lerp 0.1 toward the store position every frame, ClassNode.tsx:62-71"]
    PE --> PEL["straight line between store positions<br/>PropertyEdge.tsx:27-34<br/>colour and dash by edge type, PropertyEdge.tsx:39-69"]
    N1["DEBT: PropertyEdge styles subclass and disjoint edges, PropertyEdge.tsx:43-50,<br/>but the store only ever creates objectProperty or datatypeProperty, useGraphStore.ts:211"]
    N2["DEBT: the hook writes z = 0 for every node, useWasmSimulation.ts:154,<br/>so 3d mode tilts the camera over a flat layout"]
    PEL --- N1
    MODE --- N2
```

## WV-02.4 Menu, sidebar and export controls

```mermaid
flowchart TB
    TMB["TopMenuBar<br/>TopMenuBar.tsx:10-12"] --> NEW["New file: confirm then clear, App falls back to the drop zone<br/>TopMenuBar.tsx:14-16, App.tsx:23"]
    TMB --> ZOOM["zoom in, out, reset write viewport.zoom, rotation, target<br/>TopMenuBar.tsx:84-95"]
    TMB --> EXP["SVG and PNG export of all nodes and edges, 1920 by 1080<br/>TopMenuBar.tsx:29-33,60-65"]
    EXP --> EXF["exportSVG builds markup from store positions, export.ts:10<br/>exportPNG draws on an off-screen canvas, export.ts:100,120<br/>download through an object URL, export.ts:246-254"]
    TMB --> VM["toggleViewMode and showLabels<br/>TopMenuBar.tsx:165,176"]
    SB["Sidebar tabs details, filters, statistics<br/>Sidebar.tsx:33-46"] --> FIL["three fixed filters: classes only, degree 2 or more, object properties<br/>Sidebar.tsx:163,173,183"]
    CLK["ClassNode click toggles selectNode, ClassNode.tsx:43-46<br/>selectNode sets selectedNode, clears selectedEdge, useGraphStore.ts:286"] --> DET["details tab shows id, type, label, IRI, instances, edge counts<br/>of the selected node, Sidebar.tsx:26,54-96"]
    SB --> DET
    HOV["pointer over and out call hoverNode, ClassNode.tsx:48-58<br/>hoverNode sets hoveredNode, useGraphStore.ts:288"] --> D4["DEBT: nothing under src reads hoveredNode, the node colour comes from<br/>ClassNode local hovered state, ClassNode.tsx:19,36-40.<br/>selectEdge and hoverEdge have no caller under src, useGraphStore.ts:287,289"]
    CLK --- HOV
    ZOOM --> D1["DEBT: GraphCanvas reads only viewport.mode, GraphCanvas.tsx:11-12,<br/>so the zoom and reset buttons change state that nothing renders"]
    EXF --> D2["DEBT: export ignores the active filters and draws hidden nodes too"]
    FIL --> D3["DOC-DRIFT: CLAUDE.md:90 says LOD for performance, no LOD code in src"]
```

No control writes `linkDistance` or `chargeStrength` (`TopMenuBar.tsx:176` is the only `updateSettings` call), so the hook's settings effect only ever sees the defaults (`useUIStore.ts:36-37`).
