---
id: WV-02
title: Modern explorer runtime - load, store, scene, camera and controls
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
  - ../WasmVOWL/modern/src/types/ui.ts
  - ../WasmVOWL/modern/src/components/Canvas/Camera/CameraController.tsx
  - ../WasmVOWL/modern/src/components/Canvas/Camera/LabelDeclutter.tsx
  - ../WasmVOWL/modern/src/lib/labelDeclutter.ts
  - ../WasmVOWL/modern/src/lib/labelCandidates.ts
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
verified_commit: {wasmvowl: 2bb54b3ee2dc3a7eda34b60d9f08fdb0fb535aab}
---

## WV-02.1 Loading a file or a bundled sample into the graph store

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant FDZ as loadFile<br/>FileDropZone.tsx:57
    participant IN as ingest<br/>FileDropZone.tsx:29
    participant GS as loadOntology<br/>useGraphStore.ts:216
    participant UI as addNotification<br/>useUIStore.ts:100
    participant APP as App<br/>App.tsx:14
    U->>FDZ: drop or pick a file, FileDropZone.tsx:107-125
    FDZ->>FDZ: reject unless the name ends in .json, FileDropZone.tsx:63-64
    FDZ->>FDZ: file.text then JSON.parse, FileDropZone.tsx:68-69
    FDZ->>IN: parsed document
    IN->>IN: normaliseOwl2Vowl, throws on a malformed document, FileDropZone.tsx:30
    IN->>GS: loadOntology(normalised), FileDropZone.tsx:31
    IN->>UI: zero nodes is a warning and no load, FileDropZone.tsx:35-42<br/>else node and edge counts, a warning when records were dropped, FileDropZone.tsx:44-50
    FDZ->>APP: onFileLoaded sets hasData, FileDropZone.tsx:71-72, App.tsx:19-21
    Note over U,FDZ: Samples take the same ingest path from fetch of /ontologies/name, FileDropZone.tsx:135-142<br/>three buttons, foaf, sioc and minimal, FileDropZone.tsx:208-228
    Note over IN,APP: CHANGED at 2bb54b3: the normaliser replaced the class-array check, see WV-03.1<br/>errors still surface inline and as an error toast, FileDropZone.tsx:74-82
```

Input formats, the normaliser and the layout mapping are WV-03.1 and WV-03.2.

## WV-02.2 The two stores and who reads them

```mermaid
flowchart TB
    subgraph GSB["useGraphStore - zustand plus immer"]
        MS["enableMapSet before the store is created<br/>useGraphStore.ts:17"]
        ST["nodes, edges Maps, filteredNodes, filteredEdges Sets<br/>useGraphStore.ts:21-24"]
        GV["ontology and graphVersion, bumped by loadOntology and clear only<br/>useGraphStore.ts:41,45,283,303"]
        HELP["recomputeFilters and recomputeStatistics edit the caller draft<br/>useGraphStore.ts:94,150"]
        UPD["updateNodePositions writes a whole frame in one set, z = 0<br/>useGraphStore.ts:331-339"]
    end
    subgraph USB["useUIStore - plain zustand"]
        SET["settings defaults, layout from defaultLayout<br/>useUIStore.ts:50-64"]
        CR["cameraRequest queued by zoomCamera and fitCamera<br/>useUIStore.ts:142-152"]
    end
    MS --> ST --> GV --> HELP --> UPD
    UPD --> SET
    SET --> CR
    N1["INVARIANT: helpers are plain functions so a nested action cannot run on stale state<br/>comment at useGraphStore.ts:84-93"]
    N2["CHANGED at 2bb54b3: loadOntology now re-applies activeFilters to the new graph,<br/>useGraphStore.ts:286, and clear empties them, useGraphStore.ts:300"]
    N3["DEBT: maxFPS, lodEnabled, highContrast, reducedMotion are declared<br/>but nothing outside the store reads them, useUIStore.ts:60-63.<br/>gravity and friction are gone, see WV-03.3"]
    HELP --- N1
    GV --- N2
    SET --- N3
```

## WV-02.3 Scene graph inside the R3F canvas

```mermaid
flowchart TB
    GC["GraphCanvas selects only the view mode<br/>GraphCanvas.tsx:14"] --> MODE["initial camera at 0,0,500 in 2d, 0,300,500 in 3d, far 200000<br/>GraphCanvas.tsx:16-19,32-38<br/>OrbitControls makeDefault, distance 50 to 50000, rotate only in 3d, GraphCanvas.tsx:42-48"]
    GC --> GSC["GraphScene runs useWasmSimulation, see WV-03.3<br/>GraphScene.tsx:15<br/>mounts CameraController and LabelDeclutter, GraphScene.tsx:28-29, see WV-02.5"]
    GSC --> VIS["visible = nodes and edges whose id is in the filtered Sets<br/>GraphScene.tsx:18-24"]
    VIS --> PE["PropertyEdge per edge, drawn first<br/>GraphScene.tsx:32-34"]
    VIS --> CN["ClassNode per node<br/>GraphScene.tsx:37-39"]
    CN --> CNL["VOWL fills by kind, ClassNode.tsx:34-40<br/>lerp 0.1 toward the store position every frame, ClassNode.tsx:67-75"]
    PE --> PEL["colour, width and dash by edge type, PropertyEdge.tsx:42-74"]
    GSC --> OV["Laying out overlay while isRunning, no percentage<br/>GraphScene.tsx:42-63"]
    N1["CHANGED at 2bb54b3: subclass, disjoint and set-operator edges now exist,<br/>the normaliser emits them, see WV-03.1, so their styles are live, PropertyEdge.tsx:46,52,69"]
    N2["DEBT: every position is written with z = 0, useGraphStore.ts:336,<br/>so 3d mode tilts the camera over a flat layout"]
    PEL --- N1
    MODE --- N2
```

## WV-02.4 Menu, sidebar and export controls

```mermaid
flowchart TB
    TMB["TopMenuBar, marked data-canvas-overlay<br/>TopMenuBar.tsx:13,95"] --> NEW["New: confirm then clear, App falls back to the drop zone<br/>TopMenuBar.tsx:18-19, App.tsx:23"]
    TMB --> ZOOM["zoom out and in queue a 1.25x cameraRequest, Fit and reset queue a fit<br/>TopMenuBar.tsx:11,89-92,154-161"]
    TMB --> EXP["SVG and PNG export of all nodes and edges<br/>TopMenuBar.tsx:32-35,63-66"]
    EXP --> EXF["exportSVG builds markup from store positions, export.ts:10<br/>exportPNG draws on an off-screen canvas, export.ts:100,120<br/>download through an object URL, export.ts:246-254"]
    TMB --> VM["toggleViewMode and showLabels<br/>TopMenuBar.tsx:168,179"]
    SB["Sidebar, data-canvas-overlay, tabs details, filters, statistics<br/>Sidebar.tsx:29,33-46"] --> FIL["three fixed filters: classes only, degree 2 or more, object properties<br/>Sidebar.tsx:163,173,183"]
    CLK["ClassNode click toggles selectNode, ClassNode.tsx:48-50<br/>selectNode sets selectedNode, clears selectedEdge, useGraphStore.ts:360"] --> DET["details tab shows the selected node<br/>Sidebar.tsx:26,54-96"]
    SB --> DET
    HOV["pointer over and out call hoverNode, ClassNode.tsx:55-62"] --> D4["CHANGED at 2bb54b3: hoveredNode now pins a label in the declutter pass,<br/>labelCandidates.ts:53. DEBT: selectEdge and hoverEdge still have no caller<br/>under src, useGraphStore.ts:361,363"]
    CLK --- HOV
    ZOOM --> D1["CHANGED at 2bb54b3: the zoom buttons move the camera, see WV-02.5,<br/>they used to write a viewport.zoom nothing read"]
    EXF --> D2["DEBT: export ignores the active filters and draws hidden nodes too"]
    FIL --> D3["DOC-DRIFT: CLAUDE.md:90 says LOD for performance, lodEnabled is unread"]
```

`updateLayout` (`useUIStore.ts:159-161`) has no caller in the UI: the layout parameters change only from code or the `?debug` handle (WV-03.3), and `TopMenuBar.tsx:179` is still the only `updateSettings` call.

## WV-02.5 Camera following, menu zoom and label declutter

```mermaid
flowchart TB
    LOAD["new graphVersion: queue a fit from the default angle, start following<br/>CameraController.tsx:65-69"] --> FOL["while followed and running: re-fit every 300 ms<br/>CameraController.tsx:44,109-112"]
    FOL --> SETL["isRunning falls: one last fit, stop following<br/>CameraController.tsx:87-93"]
    USER["drag or wheel fires the controls start event: stop following<br/>CameraController.tsx:72-79"] --> FOL
    REQ["menu cameraRequest, a zoom also stops following<br/>CameraController.tsx:96-105"] --> FIT
    SETL --> FIT["fit: bounds of discs and label boxes, inside the canvas<br/>not covered by data-canvas-overlay elements<br/>CameraController.tsx:35-38,129-139"]
    REQ --> ZM["zoom: move along the view direction, clamped by zoomedDistance<br/>CameraController.tsx:121-126"]
    FIT --> FAR["far plane and orbit limit grow past the camera distance<br/>CameraController.tsx:155-163"]
    ZM --> FAR
    LD["LabelDeclutter: at most every 250 ms, only when nodes, selection or zoom changed<br/>LabelDeclutter.tsx:18,40-54"] --> RULE["hide edge labels under 8 px and minor labels under 7 px,<br/>then any label overlapping a more important one<br/>labelDeclutter.ts:41-44,81-89"]
    RULE --> PIN["selected and hovered nodes stay labelled<br/>labelCandidates.ts:53, ClassNode.tsx:113"]
    FAR --> LD
```

The camera and declutter landed in WasmVOWL PR #2 (`ab9122e`); the decision record's own account is its "Camera, labels and CI" section (`ADR-001-explorer-consumer-contract-closeout.md:139-152`).
