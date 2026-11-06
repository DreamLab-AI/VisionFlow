---
id: VC-31
title: R3F/Three.js graph render pipeline, overview layer and WASM scene effects
area: visionclaw
verified_commit: {visionclaw: ec785afa9331be1fa049cee6282a4dcda16ee86d}
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2135, ADR-2142, ADR-2179, ADR-2181, ADR-2182, ADR-2183, ADR-2184, ADR-2186, ADR-2189]
sources:
  - ../project/client/src/app/MainLayout.tsx
  - ../project/client/src/features/graph/components/GraphCanvas.tsx
  - ../project/client/src/features/graph/components/GraphCanvasWrapper.tsx
  - ../project/client/src/features/graph/components/GraphManager.tsx
  - ../project/client/src/features/graph/components/GemNodes.tsx
  - ../project/client/src/features/bots/components/AgentOverlay.tsx
  - ../project/client/src/features/bots/agentMarkerPass.ts
  - ../project/client/src/store/agentTargetStore.ts
  - ../project/client/src/features/graph/utils/drawnPositions.ts
  - ../project/client/src/features/graph/contexts/NodePositionContext.tsx
  - ../project/client/src/features/graph/components/GlassEdges.tsx
  - ../project/client/src/features/graph/components/InstancedLabels.tsx
  - ../project/client/src/features/graph/components/KnowledgeRings.tsx
  - ../project/client/src/features/graph/components/ClusterHulls.tsx
  - ../project/client/src/features/graph/components/InferredEdges.tsx
  - ../project/client/src/features/graph/components/TimelineScrubber.tsx
  - ../project/client/src/features/graph/components/PerfProbe.tsx
  - ../project/client/src/features/graph/components/NodeContextMenu.tsx
  - ../project/client/src/features/graph/components/NodeDetailPanel.tsx
  - ../project/client/src/features/control-center/drawer/RightDrawer.tsx
  - ../project/client/src/features/graph/hooks/useEdgeBufferComputation.ts
  - ../project/client/src/features/graph/hooks/edgeBufferPlan.ts
  - ../project/client/src/features/graph/hooks/useGraphFiltering.ts
  - ../project/client/src/features/graph/explore/nodeFilter.ts
  - ../project/client/src/features/graph/explore/filterRegistry.ts
  - ../project/client/src/features/graph/overview/OverviewLayer.tsx
  - ../project/client/src/features/graph/overview/OverviewEdgeLines.tsx
  - ../project/client/src/features/graph/overview/OverviewNodes.tsx
  - ../project/client/src/features/graph/overview/OverviewEdgeNotice.tsx
  - ../project/client/src/features/graph/overview/nodePositionTexture.ts
  - ../project/client/src/features/graph/overview/overviewEdgeStore.ts
  - ../project/client/src/features/graph/overview/overviewLive.ts
  - ../project/client/src/features/graph/overview/useOverviewMaterial.ts
  - ../project/client/src/features/graph/overview/useRenderFaultWatch.ts
  - ../project/client/src/features/graph/overview/useNodeEncodingSync.ts
  - ../project/client/src/features/graph/overview/tubeSelect.ts
  - ../project/client/src/features/graph/overview/gemLod.ts
  - ../project/client/src/features/graph/overview/edgeStyle.ts
  - ../project/client/src/features/graph/overview/__tests__/edgeOpacity.test.ts
  - ../project/client/src/features/control-center/registry/groups/look.ts
  - ../project/client/src/api/settings/defaults.ts
  - ../project/client/src/features/graph/overview/highlightVisual.ts
  - ../project/client/src/features/graph/overview/labelBudget.ts
  - ../project/client/src/features/graph/overview/labelRunner.ts
  - ../project/client/src/features/graph/state/highlightMask.ts
  - ../project/client/src/features/graph/state/nodeEncoding.ts
  - ../project/client/src/features/graph/state/nodeIndex.ts
  - ../project/client/src/features/graph/state/overviewEdgeCapacity.ts
  - ../project/client/src/features/graph/utils/agentNudge.ts
  - ../project/client/src/features/visualisation/components/VirtualAgentMarkers.tsx
  - ../project/client/src/features/graph/utils/nodeScaling.ts
  - ../project/client/src/features/graph/hooks/useFpsMonitor.ts
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/client/src/features/graph/hooks/useGraphSelection.ts
  - ../project/client/src/store/transientBeamStore.ts
  - ../project/client/src/store/websocket/binaryProtocol.ts
  - ../project/client/src/rendering/GemPostProcessing.tsx
  - ../project/client/src/rendering/postProcessingGuard.ts
  - ../project/client/src/rendering/rendererFactory.ts
  - ../project/client/src/rendering/renderFaults.ts
  - ../project/client/src/rendering/blendSpace.ts
  - ../project/client/src/rendering/bloomLayer.ts
  - ../project/client/src/rendering/troikaConfig.ts
  - ../project/client/src/rendering/materials/GemNodeMaterial.ts
  - ../project/client/src/rendering/materials/CrystalOrbMaterial.ts
  - ../project/client/src/rendering/materials/AgentCapsuleMaterial.ts
  - ../project/client/src/features/bots/agentVisualConstants.ts
  - ../project/client/src/rendering/materials/GlassEdgeMaterial.ts
  - ../project/client/src/rendering/materials/OverviewEdgeMaterial.ts
  - ../project/client/src/rendering/materials/OverviewNodeMaterial.ts
  - ../project/client/src/rendering/materials/__tests__/overviewMaterials.test.ts
  - ../project/client/src/rendering/text/createTextMaterial.ts
  - ../project/client/src/rendering/text/GlyphAtlas.ts
  - ../project/client/src/rendering/text/textLayout.ts
  - ../project/client/src/wasm/scene-effects-bridge.ts
  - ../project/client/src/hooks/useWasmSceneEffects.ts
  - ../project/client/src/features/visualisation/components/WasmSceneEffects.tsx
  - ../project/client/src/features/visualisation/components/TransientBeamsLayer.tsx
  - ../project/client/src/features/visualisation/components/EmbeddingCloudLayer.tsx
  - ../project/client/src/features/visualisation/hooks/useTransientBeams.ts
  - ../project/client/src/features/visualisation/attentionHeat.ts
  - ../project/client/src/features/visualisation/heatColor.ts
  - ../project/client/src/features/visualisation/semanticEncoding.ts
  - ../project/client/crates/scene-effects/src/lib.rs
  - ../project/client/crates/scene-effects/src/particles.rs
  - ../project/client/crates/scene-effects/src/energy_wisps.rs
  - ../project/client/crates/scene-effects/src/atmosphere.rs
  - ../project/client/src/wasm/scene-effects/scene_effects.js
---

## For developers

The desktop graph is one R3F `Canvas` whose `GraphManager` copies the worker's shared positions once a frame and feeds two edge renderers, two node renderers and one label path. The desktop graph-view upgrade (ADR-2179 suite, landed by 16d3bb49e) adds the overview layer (VC-31.12): a float position texture with the highlight level in `w`, screen-space edge quads and pixel-sized node sprites, with glass tubes and gems kept as close-zoom detail only while that layer is live (VC-31.14). Two render-indexed buffers, the highlight mask and the node encoding, are the contracts every drawer reads (VC-31.13). The explore widgets that write the mask (ADR-2185) and the camera fit (ADR-2180, VC-37) belong to other topics; the explore layer is VC-56. Since ec785afa9 the overview line alpha follows the one Edge Opacity setting (VC-31.12).

**Drift (BASELINE vs code):** `BASELINE-architecture.md:292` still calls the suite "proposed, not built". It lists as open gaps, by its own anchors, the 65,536 tube cap, the removal at `useGraphFiltering.ts:116` and the WebGPU DOM labels at `InstancedLabels.tsx:302-304`. At this commit `GraphManager.tsx:735-742` mounts the overview layer, and the DOM path is only a fallback (`InstancedLabels.tsx:305-307`).

## For the business

At the default view a visitor now sees the whole knowledge graph as dots and fine lines, not a grey ball. Filtering greys the rest out without removing it. If the graph is larger than the screen can draw, or the graphics card cannot run the new drawing, a notice says so on the canvas instead of parts disappearing quietly. Rich glass gems and tubes now appear only when you zoom in.

## VC-31.1 Canvas consumers and shared child tree
```mermaid
flowchart LR
    ML["MainLayoutContent<br/>MainLayout.tsx:15"] -->|mounts| GCW["GraphCanvasWrapper<br/>GraphCanvasWrapper.tsx:134"]
    GCW -->|CanvasErrorBoundary wraps| GC["GraphCanvas<br/>GraphCanvas.tsx:180"]
    GC -->|"gl = createGemRenderer"| CANVAS["R3F Canvas<br/>GraphCanvas.tsx:266-267"]
    CANVAS --> CAS["CameraAspectSync<br/>GraphCanvas.tsx:64"]
    CANVAS --> EEC["EnsureEventsConnected<br/>GraphCanvas.tsx:108"]
    CANVAS --> PP["PerfProbe, DEV only, lazy<br/>GraphCanvas.tsx:15, PerfProbe.tsx:386"]
    CANVAS --> ENV["Environment, skipped on a software renderer<br/>GraphCanvas.tsx:334"]
    CANVAS --> WSE["WasmSceneEffects<br/>WasmSceneEffects.tsx:628"]
    CANVAS --> EPL["EchoPulseLayer<br/>GraphCanvas.tsx:366"]
    CANVAS --> SPL["SplatScene, world-locked, outside the graph root<br/>GraphCanvas.tsx:369"]
    CANVAS --> ECL["EmbeddingCloudLayer<br/>EmbeddingCloudLayer.tsx:123"]
    CANVAS -->|"canvasReady and nodeCount above 0"| GM["GraphManager<br/>GraphCanvas.tsx:379-383"]
    GM --> GN["GemNodes x3 populations<br/>GraphManager.tsx:679-706"]
    GM --> GE["GlassEdges x2, main and selection highlight<br/>GraphManager.tsx:708-730"]
    GM --> OL["OverviewLayer, edges and sprites (VC-31.12)<br/>GraphManager.tsx:735-742"]
    OL --> IE["InferredEdges on the overview path<br/>InferredEdges.tsx:54"]
    GM --> KR["KnowledgeRings<br/>KnowledgeRings.tsx:32"]
    GM --> CH["ClusterHulls<br/>ClusterHulls.tsx:210"]
    GM --> VAM["VirtualAgentMarkers, agents are not graph nodes (ADR-2142)<br/>VirtualAgentMarkers.tsx:35"]
    GM --> TBL["TransientBeamsLayer<br/>GraphManager.tsx:768-773"]
    GM --> IL["InstancedLabels<br/>GraphManager.tsx:775-787"]
    CANVAS --> AO["AgentOverlay, nameplates, trails, collab edges, payment labels<br/>GraphCanvas.tsx:388, AgentOverlay.tsx:23"]
    CANVAS --> OC["OrbitControls makeDefault<br/>GraphCanvas.tsx:393"]
    CANVAS --> GPP["GemPostProcessing<br/>GraphCanvas.tsx:414"]
    GC --> OEN["OverviewEdgeNotice, outside the Canvas<br/>GraphCanvas.tsx:426, OverviewEdgeNotice.tsx:18"]
    GC -->|"provenance.enableTimeline"| TS["TimelineScrubber<br/>GraphCanvas.tsx:439, TimelineScrubber.tsx:76"]
    GC --> SIP["SplatInfoPanel<br/>GraphCanvas.tsx:442"]
    ML --> NCM["NodeContextMenu, HTML overlay<br/>MainLayout.tsx:83, NodeContextMenu.tsx:41"]
    GM -.->|"visionclaw:node-contextmenu CustomEvent"| NCM
    RD["RightDrawer tabs: explore, Memory explorer, node detail (ADR-2189)<br/>RightDrawer.tsx:86, :287, :301, :317"] --> NDP["NodeDetailPanel embedded<br/>NodeDetailPanel.tsx:65"]
    Note1["Note: the Memory explorer and node detail left the canvas and MainLayout for<br/>the drawer (MainLayout.tsx:80). No GraphViewport.tsx exists, it was retired<br/>(GraphCanvas.tsx:375)."]
```

**What it shows.** Who mounts what inside and around the canvas at ec785afa9. **Why it is this way.** ADR-2189 moved the detail, explore and memory panels into one right-hand drawer, and ADR-2181 D4 needs an in-canvas notice that sits outside the WebGL surface.

## VC-31.2 GraphManager per-frame hot loop (priority -2) and the edge pass (-0.5)
```mermaid
sequenceDiagram
    autonumber
    participant R3F as R3F render loop
    participant GM as GraphManager.useFrame priority -2<br/>GraphManager.tsx:385
    participant WP as getPositionsSync<br/>graphWorkerProxy.ts:316
    participant DP as drawnPositions<br/>drawnPositions.ts:17,46
    participant MP as agentMarkers pass<br/>agentMarkerPass.ts:71,144
    participant SEL as useGraphSelection focusCameraOnNode writes the fly-to refs<br/>useGraphSelection.ts:120-133
    participant EBC as useEdgeBufferComputation useFrame<br/>useEdgeBufferComputation.ts:121, registered at priority -0.5 (:247)
    participant GE as GlassEdges imperative handle<br/>GlassEdges.tsx:483-571

    rect rgb(232,238,250)
    Note over GM: INVARIANT frame order by priority: GraphManager -2 (GraphManager.tsx:514), GemNodes -1 (GemNodes.tsx:1229),<br/>edges -0.5 (useEdgeBufferComputation.ts:247), position texture -0.4 (OverviewLayer.tsx:90), sprites -0.3 (OverviewNodes.tsx:177),<br/>labels 0, post-processing 1 (GemPostProcessing.tsx:438). Every reader sees this frame's drawn copy.
    R3F->>GM: frame(state, delta)
    opt flyToTargetRef.current set (GraphManager.tsx:392)
        GM->>SEL: read flyToTargetRef, flyToLookAtRef, flyToProgressRef
        GM->>GM: progress += delta / 0.6, eased 1-(1-p)^3 - GraphManager.tsx:406-408
        alt p at least 1
            GM->>GM: camera.position.copy(dest), cancelFlyTo() - GraphManager.tsx:418-422
        else in flight
            GM->>GM: lerp camera and OrbitControls pivot - GraphManager.tsx:410-413
        end
    end
    GM->>GM: labelTickRef every 15 frames, frustum refresh and setLabelUpdateTick - GraphManager.tsx:427-433
    Note over GM: DEBT: labelUpdateTick and frustum have no reader (GraphManager.tsx:210, :215), so this only re-renders<br/>GraphManager about 4 times a second. Labels run their own cadence now (labelBudget.ts:36).
    GM->>WP: getPositionsSync() - GraphManager.tsx:436
    alt positions null
        GM-->>R3F: return (skip rest of frame)
    else positions available
        opt first frame and all-zero prefix
            GM-->>R3F: return - GraphManager.tsx:439-446
        end
        GM->>DP: drawn = drawnPositions.copy(positions), one copy a frame - GraphManager.tsx:449, drawnPositions.ts:26
        GM->>GM: nodePositionsRef.current = drawn, setSharedNodePositions(drawn) - GraphManager.tsx:450-451
        Note over GM: the module-level singleton publishes the drawn copy without a React context (NodePositionContext.tsx:7-8)
        opt transitionRef.current.active (layout mode change)
            GM->>GM: easeInOutQuad progress, massFactor = 1 / (1 + sqrt(cc)*0.3), lerp drawn - GraphManager.tsx:454-471
        end
        Note over GM: DEBT: the mass lookup is connectionCountMap.get(String(i)), a render index (GraphManager.tsx:465), while<br/>the map is keyed by node id everywhere else (nodeEncoding.ts:108), so it matches only where an id equals its index (labelRunner.ts:114)
        GM->>GM: every 30 frames the scene edge from sceneFitBounds - GraphManager.tsx:475-480
        GM->>MP: agentMarkers.step(..., edge, clocks), the only agent position writer - GraphManager.tsx:484-491 (ADR-2142, ADR-2135)
        GM->>DP: applyDrag(dragIndex, hand) once, into the copy - GraphManager.tsx:493-495, drawnPositions.ts:36
        GM->>GM: tickCameraFit() (fit itself: VC-37) - GraphManager.tsx:497
        GM->>GM: fillLabelPositions(drawn, labelCount, labelArr) - GraphManager.tsx:499-512
    end
    end

    rect rgb(240,232,250)
    Note over EBC: the memoised plan is rebuilt only on topology, filter or scaling change (useEdgeBufferComputation.ts:80, edgeBufferPlan.ts:67)
    R3F->>EBC: frame()
    alt positions short of nodes.length*3
        EBC-->>R3F: return - useEdgeBufferComputation.ts:130
    else sufficient
        opt tubeLod set, the overview edges are live (GraphManager.tsx:99-102)
            EBC->>EBC: selectTubeEdges on mask or LOD change, or every 6 frames while camera or layout moves<br/>useEdgeBufferComputation.ts:136-159, tubeSelect.ts:110
        end
        EBC->>EBC: EdgeFrameGate.shouldSkip(frame), same plan, handles, selection, subset and positions<br/>useEdgeBufferComputation.ts:172, edgeBufferPlan.ts:306
        EBC->>EBC: fillEdgeFrame over the whole plan, or over the tube subset only - useEdgeBufferComputation.ts:176-178
        opt selectedNodeId set
            EBC->>GE: fillHighlightFrame then highlightEdgeFlowRef.updatePoints - useEdgeBufferComputation.ts:182-191, edgeBufferPlan.ts:214
        end
        EBC->>GE: updateWidths BEFORE updatePoints, then updateColors - useEdgeBufferComputation.ts:205-208
    end
    end
    Note over GM,EBC: the plan is returned to GraphManager, which hands it to OverviewLayer (useEdgeBufferComputation.ts:249, GraphManager.tsx:517). See VC-30 for the worker SAB.
```

**What it shows.** One frame of the CPU side: drawn copy, agent markers, drag, then the edge pass that now feeds either every edge or only the tube subset. **Why it is this way.** One copy of the shared buffer per frame keeps every drawer on the same positions (ADR-2135, ADR-2142), and ADR-2181 D3 makes tubes a close-zoom subset whose CPU cost scales with the tube budget, not the edge count.

## VC-31.3 Labels: priority budget, one glyph path, DOM fallback
```mermaid
flowchart TB
    subgraph SW["InstancedLabelsSwitch, InstancedLabels.tsx:309-317"]
        ST["GlyphStatus: WebGL ready at once, WebGPU building<br/>InstancedLabels.tsx:299, :310"]
        ST --> GLY["InstancedLabelsGlyph, unless failed<br/>InstancedLabels.tsx:313, :509"]
        ST --> DOM["InstancedLabelsWebGPU, Html overlay<br/>only on WebGPU while not ready<br/>InstancedLabels.tsx:305-307, :314, :326"]
    end
    GLY --> CAP["glyph buffer from the budget, not MAX_GLYPHS<br/>InstancedLabels.tsx:519, labelBudget.ts:51-54"]
    GLY --> MOT["camera moving fast: instanceCount 0, then 150 ms still<br/>InstancedLabels.tsx:657-670"]
    MOT --> PATCH["every still frame: patch aLabelPos from the drawn array<br/>InstancedLabels.tsx:672-703"]
    PATCH --> DUE["runner.due: 250 ms, camera stop or a mask version change<br/>InstancedLabels.tsx:662, labelRunner.ts:67-71"]
    DUE --> RUN["LabelBudgetRunner.run: rank on population or analytics change,<br/>project every node, read mask bits<br/>labelRunner.ts:89-148"]
    RUN --> SEL["selectLabels<br/>labelBudget.ts:160"]
    SEL --> T1["1. selection tier: selected and hovered always, then search hits,<br/>then focus neighbours by priority<br/>labelBudget.ts:176-193"]
    T1 --> T1B["1b. cluster labels out of the overview budget<br/>labelBudget.ts:195-206"]
    T1B --> T2["2. overview tier: global top-budget by centrality or degree<br/>labelBudget.ts:208-216, :61-80"]
    T2 --> T3["3. dynamic tier: best remaining on-screen nodes<br/>labelBudget.ts:218-225"]
    T3 --> GREY["greyed nodes are never eligible, a taken grid cell skips<br/>labelBudget.ts:174, :214"]
    GREY --> LAY["layoutTextInline per chosen node, font sized in CSS px at depth<br/>InstancedLabels.tsx:765-825"]
    LAY --> CL["cluster labels at centroids, not position-patched<br/>InstancedLabels.tsx:827-844"]
    LAY -.-> OFF["parallel sizing: label lift uses computeNodeScale times nodeSize<br/>InstancedLabels.tsx:792-793, while gems size from the encoding<br/>and the emphasis scale, GemNodes.tsx:939-944"]
    DOM -.-> DUP["parallel runner: the DOM fallback keeps its own LabelBudgetRunner<br/>and projection loop beside the glyph one<br/>InstancedLabels.tsx:333-334, :517"]
    DOM -.-> TEN["ADR-2186 D4 gap: D4 keeps DOM only for the tooltip,<br/>the code keeps a DOM label path for build and fault<br/>InstancedLabels.tsx:253-259"]
```

**What it shows.** The label pipeline after ADR-2186: one importance-ranked budget, refreshed about 4 times a second, drawn as glyph quads on both renderers. **Why it is this way.** Proximity labels crowded the overview; a fixed budget ranked by centrality (ADR-2186 D1-D2) keeps the count and the glyph buffer bounded, and the DOM layer survives only as the WebGPU safety net.

**Tension (ADR-2186 D4 vs code):** D4 keeps the DOM `<Html>` only for the node tooltip, but the code keeps a whole DOM label layer as the WebGPU build and fault fallback (`InstancedLabels.tsx:253-259`, `:305-307`).

**Debt (parallel):** the DOM fallback has its own `LabelBudgetRunner` and projection loop beside the glyph one (`InstancedLabels.tsx:333-334`, `:517`).

**Debt (parallel):** labels are lifted by `computeNodeScale × nodeSize` (`InstancedLabels.tsx:792-793`), but gems are sized from the encoding and the emphasis scale (`GemNodes.tsx:939-944`), so a label sits wrong on a size-by column or an emphasised node.

## VC-31.4 GlassEdges instancing, now the close-zoom tube LOD
```mermaid
sequenceDiagram
    autonumber
    participant CALLER as useEdgeBufferComputation useFrame<br/>useEdgeBufferComputation.ts:121
    participant TS as selectTubeEdges<br/>tubeSelect.ts:110
    participant GE as GlassEdges<br/>GlassEdges.tsx:210
    participant GEOM as createGlassEdgeGeometry<br/>GlassEdgeMaterial.ts:197
    participant CIM as computeInstanceMatrices<br/>GlassEdges.tsx:133
    participant MESH as THREE.InstancedMesh

    Note over GEOM: CylinderGeometry radius, height 1, open-ended. Default radius 0.03 (GlassEdgeMaterial.ts:197),<br/>but GlassEdges passes settings.baseWidth, falling back to 0.1 (GlassEdges.tsx:241)
    Note over GE: EDGE_INITIAL 1024, DEFAULT_EDGE_CEILING 65536 (GlassEdges.tsx:40-41). The highlight instance sits on the bloom layer (GlassEdges.tsx:294, :602)

    rect rgb(240,236,252)
    Note over CALLER,TS: only while tubeLod is set: maxDistance = gemLodDistance, or 0 when glassAtCloseZoom is off, budget 4096 (GraphManager.tsx:99-102, tubeSelect.ts:34)
    CALLER->>TS: selectTubeEdges(plan, adjacency, positions, mask, emphasis, camera, lod) - useEdgeBufferComputation.ts:148-152
    TS->>TS: 1. edges of hovered nodes and focus-to-neighbour edges - tubeSelect.ts:134-145
    TS->>TS: 2. edges of the nodes nearest the camera within maxDistance, nearest first - tubeSelect.ts:147-172
    TS->>TS: selected-node edges skipped, the highlight tubes draw them - tubeSelect.ts:126-127
    TS-->>CALLER: subset indices in plan order, never above budget - tubeSelect.ts:174-176
    Note over TS: ADR-2181 D3 gap, D3 picks close-zoom edges by on-screen length, the code by world distance<br/>of a node to the camera (tubeSelect.ts:148-156)
    Note over TS: ADR-2183 D1 gap, take() checks only the SELECTED bit (tubeSelect.ts:123-132) and GlassEdges reads no mask,<br/>so a greyed edge near the camera is drawn as a full-colour tube
    end

    rect rgb(236,250,232)
    CALLER->>GE: updateWidths(weights, count), cached factors = clamp(sqrt(w), 0.5, 2) - GlassEdges.tsx:512, :121-125
    CALLER->>GE: updatePoints(points, count) - GlassEdges.tsx:483-500
    GE->>GE: imperativeActiveRef = true, the prop-driven reveal stops for good - GlassEdges.tsx:489
    GE->>GE: ensureCapacity, grow x2 up to the ceiling, else clamp - GlassEdges.tsx:405-417
    GE->>CIM: computeInstanceMatrices(mesh, pts, capacity, undefined, len, radiusFactors) - GlassEdges.tsx:499
    loop for each edge
        alt length below 1e-6
            CIM->>MESH: zero-scale matrix - GlassEdges.tsx:155
        else
            CIM->>MESH: compose(midpoint, quat from up to dir, scale r,len,r) - GlassEdges.tsx:166-179
        end
    end
    CIM->>MESH: mesh.count = renderCount, instanceMatrix.needsUpdate - GlassEdges.tsx:183-184
    end
    Note over GE: own useFrame, default priority: emissive pulse, plus the reveal ramp until the imperative path takes over (GlassEdges.tsx:576-597).<br/>Historical fix: no hash gate on updatePoints, a 3-sample hash froze edges (GlassEdges.tsx:480-482)
```

**What it shows.** How tubes are chosen and instanced once the overview edges are live. **Why it is this way.** ADR-2181 D3 demotes world-space tubes, which were sub-pixel at fit and capped at 65,536, to close-zoom detail within a 4,096 budget.

**Tension (ADR-2181 D3 vs code):** D3 picks close-zoom tubes by on-screen edge length. The code picks the edges of nodes within `gemLodDistance` world units of the camera (`tubeSelect.ts:148-156`, `GraphManager.tsx:101`).

**Tension (ADR-2183 D1 vs code):** the tube subset skips only SELECTED edges (`tubeSelect.ts:123-132`), and `GlassEdges` reads no mask, so at close zoom a greyed edge near the camera is drawn as a full-colour tube over its greyed overview line.

## VC-31.5 Node geometries, materials, LOD and colour order per population
```mermaid
classDiagram
    class GemNodes {
        GemNodes.tsx:196 GemNodesInner
        GemNodes.tsx:330 capacity nextPowerOf2 of max nodes 4096
        GemNodes.tsx:524-530 getOverviewSource for the sprites
        GemNodes.tsx:1006 gem LOD only for knowledge while sprites are live
        GemNodes.tsx:1116 a non-LOD gem gets scale 0
    }
    class GemLod {
        gemLod.ts:51 selectGemLod
        gemLod.ts:12 GEM_BUDGET 4096, nearest within gemLodDistance
        GemNodes.tsx:1029 selected and hovered are forced
    }
    class ColourPass {
        GemNodes.tsx:974 computeColor, knowledge from the encoding at GemNodes.tsx:178-181
        GemNodes.tsx:975-979 timeline tint
        GemNodes.tsx:983-984 highlightVisualFor grey or emphasis
        GemNodes.tsx:989-991 heat brighten, then setColorAt
        GemNodes.tsx:1214-1224 2 Hz heat repaint from the cached base
    }
    class GemGeometry {
        GemNodeMaterial.ts:278 createGemGeometry
        THREE.IcosahedronGeometry radius 0.5 detail 1, knowledge
    }
    class CrystalOrbGeometry {
        CrystalOrbMaterial.ts:165 createCrystalOrbGeometry
        THREE.IcosahedronGeometry radius 0.5 detail 1, ontology
    }
    class AgentCapsuleGeometry {
        AgentCapsuleMaterial.ts:183 createAgentCapsuleGeometry
        THREE.CapsuleGeometry radius 0.3 length 0.6, agent
    }
    class GemNodeMaterial {
        GemNodeMaterial.ts:39 createGemNodeMaterial
        MeshStandardMaterial, TSL emissive and opacity on WebGPU only
    }
    class CrystalOrbMaterial {
        CrystalOrbMaterial.ts:36 createCrystalOrbMaterial
    }
    class AgentCapsuleMaterial {
        AgentCapsuleMaterial.ts:36 createAgentCapsuleMaterial
        AgentCapsuleMaterial.ts:78 createTslAgentCapsuleMaterial
        AgentCapsuleMaterial.ts:165-167 activity below 0.1 fades alpha to 0.3
    }
    class AgentStatusActivity {
        agentVisualConstants.ts:96-104 AGENT_STATUS_ACTIVITY table
        agentVisualConstants.ts:107 agentStatusActivity
        GemNodes.tsx:1190 packed into meta.w for agents only
    }
    GemNodes ..> GemLod : knowledge only
    GemNodes ..> ColourPass : on colour hash change
    GemNodes ..> GemGeometry : dominant knowledge
    GemNodes ..> CrystalOrbGeometry : dominant ontology
    GemNodes ..> AgentCapsuleGeometry : dominant agent
    GemGeometry --> GemNodeMaterial : paired
    CrystalOrbGeometry --> CrystalOrbMaterial : paired
    AgentCapsuleGeometry --> AgentCapsuleMaterial : paired
    GemNodes ..> AgentStatusActivity : meta.w
    AgentCapsuleMaterial ..> AgentStatusActivity : done fade
    note for ColourPass "ADR-2183 D3 gap: heat brightens<br/>after the grey-out, so a greyed node an agent<br/>touches lights up again, and the sprites copy<br/>these colours (OverviewNodes.tsx:128)"
    note for AgentStatusActivity "DRIFT: AgentCapsuleMaterial.ts:163-164 says a done<br/>agent has no recent work-target beam, but only<br/>offline (0.0) falls below the 0.1 threshold"
```

**What it shows.** The three populations' meshes and the colour pass's order: encoding, timeline, highlight, heat. **Why it is this way.** ADR-2182 D2 keeps the gem look as close-zoom detail for knowledge nodes only, and ADR-2184 D4 moves the knowledge base colour into the shared encoding.

**Tension (ADR-2183 D3 vs code):** greyed nodes get no glow, yet attention heat is applied after the grey-out (`GemNodes.tsx:989-991`, `:1214-1224`), so a greyed node an agent touches brightens, and the sprites copy that colour (`OverviewNodes.tsx:128`).

## VC-31.6 Renderer selection, render guard and bloom scope
```mermaid
sequenceDiagram
    autonumber
    participant CANVAS as R3F Canvas gl prop
    participant RF as createGemRenderer<br/>rendererFactory.ts:119
    participant WGPU as WebGPURenderer three/webgpu
    participant GPP as GemPostProcessing<br/>GemPostProcessing.tsx:79
    participant GUARD as initRenderGuard<br/>postProcessingGuard.ts:44
    participant BS as setSceneComposited<br/>blendSpace.ts:21

    CANVAS->>RF: await createGemRenderer(defaultProps)
    alt forceWebGLOverride or no navigator.gpu (rendererFactory.ts:93, :129)
        RF->>RF: WebGL directly - rendererFactory.ts:253
    else WebGPU attempt
        RF->>WGPU: new WebGPURenderer, race init() against 5 s - rendererFactory.ts:138-151
        alt init throws, times out, or backendIsWebGPU false (rendererFactory.ts:114-117, :156-160)
            RF->>RF: dispose, fall through to WebGL - rendererFactory.ts:249-251
        else real WebGPU backend
            RF->>WGPU: ACES tone map, sRGB output, setSize, ResizeObserver - rendererFactory.ts:162-195
            RF->>RF: installPipelineErrorWatch on three's console hook - rendererFactory.ts:209-212
            RF->>WGPU: wrap renderObject: catch, warn once, notifyRenderFault - rendererFactory.ts:213-232
            RF-->>CANVAS: WebGPURenderer
        end
    end
    opt WebGL fallback
        RF-->>CANVAS: new THREE.WebGLRenderer - rendererFactory.ts:256-288
    end

    Note over GPP: bloomScope emphasis (default), scene or off. off drops the pipeline (GemPostProcessing.tsx:113-115)
    alt WebGPU, emphasis
        GPP->>GPP: second pass from a camera on BLOOM_LAYER only, background nulled, bloom threshold 0 - GemPostProcessing.tsx:193-212
    else WebGPU, scene
        GPP->>GPP: scenePass to texture, bloom with the threshold, add - GemPostProcessing.tsx:178-214
    else WebGL, emphasis
        GPP->>GPP: bloom-only composer on bloomCamera, mixed over the full scene with sRGB encode - GemPostProcessing.tsx:305-334
    else WebGL, scene
        GPP->>GPP: EffectComposer, RenderPass plus UnrealBloomPass, target capped at 2048 - GemPostProcessing.tsx:283-304
    end
    GPP->>BS: setSceneComposited(pipelineReady) - GemPostProcessing.tsx:100-103
    loop priority 1 while pipelineReady and not ppDisabled (GemPostProcessing.tsx:387-438)
        GPP->>GPP: copy the scene camera into bloomCamera, keep the bloom-only layer mask - GemPostProcessing.tsx:390-395
        alt render succeeds
            GPP->>GUARD: recordRenderSuccess
        else throws
            GPP->>GUARD: recordRenderFailure, MAX 8 (GemPostProcessing.tsx:34)
            opt 8 in a row
                GPP->>GPP: dispose, setPpDisabled(true), R3F renders from now on - GemPostProcessing.tsx:420-436
            end
        end
    end
```

**What it shows.** How the renderer is chosen and wrapped, and how bloom now covers only the bloom layer by default. **Why it is this way.** ADR-2182 D4 restricts glow to emphasis through a selective-bloom layer (`bloomLayer.ts:13`) so that the memory cloud and dense clusters stop feeding it, and ADR-2181 D4 requires the renderObject guard to report faults instead of swallowing them.

## VC-31.7 Text rendering: one glyph atlas, two material dialects
```mermaid
sequenceDiagram
    autonumber
    participant BOOT as troika configureTextBuilder<br/>troikaConfig.ts:19
    participant IL as InstancedLabelsGlyph useMemo<br/>InstancedLabels.tsx:539
    participant ATLAS as createGlyphAtlas<br/>GlyphAtlas.ts:32
    participant MAT as createTextMaterial<br/>createTextMaterial.ts:65
    participant TSL as createTslTextMaterial<br/>createTextMaterial.ts:96
    participant LAY as layoutTextInline<br/>textLayout.ts:87

    BOOT->>BOOT: useWorker false when crossOriginIsolated - troikaConfig.ts:21-22, for small agent labels only
    IL->>ATLAS: createGlyphAtlas(), 48 px default, module cache - GlyphAtlas.ts:27, :33
    ATLAS->>ATLAS: 1024 canvas, strokeText outline then fillText per char - GlyphAtlas.ts:26, :70, :74
    ATLAS-->>IL: CanvasTexture plus metrics plus lineHeight - GlyphAtlas.ts:90-100
    Note over ATLAS: ADR-2186 D4 gap, D4 names an SDF or MSDF atlas, the atlas is a stroked bitmap (GlyphAtlas.ts:70, :74)
    alt WebGL
        IL->>MAT: createTextMaterial(atlas), module-cached ShaderMaterial - InstancedLabels.tsx:543, createTextMaterial.ts:66
    else WebGPU
        IL->>IL: fresh uniforms per glyph mesh - InstancedLabels.tsx:544, createTextMaterial.ts:152
        IL->>TSL: createTslTextMaterial(uniforms, geometry), bound to that geometry - InstancedLabels.tsx:603
        TSL->>TSL: same billboard and atlas sampling in TSL, discard below 0.01 alpha - createTextMaterial.ts:121-136
        alt build fails
            TSL-->>IL: status failed, the DOM fallback takes over - InstancedLabels.tsx:608-611
        end
    end
    Note over MAT,TSL: hand-kept twins, the GLSL and TSL shaders are hand-kept twins, the test checks the TSL one builds, not that they agree (createTextMaterial.ts:89-95)
    IL->>LAY: layoutTextInline into the six attribute buffers at the label cadence - InstancedLabels.tsx:805-811
    LAY-->>IL: glyphCount, zero allocation
```

**What it shows.** The glyph atlas and the two material dialects behind the single glyph path. **Why it is this way.** ADR-2186 D4 retired the WebGPU DOM labels, so WebGPU needed a TSL twin of the GLSL glyph material.

**Tension (ADR-2186 D4 vs code):** D4 specifies an SDF or MSDF atlas, but the atlas is a stroked bitmap raster (`GlyphAtlas.ts:70`, `:74`), so glyphs soften when scaled.

**Debt (parallel):** the glyph shader exists as GLSL (`createTextMaterial.ts:65`) and as a hand-kept TSL twin (`createTextMaterial.ts:96`). Its test builds the TSL one but compares no output.

## VC-31.8 WASM scene-effects zero-copy -- init, tick, view rebuild, dispose
```mermaid
sequenceDiagram
    autonumber
    participant COMP as WasmSceneEffects<br/>WasmSceneEffects.tsx:628
    participant HOOK as useWasmSceneEffects<br/>useWasmSceneEffects.ts:53
    participant BRIDGE as initSceneEffects<br/>scene-effects-bridge.ts:453
    participant WASM as scene_effects.js glue (built output)<br/>client/src/wasm/scene-effects
    participant RUST as ParticleField / EnergyWisps / AtmosphereField<br/>particles.rs:65, energy_wisps.rs:80, atmosphere.rs:72
    participant PFB as ParticleFieldBridge<br/>scene-effects-bridge.ts:84
    participant PARTINST as WasmParticleInstances useFrame<br/>WasmSceneEffects.tsx:200

    COMP->>HOOK: useWasmSceneEffects({particleCount, wispCount, atmosphereWidth,<br/>atmosphereHeight, enabled})
    HOOK->>BRIDGE: initSceneEffects() (module singleton, cachedAPI or initPromise)
    alt already cached or in-flight
        BRIDGE-->>HOOK: return cachedAPI or shared initPromise
    else first call
        BRIDGE->>WASM: dynamic import scene-effects/scene_effects.js, wasmModule.default()
        WASM-->>BRIDGE: initOutput.memory (WebAssembly.Memory)
        BRIDGE-->>HOOK: SceneEffectsAPI {createParticleField, createAtmosphereField,<br/>createWispField, version}
    end
    HOOK->>BRIDGE: api.createParticleField(particleCount), createAtmosphereField(w,h),<br/>createWispField(wispCount)
    BRIDGE->>RUST: new wasmModule.ParticleField(count) etc (wasm_bindgen constructor)
    BRIDGE->>PFB: new ParticleFieldBridge(inner, memory)
    HOOK-->>COMP: {ready:true, particles, atmosphere, wisps, update}

    rect rgb(232,244,250)
    Note over PARTINST,RUST: per frame -- WasmParticleInstances.useFrame (no explicit<br/>priority, default 0)
    PARTINST->>HOOK: update(dt clamped to 0.05, camera.x, camera.y, camera.z)
    HOOK->>PFB: particles.update(dt, camX, camY, camZ)
    PFB->>RUST: inner.update(dt, camera_x, camera_y, camera_z) -- particles.rs:109
    PARTINST->>PFB: getPositions(), getOpacities(), getSizes()
    PFB->>PFB: refreshViews() -- compare current byteLength/ptr/len vs cached
    alt unchanged since last frame (steady state, expected every frame)
        PFB-->>PARTINST: return cached Float32Array view (zero allocation)
    else WebAssembly.Memory grew or wasm-side buffer relocated
        PFB->>PFB: bounds check ptr+len*4 <= byteLength, throw if violated
        PFB->>PFB: logger.warn rebuilding views (byteLength changed, posPtr changed)
        PFB->>PFB: new Float32Array(memory.buffer, ptr, len) for positions/opacities/sizes
        Note over PFB: INVARIANT this IS the memory-growth view-invalidation hazard handled --<br/>every accessor calls refreshViews() before returning a view, so a stale view (from<br/>before Memory.grow detached the old ArrayBuffer) is never returned
    end
    loop for i in 0..min(count, mesh.count)
        PARTINST->>PARTINST: setMatrixAt(i, compose(position, identityQuat, scale=size*0.12))
        PARTINST->>PARTINST: instanceColor[i] = baseColor * opacity[i]
    end
    PARTINST->>PARTINST: instanceMatrix.needsUpdate = true, instanceColor.needsUpdate = true
    end

    rect rgb(252,238,230)
    Note over COMP,RUST: unmount -- useWasmSceneEffects cleanup (useWasmSceneEffects.ts:122-140)
    HOOK->>PFB: particles.dispose(), atmosphere.dispose(), wisps.dispose()
    PFB->>PFB: _disposed=true, views nulled
    PFB->>RUST: inner.free() (wasm_bindgen drop)
    end
    Note over RUST: get_positions_ptr/get_positions_len (particles.rs:189,194),<br/>get_pixels_ptr/get_pixels_len (atmosphere.rs:160,165), get_hues_ptr/get_hues_len<br/>(energy_wisps.rs:269,272) are the raw pointer/length exports underlying every<br/>Float32Array/Uint8Array view above
```

**What it shows.** WASM particles, wisps and atmosphere are read through zero-copy views that rebuild when WebAssembly memory grows. **Why it is this way.** Copying them every frame was avoidable. Rebuilding the views on every growth stops a detached `ArrayBuffer` from ever being read. The sources are unchanged since the last stamp.

## VC-31.9 Transient agent-action beam -- store event to expiry
```mermaid
sequenceDiagram
    autonumber
    participant WS as handleAgentActionTagged<br/>store/websocket/binaryProtocol.ts:427
    participant STORE as transientBeamStore.pushBeams<br/>transientBeamStore.ts:68
    participant LAYER as TransientBeamsLayer<br/>TransientBeamsLayer.tsx:69
    participant HOOK as useTransientBeams<br/>useTransientBeams.ts:24
    participant MESH as TransientBeamMesh.updateBeam useFrame<br/>TransientBeamsLayer.tsx:166
    participant ENC as semanticEncoding agentActionColorHex/Shape<br/>semanticEncoding.ts:126,131

    Note over WS: 0x23 wire frame is a bare type tag, not a V4 header colon separated fields<br/>count:u16 then repeated len:u16 plus event bytes (store/websocket/binaryProtocol.ts:413-417, decoded at :427-430). See VC-30<br/>for the surrounding websocket store and worker plumbing.
    WS->>WS: dispatchAgentActions drops state frames (presence and lifecycle reach only the registry)<br/>store/websocket/binaryProtocol.ts:434-438, :447
    WS->>STORE: pushTransientBeams(work) exported non-React entry point<br/>(store/websocket/binaryProtocol.ts:444, transientBeamStore.ts:119)
    STORE->>STORE: pushBeams -- clampDuration(durationMs) floor MIN_BEAM_DURATION_MS=400,<br/>default DEFAULT_BEAM_DURATION_MS=1500
    STORE->>STORE: beams.concat(incoming), FIFO trim to MAX_TRANSIENT_BEAMS=256 (oldest<br/>evicted first)

    LAYER->>HOOK: useTransientBeams() subscribes to beams array
    loop every frame (LAYER's own useFrame, no explicit priority)
        LAYER->>STORE: prune() calls pruneExpired()
        STORE->>STORE: filter beams where now - startTime < durationMs
        opt something expired
            STORE->>STORE: set({beams: alive}) -- only commits when count actually changed, avoids<br/>gratuitous re-render
        end
    end

    alt beams.length === 0
        LAYER-->>LAYER: return null (nothing rendered)
    else beams present
        LAYER->>MESH: mount one TransientBeamMesh per beam (React key=beam.id)
        MESH->>ENC: agentActionShape(actionType) -- radiusTop/radiusBottom/radialSegments per verb
        Note over ENC: Create widens into target, Delete narrows into it, Query thin probe, Link<br/>thick tie, Transform rounder beam. Colours QUERY blue UPDATE yellow CREATE green DELETE<br/>red LINK purple TRANSFORM cyan
        MESH->>MESH: CylinderGeometry(radiusTop, radiusBottom, height=1, radialSegments, 1,<br/>openEnded=true) built once in useMemo
        loop every frame (own useFrame, per beam mesh)
            MESH->>MESH: resolveAgentPosition(sourceAgentId), resolveNodePosition(targetNodeId) - TransientBeamsLayer.tsx:166-172<br/>the agent resolver is agentMarkerPosition, so a beam starts at the drawn agent marker (GraphManager.tsx:174)
            alt either resolver returns false
                MESH->>MESH: mesh.visible = false, skip silently (never throws)
            else both resolved
                MESH->>MESH: progress = (now - startTime) / durationMs
                MESH->>MESH: opacity = opacityEnvelope(progress) * maxOpacity -- fade-in 0..0.18, hold,<br/>fade-out 0.68..1
                alt opacity <= 0
                    MESH->>MESH: mesh.visible = false
                else visible
                    MESH->>MESH: compose midpoint, quaternion src-to-tgt direction, scale Y = distance
                    MESH->>MESH: material.opacity = opacity, mesh.visible = true
                end
            end
        end
    end
    Note over STORE,MESH: beam removed from the React tree once pruneExpired drops it from<br/>the store array -- React unmount runs the geometry.dispose()/material.dispose() cleanup<br/>(TransientBeamsLayer.tsx:159-164)
```

**What it shows.** An agent-action event, from the 0x23 frame through the store to a fading beam. **Why it is this way.** A beam starts at the drawn agent marker, which is the only agent position (ADR-2142). A bounded FIFO and a 400 ms floor keep bursts cheap.


## VC-31.10 Frame budget instrumentation -- FPS auto-degrade and PerfProbe
```mermaid
sequenceDiagram
    autonumber
    participant R3F as R3F render loop
    participant FPS as useFpsMonitor.useFrame<br/>useFpsMonitor.ts:36
    participant SETSTORE as settingsStore.updateSettings<br/>useFpsMonitor.ts:117
    participant PP as PerfProbe.useFrame priority -10000<br/>PerfProbe.tsx:405
    participant CTRL as PerfController singleton<br/>PerfProbe.tsx:48, window.__perf

    R3F->>FPS: frame(state, delta) -- mounted inside GraphManager, GraphManager.tsx:155
    alt qualityGates.autoAdjust is false
        FPS->>FPS: reset internal tracking (samples, degradeLevel) but leave visual settings alone
    else autoAdjust enabled
        FPS->>FPS: fps = 1/delta, push to ring buffer, cap SAMPLE_SIZE=60 (about 1s at 60fps)
        alt fewer than 30 samples collected (SAMPLE_SIZE/2)
            FPS-->>R3F: wait for meaningful window
        else enough samples
            FPS->>FPS: avgFps = mean(samples)
            alt avgFps below minFpsThreshold (default 30) and degradeLevel < 3
                alt sustained below threshold for DEGRADE_SUSTAIN_MS=2000ms
                    FPS->>SETSTORE: updateSettings -- degradeLevel++, applyDegradeLevel(level)
                    Note over SETSTORE: level1 particleCount 128 atmosphereResolution 64. level2 wispsEnabled false particleCount 64. level3 sceneEffects.enabled false entirely
                end
            else avgFps recovered above threshold
                FPS->>FPS: reset lastDegradeTime = 0
            end
            alt avgFps above minFpsThreshold+10 (hysteresis band) and degradeLevel > 0
                alt sustained above band for RESTORE_SUSTAIN_MS=3000ms
                    FPS->>SETSTORE: updateSettings -- degradeLevel--, applyDegradeLevel(level)
                end
            end
        end
    end

    Note over PP: DEV-only, lazy-loaded (GraphCanvas.tsx:15), master-gated OFF by default -- zero overhead until window.__perf.on() or query perf=1 or localStorage perfProbe=1
    R3F->>PP: frame() -- priority -10000 runs FIRST every frame, before all other useFrame subscribers
    PP->>CTRL: controller.tick()
    alt controller.enabled false
        CTRL-->>PP: no-op, return immediately
    else armed
        CTRL->>CTRL: frameBuf ring buffer (FRAME_BUF=240 samples) records now-lastFrameTs
        CTRL->>CTRL: snapshot gl.info.render calls/triangles/programs, then gl.info.reset()
        opt now - lastRescan > 1000ms
            CTRL->>CTRL: rescanInstanced -- wrap any new InstancedMesh.setMatrixAt/setColorAt to count per-frame uploads
        end
        loop every reportIntervalMs=4000ms (setInterval, not useFrame)
            CTRL->>CTRL: report() -- console.table of frame p50/p90/p99, rAF callback costs, instanced upload counts, scene census
        end
    end
```

**What it shows.** The FPS auto-degrade ladder and the opt-in DEV PerfProbe. **Why it is this way.** Scene effects are the first thing shed under load. The probe runs first in every frame so it can time the rest.


## VC-31.11 Agent marker pass, a purely visual position over the drawn node positions

```mermaid
sequenceDiagram
    autonumber
    participant WS as 0x23 agent-action dispatch<br/>client/src/store/websocket/binaryProtocol.ts:434
    participant REG as agentTargetStore registry<br/>client/src/store/agentTargetStore.ts:43
    participant GM as GraphManager.useFrame<br/>client/src/features/graph/components/GraphManager.tsx:484
    participant MP as AgentMarkerPass.step<br/>client/src/features/bots/agentMarkerPass.ts:77
    participant NUDGE as stepAgentOffset<br/>client/src/features/graph/utils/agentNudge.ts:62
    participant OUT as positions map, wire id to world point<br/>client/src/features/bots/agentMarkerPass.ts:73

    WS->>REG: pushActions(actions) - presence, targets and lifecycle state<br/>client/src/store/websocket/binaryProtocol.ts:447
    GM->>MP: agentMarkers.step(liveMarkerDeps, resolveNode, triFrame, edge, clocks) once per frame, after the drawn copy<br/>client/src/features/graph/components/GraphManager.tsx:484-491, agentMarkerPass.ts:152-154
    loop each agent in registry.seenAt with live presence (agentMarkerPass.ts:82-83)
        alt no work node on record
            MP->>MP: base = scene centre - agentMarkerPass.ts:87-88
        else work node on screen
            MP->>MP: base = that node's drawn position - agentMarkerPass.ts:89-90
        else work node not on screen
            MP->>MP: skip - the marker is hidden, never drawn elsewhere - agentMarkerPass.ts:91-92
        end
        MP->>MP: plus the id-hashed hover ring and the activity drift (drift stepped once per pass)<br/>agentMarkerPass.ts:84, :94-99
        alt offsetsEnabled
            MP->>NUDGE: stepAgentOffset(off, 0, base, done or null, edge, finish, nowMs) - agentMarkerPass.ts:107-110
            alt state done and the scene edge is known
                NUDGE->>NUDGE: run outward to the scene edge over AGENT_EDGE_SECONDS 30 s, easeOutCubic on elapsed time,<br/>never past the edge, then hold - agentNudge.ts:72-92, :29
            else not done
                NUDGE->>NUDGE: offset relaxes by AGENT_RELAX_SPEED 0.03 a frame, snaps to zero under AGENT_RELAX_SNAP<br/>agentNudge.ts:94-100, :33, :38
            end
        else offsets off
            MP->>MP: offset zeroed and the 30 s run clock paused, not elapsed - agentMarkerPass.ts:111-115
        end
        MP->>OUT: positions[id] = base + offset - agentMarkerPass.ts:116-118
    end
    MP->>OUT: retired, expired or hidden ids leave the map and the state, so a reused id starts from zero<br/>agentMarkerPass.ts:120-126
    Note over OUT: every agent drawing reads this map through agentMarkerPosition (agentMarkerPass.ts:147-149): the marker<br/>spheres, the beam source (VC-31.9), nameplates, trails, collaboration edges and payment labels
    Note over MP,OUT: INVARIANT the pass is VISUAL ONLY. It writes its own map, never the drawn positions array or the shared<br/>buffer, so physics and the wire remain the single source of node position (agentMarkerPass.ts:1-8, :72-73)
    Note over MP,NUDGE: RESOLVED (VisionClaw 15acb9219, 6cc5c153d): the agentWorkTargets registry, the working nudge toward the<br/>target and the GemNodes per-instance offset array are deleted. Pass state is keyed by wire id (agentMarkerPass.ts:74, :100-104),<br/>which retires the old DEBT of a grow-only offset array indexed by visible instance.
    Note over NUDGE: the headset runs the same 30 s edge rule in its own choreography and a test fails if the two constants<br/>differ (agentNudge.ts:17-19) - see VC-36.22
```

**What it shows.** The visual-only agent marker pass over the drawn positions. **Why it is this way.** Agents are not graph nodes (ADR-2142). The pass writes its own map and never the physics buffer.

## VC-31.12 Overview layer: one GPU node source for edges and sprites
```mermaid
flowchart TB
    subgraph SRC["Inputs, render-indexed (VC-31.13)"]
        DRAWN["drawn positions, nodePositionsRef<br/>GraphManager.tsx:450"]
        MASK["highlight mask bytes and version<br/>highlightMask.ts:101, :221"]
        PLAN["edge plan: srcIdx, tgtIdx, colours, weights<br/>useEdgeBufferComputation.ts:80-87"]
    end
    DRAWN --> TEX["NodePositionTexture.update, priority -0.4<br/>OverviewLayer.tsx:86-90"]
    MASK --> TEX
    TEX --> SKIP["INVARIANT: no upload unless positions, mask version or count changed<br/>nodePositionTexture.ts:75-76"]
    SKIP --> FILL["RGBA float texel i = x, y, z, highlightLevel, 2048 wide, nearest<br/>nodePositionTexture.ts:21, :57-66, :84-92"]
    FILL --> GROW["grow-only power-of-two rows, listeners rebind uPos<br/>nodePositionTexture.ts:77-83, OverviewEdgeLines.tsx:90"]
    PLAN --> CAPN["resolveOverviewEdgeCapacity: auto = next power of two, a number caps it<br/>OverviewLayer.tsx:94-99, overviewEdgeCapacity.ts:27-33"]
    CAPN --> PICK["truncated: selectDrawnEdges keeps the heaviest, ties by plan order<br/>edgeStyle.ts:63-81"]
    CAPN --> REP["publishOverviewEdgeReport, N of M edges drawn<br/>OverviewLayer.tsx:101-105, overviewEdgeStore.ts:56-59"]
    PICK --> EL["OverviewEdgeLines: instance attributes only on data or width change<br/>OverviewEdgeLines.tsx:116-142"]
    FILL --> EL
    SET["Edge Opacity setting, the one slider 0 to 0.3, default 0.15<br/>look.ts:15, defaults.ts:316"] --> OPA["overviewEdgeOpacity: setting over default times 0.6, clamped 0 to 1<br/>edgeStyle.ts:32-38, OverviewLayer.tsx:111-112"]
    OPA --> EL
    OPA -.-> O2["OPEN: lines saturate at setting 0.25, the slider top changes only the tubes,<br/>which take the setting verbatim (GlassEdges.tsx:375-376)"]
    OPA -.-> I2["INVARIANT: the shipped default reproduces base alpha 0.6, bad input falls back to it<br/>edgeOpacity.test.ts:10-22"]
    EL --> EUN["per frame only uniforms: resolution, DPR, near, fade, opacity, grey alpha<br/>OverviewEdgeLines.tsx:144-158"]
    EUN --> EMAT["edge quad: fetch both endpoints, extrude by half width in px, length fade 50 to 150 px<br/>OverviewEdgeMaterial.ts:108, :140-143, :179"]
    FILL --> NODES["OverviewNodes, priority -0.3: mirror the knowledge gem mesh<br/>OverviewNodes.tsx:110-135"]
    NODES --> HALO["halo batch: emphasised nodes only, additive, on the bloom layer<br/>OverviewNodes.tsx:88, :137-169"]
    NODES --> NMAT["sprite size = spriteSizePx scaled by the encoded size, clamped 2 to 24 px<br/>OverviewNodeMaterial.ts:247-251"]
    FILL --> INF["InferredEdges: dashed amber, no length fade, same texture<br/>InferredEdges.tsx:55, :61-80"]
    REP --> NOTICE["OverviewEdgeNotice joins fault, node-buffer and edge text<br/>OverviewEdgeNotice.tsx:18-25"]
    WRK["worker SAB grows from the node count, else N of M nodes drawn<br/>graphWorkerProxy.ts:344-363"] --> NOTICE
    EMAT -.-> D1["three copies: the length fade is written three times, TS, GLSL and TSL<br/>edgeStyle.ts:14, OverviewEdgeMaterial.ts:140, :320-322"]
    D1 -.-> D2["DRIFT: edgeStyle.ts:4-6 says fadeAlpha drives the tube selection,<br/>only tests call it and tubes select by distance (tubeSelect.ts:148-156)"]
    NODES -.-> O1["scope accepted in ADR-2182 notes: sprites mirror only the knowledge<br/>population, ontology and agent stay gems (OverviewLayer.tsx:13, :134)"]
```

**What it shows.** How the screen-space overview gets its endpoints and centres from one texture refilled only on change, and how capacity limits become a visible notice. **Why it is this way.** ADR-2181 D1/D4 and ADR-2182 D1/D5 ask for no per-edge CPU loop, no per-node work on a settled graph and no silent truncation. `useBackendGreyoutOpacity` (`OverviewLayer.tsx:68-75`) is the single place the greyed alpha meets the blend space.

**Debt (loose end):** `OVERVIEW_EDGE_OPACITY` (`OverviewLayer.tsx:40`) survives as an alias of `OVERVIEW_EDGE_BASE_OPACITY` (`edgeStyle.ts:23`) after ec785afa9 moved the alpha onto the Edge Opacity setting (`OverviewLayer.tsx:111-112`); nothing in the client imports it.

**Debt (parallel):** the on-screen length fade is written three times: TS (`edgeStyle.ts:14`), GLSL (`OverviewEdgeMaterial.ts:140`) and TSL (`OverviewEdgeMaterial.ts:320-322`). Only the TS copy is unit-tested.

**Tension (ADR-2182 D1 vs code):** D1 names `THREE.Points` with a sized `PointsMaterial` for WebGL nodes. The code draws instanced GLSL3 quads (`OverviewNodeMaterial.ts:142`), and its test asserts "never LineSegments/Points" (`overviewMaterials.test.ts:33`). ADR-2181 records the matching edge deviation as accepted. ADR-2182 records none.

**Debt (parallel):** both overview materials are hand-written in two dialects, GLSL (`OverviewEdgeMaterial.ts:216`, `OverviewNodeMaterial.ts:142`) and TSL (`OverviewEdgeMaterial.ts:242`, `OverviewNodeMaterial.ts:158`). The tests only check that each one builds (`overviewMaterials.test.ts:1-5`), so nothing checks that the two produce the same pixels.

## VC-31.13 Highlight mask and node encoding: the render-index contracts
```mermaid
flowchart TB
    IDX["INVARIANT: one render index, a node's slot in graphData.nodes, shared by positions,<br/>analytics, mask and encoding (nodeIndex.ts:4-10, :24)"]
    subgraph W["Writers, through named sources (explore lane: other topic)"]
        WF["settings node filter greys via writeFilter exclude<br/>useGraphFiltering.ts:115-132, filterRegistry.ts:92"]
        WS["selection, neighbours, search via writeEmphasis<br/>useGraphSelection.ts:188-189, :215, filterRegistry.ts:117"]
        WH["hover<br/>GraphManager.tsx:606, filterRegistry.ts:175"]
    end
    IDX --> W
    W --> REC["HighlightMask.recompose: FILTERED_OUT if any filter excludes, emphasis flags ORed,<br/>version++ (highlightMask.ts:193-222)"]
    REC --> LVL["highlightLevel: selected or hovered beat filtered-out, which beats search and neighbour<br/>highlightMask.ts:54-58"]
    LVL --> HV["highlightVisualFor: greyed = luma mixed to the background, no glow, no label,<br/>emphasised = x1.35 size, x1.25 colour (highlightVisual.ts:59-102, :23-25)"]
    HV --> GN["GemNodes colour and scale passes<br/>GemNodes.tsx:944, :983-984"]
    LVL --> TEXL["texel w of the position texture<br/>nodePositionTexture.ts:91"]
    LVL --> LBL["label eligibility<br/>labelBudget.ts:174"]
    ENC["useNodeEncodingSync, the one encodeAll caller over all delivered nodes<br/>GraphManager.tsx:90, useNodeEncodingSync.ts:51, :103"] --> NE["nodeEncoding colours and sizes by render index, version++<br/>nodeEncoding.ts:206-227"]
    NE --> GN
    GN --> SPR["instanceColor is the sprite colour source<br/>OverviewNodes.tsx:111, :128"]
    TEXL --> BLEND["greyed alpha: as set under linear blending, corrected when WebGL<br/>draws straight to an sRGB canvas (OverviewLayer.tsx:68-75, blendSpace.ts:27-29)"]
    BLEND --> SOLVE["srgbBlendAlpha solves for a 0.5 reference colour over the background<br/>highlightVisual.ts:146, :161-170"]
    WF -.-> T1["opt-in removal accepted in ADR-2183 notes: hideFiltered drops greyed<br/>nodes from the render set (useGraphFiltering.ts:140-147)"]
    WF -.-> D1["DEBT: _hierarchyMap is unused but still passed and still a memo dependency<br/>nodeFilter.ts:112, useGraphFiltering.ts:118, :125"]
    NE -.-> D2["two copies: anomaly, community, cluster and centrality colours exist twice<br/>nodeEncoding.ts:88-102 for knowledge, GemNodes.tsx:570-593 for the rest"]
    NE -.-> D3["DRIFT: nodeEncoding.ts:10-11 cites GemNodes.tsx:454-583 and :863 for computeColor<br/>and scale, which are now at GemNodes.tsx:543 and :942"]
```

**What it shows.** The two render-indexed buffers that every drawer reads, who writes them and how a highlight bit becomes a look. **Why it is this way.** ADR-2183 D1-D3 has filters grey nodes out through one combinator instead of removing them, and ADR-2184 D4 makes the encoding the single source of the knowledge base colour, so sprites, gems and labels agree.

**Debt (parallel):** the anomaly, community, cluster and centrality colour formulas exist twice: in `nodeEncoding.ts:88-102` for knowledge nodes, and in `GemNodes.tsx:570-593` for ontology and agent nodes.

## VC-31.14 Overview-live gating and the GPU fault fallback
```mermaid
flowchart TB
    MB["useOverviewMaterial: GLSL at once on WebGL, TSL async on WebGPU,<br/>paired with its geometry (useOverviewMaterial.ts:14-44)"]
    MB --> LIVE["setOverviewLive edges or nodes when the mesh exists<br/>OverviewEdgeLines.tsx:108-113, OverviewNodes.tsx:101-107"]
    LIVE --> STATE["isOverviewLive = mounted and not faulted<br/>overviewLive.ts:41-45, :65-67"]
    STATE --> TUBES["tubeLod set only when edges are enabled and live<br/>GraphManager.tsx:96-102"]
    STATE --> GEMS["gem LOD only when the knowledge sprites are live<br/>GemNodes.tsx:1004-1007"]
    TUBES --> INV["INVARIANT: until the overview draws, or once it faults, every edge stays a tube<br/>and every gem stays drawn, so the graph never vanishes at fit (overviewLive.ts:5-11)"]
    GEMS --> INV
    subgraph F["Fault sources, useRenderFaultWatch (useRenderFaultWatch.ts:19-53)"]
        G1["WebGPU draw throws: renderObject guard calls the mesh hook<br/>rendererFactory.ts:230, renderFaults.ts:46-50"]
        G1 --> DEB["debounceFault: 3 separate frames, throws within 8 ms are one frame<br/>renderFaults.ts:24-43"]
        G2["WebGPU pipeline error: no throw, routed from three's console by material id<br/>renderFaults.ts:79-96, :106-119"]
        G3["WebGL: read program diagnostics once, within 120 frames<br/>useRenderFaultWatch.ts:17, :42-52, renderFaults.ts:59-65"]
    end
    DEB --> MARK["markOverviewFault, first reason wins, sticky until unmount<br/>overviewLive.ts:48-52, :58-62"]
    G2 --> MARK
    G3 --> MARK
    MARK --> HIDE["mesh hidden, live state drops, tubes and gems return<br/>useRenderFaultWatch.ts:25-27, overviewLive.ts:32-37"]
    HIDE --> TXT["notice: Overview ... failed on this GPU, drawing gems and glass tubes instead<br/>overviewLive.ts:78-83"]
    G1 -.-> D1["DEBT: the WebGPU guard catches every throw and warns once per object. An object<br/>with no fault hook just stops drawing, and WebGL has no guard (rendererFactory.ts:201-226)"]
    TXT -.-> T1["P4 gap: back on all-edge tubes, the 65,536 ceiling<br/>clips with a console line only (GlassEdges.tsx:412-430), which the notice never mentions"]
```

**What it shows.** The overview layers count as live only once their material has built and no fault has been seen. A throw, a pipeline error or a failed WebGL link hands the frame back to gems and tubes, and the canvas notice says so. **Why it is this way.** ADR-2181 D4 rules out a silent skip. Measured WebGPU pipeline failures log through three's console instead of throwing (`renderFaults.ts:67-76`), so the fallback has to listen there as well as wrap `renderObject`.

**Tension (ADR-2179 P4 vs fallback):** after a fault the tubes carry every edge again, and above the 65,536 ceiling they clip with only a console line (`GlassEdges.tsx:412-430`). The canvas notice reports the fault but not that clip (`overviewLive.ts:78-83`).
