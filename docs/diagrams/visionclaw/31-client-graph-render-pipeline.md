---
id: VC-31
title: R3F/Three.js graph render pipeline, overview layer and WASM scene effects
area: visionclaw
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
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
  - ../project/client/src/features/graph/overview/__tests__/overdraw.test.ts
  - ../project/client/src/features/graph/overview/__tests__/tubeSelect.test.ts
  - ../project/client/src/features/graph/utils/layoutTransition.ts
  - ../project/client/src/features/graph/state/analyticColours.ts
  - ../project/client/src/features/bots/components/lineDrawCount.ts
  - ../project/client/src/features/bots/components/AgentTrail.tsx
  - ../project/client/src/features/bots/components/AgentCollabEdges.tsx
  - ../project/client/src/features/control-center/registry/groups/motion.ts
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
  - ../project/docs/adr/ADR-2181-overview-edges-are-screen-space-lines-with-no-silent-cap.md
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
  - ../project/client/src/features/graph/hooks/useFpsMonitor.ts
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/client/src/features/graph/hooks/useGraphSelection.ts
  - ../project/client/src/store/transientBeamStore.ts
  - ../project/client/src/store/websocket/binaryProtocol.ts
  - ../project/client/src/rendering/GemPostProcessing.tsx
  - ../project/client/src/rendering/postProcessingGuard.ts
  - ../project/client/src/rendering/rendererFactory.ts
  - ../project/client/src/rendering/renderFaults.ts
  - ../project/client/src/rendering/glowEffect.ts
  - ../project/client/src/rendering/StudioEnvironment.tsx
  - ../project/client/src/rendering/bloomLayer.ts
  - ../project/client/src/rendering/troikaConfig.ts
  - ../project/client/src/rendering/materials/GemNodeMaterial.ts
  - ../project/client/src/rendering/materials/CrystalOrbMaterial.ts
  - ../project/client/src/rendering/materials/AgentCapsuleMaterial.ts
  - ../project/client/src/features/bots/agentVisualConstants.ts
  - ../project/client/src/rendering/materials/GlassEdgeMaterial.ts
  - ../project/client/src/rendering/materials/OverviewEdgeMaterial.ts
  - ../project/client/src/rendering/materials/OverviewNodeMaterial.ts
  - ../project/client/src/rendering/text/createTextMaterial.ts
  - ../project/client/src/rendering/text/GlyphAtlas.ts
  - ../project/client/src/rendering/text/textLayout.ts
  - ../project/client/src/wasm/scene-effects-bridge.ts
  - ../project/client/src/hooks/useWasmSceneEffects.ts
  - ../project/client/src/features/visualisation/components/WasmSceneEffects.tsx
  - ../project/client/src/features/visualisation/components/TransientBeamsLayer.tsx
  - ../project/client/src/features/visualisation/components/EmbeddingCloudLayer.tsx
  - ../project/client/src/features/visualisation/hooks/useTransientBeams.ts
  - ../project/client/src/features/visualisation/semanticEncoding.ts
  - ../project/client/crates/scene-effects/src/particles.rs
  - ../project/client/crates/scene-effects/src/energy_wisps.rs
  - ../project/client/crates/scene-effects/src/atmosphere.rs
  - ../project/client/src/wasm/scene-effects/scene_effects.js
---

## For developers

The desktop graph is one R3F `Canvas` whose `GraphManager` copies the worker's shared positions once a frame and feeds two edge renderers, two node renderers and one label path. Since 809a859 there is one desktop renderer, three's `WebGPURenderer`, which runs on its own WebGL2 backend where WebGPU is missing (VC-31.6). Every material is TSL; the WebGLRenderer branch, the GLSL twins, the EffectComposer bloom, the DOM label layer and the blend-space correction are deleted (renderer consolidation, 5d1b2d04d). The overview layer (VC-31.12) draws screen-space edge quads and pixel-sized sprites from one float position texture, with glass tubes and gems kept as close-zoom detail (VC-31.4, VC-31.14). A shader that fails on either backend now faults the layer that owns its material, labels included (VC-31.14). Two render-indexed buffers, the highlight mask and the node encoding, are the contracts every drawer reads (VC-31.13). The overview "slab" was a layout pinned to the 400-unit physics bounds, not edge overdraw (VC-31.15). The explore widgets (ADR-2185) and the camera fit (ADR-2180, VC-37) belong to other topics; the explore layer is VC-56.

## For the business

At the default view a visitor sees the whole knowledge graph as dots and fine lines, now as a round cloud rather than a crushed square. Filtering greys the rest out without removing it. Browsers without WebGPU get the same picture through a built-in fallback, not a second, older renderer. If the graph is larger than the screen can draw, or the graphics card cannot build part of the drawing, labels included, a notice says so on the canvas instead of parts disappearing quietly. Rich glass gems and tubes appear only when you zoom in.

## VC-31.1 Canvas consumers and shared child tree
```mermaid
flowchart LR
    ML["MainLayoutContent<br/>MainLayout.tsx:15"] -->|mounts| GCW["GraphCanvasWrapper<br/>GraphCanvasWrapper.tsx:134"]
    GCW -->|CanvasErrorBoundary wraps| GC["GraphCanvas<br/>GraphCanvas.tsx:181"]
    GC -->|"gl = createGemRenderer"| CANVAS["R3F Canvas, shadows off as PCFShadowMap<br/>GraphCanvas.tsx:267-271"]
    CANVAS --> CAS["CameraAspectSync<br/>GraphCanvas.tsx:66"]
    CANVAS --> EEC["EnsureEventsConnected<br/>GraphCanvas.tsx:110"]
    CANVAS --> PP["PerfProbe, DEV only, lazy<br/>GraphCanvas.tsx:16, PerfProbe.tsx:388"]
    CANVAS --> ENV["StudioEnvironment cube map, skipped on a software WebGL2 backend<br/>GraphCanvas.tsx:338-340, StudioEnvironment.tsx:53"]
    CANVAS --> WSE["WasmSceneEffects<br/>GraphCanvas.tsx:344, WasmSceneEffects.tsx:538"]
    CANVAS --> EPL["EchoPulseLayer<br/>GraphCanvas.tsx:360"]
    CANVAS --> SPL["SplatScene, world-locked, outside the graph root<br/>GraphCanvas.tsx:363"]
    CANVAS --> ECL["EmbeddingCloudLayer<br/>EmbeddingCloudLayer.tsx:122"]
    CANVAS -->|"canvasReady and nodeCount above 0"| GM["GraphManager<br/>GraphCanvas.tsx:373-374"]
    GM --> GN["GemNodes x3 populations<br/>GraphManager.tsx:674"]
    GM --> GE["GlassEdges x2, main and selection highlight<br/>GraphManager.tsx:702, :718"]
    GM --> OL["OverviewLayer, edges and sprites (VC-31.12)<br/>GraphManager.tsx:729-736"]
    OL --> IE["InferredEdges on the overview path<br/>GraphManager.tsx:735, InferredEdges.tsx:54"]
    GM --> KR["KnowledgeRings<br/>KnowledgeRings.tsx:32"]
    GM --> CH["ClusterHulls<br/>ClusterHulls.tsx:206"]
    GM --> VAM["VirtualAgentMarkers, agents are not graph nodes (ADR-2142)<br/>VirtualAgentMarkers.tsx:35"]
    GM --> TBL["TransientBeamsLayer<br/>GraphManager.tsx:762"]
    GM --> IL["InstancedLabels<br/>GraphManager.tsx:769"]
    CANVAS --> AO["AgentOverlay, nameplates, trails, collab edges, payment labels<br/>GraphCanvas.tsx:382, AgentOverlay.tsx:23"]
    CANVAS --> OC["OrbitControls makeDefault<br/>GraphCanvas.tsx:387"]
    CANVAS --> GPP["GemPostProcessing enabled by glow.enabled<br/>GraphCanvas.tsx:196, :408"]
    GC --> OEN["OverviewEdgeNotice, outside the Canvas<br/>GraphCanvas.tsx:420, OverviewEdgeNotice.tsx:18"]
    GC -->|"provenance.enableTimeline"| TS["TimelineScrubber<br/>GraphCanvas.tsx:433, TimelineScrubber.tsx:76"]
    GC --> SIP["SplatInfoPanel<br/>GraphCanvas.tsx:436"]
    ML --> NCM["NodeContextMenu, HTML overlay<br/>MainLayout.tsx:83, NodeContextMenu.tsx:41"]
    GM -.->|"visionclaw:node-contextmenu CustomEvent"| NCM
    RD["RightDrawer tabs: explore, Memory explorer, node detail (ADR-2189)<br/>RightDrawer.tsx:86, :287, :301, :317"] --> NDP["NodeDetailPanel embedded<br/>NodeDetailPanel.tsx:65"]
    Note1["Note: the Memory explorer and node detail left the canvas and MainLayout for<br/>the drawer (MainLayout.tsx:80). No GraphViewport.tsx exists, it was retired<br/>(GraphCanvas.tsx:369)."]
```

**What it shows.** Who mounts what inside and around the canvas at 809a859. **Why it is this way.** ADR-2189 moved the detail, explore and memory panels into one right-hand drawer, and ADR-2181 D4 needs an in-canvas notice outside the drawing surface. drei's `<Environment>` was replaced because its WebGL cube target fails on the WebGL2 backend (`StudioEnvironment.tsx:6-10`).

**Open (risk):** `StudioEnvironment` was verified only on the WebGL2 backend; the renderer lane had no WebGPU adapter, so the cube-map path on real WebGPU (`StudioEnvironment.tsx:63-69`) is unobserved.

## VC-31.2 GraphManager per-frame hot loop (priority -2) and the edge pass (-0.5)
```mermaid
sequenceDiagram
    autonumber
    participant R3F as R3F render loop
    participant GM as GraphManager.useFrame priority -2<br/>GraphManager.tsx:391
    participant WP as getPositionsSync<br/>graphWorkerProxy.ts:316
    participant DP as drawnPositions<br/>drawnPositions.ts:17,46
    participant LT as lerpLayoutTransition<br/>layoutTransition.ts:37
    participant MP as agentMarkers pass<br/>agentMarkerPass.ts:71,144
    participant SEL as useGraphSelection focusCameraOnNode writes the fly-to refs<br/>useGraphSelection.ts:120-133
    participant EBC as useEdgeBufferComputation useFrame<br/>useEdgeBufferComputation.ts:122, registered at priority -0.5 (:248)
    participant GE as GlassEdges imperative handle<br/>GlassEdges.tsx:483-571

    rect rgb(232,238,250)
    Note over GM: INVARIANT frame order by priority: GraphManager -2 (GraphManager.tsx:501), GemNodes -1 (GemNodes.tsx:1166),<br/>edges -0.5 (useEdgeBufferComputation.ts:248), position texture -0.4 (OverviewLayer.tsx:74), sprites -0.3 (OverviewNodes.tsx:198),<br/>labels 0, post-processing 1 (GemPostProcessing.tsx:270). Every reader sees this frame's drawn copy.
    R3F->>GM: frame(state, delta)
    opt flyToTargetRef.current set (GraphManager.tsx:398)
        GM->>SEL: read flyToTargetRef, flyToLookAtRef, flyToProgressRef
        GM->>GM: progress += delta / 0.6, eased 1-(1-p)^3 - GraphManager.tsx:412-414
        alt p at least 1
            GM->>GM: camera.position.copy(dest), cancelFlyTo() - GraphManager.tsx:424-428
        else in flight
            GM->>GM: lerp camera and OrbitControls pivot - GraphManager.tsx:416-419
        end
    end
    GM->>WP: getPositionsSync() - GraphManager.tsx:433
    alt positions null
        GM-->>R3F: return (skip rest of frame)
    else positions available
        opt nothing published yet and the prefix is all zero
            GM-->>R3F: return, so the overview never sees an all-zero texture - GraphManager.tsx:436-442
        end
        GM->>DP: drawn = drawnPositions.copy(positions), one copy a frame - GraphManager.tsx:446, drawnPositions.ts:26
        GM->>GM: nodePositionsRef.current = drawn, setSharedNodePositions(drawn) - GraphManager.tsx:447-448
        Note over GM: the module-level singleton publishes the drawn copy without a React context (NodePositionContext.tsx:7-8)
        opt transitionRef.current.active (layout mode change)
            GM->>LT: easeInOutQuad progress, lerp every slot by its own degree - GraphManager.tsx:451-459
            LT->>LT: progress / transitionMassFactor(degree), 1 / (1 + sqrt(cc)*0.3) - layoutTransition.ts:9-11, :48
        end
        Note over GM,LT: degrees come from slotDegrees, keyed by the id in each slot, built once per transition<br/>(GraphManager.tsx:289, layoutTransition.ts:18-31). The old render-index lookup is gone.
        GM->>GM: every 30 frames the scene edge from sceneFitBounds - GraphManager.tsx:462-467
        GM->>MP: agentMarkers.step(..., edge, clocks), the only agent position writer - GraphManager.tsx:471-478 (ADR-2142, ADR-2135)
        GM->>DP: applyDrag(dragIndex, hand) once, into the copy - GraphManager.tsx:480-482, drawnPositions.ts:36
        GM->>GM: tickCameraFit() (fit itself: VC-37) - GraphManager.tsx:484
        GM->>GM: fillLabelPositions(drawn, labelCount, labelArr) - GraphManager.tsx:486-499
    end
    end

    rect rgb(240,232,250)
    Note over EBC: the memoised plan is rebuilt only on topology, filter or scaling change (useEdgeBufferComputation.ts:81-88, edgeBufferPlan.ts:67)
    R3F->>EBC: frame()
    alt positions short of nodes.length*3
        EBC-->>R3F: return - useEdgeBufferComputation.ts:131
    else sufficient
        opt tubeLod set: the overview edges are enabled and not faulted, pending included (VC-31.4)
            EBC->>EBC: selectTubeEdges on mask or LOD change, or every 6 frames while camera or layout moves<br/>useEdgeBufferComputation.ts:137-159, tubeSelect.ts:134
        end
        EBC->>EBC: EdgeFrameGate.shouldSkip(frame), same plan, handles, selection, subset and positions<br/>useEdgeBufferComputation.ts:173, edgeBufferPlan.ts:306
        EBC->>EBC: fillEdgeFrame over the whole plan, or over the tube subset only - useEdgeBufferComputation.ts:177-179
        opt selectedNodeId set
            EBC->>GE: fillHighlightFrame then highlightEdgeFlowRef.updatePoints - useEdgeBufferComputation.ts:185-188, edgeBufferPlan.ts:214
        end
        EBC->>GE: updateWidths BEFORE updatePoints, then updateColors - useEdgeBufferComputation.ts:206-209
    end
    end
    Note over GM,EBC: the plan is returned to GraphManager, which hands it to OverviewLayer (useEdgeBufferComputation.ts:250, GraphManager.tsx:504, :730). See VC-30 for the worker SAB.
```

**What it shows.** One frame of the CPU side: drawn copy, layout transition, agent markers, drag, then the edge pass that feeds either every edge or only the tube subset. **Why it is this way.** One copy of the shared buffer per frame keeps every drawer on the same positions (ADR-2135, ADR-2142). ADR-2181 D3 makes tubes a close-zoom subset whose CPU cost scales with the tube budget, not the edge count. The transition mass now follows the node in each slot (a2ecefdb0), and the unread label tick and frustum are deleted.

## VC-31.3 Labels: priority budget, one glyph path, a labels fault
```mermaid
flowchart TB
    subgraph MATB["glyph material, InstancedLabels.tsx:382-397"]
        BUILD["createTslTextMaterial for either backend, paired with its geometry<br/>InstancedLabels.tsx:387-390, createTextMaterial.ts:18"]
        BUILD -->|"build rejects"| BF["markOverviewFault labels, labels hidden<br/>InstancedLabels.tsx:391-394"]
        BUILD --> WATCH["draw throw (debounced) or shader failure: fault labels<br/>InstancedLabels.tsx:401-412"]
    end
    WATCH --> GLY["useFrame: a faulted labels layer stays hidden<br/>InstancedLabels.tsx:416-419"]
    BUILD -.->|"while building nothing draws, no DOM layer"| NOTE0["ADR-2186 D4 one label path<br/>InstancedLabels.tsx:257-262"]
    GLY --> CAP["glyph buffer from the budget, not MAX_GLYPHS<br/>InstancedLabels.tsx:308-309, labelBudget.ts:51-54"]
    GLY --> MOT["camera moving fast: instanceCount 0, then 150 ms still<br/>InstancedLabels.tsx:441-454"]
    MOT --> PATCH["every still frame: patch aLabelPos from the drawn array<br/>InstancedLabels.tsx:456-477"]
    PATCH --> DUE["runner.due: 250 ms, camera stop or a mask version change<br/>InstancedLabels.tsx:446, labelRunner.ts:67-71"]
    DUE --> RUN["LabelBudgetRunner.run: rank on population or analytics change,<br/>project every node, read mask bits<br/>InstancedLabels.tsx:545, labelRunner.ts:89-148"]
    RUN --> SEL["selectLabels<br/>labelBudget.ts:160"]
    SEL --> T1["1. selection tier: selected and hovered always, then search hits,<br/>then focus neighbours by priority<br/>labelBudget.ts:176-193"]
    T1 --> T1B["1b. cluster labels out of the overview budget<br/>labelBudget.ts:195-206"]
    T1B --> T2["2. overview tier: global top-budget by centrality or degree<br/>labelBudget.ts:208-216, :61-80"]
    T2 --> T3["3. dynamic tier: best remaining on-screen nodes<br/>labelBudget.ts:218-225"]
    T3 --> GREY["greyed nodes are never eligible, a taken grid cell skips<br/>labelBudget.ts:174, :214"]
    GREY --> LAY["layoutTextInline per chosen node, font sized in CSS px at depth<br/>InstancedLabels.tsx:568-569, :587"]
    LAY --> CL["cluster labels at centroids, not position-patched<br/>InstancedLabels.tsx:609-625"]
    LAY -.-> OFF["parallel sizing: label lift uses computeNodeScale times nodeSize<br/>InstancedLabels.tsx:574-575, while gems size from the encoding<br/>and the emphasis scale, GemNodes.tsx:883"]
    BF -.-> DUPF["parallel fault wiring: the glyph material builds and watches itself,<br/>beside useOverviewMaterial and useRenderFaultWatch<br/>InstancedLabels.tsx:382-412, useOverviewMaterial.ts:14, useRenderFaultWatch.ts:16"]
```

**What it shows.** The label pipeline after ADR-2186 and the renderer consolidation: one importance-ranked budget, refreshed about 4 times a second, drawn as glyph quads with one TSL material on both backends; a failed build or draw becomes a `labels` fault on the canvas notice. **Why it is this way.** Proximity labels crowded the overview, and a fixed budget ranked by centrality (ADR-2186 D1-D2) keeps the count and the glyph buffer bounded. With one renderer the DOM fallback had no job left, so ADR-2186 D4's single label path now holds.

**Debt (parallel):** labels are lifted by `computeNodeScale × nodeSize` (`InstancedLabels.tsx:574-575`), but gems are sized from the encoding and the emphasis scale (`GemNodes.tsx:883`), so a label sits wrong on a size-by column or an emphasised node.

**Debt (parallel):** the glyph material builds, pairs and fault-watches itself (`InstancedLabels.tsx:382-412`), a hand copy of `useOverviewMaterial` (`useOverviewMaterial.ts:14-46`) and `useRenderFaultWatch` (`useRenderFaultWatch.ts:16-33`); `buildFault` says it serves "the glyph material" (`overviewLive.ts:77`), yet the labels call `markOverviewFault` directly (`InstancedLabels.tsx:394`).

**Drift:** the comment at `InstancedLabels.tsx:377-379` still says WebGL uses "the GLSL one at once" and that "the DOM fallback takes over"; neither exists at 809a859 (`InstancedLabels.tsx:257-262`, `:387`).

## VC-31.4 GlassEdges instancing, now the close-zoom tube LOD
```mermaid
sequenceDiagram
    autonumber
    participant CALLER as useEdgeBufferComputation useFrame<br/>useEdgeBufferComputation.ts:122
    participant TS as selectTubeEdges<br/>tubeSelect.ts:134
    participant GE as GlassEdges<br/>GlassEdges.tsx:210
    participant GEOM as createGlassEdgeGeometry<br/>GlassEdgeMaterial.ts:128
    participant CIM as computeInstanceMatrices<br/>GlassEdges.tsx:133
    participant MESH as THREE.InstancedMesh

    Note over GEOM: CylinderGeometry radius, height 1, open-ended. Default radius 0.03 (GlassEdgeMaterial.ts:128),<br/>but GlassEdges passes settings.baseWidth, falling back to 0.1 (GlassEdges.tsx:241)
    Note over GE: EDGE_INITIAL 1024, DEFAULT_EDGE_CEILING 65536 (GlassEdges.tsx:40-41). The highlight instance sits on the bloom layer (GlassEdges.tsx:294, :602)

    rect rgb(240,236,252)
    Note over CALLER,TS: tubeLod from resolveTubeLod: null only when the overview edges are off or faulted, so a pending overview<br/>(startup) already gets the subset. maxDistance = gemLodDistance, or 0 when glassAtCloseZoom is off, budget 4096<br/>(GraphManager.tsx:100-107, tubeSelect.ts:49-58, :36). Pinned by tubeSelect.test.ts:122-149
    CALLER->>TS: selectTubeEdges(plan, adjacency, positions, mask, emphasis, camera, lod) - useEdgeBufferComputation.ts:149-153
    TS->>TS: 1. edges of hovered nodes and focus-to-neighbour edges - tubeSelect.ts:158-169
    TS->>TS: 2. edges of the nodes nearest the camera within maxDistance, nearest first - tubeSelect.ts:171-196
    TS->>TS: selected-node edges skipped, the highlight tubes draw them - tubeSelect.ts:150-151
    TS-->>CALLER: subset indices in plan order, never above budget - tubeSelect.ts:198-200
    Note over TS: ADR-2181 D3 gap, D3 picks close-zoom edges by on-screen length, the code by world distance<br/>of a node to the camera (tubeSelect.ts:172-180)
    Note over TS: ADR-2183 D1 gap, take() checks only the SELECTED bit (tubeSelect.ts:147-156) and GlassEdges reads no mask,<br/>so a greyed edge near the camera is drawn as a full-colour tube
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

**Tension (ADR-2181 D3 vs code):** D3 picks close-zoom tubes by on-screen edge length. The code picks the edges of nodes within `gemLodDistance` world units of the camera (`tubeSelect.ts:172-180`, `GraphManager.tsx:106`).

**Tension (ADR-2183 D1 vs code):** the tube subset skips only SELECTED edges (`tubeSelect.ts:147-156`), and `GlassEdges` reads no mask, so at close zoom a greyed edge near the camera is drawn as a full-colour tube over its greyed overview line.

## VC-31.5 Node geometries, materials, LOD and colour order per population
```mermaid
classDiagram
    class GemNodes {
        GemNodes.tsx:201 GemNodesInner
        GemNodes.tsx:326 capacity nextPowerOf2 of max nodes 4096
        GemNodes.tsx:484-490 getOverviewSource for the sprites
        GemNodes.tsx:947 gem LOD only for the sprite population (props.spriteMode, chosen by size) while sprites are live
        GemNodes.tsx:1057 a non-LOD gem gets scale 0
    }
    class GemLod {
        gemLod.ts:80 selectGemLod
        gemLod.ts:12 GEM_BUDGET 4096, nearest within gemLodDistance
        GemNodes.tsx:970 selected and hovered are forced
    }
    class ColourPass {
        GemNodes.tsx:913 computeColor, knowledge from the encoding at GemNodes.tsx:183-186
        GemNodes.tsx:914-918 timeline tint
        GemNodes.tsx:922-923 highlightVisualFor grey or emphasis
        GemNodes.tsx:928-930 heat brighten, then setColorAt
        GemNodes.tsx:1151-1161 2 Hz heat repaint from the cached base
    }
    class GemGeometry {
        GemNodeMaterial.ts:269 createGemGeometry
        THREE.IcosahedronGeometry radius 0.5 detail 1, knowledge
    }
    class CrystalOrbGeometry {
        CrystalOrbMaterial.ts:161 createCrystalOrbGeometry
        THREE.IcosahedronGeometry radius 0.5 detail 1, ontology
    }
    class AgentCapsuleGeometry {
        AgentCapsuleMaterial.ts:179 createAgentCapsuleGeometry
        THREE.CapsuleGeometry radius 0.3 length 0.6, agent
    }
    class GemNodeMaterial {
        GemNodeMaterial.ts:38 createGemNodeMaterial
        MeshStandardMaterial, TSL emissive and opacity on WebGPU only
    }
    class CrystalOrbMaterial {
        CrystalOrbMaterial.ts:35 createCrystalOrbMaterial
    }
    class AgentCapsuleMaterial {
        AgentCapsuleMaterial.ts:35 createAgentCapsuleMaterial
        AgentCapsuleMaterial.ts:76 createTslAgentCapsuleMaterial
        AgentCapsuleMaterial.ts:161-163 activity below 0.1 fades alpha to 0.3
    }
    class AgentStatusActivity {
        agentVisualConstants.ts:96-104 AGENT_STATUS_ACTIVITY table
        agentVisualConstants.ts:107 agentStatusActivity
        GemNodes.tsx:1131 packed into meta.w for agents only
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
    note for ColourPass "ADR-2183 D3 gap: heat brightens<br/>after the grey-out, so a greyed node an agent<br/>touches lights up again, and the sprites copy<br/>these colours (OverviewNodes.tsx:149)"
    note for AgentStatusActivity "DRIFT: AgentCapsuleMaterial.ts:159-160 says a done<br/>agent has no recent work-target beam, but only<br/>offline (0.0) falls below the 0.1 threshold"
```

**What it shows.** The three populations' meshes and the colour pass's order: encoding, timeline, highlight, heat. **Why it is this way.** ADR-2182 D2 keeps the gem look as close-zoom detail for knowledge nodes only, and ADR-2184 D4 moves the knowledge base colour into the shared encoding.

**Tension (ADR-2183 D3 vs code):** greyed nodes get no glow, yet attention heat is applied after the grey-out (`GemNodes.tsx:928-930`, `:1151-1161`), so a greyed node an agent touches brightens, and the sprites copy that colour (`OverviewNodes.tsx:149`).

## VC-31.6 One renderer: WebGPURenderer, its WebGL2 fallback, the guards and bloom scope
```mermaid
sequenceDiagram
    autonumber
    participant CANVAS as R3F Canvas gl prop
    participant RF as createGemRenderer<br/>rendererFactory.ts:106
    participant WGPU as WebGPURenderer three/webgpu
    participant RFX as renderFaults watches<br/>renderFaults.ts:101, :145
    participant GPP as GemPostProcessing<br/>GemPostProcessing.tsx:48
    participant GUARD as initRenderGuard<br/>postProcessingGuard.ts:44

    CANVAS->>RF: await createGemRenderer(defaultProps)
    RF->>WGPU: new WebGPURenderer(forceWebGL = forceWebGLOverride) - rendererFactory.ts:112-121
    Note over RF,WGPU: no navigator.gpu or no adapter: three's own getFallback runs the WebGL2 backend,<br/>so there is no second renderer class (rendererFactory.ts:97-105)
    RF->>WGPU: race init() against 5 s - rendererFactory.ts:122-127
    alt init throws or times out, and WebGL2 was not forced
        RF->>WGPU: dispose, new WebGPURenderer(forceWebGL true), await init() - rendererFactory.ts:128-134
    end
    RF->>WGPU: ACES tone map, sRGB output, pixel ratio, setSize, ResizeObserver - rendererFactory.ts:136-159
    RF->>RFX: installPipelineErrorWatch on three's console hook, installShaderErrorWatch - rendererFactory.ts:163-167
    RF->>WGPU: wrap renderObject: catch, warn once per object, notifyRenderFault - rendererFactory.ts:173-189
    RF->>RF: rendererCapabilities backend from backendIsWebGPU, never constructor.name - rendererFactory.ts:84-87, :191-212
    RF-->>CANVAS: the one WebGPURenderer, on the WebGPU or the WebGL2 backend

    Note over GPP: glow.* alone drives the effect (glowEffect.ts:22-32), bloom.* is retired. bloomScope emphasis (default), scene or off<br/>GemPostProcessing.tsx:67-75
    alt emphasis
        GPP->>GPP: second pass from a camera on BLOOM_LAYER only, background nulled, threshold 0 - GemPostProcessing.tsx:141-160
    else scene
        GPP->>GPP: scenePass to texture, bloom with the threshold, add - GemPostProcessing.tsx:126-137, :160-162
    else off or glow disabled
        GPP->>GPP: no pipeline, R3F renders the frame - GemPostProcessing.tsx:83-84, :98-107
    end
    GPP->>GPP: RenderPipeline(gl, outputNode), pipelineReady - GemPostProcessing.tsx:165-171
    loop priority 1 while pipelineReady and not ppDisabled (GemPostProcessing.tsx:222-270)
        GPP->>GPP: copy the scene camera into bloomCamera, keep the bloom-only layer mask - GemPostProcessing.tsx:225-230
        alt render succeeds
            GPP->>GUARD: recordRenderSuccess
        else throws
            GPP->>GUARD: recordRenderFailure, MAX 8 (GemPostProcessing.tsx:24)
            opt 8 in a row
                GPP->>GPP: dispose, setPpDisabled(true), R3F renders from now on - GemPostProcessing.tsx:253-268
            end
        end
    end
    Note over GPP: OPEN: a pipeline that fails to BUILD is only logged (GemPostProcessing.tsx:179-181), glow is then<br/>silently absent and no notice says so
```

**What it shows.** One renderer class for the desktop, with three's WebGL2 backend as the fallback, the two failure watches installed for both backends, and glow limited to the bloom layer by default. **Why it is this way.** The operator chose WebGPURenderer only (renderer consolidation, 5d1b2d04d; ADR-2181/2182 implementation notes): every material is TSL, so the WebGL2 backend draws the same scene and the GLSL twins could go. ADR-2182 D4 keeps selective bloom on `BLOOM_LAYER` (`bloomLayer.ts:13`), and ADR-2181 D4 requires the renderObject guard to report faults.

**Tension (parallel):** RESOLVED at visionclaw 46edc196 — the two readers of `glow.enabled` now agree: `GraphCanvas.tsx:196` treats a missing value as on (`!== false`) and `glowEffect.ts:25-27` applies the same rule, so the AND at `GemPostProcessing.tsx:84` no longer lets one default silence the other.

**Open (risk):** `installShaderErrorWatch` patches three's private `backend.createRenderPipeline` and reads `backend.get(...).programGPU` (`renderFaults.ts:148-163`). If a three upgrade renames either, the guard at `:148` turns the watch into a silent no-op, and WebGL2 link failures stop faulting their layer.

**Open (risk):** the WebGL2 retry after a failed WebGPU init is not itself guarded (`rendererFactory.ts:132-133`), and a forced-WebGL init failure rethrows (`:129`); either reaches the canvas error boundary rather than a notice.

## VC-31.7 Text rendering: one glyph atlas, one TSL material
```mermaid
sequenceDiagram
    autonumber
    participant BOOT as troika configureTextBuilder<br/>troikaConfig.ts:19
    participant IL as InstancedLabels useMemo<br/>InstancedLabels.tsx:329
    participant ATLAS as createGlyphAtlas<br/>GlyphAtlas.ts:32
    participant UNI as createTextUniforms<br/>createTextMaterial.ts:74
    participant TSL as createTslTextMaterial<br/>createTextMaterial.ts:18
    participant LAY as layoutTextInline<br/>textLayout.ts:87

    BOOT->>BOOT: useWorker false when crossOriginIsolated - troikaConfig.ts:21-22, for small agent labels only
    IL->>ATLAS: createGlyphAtlas(), 48 px default, module cache - InstancedLabels.tsx:330, GlyphAtlas.ts:27, :33
    ATLAS->>ATLAS: 1024 canvas, strokeText outline then fillText per char - GlyphAtlas.ts:26, :70, :74
    ATLAS-->>IL: CanvasTexture plus metrics plus lineHeight - GlyphAtlas.ts:90-100
    Note over ATLAS: ADR-2186 D4 gap, D4 names an SDF or MSDF atlas, the atlas is a stroked bitmap (GlyphAtlas.ts:70, :74)
    IL->>UNI: uniforms per glyph mesh, the TextUniforms type - InstancedLabels.tsx:332, createTextMaterial.ts:4
    IL->>TSL: createTslTextMaterial(uniforms, geometry), bound to that geometry - InstancedLabels.tsx:387
    TSL->>TSL: billboard and atlas sampling in TSL, discard below 0.01 alpha - createTextMaterial.ts:56
    alt build rejects
        TSL-->>IL: labels fault, labels hidden (VC-31.3) - InstancedLabels.tsx:391-394
    end
    IL->>LAY: layoutTextInline into the attribute buffers at the label cadence - InstancedLabels.tsx:587
    LAY-->>IL: glyphCount, zero allocation
```

**What it shows.** The glyph atlas and the single TSL glyph material, which WebGPURenderer compiles to WGSL or, on its WebGL2 backend, to GLSL. **Why it is this way.** ADR-2186 D4 asks for one label path. With one renderer the hand-kept GLSL twin was deleted (5d1b2d04d), so only the TSL material remains.

**Tension (ADR-2186 D4 vs code):** D4 specifies an SDF or MSDF atlas, but the atlas is a stroked bitmap raster (`GlyphAtlas.ts:70`, `:74`), so glyphs soften when scaled.

## VC-31.8 WASM scene-effects zero-copy -- init, tick, view rebuild, dispose
```mermaid
sequenceDiagram
    autonumber
    participant COMP as WasmSceneEffects<br/>WasmSceneEffects.tsx:538
    participant HOOK as useWasmSceneEffects<br/>useWasmSceneEffects.ts:53
    participant BRIDGE as initSceneEffects<br/>scene-effects-bridge.ts:453
    participant WASM as scene_effects.js glue (built output)<br/>client/src/wasm/scene-effects
    participant RUST as ParticleField / EnergyWisps / AtmosphereField<br/>particles.rs:65, energy_wisps.rs:80, atmosphere.rs:72
    participant PFB as ParticleFieldBridge<br/>scene-effects-bridge.ts:84
    participant PARTINST as WasmParticleInstances useFrame<br/>WasmSceneEffects.tsx:151

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
            MESH->>MESH: resolveAgentPosition(sourceAgentId), resolveNodePosition(targetNodeId) - TransientBeamsLayer.tsx:166-172<br/>the agent resolver is agentMarkerPosition, so a beam starts at the drawn agent marker (GraphManager.tsx:179)
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
    participant PP as PerfProbe.useFrame priority -10000<br/>PerfProbe.tsx:407
    participant CTRL as PerfController singleton<br/>PerfProbe.tsx:49, window.__perf

    R3F->>FPS: frame(state, delta) -- mounted inside GraphManager, GraphManager.tsx:160
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

    Note over PP: DEV-only, lazy-loaded (GraphCanvas.tsx:16), master-gated OFF by default -- zero overhead until window.__perf.on() or query perf=1 or localStorage perfProbe=1
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
    participant GM as GraphManager.useFrame<br/>client/src/features/graph/components/GraphManager.tsx:471
    participant MP as AgentMarkerPass.step<br/>client/src/features/bots/agentMarkerPass.ts:77
    participant NUDGE as stepAgentOffset<br/>client/src/features/graph/utils/agentNudge.ts:62
    participant OUT as positions map, wire id to world point<br/>client/src/features/bots/agentMarkerPass.ts:73

    WS->>REG: pushActions(actions) - presence, targets and lifecycle state<br/>client/src/store/websocket/binaryProtocol.ts:447
    GM->>MP: agentMarkers.step(liveMarkerDeps, resolveNode, triFrame, edge, clocks) once per frame, after the drawn copy<br/>client/src/features/graph/components/GraphManager.tsx:471-478, agentMarkerPass.ts:152-154
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
        DRAWN["drawn positions, nodePositionsRef<br/>GraphManager.tsx:447"]
        MASK["highlight mask bytes and version<br/>highlightMask.ts:101, :221"]
        PLAN["edge plan: srcIdx, tgtIdx, colours, weights<br/>useEdgeBufferComputation.ts:81-88"]
    end
    DRAWN --> TEX["NodePositionTexture.update, priority -0.4, then setOverviewPositions(true)<br/>OverviewLayer.tsx:67-74"]
    MASK --> TEX
    TEX --> SKIP["INVARIANT: no upload unless positions, mask version or count changed<br/>nodePositionTexture.ts:75-76"]
    SKIP --> FILL["RGBA float texel i = x, y, z, highlightLevel, 2048 wide, nearest<br/>nodePositionTexture.ts:21, :57-66, :84-92"]
    FILL --> GROW["grow-only power-of-two rows, listeners rebind uPos<br/>nodePositionTexture.ts:77-83, OverviewEdgeLines.tsx:89"]
    PLAN --> CAPN["resolveOverviewEdgeCapacity: auto = next power of two, a number caps it<br/>OverviewLayer.tsx:79-84, overviewEdgeCapacity.ts:27-33"]
    CAPN --> PICK["truncated: selectDrawnEdges keeps the heaviest, ties by plan order<br/>edgeStyle.ts:145-163"]
    CAPN --> REP["publishOverviewEdgeReport, N of M edges drawn<br/>OverviewLayer.tsx:86-90, overviewEdgeStore.ts:56-59"]
    PICK --> EL["OverviewEdgeLines: instance attributes only on data or width change<br/>OverviewEdgeLines.tsx:115-141"]
    FILL --> EL
    SET["Edge Opacity (all edges), Look - Edges, 0 to 0.3, default 0.15<br/>look.ts:15, defaults.ts:305"] --> OPA["overviewEdgeOpacity: piecewise linear, default maps to 0.6, slider top to 1<br/>edgeStyle.ts:107-120, OverviewLayer.tsx:98-99"]
    OPA --> EL
    OPA -.-> I2["INVARIANT: the default reproduces base alpha 0.6, the whole slider is live,<br/>bad input falls back (edgeOpacity.test.ts:12-35)"]
    EL --> EUN["per frame only uniforms, and hidden until positions are ready or once faulted<br/>OverviewEdgeLines.tsx:143-158"]
    EUN --> EMAT["edge quad: fetch both endpoints, extrude by half width in px,<br/>alpha = fadeAlphaWith on TSL nodes times opacity<br/>OverviewEdgeMaterial.ts:169-214, :200"]
    EMAT -.-> ONE["one fade: fadeAlphaWith over ScalarOps, numbers for the tests,<br/>TSL nodes for the shader (edgeStyle.ts:16-26, :79-85, :51-68)"]
    FILL --> NODES["OverviewNodes, priority -0.3: mirror the knowledge gem mesh,<br/>hidden until positions are ready (OverviewNodes.tsx:120-152)"]
    NODES --> HALO["halo batch: emphasised nodes only, additive, on the bloom layer<br/>OverviewNodes.tsx:98, :109, :159-170"]
    NODES --> NMAT["sprite size = spriteSizePx scaled by the encoded size, clamped 2 to 24 px<br/>OverviewNodeMaterial.ts:157-166"]
    FILL --> INF["InferredEdges: dashed amber, same texture and edge component<br/>InferredEdges.tsx:54-55, :76-84"]
    REP --> NOTICE["OverviewEdgeNotice joins fault, node-buffer and edge text<br/>OverviewEdgeNotice.tsx:18-25"]
    WRK["worker SAB grows from the node count, else N of M nodes drawn<br/>graphWorkerProxy.ts:344-363"] --> NOTICE
    NODES -.-> O1["scope moved since the ADR-2182 note: the sprites mirror one population, the largest at 1000 nodes or more,<br/>so an ontology-dominant scene hands off as a knowledge one, the rest stay gems<br/>(gemLod.ts:23, :33-41, GraphManager.tsx:667-668, OverviewLayer.tsx:53-54, :121, the header at :13 still says knowledge)"]
    OPA -.-> O2["OPEN: the tubes take the same setting verbatim (GlassEdges.tsx:375-376),<br/>so one slider means two different curves"]
```

**What it shows.** How the screen-space overview gets its endpoints and centres from one texture refilled only on change, how one Edge Opacity setting maps onto the line alpha, and how capacity limits become a visible notice. **Why it is this way.** ADR-2181 D1/D4 and ADR-2182 D1/D5 ask for no per-edge CPU loop, no per-node work on a settled graph and no silent truncation. The renderer consolidation left one TSL material per layer and one length fade (`edgeStyle.ts:1-10`); the scene blends in linear space on both backends, so the grey-out alpha is used as set (`OverviewLayer.tsx:93-95`).

**Invariant:** the overview meshes stay hidden until real positions reach the texture (`overviewLive.ts:17-20`, `OverviewEdgeLines.tsx:146`, `OverviewNodes.tsx:123-125`), and GraphManager publishes `nodePositionsRef` only once the stream has non-zero positions (`GraphManager.tsx:436-442`), so the sprites never collapse onto the origin while the gems are hidden.

**Open (risk):** `setOverviewPositions(true)` runs every frame once `nodePositionsRef` is set (`OverviewLayer.tsx:67-73`), relying on GraphManager's first-frame all-zero check (`GraphManager.tsx:436-442`); that check runs only while nothing is published, so a later all-zero buffer (a reset stream) still counts as ready.

## VC-31.13 Highlight mask and node encoding: the render-index contracts
```mermaid
flowchart TB
    IDX["INVARIANT: one render index, a node's slot in graphData.nodes, shared by positions,<br/>analytics, mask and encoding (nodeIndex.ts:4-10, :24)"]
    subgraph W["Writers, through named sources (explore lane: other topic)"]
        WF["settings node filter greys via writeFilter exclude<br/>useGraphFiltering.ts:115-131, filterRegistry.ts:92"]
        WS["selection, neighbours, search via writeEmphasis<br/>useGraphSelection.ts:188-189, :215, filterRegistry.ts:117"]
        WH["hover<br/>GraphManager.tsx:593, filterRegistry.ts:175"]
    end
    IDX --> W
    W --> REC["HighlightMask.recompose: FILTERED_OUT if any filter excludes, emphasis flags ORed,<br/>version++ (highlightMask.ts:193-222)"]
    REC --> LVL["highlightLevel: selected or hovered beat filtered-out, which beats search and neighbour<br/>highlightMask.ts:54-58"]
    LVL --> HV["highlightVisualFor: greyed = luma mixed to the background, no glow, no label,<br/>emphasised = x1.35 size, x1.25 colour (highlightVisual.ts:58-88, :22-24)"]
    HV --> GN["GemNodes colour and scale passes<br/>GemNodes.tsx:883, :922"]
    LVL --> TEXL["texel w of the position texture<br/>nodePositionTexture.ts:91"]
    LVL --> LBL["label eligibility<br/>labelBudget.ts:174"]
    ENC["useNodeEncodingSync, the one encodeAll caller over all delivered nodes<br/>GraphManager.tsx:92, useNodeEncodingSync.ts:48, :100"] --> NE["nodeEncoding colours and sizes by render index, version++<br/>nodeEncoding.ts:206-226"]
    AC["analyticColours: communityHsl, clusterHsl, centralityHsl, anomalyRgb<br/>analyticColours.ts:12-28"] --> NE
    AC --> GN
    NE --> GN
    GN --> SPR["instanceColor is the sprite colour source<br/>OverviewNodes.tsx:126, :149"]
    TEXL --> LIN["greyed alpha used as set, both backends blend in linear space<br/>OverviewLayer.tsx:93-95, OverviewEdgeMaterial.ts:205"]
    WF -.-> T1["opt-in removal accepted in ADR-2183 notes: hideFiltered drops greyed<br/>nodes from the render set (useGraphFiltering.ts:140-149)"]
```

**What it shows.** The two render-indexed buffers that every drawer reads, who writes them, and how a highlight bit becomes a look. The analytic colour formulas now live once, in `analyticColours.ts`, read by the encoding for knowledge nodes (`nodeEncoding.ts:36`, `:91-100`) and by GemNodes for ontology and agent nodes (`GemNodes.tsx:36`, `:529-540`). **Why it is this way.** ADR-2183 D1-D3 has filters grey nodes out through one combinator instead of removing them, and ADR-2184 D4 makes the encoding the single source of the knowledge base colour, so sprites, gems and labels agree. The blend-space correction went with the WebGLRenderer, its only cause (5d1b2d04d).

## VC-31.14 Overview-live gating and the GPU fault fallback
```mermaid
flowchart TB
    MB["useOverviewMaterial: one TSL build, async, paired with its geometry,<br/>a rejected build calls onError (useOverviewMaterial.ts:14-46)"]
    MB -->|"onError = buildFault(layer)"| MARK
    MB --> LIVE["setOverviewLive edges or nodes when the mesh exists<br/>OverviewEdgeLines.tsx:107-112, OverviewNodes.tsx:111-117"]
    POS["setOverviewPositions: first upload of real positions<br/>OverviewLayer.tsx:73, overviewLive.ts:59-63"] --> STATE
    LIVE --> STATE["live = positions ready and mounted and not faulted<br/>overviewLive.ts:43-50, :93-95"]
    STATE --> GEMS["gem LOD only when the sprites of the chosen population are live<br/>GemNodes.tsx:947, gemLod.ts:23, :33-41"]
    TUBES["tubeLod null only when overview edges are off or faulted,<br/>pending already takes the subset (tubeSelect.ts:49-58)"]
    GEMS --> INV["INVARIANT: until the sprites draw, or once they fault, every gem stays drawn,<br/>so the graph never vanishes at fit (overviewLive.ts:5-12)"]
    subgraph F["Fault sources, all routed per material or per mesh"]
        G1["draw throws: renderObject guard calls the mesh hook<br/>rendererFactory.ts:175-189, renderFaults.ts:46-50"]
        G1 --> DEB["debounceFault: 3 separate frames, throws within 8 ms are one frame<br/>renderFaults.ts:24-43"]
        G2["WebGPU backend: pipeline error never throws, routed from three's console by material id<br/>renderFaults.ts:73-91, :101-114"]
        G3["WebGL2 backend: renderer.debug.onShaderError, program to material recorded in createRenderPipeline<br/>renderFaults.ts:145-171"]
        G4["material build rejects: buildFault<br/>overviewLive.ts:78-80"]
    end
    DEB --> MARK["markOverviewFault, first reason wins, sticky until unmount<br/>overviewLive.ts:71-75, :86-90"]
    G2 --> MARK
    G3 --> MARK
    G4 --> MARK
    MARK --> HIDE["mesh hidden every frame, live drops, gems return, tubes get every edge<br/>useRenderFaultWatch.ts:19-32, tubeSelect.ts:56"]
    HIDE --> TXT["notice: Overview ... failed on this GPU, drawing gems and glass tubes instead,<br/>labels named without a fallback (overviewLive.ts:106-112)"]
    G1 -.-> D1["DEBT: the guard catches every throw and warns once per object; an object<br/>with no fault hook just stops drawing (rendererFactory.ts:173-189)"]
    TXT -.-> T1["P4 gap: back on all-edge tubes, the 65,536 ceiling<br/>clips with a console line only (GlassEdges.tsx:412-430), which the notice never mentions"]
    TUBES -.-> TR["trade-off: the far edges are blank for the pending frames<br/>instead of drawn as tubes (tubeSelect.ts:42-47)"]
```

**What it shows.** An overview layer counts as live only once its material has built, real positions are in the texture and no fault has been seen. A throw, a WebGPU pipeline error, a WebGL2 link failure or a rejected build hands the frame back to gems and tubes, and the canvas notice says so; the labels layer reports the same way with no fallback. **Why it is this way.** ADR-2181 D4 rules out a silent skip. Pipeline and program failures never throw on either backend (`renderFaults.ts:52-71`), so the fallback listens to three's console and its shader-error hook as well as wrapping `renderObject`. The tube trigger stopped waiting for "live" because feeding every edge during startup grew `GlassEdges` to its ceiling (`tubeSelect.ts:38-48`, f3738825c).

**Tension (ADR-2179 P4 vs fallback):** after a fault the tubes carry every edge again, and above the 65,536 ceiling they clip with only a console line (`GlassEdges.tsx:412-430`). The canvas notice reports the fault but not that clip (`overviewLive.ts:106-112`).

**Debt (loose end):** only meshes that set a fault hook are protected; any other object whose draw throws is skipped every frame after one console warning (`rendererFactory.ts:174-186`, `renderFaults.ts:46-48`). The renderer lane records this as the remaining part of the old fault gap.

## VC-31.15 The overview "slab": overdraw model, boundary-pinned layout, empty line draws
```mermaid
flowchart TB
    SYM["symptom at default fit: about 140k overview edges read as a near-opaque square<br/>overdraw.test.ts:4-7"]
    SYM --> M1["edgeOverdrawCoverage: k = edges x length x width / area, cover = 1 - exp(-k alpha)<br/>edgeStyle.ts:176-187"]
    M1 --> M2["inverses: maxEdgeAlphaForCoverage, maxEdgesForCoverage<br/>edgeStyle.ts:189-199"]
    M2 --> M3["at the D2 defaults the box saturates, still at a quarter of the alpha<br/>overdraw.test.ts:52-58"]
    M3 --> M4["half coverage needs about 0.004 per-edge alpha, a few thousand edges at 0.15<br/>overdraw.test.ts:60-69"]
    SYM --> L1["root cause: knowledge nodes pinned at the 400-unit soft cube, p5 and p95 on y at +-400<br/>ADR-2181-overview-edges-are-screen-space-lines-with-no-silent-cap.md:65"]
    L1 --> L2["fix b11d8cba1: default_bounds_size 1600, PhysicsSettings default 1600<br/>physics_config.rs:113-115, :550"]
    L2 --> L3["client copies follow: defaults.ts:57, registry slider max 2000 at motion.ts:42"]
    L2 --> L4["INVARIANT: test_physics_settings_default asserts 1600<br/>physics_config.rs:833"]
    L2 -.-> DR["DRIFT RESOLVED at visionclaw 46edc196 for the comment on the default, physics_config.rs:547-549 now says it was measured under the earlier 400u bounds<br/>the test comment still says the graph settles within the 400u cube, physics_config.rs:828"]
    E1["Dawn Draw(0) warnings: empty trails and collab edges"] --> E2["setLineDrawCount hides a line with fewer than 2 vertices<br/>lineDrawCount.ts:12-15, AgentTrail.tsx:26, AgentCollabEdges.tsx:41"]
```

**What it shows.** Two explanations of the overview "slab" side by side: an overdraw planning model that says the default alpha saturates a dense fit box, and the measured root cause, a layout crushed against the 400-unit physics bounds; plus the empty-line fix from the same WebGPU fault lane. **Why it is this way.** The overdraw model is a planning tool for the D2/D4 thresholds, not a renderer path (`edgeStyle.ts:180-181`). The real-WebGPU check found the slab survived an edge alpha of 0.05 and that the nodes themselves formed the square, so the bounds default was raised (b11d8cba1; `ADR-2181-overview-edges-are-screen-space-lines-with-no-silent-cap.md:65`).

**Tension (ADR-2181 D2 vs code, resolved):** the slab at default fit was read as a D2 alpha failure, but it was a boundary-pinned layout; `default_bounds_size` went from 400 to 1600 (`physics_config.rs:113-115`, `:550`), with the ADR-2181 note recording "check the position quantiles before touching edge alpha" (`ADR-2181-overview-edges-are-screen-space-lines-with-no-silent-cap.md:65`).

**Tension (overdraw test vs ADR-2181 note):** `overdraw.test.ts:52` names its case "the slab is overdraw, on any backend", while the ADR note says the slab was the nodes crushed into the cube (`ADR-2181-overview-edges-are-screen-space-lines-with-no-silent-cap.md:65`).

**Debt (parallel):** the bounds default is written four times, `default_bounds_size` (`physics_config.rs:113-115`), the `PhysicsSettings` literal (`:550`), the client defaults (`defaults.ts:57`) and the registry description (`motion.ts:42`); only the Rust test pins one (`physics_config.rs:833`).

**Drift:** RESOLVED at visionclaw 46edc196 for the comment on the default (`physics_config.rs:547-549` now says it was measured under the earlier 400u bounds), but `:828` still says the graph settles "within the ~400u soft-cube bounds", beside the 1600 it annotates.
