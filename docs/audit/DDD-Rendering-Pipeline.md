# DDD: Node/Edge Rendering Pipeline & WebGPU/WebGL Feature Matrix

## 1. Rendering Pipeline Detail

### 1.1 Material Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │        createGemRenderer()          │
                    │  rendererFactory.ts:27               │
                    │                                      │
                    │  navigator.gpu? ─── YES ──┐         │
                    │       │                    │         │
                    │       NO              WebGPURenderer │
                    │       │              + backend check │
                    │       ▼                    │         │
                    │  WebGLRenderer      WebGLBackend? ───┤
                    │  (clean path)         YES → discard  │
                    │                       NO → keep GPU  │
                    └──────────┬──────────────────┬───────┘
                               │                  │
                         isWebGPURenderer    isWebGPURenderer
                           = false              = true
                               │                  │
                    ┌──────────┴──────────────────┴───────┐
                    │         Material Factory            │
                    ├──────────────────┬─────────────────┤
                    │   WebGL Path     │   WebGPU Path   │
                    ├──────────────────┼─────────────────┤
                    │ MeshPhysical     │ MeshPhysicalNode│
                    │ Material         │ Material (TSL)  │
                    │                  │                  │
                    │ transmission=0.6 │ transmission=0  │
                    │ opacity=0.85     │ opacityNode=TSL │
                    │ emissive=uniform │ emissiveNode=TSL│
                    │ iridescence=0.3  │ iridescence=0.4│
                    │                  │ colorNode=TSL   │
                    │                  │ + DataTexture   │
                    │                  │   metadata      │
                    └──────────────────┴─────────────────┘
```

### 1.2 TSL Material Node Graph (WebGPU)

```
instanceIndex ──→ texU = (idx + 0.5) / texWidth
                         │
                    DataTexture(Nx1, RGBA Float)
                         │
                    ┌─────┴─────┐
                    │ meta.xyzw │
                    ├───────────┤
                    │ .x quality    → qualityBrightness = mix(0.08, 0.5, quality)
                    │ .y authority  → pulseSpeed = mix(0.8, 3.0, authority)
                    │ .z connections → warmShift = connections * 0.25
                    │ .w recency    → recencyBoost = mix(0.5, 1.0, recency)
                    └───────────┘

phase = fract(sin(instanceIndex * 43758.5453)) * 2π
pulse = sin(time * pulseSpeed + phase) * 0.5 + 0.5

viewDir = normalize(-positionView)
nDotV = saturate(dot(normalView, viewDir))
fresnel = pow(1 - nDotV, 3.0)

emissiveNode = baseEmissive * qualityBrightness * mix(0.4, 1.0, pulse) * recencyBoost
opacityNode = mix(mix(0.35, 0.55, authority), 0.92, fresnel)
colorNode = mix(vertexColor, white, fresnel * 0.35)
```

### 1.3 Post-Processing Paths

```
WebGPU:                              WebGL:
  PostProcessing (three/webgpu)        EffectComposer (three/examples/jsm)
  └─ bloom() node                      └─ RenderPass
     strength: settings.bloom            └─ UnrealBloomPass
     radius: settings.radius               strength, radius, threshold
     threshold: settings.threshold

  Priority 1 in R3F render loop        Priority 1 in R3F render loop
  (sole renderer — prevents double)    (sole renderer — prevents double)
```

## 2. Node Rendering Detail (GemNodes.tsx)

### 2.1 InstancedMesh Allocation
- Count: `nextPowerOf2(nodes.length)`, minimum 64
- Re-created when dominant mode changes or node count crosses power-of-2 boundary
- `frustumCulled = false` for guaranteed visibility

### 2.2 Per-Frame Update (useFrame)
```
for each node i:
  1. Compute scale = baseSize * modeMultiplier * settingsScale
     - knowledge_graph: log(connections+1) * authority
     - ontology: depth-based shrinking
     - agent: workload + tokenRate
  2. Read position from nodePositionsRef.current[i*3..i*3+2]
  3. Compose matrix: makeScale(s,s,s) → setPosition(x,y,z)
  4. Compute color based on mode/SSSP/selection
  5. setMatrixAt(i, matrix)
  6. setColorAt(i, color)

Update metadata texture only when metaHash changes
Upload: instanceMatrix.needsUpdate = instanceColor.needsUpdate = true
```

### 2.3 Metadata Texture Update
```
texBuf[i*4 + 0] = quality    (node.metadata.quality || authorityScore || 0.5)
texBuf[i*4 + 1] = authority  (node.metadata.authority || authorityScore || 0)
texBuf[i*4 + 2] = connections (min(connectionCount / 20, 1.0))
texBuf[i*4 + 3] = recency    (exp(-ageSec / 3600))
```

## 3. Edge Rendering Detail (GlassEdges.tsx)

### 3.1 Geometry
- `CylinderGeometry(radius, radius, 1, 8, 1)` — 8 radial segments
- Radius configurable via `settings.edgeRadius` (default 0.03)
- Max 10,000 edges (hardcoded)

### 3.2 Matrix Composition
```
for each edge (src, tgt):
  midpoint = (src + tgt) / 2
  direction = normalize(tgt - src)
  length = |tgt - src|

  if dot(up, direction) < -0.9999:
    quaternion = (1, 0, 0, 0)  // 180° X-axis rotation
  else:
    quaternion = setFromUnitVectors(up, direction)

  scale = (1, length, 1)  // stretch Y axis
  matrix = compose(midpoint, quaternion, scale)
```

### 3.3 Material Properties
```
GlassEdgeMaterial:
  color: (0.7, 0.85, 1.0)  // blue-white
  ior: 1.5
  transmission: WebGPU ? 0 : 0.7
  opacity: WebGPU ? 0.4 : 0.5
  roughness: 0.15
  iridescence: WebGPU ? 0.2 : 0.1
  depthWrite: false  // edges behind nodes

  TSL (WebGPU):
    flow uniform → animated emissive pulse along edge
```

## 4. Quest 3 / VR Rendering Adaptations

### 4.1 LOD Thresholds (WebXRScene.tsx)
```
distance < 5m   → high detail (40 curve segments, glow)
5m - 15m        → medium detail (reduced segments)
15m - 30m       → low detail (minimal geometry)
distance > 30m  → culled (invisible)
```

### 4.2 VR Performance Budget
```
Target: 72 fps (11.1ms per frame)

Budget:
  Node rendering:  3ms  (instanced, no TSL in VR)
  Edge rendering:  2ms  (simplified cylinders)
  Post-processing: 0ms  (disabled in VR)
  Action effects:  1ms  (max 20 connections)
  Overhead:        5ms  (swap, compositor, tracking)
```

### 4.3 VR Material Simplification
- No transmission (performance)
- No bloom/post-processing
- Reduced iridescence
- Simplified geometry (fewer curve segments)
- Max 20 concurrent action connections (vs 50 desktop)

## 5. Audit Checklist

### WebGPU Feature Completeness
- [ ] TSL metadata material activates on all WebGPU browsers
- [ ] DataTexture metadata correctly sampled per instance
- [ ] Fresnel rim lighting visible on all node types
- [ ] Authority pulse animation smooth and continuous
- [ ] Quality-driven emissive glow differentiates node importance
- [ ] Connection density warm-shift visible
- [ ] Recency decay updates correctly
- [ ] Node-based bloom renders correctly
- [ ] Edge flow animation visible

### WebGL Fallback Completeness
- [ ] Clean WebGLRenderer init (no hybrid path)
- [ ] MeshPhysicalMaterial renders all node types
- [ ] Transmission enabled and visible
- [ ] UnrealBloomPass produces comparable bloom
- [ ] No visual regression vs WebGPU
- [ ] Performance comparable (no extra draw calls)

### Server-Authoritative Layout
- [ ] Client worker receives server binary updates
- [ ] No client-side force computation for any graph type
- [ ] Interpolation smooth and snap threshold correct
- [ ] User drag → server → broadcast → reconcile works
- [ ] Pinned nodes respected
- [ ] No position drift between clients

### Quest 3 / Vircadia
- [ ] LOD thresholds produce 72fps
- [ ] Entity sync maps to graph nodes
- [ ] Position updates reach Vircadia clients
- [ ] Avatar rendering integrates
- [ ] Optimistic tweening matches desktop
