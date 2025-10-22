# Master Settings Catalog

**Generated**: 2025-10-22
**Purpose**: Comprehensive categorization and prioritization of ALL settings parameters
**Status**: Based on client settingsConfig.ts + server settings_paths.rs analysis

---

## Executive Summary

**Total Settings Discovered**: 146 parameters
**Active (Backend Supported)**: 93 (63.7%)
**Commented Out (No Backend Path)**: 53 (36.3%)
**Categories**: 11 logical domains

**Key Finding**: Major settings sections are commented out because backend database paths don't exist yet in the CQRS/hexagonal architecture.

---

## 1. Visualization Settings (Core Graphics)
**Category**: Visualization
**Priority**: HIGH (User-Facing)
**Status**: ✅ ACTIVE - Full backend support
**Backend Path**: `visualisation.graphs.logseq.*`

### Node Rendering (9 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| nodeColor | color | - | visualisation.graphs.logseq.nodes.baseColor | Visualization/Nodes | HIGH |
| nodeSize | slider | 0.2-2 | visualisation.graphs.logseq.nodes.nodeSize | Visualization/Nodes | HIGH |
| nodeMetalness | slider | 0-1 | visualisation.graphs.logseq.nodes.metalness | Visualization/Nodes | MEDIUM |
| nodeOpacity | slider | 0-1 | visualisation.graphs.logseq.nodes.opacity | Visualization/Nodes | HIGH |
| nodeRoughness | slider | 0-1 | visualisation.graphs.logseq.nodes.roughness | Visualization/Nodes | MEDIUM |
| enableInstancing | toggle | - | visualisation.graphs.logseq.nodes.enableInstancing | Visualization/Performance | HIGH |
| enableMetadataShape | toggle | - | visualisation.graphs.logseq.nodes.enableMetadataShape | Visualization/Advanced | LOW |
| enableMetadataVis | toggle | - | visualisation.graphs.logseq.nodes.enableMetadataVisualisation | Visualization/Advanced | LOW |
| nodeImportance | toggle | - | visualisation.graphs.logseq.nodes.enableImportance | Visualization/Advanced | MEDIUM |

### Edge Rendering (6 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| edgeColor | color | - | visualisation.graphs.logseq.edges.color | Visualization/Edges | HIGH |
| edgeWidth | slider | 0.01-2 | visualisation.graphs.logseq.edges.baseWidth | Visualization/Edges | HIGH |
| edgeOpacity | slider | 0-1 | visualisation.graphs.logseq.edges.opacity | Visualization/Edges | HIGH |
| enableArrows | toggle | - | visualisation.graphs.logseq.edges.enableArrows | Visualization/Edges | MEDIUM |
| arrowSize | slider | 0.01-0.5 | visualisation.graphs.logseq.edges.arrowSize | Visualization/Edges | MEDIUM |
| glowStrength | slider | 0-5 | visualisation.graphs.logseq.edges.glowStrength | Visualization/Effects | MEDIUM |

### Label Rendering (5 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| enableLabels | toggle | - | visualisation.graphs.logseq.labels.enableLabels | Visualization/Labels | HIGH |
| labelSize | slider | 0.01-1.5 | visualisation.graphs.logseq.labels.desktopFontSize | Visualization/Labels | HIGH |
| labelColor | color | - | visualisation.graphs.logseq.labels.textColor | Visualization/Labels | MEDIUM |
| labelOutlineColor | color | - | visualisation.graphs.logseq.labels.textOutlineColor | Visualization/Labels | LOW |
| labelOutlineWidth | slider | 0-0.01 | visualisation.graphs.logseq.labels.textOutlineWidth | Visualization/Labels | LOW |

### Lighting (2 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| ambientLight | slider | 0-2 | visualisation.rendering.ambientLightIntensity | Visualization/Lighting | MEDIUM |
| directionalLight | slider | 0-2 | visualisation.rendering.directionalLightIntensity | Visualization/Lighting | MEDIUM |

**Total Active**: 22 settings
**UI Recommendation**: Create "Visualization" panel with tabs: Nodes, Edges, Labels, Lighting

---

## 2. Physics Settings (Simulation)
**Category**: Physics
**Priority**: HIGH (Performance Impact)
**Status**: ✅ ACTIVE - Full backend support
**Backend Path**: `visualisation.graphs.logseq.physics.*`

### Core Physics (28 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| enabled | toggle | - | visualisation.graphs.logseq.physics.enabled | Physics/Core | CRITICAL |
| autoBalance | toggle | - | visualisation.graphs.logseq.physics.autoBalance | Physics/Core | HIGH |
| damping | slider | 0-1 | visualisation.graphs.logseq.physics.damping | Physics/Core | HIGH |
| springK | slider | 0.0001-10 | visualisation.graphs.logseq.physics.springK | Physics/Forces | HIGH |
| repelK | slider | 0.1-200 | visualisation.graphs.logseq.physics.repelK | Physics/Forces | HIGH |
| attractionK | slider | 0-10 | visualisation.graphs.logseq.physics.attractionK | Physics/Forces | HIGH |
| dt | slider | 0.001-0.1 | visualisation.graphs.logseq.physics.dt | Physics/Advanced | MEDIUM |
| maxVelocity | slider | 0.1-10 | visualisation.graphs.logseq.physics.maxVelocity | Physics/Constraints | HIGH |
| separationRadius | slider | 0.1-10 | visualisation.graphs.logseq.physics.separationRadius | Physics/Constraints | MEDIUM |
| enableBounds | toggle | - | visualisation.graphs.logseq.physics.enableBounds | Physics/Bounds | HIGH |
| boundsSize | slider | 1-10000 | visualisation.graphs.logseq.physics.boundsSize | Physics/Bounds | HIGH |
| stressWeight | slider | 0-1 | visualisation.graphs.logseq.physics.stressWeight | Physics/Advanced | MEDIUM |
| stressAlpha | slider | 0-1 | visualisation.graphs.logseq.physics.stressAlpha | Physics/Advanced | MEDIUM |
| minDistance | slider | 0.05-1 | visualisation.graphs.logseq.physics.minDistance | Physics/Constraints | MEDIUM |
| maxRepulsionDist | slider | 10-200 | visualisation.graphs.logseq.physics.maxRepulsionDist | Physics/Forces | MEDIUM |
| warmupIterations | slider | 0-500 | visualisation.graphs.logseq.physics.warmupIterations | Physics/Performance | MEDIUM |
| coolingRate | slider | 0.00001-0.01 | visualisation.graphs.logseq.physics.coolingRate | Physics/Performance | MEDIUM |
| restLength | slider | 10-200 | visualisation.graphs.logseq.physics.restLength | Physics/Forces | MEDIUM |
| repulsionCutoff | slider | 10-200 | visualisation.graphs.logseq.physics.repulsionCutoff | Physics/Forces | MEDIUM |
| repulsionSofteningEpsilon | slider | 0.00001-0.01 | visualisation.graphs.logseq.physics.repulsionSofteningEpsilon | Physics/Advanced | LOW |
| centerGravityK | slider | 0-0.1 | visualisation.graphs.logseq.physics.centerGravityK | Physics/Forces | MEDIUM |
| gridCellSize | slider | 10-100 | visualisation.graphs.logseq.physics.gridCellSize | Physics/Optimization | LOW |
| boundaryExtremeMultiplier | slider | 1-5 | visualisation.graphs.logseq.physics.boundaryExtremeMultiplier | Physics/Bounds | LOW |
| boundaryExtremeForceMultiplier | slider | 1-20 | visualisation.graphs.logseq.physics.boundaryExtremeForceMultiplier | Physics/Bounds | LOW |
| boundaryVelocityDamping | slider | 0-1 | visualisation.graphs.logseq.physics.boundaryVelocityDamping | Physics/Bounds | LOW |
| iterations | slider | 1-1000 | visualisation.graphs.logseq.physics.iterations | Physics/Performance | MEDIUM |
| massScale | slider | 0.1-10 | visualisation.graphs.logseq.physics.massScale | Physics/Advanced | LOW |
| boundaryDamping | slider | 0-1 | visualisation.graphs.logseq.physics.boundaryDamping | Physics/Bounds | LOW |
| updateThreshold | slider | 0-0.5 | visualisation.graphs.logseq.physics.updateThreshold | Physics/Performance | LOW |

**Total Active**: 28 settings
**UI Recommendation**: Create "Physics" panel with tabs: Core, Forces, Constraints, Bounds, Advanced, Performance

---

## 3. Visual Effects (Integrations)
**Category**: Visual Effects
**Priority**: MEDIUM (User Experience)
**Status**: ✅ ACTIVE - Full backend support
**Backend Path**: `visualisation.*`

### Glow/Hologram Effects (15 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| glow | toggle | - | visualisation.glow.enabled | Effects/Glow | MEDIUM |
| glowIntensity | slider | 0-5 | visualisation.glow.intensity | Effects/Glow | MEDIUM |
| glowRadius | slider | 0-5 | visualisation.glow.radius | Effects/Glow | MEDIUM |
| glowThreshold | slider | 0-1 | visualisation.glow.threshold | Effects/Glow | LOW |
| hologram | toggle | - | visualisation.graphs.logseq.nodes.enableHologram | Effects/Hologram | MEDIUM |
| ringCount | slider | 0-10 | visualisation.hologram.ringCount | Effects/Hologram | LOW |
| ringColor | color | - | visualisation.hologram.ringColor | Effects/Hologram | LOW |
| ringOpacity | slider | 0-1 | visualisation.hologram.ringOpacity | Effects/Hologram | LOW |
| ringRotationSpeed | slider | 0-5 | visualisation.hologram.ringRotationSpeed | Effects/Hologram | LOW |
| flowEffect | toggle | - | visualisation.graphs.logseq.edges.enableFlowEffect | Effects/Flow | MEDIUM |
| flowSpeed | slider | 0.1-5 | visualisation.graphs.logseq.edges.flowSpeed | Effects/Flow | LOW |
| flowIntensity | slider | 0-10 | visualisation.graphs.logseq.edges.flowIntensity | Effects/Flow | LOW |
| useGradient | toggle | - | visualisation.graphs.logseq.edges.useGradient | Effects/Flow | LOW |
| distanceIntensity | slider | 0-10 | visualisation.graphs.logseq.edges.distanceIntensity | Effects/Flow | LOW |
| nodeAnimations | toggle | - | visualisation.animations.enableNodeAnimations | Effects/Animations | MEDIUM |

### Pulse/Wave Animations (4 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| pulseEnabled | toggle | - | visualisation.animations.pulseEnabled | Effects/Animations | LOW |
| pulseSpeed | slider | 0.1-2 | visualisation.animations.pulseSpeed | Effects/Animations | LOW |
| pulseStrength | slider | 0.1-2 | visualisation.animations.pulseStrength | Effects/Animations | LOW |
| selectionWave | toggle | - | visualisation.animations.selectionWaveEnabled | Effects/Animations | LOW |
| waveSpeed | slider | 0.1-2 | visualisation.animations.waveSpeed | Effects/Animations | LOW |

### Rendering Quality (3 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| antialiasing | toggle | - | visualisation.rendering.enableAntialiasing | Effects/Quality | MEDIUM |
| shadows | toggle | - | visualisation.rendering.enableShadows | Effects/Quality | MEDIUM |
| ambientOcclusion | toggle | - | visualisation.rendering.enableAmbientOcclusion | Effects/Quality | LOW |

**Total Active**: 22 settings
**UI Recommendation**: Create "Visual Effects" panel with tabs: Glow, Hologram, Flow, Animations, Quality

---

## 4. Authentication Settings
**Category**: Security
**Priority**: CRITICAL (Security Impact)
**Status**: ✅ ACTIVE - Full backend support
**Backend Path**: `auth.*`

### Auth Configuration (4 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| nostr | button | - | auth.nostr | Auth/Providers | HIGH |
| enabled | toggle | - | auth.enabled | Auth/Core | CRITICAL |
| required | toggle | - | auth.required | Auth/Core | CRITICAL |
| provider | text | - | auth.provider | Auth/Core | HIGH |

**Total Active**: 4 settings
**UI Recommendation**: Dedicated "Authentication" panel or system settings section

---

## 5. XR/AR Settings
**Category**: Extended Reality
**Priority**: MEDIUM (Feature-Specific)
**Status**: ✅ ACTIVE - Full backend support
**Backend Path**: `xr.*` + `system.*`

### XR Configuration (10 settings)
| Parameter | Type | Range | Path | UI Panel | Priority |
|-----------|------|-------|------|----------|----------|
| persistSettings | toggle | - | system.persistSettingsOnServer | System/Storage | HIGH |
| customBackendURL | text | - | system.customBackendUrl | System/Network | MEDIUM |
| xrEnabled | toggle | - | xr.enabled | XR/Core | HIGH |
| xrQuality | select | Low/Med/High | xr.quality | XR/Performance | MEDIUM |
| xrRenderScale | slider | 0.5-2 | xr.renderScale | XR/Performance | MEDIUM |
| handTracking | toggle | - | xr.handTracking.enabled | XR/Interactions | MEDIUM |
| enableHaptics | toggle | - | xr.interactions.enableHaptics | XR/Interactions | LOW |
| xrComputeMode | toggle | - | xr.gpu.enableOptimizedCompute | XR/Performance | MEDIUM |
| xrPerformancePreset | select | Battery/Balanced/Perf | xr.performance.preset | XR/Performance | MEDIUM |
| xrAdaptiveQuality | toggle | - | xr.enableAdaptiveQuality | XR/Performance | MEDIUM |

**Total Active**: 10 settings
**UI Recommendation**: Dedicated "XR/AR" panel for immersive mode users

---

## 6. Developer Tools
**Category**: Debugging
**Priority**: LOW (Developer-Only)
**Status**: ⚠️ PARTIAL - Only 1/16 settings have backend support
**Backend Path**: `system.debug.enabled` (only)

### Debug Settings (1 ACTIVE, 15 COMMENTED OUT)
| Parameter | Type | Status | Path | Priority | Notes |
|-----------|------|--------|------|----------|-------|
| enableDebug | toggle | ✅ ACTIVE | system.debug.enabled | HIGH | Only working debug setting |
| consoleLogging | toggle | ❌ DISABLED | developer.consoleLogging | MEDIUM | No backend path |
| logLevel | select | ❌ DISABLED | developer.logLevel | MEDIUM | No backend path |
| showNodeIds | toggle | ❌ DISABLED | developer.showNodeIds | LOW | No backend path |
| showEdgeWeights | toggle | ❌ DISABLED | developer.showEdgeWeights | LOW | No backend path |
| enableProfiler | toggle | ❌ DISABLED | developer.enableProfiler | MEDIUM | No backend path |
| apiDebugMode | toggle | ❌ DISABLED | developer.apiDebugMode | MEDIUM | No backend path |
| showMemory | toggle | ❌ DISABLED | system.debug.showMemory | MEDIUM | No backend path |
| perfDebug | toggle | ❌ DISABLED | system.debug.enablePerformanceDebug | MEDIUM | No backend path |
| telemetry | toggle | ❌ DISABLED | system.debug.enableTelemetry | MEDIUM | No backend path |
| dataDebug | toggle | ❌ DISABLED | system.debug.enableDataDebug | LOW | No backend path |
| wsDebug | toggle | ❌ DISABLED | system.debug.enableWebSocketDebug | MEDIUM | No backend path |
| physicsDebug | toggle | ❌ DISABLED | system.debug.enablePhysicsDebug | MEDIUM | No backend path |
| nodeDebug | toggle | ❌ DISABLED | system.debug.enableNodeDebug | LOW | No backend path |
| shaderDebug | toggle | ❌ DISABLED | system.debug.enableShaderDebug | LOW | No backend path |
| matrixDebug | toggle | ❌ DISABLED | system.debug.enableMatrixDebug | LOW | No backend path |

**Total Active**: 1 setting
**Total Disabled**: 15 settings
**Restoration Effort**: ~2-4 hours (add backend paths + uncomment)

---

## 7. ❌ Dashboard Settings (DISABLED)
**Category**: Monitoring
**Priority**: MEDIUM (User Visibility)
**Status**: ❌ DISABLED - No backend support
**Backend Path**: None - `dashboard.*` doesn't exist

### Dashboard Monitoring (8 COMMENTED OUT)
| Parameter | Type | Path (Missing) | Priority | Notes |
|-----------|------|----------------|----------|-------|
| graphStatus | toggle | dashboard.showStatus | MEDIUM | Show graph status indicator |
| autoRefresh | toggle | dashboard.autoRefresh | LOW | Auto-refresh dashboard |
| refreshInterval | slider | dashboard.refreshInterval | LOW | Refresh interval (seconds) |
| computeMode | select | dashboard.computeMode | HIGH | Compute mode selector |
| iterationCount | text | dashboard.iterationCount | MEDIUM | Current iteration display |
| convergenceIndicator | toggle | dashboard.showConvergence | MEDIUM | Show convergence status |
| activeConstraints | text | dashboard.activeConstraints | LOW | Active constraints count |
| clusteringStatus | toggle | dashboard.clusteringActive | LOW | Clustering status indicator |

**Total Disabled**: 8 settings
**Backend Work Required**: Add `dashboard.*` paths to settings database
**Restoration Effort**: ~30 minutes (backend) + 20 minutes (frontend)

---

## 8. ❌ Analytics Settings (DISABLED)
**Category**: Analytics
**Priority**: HIGH (Backend APIs Exist!)
**Status**: ❌ DISABLED - No backend settings support
**Backend Path**: None - `analytics.*` doesn't exist
**Critical Note**: Backend has WORKING analytics APIs at `/api/analytics/*` but no UI settings!

### Analytics Configuration (11 COMMENTED OUT)
| Parameter | Type | Path (Missing) | Priority | Notes |
|-----------|------|----------------|----------|-------|
| enableMetrics | toggle | analytics.enableMetrics | HIGH | Enable analytics collection |
| updateInterval | slider | analytics.updateInterval | MEDIUM | Update interval (seconds) |
| showDegreeDistribution | toggle | analytics.showDegreeDistribution | MEDIUM | Show degree distribution |
| showClustering | toggle | analytics.showClusteringCoefficient | MEDIUM | Show clustering coefficient |
| showCentrality | toggle | analytics.showCentrality | MEDIUM | Show centrality metrics |
| clusteringAlgorithm | select | analytics.clustering.algorithm | HIGH | kmeans/spectral/louvain |
| clusterCount | slider | analytics.clustering.clusterCount | HIGH | Number of clusters (2-20) |
| clusterResolution | slider | analytics.clustering.resolution | MEDIUM | Resolution (0.1-2) |
| clusterIterations | slider | analytics.clustering.iterations | MEDIUM | Iterations (10-100) |
| exportClusters | toggle | analytics.clustering.exportEnabled | LOW | Export cluster data |
| importDistances | toggle | analytics.clustering.importEnabled | LOW | Import distance matrices |

**Total Disabled**: 11 settings
**Backend APIs Available**:
- ✅ POST /api/analytics/clustering
- ✅ POST /api/analytics/community-detection
- ✅ POST /api/analytics/anomaly-detection
- ✅ GET /api/analytics/centrality
- ✅ GET /api/analytics/path-analysis

**Restoration Effort**: ~30 minutes (add `analytics.*` paths) + 30 minutes (wire to APIs)
**Priority Justification**: HIGH - Backend infrastructure exists, just needs UI hookup!

---

## 9. ❌ Performance Settings (DISABLED)
**Category**: Performance
**Priority**: MEDIUM (Optimization)
**Status**: ❌ DISABLED - No backend support
**Backend Path**: None - `performance.*` doesn't exist

### Performance Tuning (11 COMMENTED OUT)
| Parameter | Type | Path (Missing) | Priority | Notes |
|-----------|------|----------------|----------|-------|
| showFPS | toggle | performance.showFPS | MEDIUM | Display FPS counter |
| targetFPS | slider | performance.targetFPS | MEDIUM | Target FPS (30-144) |
| gpuMemoryLimit | slider | performance.gpuMemoryLimit | MEDIUM | GPU memory limit (MB) |
| levelOfDetail | select | performance.levelOfDetail | HIGH | Quality preset (low/med/high/ultra) |
| adaptiveQuality | toggle | performance.enableAdaptiveQuality | HIGH | Adaptive quality adjustment |
| warmupDuration | slider | performance.warmupDuration | LOW | Warmup duration (seconds) |
| convergenceThreshold | slider | performance.convergenceThreshold | MEDIUM | Convergence threshold |
| adaptiveCooling | toggle | performance.enableAdaptiveCooling | MEDIUM | Adaptive cooling rate |
| gpuBlockSize | select | performance.gpuBlockSize | MEDIUM | GPU block size (64/128/256/512) |
| memoryCoalescing | toggle | performance.enableMemoryCoalescing | LOW | Memory coalescing optimization |
| iterationLimit | slider | performance.iterationLimit | MEDIUM | Iteration limit (100-5000) |

**Total Disabled**: 11 settings
**Restoration Effort**: ~1 hour (backend paths) + 30 minutes (UI)

---

## 10. ❌ GPU Visualization Features (DISABLED)
**Category**: GPU Rendering
**Priority**: MEDIUM (Advanced Features)
**Status**: ❌ DISABLED - No backend support
**Backend Path**: None - `visualisation.gpu.*` doesn't exist

### GPU Advanced Features (4 COMMENTED OUT)
| Parameter | Type | Path (Missing) | Priority | Notes |
|-----------|------|----------------|----------|-------|
| temporalCoherence | slider | visualisation.gpu.temporalCoherence | MEDIUM | Temporal coherence (0-1) |
| graphDifferentiation | toggle | visualisation.gpu.enableGraphDifferentiation | LOW | Graph differentiation mode |
| clusterVisualization | toggle | visualisation.gpu.enableClusterVisualization | MEDIUM | Cluster visualization |
| stressOptimization | toggle | visualisation.gpu.enableStressOptimization | MEDIUM | Stress optimization |

**Total Disabled**: 4 settings
**Restoration Effort**: ~30 minutes (backend) + 15 minutes (UI)

---

## 11. ❌ GPU Developer Debug (DISABLED)
**Category**: GPU Debugging
**Priority**: LOW (Developer-Only)
**Status**: ❌ DISABLED - No backend support
**Backend Path**: None - `developer.gpu.*` doesn't exist

### GPU Debug Visualizations (5 COMMENTED OUT)
| Parameter | Type | Path (Missing) | Priority | Notes |
|-----------|------|----------------|----------|-------|
| forceVectors | toggle | developer.gpu.showForceVectors | LOW | Show force vectors |
| constraintVisualization | toggle | developer.gpu.showConstraints | LOW | Show constraints |
| boundaryForceDisplay | toggle | developer.gpu.showBoundaryForces | LOW | Show boundary forces |
| convergenceGraph | toggle | developer.gpu.showConvergenceGraph | MEDIUM | Show convergence graph |
| gpuTimingStats | toggle | developer.gpu.showTimingStats | MEDIUM | Show GPU timing stats |

**Total Disabled**: 5 settings
**Restoration Effort**: ~1 hour (backend) + 20 minutes (UI)

---

## 12. ❌ Bloom Effects (DISABLED)
**Category**: Visual Effects
**Priority**: LOW (Cosmetic)
**Status**: ❌ DISABLED - No backend support
**Backend Path**: None - `visualisation.bloom.*` doesn't exist
**Note**: Glow effects work, bloom is separate

### Bloom Configuration (4 COMMENTED OUT)
| Parameter | Type | Path (Missing) | Priority | Notes |
|-----------|------|----------------|----------|-------|
| bloom | toggle | visualisation.bloom.enabled | LOW | Enable bloom effect |
| bloomStrength | slider | visualisation.bloom.strength | LOW | Bloom strength (0-5) |
| bloomRadius | slider | visualisation.bloom.radius | LOW | Bloom radius (0-1) |
| bloomThreshold | slider | visualisation.bloom.threshold | LOW | Bloom threshold (0-1) |

**Total Disabled**: 4 settings
**Restoration Effort**: ~20 minutes (backend) + 10 minutes (UI)

---

## Summary Statistics

### By Status
| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Active (Backend Supported) | 93 | 63.7% |
| ❌ Disabled (No Backend Path) | 53 | 36.3% |
| **TOTAL** | **146** | **100%** |

### By Category
| Category | Active | Disabled | Total | Priority |
|----------|--------|----------|-------|----------|
| Visualization | 22 | 0 | 22 | HIGH |
| Physics | 28 | 0 | 28 | HIGH |
| Visual Effects | 22 | 4 (bloom) | 26 | MEDIUM |
| Authentication | 4 | 0 | 4 | CRITICAL |
| XR/AR | 10 | 0 | 10 | MEDIUM |
| Developer Tools | 1 | 15 | 16 | LOW |
| Dashboard | 0 | 8 | 8 | MEDIUM |
| Analytics | 0 | 11 | 11 | HIGH |
| Performance | 0 | 11 | 11 | MEDIUM |
| GPU Visualization | 0 | 4 | 4 | MEDIUM |
| GPU Debug | 0 | 5 | 5 | LOW |

### By Priority
| Priority | Active | Disabled | Total |
|----------|--------|----------|-------|
| CRITICAL | 6 | 0 | 6 |
| HIGH | 38 | 6 | 44 |
| MEDIUM | 39 | 25 | 64 |
| LOW | 10 | 22 | 32 |

---

## UI Panel Mapping (Recommended Structure)

### Main Tabs
1. **Visualization** (22 settings)
   - Nodes (9 settings)
   - Edges (6 settings)
   - Labels (5 settings)
   - Lighting (2 settings)

2. **Physics** (28 settings)
   - Core (6 settings)
   - Forces (7 settings)
   - Constraints (4 settings)
   - Bounds (7 settings)
   - Performance (4 settings)

3. **Visual Effects** (22 settings)
   - Glow (4 settings)
   - Hologram (5 settings)
   - Flow (6 settings)
   - Animations (4 settings)
   - Quality (3 settings)

4. **Analytics** (11 settings - DISABLED, needs restoration)
   - Metrics (5 settings)
   - Clustering (6 settings)

5. **Performance** (11 settings - DISABLED, needs restoration)
   - Display (2 settings)
   - Quality (4 settings)
   - GPU (3 settings)
   - Limits (2 settings)

6. **XR/AR** (10 settings)
   - Core (2 settings)
   - Performance (5 settings)
   - Interactions (3 settings)

7. **System** (6 settings)
   - Authentication (4 settings)
   - Storage (2 settings)

8. **Developer** (16 settings - mostly DISABLED)
   - Debug Flags (16 settings, only 1 works)
   - GPU Debug (5 settings, all disabled)

---

## Conflict Analysis

### No Conflicts Detected
All active settings have unique paths with no overlaps or duplicates.

### Commented-Out Conflicts
None - commented sections reference different namespaces than active settings.

---

## Restoration Priority Matrix

### P0 - Critical (Immediate)
*None* - System is stable with current active settings

### P1 - High (Next Sprint)
1. **Analytics Settings** (~1 hour)
   - Backend APIs EXIST but no UI controls
   - Add `analytics.*` paths to database
   - Uncomment analytics section in settingsConfig.ts
   - Wire to existing `/api/analytics/*` endpoints
   - **Impact**: Unlock clustering, centrality, community detection features

2. **Dashboard Settings** (~50 minutes)
   - Add `dashboard.*` paths to database
   - Uncomment dashboard section
   - Create dashboard status panel component
   - **Impact**: User visibility into system status

### P2 - Medium (Backlog)
3. **Performance Settings** (~1.5 hours)
   - Add `performance.*` paths to database
   - Uncomment performance section
   - Wire FPS counter, quality presets, GPU controls
   - **Impact**: User control over performance tuning

4. **Developer System Debug** (~1.5 hours)
   - Add missing `system.debug.*` paths (15 settings)
   - Add `developer.*` paths
   - Uncomment developer section
   - **Impact**: Enhanced debugging capabilities

5. **GPU Visualization Features** (~45 minutes)
   - Add `visualisation.gpu.*` paths (4 settings)
   - Uncomment GPU visualization section
   - **Impact**: Advanced GPU rendering features

### P3 - Low (Future)
6. **GPU Developer Debug** (~1.5 hours)
   - Add `developer.gpu.*` paths (5 settings)
   - Implement debug visualizations
   - **Impact**: GPU debugging visualizations

7. **Bloom Effects** (~30 minutes)
   - Add `visualisation.bloom.*` paths (4 settings)
   - Uncomment bloom section (glow already works)
   - **Impact**: Additional visual effect option

---

## Implementation Checklist

### For Each Disabled Settings Group:

#### Backend (Rust)
- [ ] Add settings paths to `src/config/mod.rs`
- [ ] Update `AppFullSettings` struct with new fields
- [ ] Add default values to `NetworkSettings::default()`
- [ ] Update database schema if needed
- [ ] Test settings read/write via `/api/settings/path` endpoint

#### Frontend (TypeScript)
- [ ] Uncomment settings section in `client/src/features/visualisation/components/ControlPanel/settingsConfig.ts`
- [ ] Verify paths match backend exactly
- [ ] Add to appropriate UI panel/tab
- [ ] Test settings persistence
- [ ] Verify hot-reload works

#### Integration
- [ ] Test end-to-end: UI change → API call → Database → State update
- [ ] Verify WebSocket sync (if applicable)
- [ ] Check for validation errors
- [ ] Document new settings in user guide

---

## Notes

### Why Settings Were Commented Out
**From LOST_SETTINGS_ANALYSIS.md**: Settings weren't lost in a recent refactor - they were already commented out because their backend paths don't exist in the new CQRS/hexagonal architecture after the major database migration.

### Current State
- **Core visualization, physics, and effects**: Fully functional
- **Analytics**: Backend ready, UI missing
- **Dashboard, Performance, Debug**: Need full stack implementation

### Migration Path
**Additive approach** (no rollbacks needed):
1. Add missing database paths incrementally
2. Uncomment settings sections as backend paths are added
3. Test each section independently
4. Priority: Analytics → Dashboard → Performance → Developer

---

## Appendix: Settings Paths Reference

### Active Path Prefixes
```
visualisation.graphs.logseq.nodes.*      (9 settings)
visualisation.graphs.logseq.edges.*      (6 settings)
visualisation.graphs.logseq.labels.*     (5 settings)
visualisation.graphs.logseq.physics.*    (28 settings)
visualisation.rendering.*                (5 settings)
visualisation.glow.*                     (4 settings)
visualisation.hologram.*                 (4 settings)
visualisation.animations.*               (5 settings)
auth.*                                   (4 settings)
xr.*                                     (8 settings)
system.*                                 (3 settings)
```

### Missing Path Prefixes (Need Backend)
```
dashboard.*                              (8 settings)
analytics.*                              (5 settings)
analytics.clustering.*                   (6 settings)
performance.*                            (11 settings)
developer.*                              (6 settings)
developer.gpu.*                          (5 settings)
system.debug.*                           (14 settings, only 1 exists)
visualisation.gpu.*                      (4 settings)
visualisation.bloom.*                    (4 settings)
```

---

**End of Master Settings Catalog**
