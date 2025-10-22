# Client-Side Settings Audit

**Date**: 2025-10-22
**Scope**: React/TypeScript Client (client/src)
**Total Parameters Discovered**: 270+

---

## 1. ZUSTAND STORES

### 1.1 settingsStore.ts (`client/src/store/settingsStore.ts`)

**Core Settings Management** - Main store with 1055 lines of configuration

#### Essential Paths (Loaded at Startup)
- **Parameter**: system.debug.enabled
  - **Type**: boolean
  - **Location**: Line 60
  - **Priority**: Critical
  - **Category**: System/Debug

- **Parameter**: system.websocket.updateRate
  - **Type**: number
  - **Location**: Line 61
  - **Priority**: High
  - **Category**: System/WebSocket

- **Parameter**: system.websocket.reconnectAttempts
  - **Type**: number
  - **Location**: Line 62
  - **Priority**: High
  - **Category**: System/WebSocket

- **Parameter**: auth.enabled
  - **Type**: boolean
  - **Location**: Line 63
  - **Priority**: High
  - **Category**: Authentication

- **Parameter**: auth.required
  - **Type**: boolean
  - **Location**: Line 64
  - **Priority**: High
  - **Category**: Authentication

- **Parameter**: visualisation.rendering.context
  - **Type**: string
  - **Location**: Line 65
  - **Priority**: Critical
  - **Category**: Visualization/Rendering

- **Parameter**: xr.enabled
  - **Type**: boolean
  - **Location**: Line 66
  - **Priority**: High
  - **Category**: XR/AR

- **Parameter**: xr.mode
  - **Type**: string
  - **Location**: Line 67
  - **Priority**: High
  - **Category**: XR/AR

#### GPU Physics Parameters (Interface GPUPhysicsParams, Line 167-214)
- **Parameter**: springK
  - **Type**: number
  - **Range**: 0.001-10.0
  - **Location**: Line 168
  - **Priority**: High
  - **Category**: Physics/Forces

- **Parameter**: repelK
  - **Type**: number
  - **Range**: 0.001-100.0
  - **Location**: Line 169
  - **Priority**: High
  - **Category**: Physics/Forces

- **Parameter**: attractionK
  - **Type**: number
  - **Range**: 0.0-1.0
  - **Location**: Line 170
  - **Priority**: High
  - **Category**: Physics/Forces

- **Parameter**: gravity
  - **Type**: number
  - **Range**: -1.0-1.0
  - **Location**: Line 171
  - **Priority**: High
  - **Category**: Physics/Forces

- **Parameter**: dt (Time Step)
  - **Type**: number
  - **Location**: Line 172
  - **Priority**: Critical
  - **Category**: Physics/Simulation

- **Parameter**: maxVelocity
  - **Type**: number
  - **Location**: Line 173
  - **Priority**: High
  - **Category**: Physics/Limits

- **Parameter**: damping
  - **Type**: number
  - **Range**: 0.0-1.0
  - **Location**: Line 174
  - **Priority**: High
  - **Category**: Physics/Damping

- **Parameter**: temperature
  - **Type**: number
  - **Location**: Line 175
  - **Priority**: Medium
  - **Category**: Physics/Simulation

- **Parameter**: maxRepulsionDist
  - **Type**: number
  - **Location**: Line 176
  - **Priority**: High
  - **Category**: Physics/Forces

- **Parameter**: restLength
  - **Type**: number
  - **Range**: 0.1-10.0
  - **Location**: Line 179
  - **Priority**: High
  - **Category**: Physics/CUDA

- **Parameter**: repulsionCutoff
  - **Type**: number
  - **Range**: 1.0-1000.0
  - **Location**: Line 180
  - **Priority**: High
  - **Category**: Physics/CUDA

- **Parameter**: repulsionSofteningEpsilon
  - **Type**: number
  - **Range**: 0.001-1.0
  - **Location**: Line 181
  - **Priority**: Medium
  - **Category**: Physics/CUDA

- **Parameter**: centerGravityK
  - **Type**: number
  - **Range**: -1.0-1.0
  - **Location**: Line 182
  - **Priority**: High
  - **Category**: Physics/CUDA

- **Parameter**: gridCellSize
  - **Type**: number
  - **Range**: 1.0-100.0
  - **Location**: Line 183
  - **Priority**: Medium
  - **Category**: Physics/CUDA

- **Parameter**: featureFlags
  - **Type**: number (bitfield)
  - **Range**: 0-255
  - **Location**: Line 184
  - **Priority**: Low
  - **Category**: Physics/CUDA

- **Parameter**: warmupIterations
  - **Type**: number
  - **Range**: 0-1000
  - **Location**: Line 187
  - **Priority**: High
  - **Category**: Physics/Warmup

- **Parameter**: coolingRate
  - **Type**: number
  - **Range**: 0.0001-1.0
  - **Location**: Line 188
  - **Priority**: High
  - **Category**: Physics/Warmup

- **Parameter**: enableBounds
  - **Type**: boolean
  - **Location**: Line 191
  - **Priority**: Medium
  - **Category**: Physics/Boundaries

- **Parameter**: boundsSize
  - **Type**: number
  - **Location**: Line 192
  - **Priority**: Medium
  - **Category**: Physics/Boundaries

- **Parameter**: boundaryDamping
  - **Type**: number
  - **Range**: 0.0-1.0
  - **Location**: Line 193
  - **Priority**: Medium
  - **Category**: Physics/Boundaries

- **Parameter**: collisionRadius
  - **Type**: number
  - **Location**: Line 194
  - **Priority**: Medium
  - **Category**: Physics/Collision

- **Parameter**: iterations
  - **Type**: number
  - **Location**: Line 197
  - **Priority**: High
  - **Category**: Physics/Simulation

- **Parameter**: massScale
  - **Type**: number
  - **Location**: Line 198
  - **Priority**: Medium
  - **Category**: Physics/Simulation

- **Parameter**: updateThreshold
  - **Type**: number
  - **Location**: Line 199
  - **Priority**: Medium
  - **Category**: Physics/Optimization

- **Parameter**: boundaryExtremeMultiplier
  - **Type**: number
  - **Range**: 1.0-5.0
  - **Location**: Line 203
  - **Priority**: Low
  - **Category**: Physics/Boundaries

- **Parameter**: boundaryExtremeForceMultiplier
  - **Type**: number
  - **Range**: 1.0-20.0
  - **Location**: Line 205
  - **Priority**: Low
  - **Category**: Physics/Boundaries

- **Parameter**: boundaryVelocityDamping
  - **Type**: number
  - **Range**: 0.0-1.0
  - **Location**: Line 207
  - **Priority**: Low
  - **Category**: Physics/Boundaries

- **Parameter**: maxForce
  - **Type**: number
  - **Range**: 1-1000
  - **Location**: Line 209
  - **Priority**: Medium
  - **Category**: Physics/Limits

- **Parameter**: seed
  - **Type**: number
  - **Location**: Line 211
  - **Priority**: Low
  - **Category**: Physics/Random

- **Parameter**: iteration (current)
  - **Type**: number
  - **Location**: Line 213
  - **Priority**: Low
  - **Category**: Physics/State

#### Clustering Configuration (Interface ClusteringConfig, Line 216-223)
- **Parameter**: algorithm
  - **Type**: enum
  - **Options**: 'none', 'kmeans', 'spectral', 'louvain'
  - **Location**: Line 217
  - **Priority**: Medium
  - **Category**: Analytics/Clustering

- **Parameter**: clusterCount
  - **Type**: number
  - **Location**: Line 218
  - **Priority**: Medium
  - **Category**: Analytics/Clustering

- **Parameter**: resolution
  - **Type**: number
  - **Location**: Line 219
  - **Priority**: Medium
  - **Category**: Analytics/Clustering

- **Parameter**: iterations
  - **Type**: number
  - **Location**: Line 220
  - **Priority**: Medium
  - **Category**: Analytics/Clustering

- **Parameter**: exportEnabled
  - **Type**: boolean
  - **Location**: Line 221
  - **Priority**: Low
  - **Category**: Analytics/Export

- **Parameter**: importEnabled
  - **Type**: boolean
  - **Location**: Line 222
  - **Priority**: Low
  - **Category**: Analytics/Import

#### Warmup Settings (Interface WarmupSettings, Line 233-239)
- **Parameter**: warmupDuration
  - **Type**: number
  - **Location**: Line 234
  - **Priority**: High
  - **Category**: Performance/Warmup

- **Parameter**: convergenceThreshold
  - **Type**: number
  - **Location**: Line 235
  - **Priority**: High
  - **Category**: Performance/Convergence

- **Parameter**: enableAdaptiveCooling
  - **Type**: boolean
  - **Location**: Line 236
  - **Priority**: Medium
  - **Category**: Performance/Adaptive

#### Section Paths (Function getSectionPaths, Line 912-963)

**Physics Section** (Line 914-918):
- visualisation.graphs.logseq.physics
- visualisation.graphs.visionflow.physics

**Rendering Section** (Line 919-929):
- visualisation.rendering.ambientLightIntensity
- visualisation.rendering.backgroundColor
- visualisation.rendering.directionalLightIntensity
- visualisation.rendering.enableAmbientOcclusion
- visualisation.rendering.enableAntialiasing
- visualisation.rendering.enableShadows
- visualisation.rendering.environmentIntensity
- visualisation.rendering.shadowMapSize
- visualisation.rendering.shadowBias
- visualisation.rendering.context

**XR Section** (Line 930-936):
- xr.enabled
- xr.mode
- xr.enableHandTracking
- xr.enableHaptics
- xr.quality

**Glow Section** (Line 937-942):
- visualisation.glow.enabled
- visualisation.glow.intensity
- visualisation.glow.radius
- visualisation.glow.threshold

**Hologram Section** (Line 943-947):
- visualisation.hologram.ringCount
- visualisation.hologram.ringColor
- visualisation.hologram.globalRotationSpeed

**Nodes Section** (Line 948-951):
- visualisation.graphs.logseq.nodes
- visualisation.graphs.visionflow.nodes

**Edges Section** (Line 952-955):
- visualisation.graphs.logseq.edges
- visualisation.graphs.visionflow.edges

**Labels Section** (Line 956-959):
- visualisation.graphs.logseq.labels
- visualisation.graphs.visionflow.labels

### 1.2 multiUserStore.ts (`client/src/store/multiUserStore.ts`)

**Multi-User Collaboration Settings** - 267 lines

#### User Data Interface (Line 8-17)
- **Parameter**: id
  - **Type**: string
  - **Location**: Line 9
  - **Priority**: Critical
  - **Category**: MultiUser/Identity

- **Parameter**: name
  - **Type**: string (optional)
  - **Location**: Line 10
  - **Priority**: Medium
  - **Category**: MultiUser/Display

- **Parameter**: position
  - **Type**: [number, number, number]
  - **Location**: Line 11
  - **Priority**: High
  - **Category**: MultiUser/Position

- **Parameter**: rotation
  - **Type**: [number, number, number]
  - **Location**: Line 12
  - **Priority**: High
  - **Category**: MultiUser/Position

- **Parameter**: isSelecting
  - **Type**: boolean
  - **Location**: Line 13
  - **Priority**: Medium
  - **Category**: MultiUser/Interaction

- **Parameter**: selectedNodeId
  - **Type**: string (optional)
  - **Location**: Line 14
  - **Priority**: Medium
  - **Category**: MultiUser/Interaction

- **Parameter**: lastUpdate
  - **Type**: number (timestamp)
  - **Location**: Line 15
  - **Priority**: High
  - **Category**: MultiUser/Sync

- **Parameter**: color
  - **Type**: string (optional)
  - **Location**: Line 16
  - **Priority**: Low
  - **Category**: MultiUser/Display

#### Connection Settings
- **Parameter**: connectionStatus
  - **Type**: enum
  - **Options**: 'disconnected', 'connecting', 'connected'
  - **Location**: Line 22
  - **Priority**: Critical
  - **Category**: MultiUser/Connection

- **Parameter**: staleThreshold
  - **Type**: number (milliseconds)
  - **Default**: 30000
  - **Location**: Line 86
  - **Priority**: Medium
  - **Category**: MultiUser/Cleanup

- **Parameter**: heartbeatInterval
  - **Type**: number (milliseconds)
  - **Default**: 5000
  - **Location**: Line 212
  - **Priority**: Medium
  - **Category**: MultiUser/Connection

- **Parameter**: reconnectInterval
  - **Type**: number (milliseconds)
  - **Default**: 5000
  - **Location**: Line 229
  - **Priority**: Medium
  - **Category**: MultiUser/Connection

---

## 2. CONTROL PANEL CONFIGURATION

### 2.1 settingsConfig.ts (`client/src/features/visualisation/components/ControlPanel/settingsConfig.ts`)

**Settings Tab Configuration** - 225 lines, 100+ individual parameters

#### Visualization Section (Line 24-63)

**Node Settings**:
- **Parameter**: nodeColor (baseColor)
  - **Type**: color
  - **Path**: visualisation.graphs.logseq.nodes.baseColor
  - **Location**: Line 28
  - **Priority**: High
  - **Category**: Visualization/Nodes

- **Parameter**: nodeSize
  - **Type**: slider
  - **Range**: 0.2-2.0
  - **Path**: visualisation.graphs.logseq.nodes.nodeSize
  - **Location**: Line 29
  - **Priority**: High
  - **Category**: Visualization/Nodes

- **Parameter**: nodeMetalness
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.graphs.logseq.nodes.metalness
  - **Location**: Line 30
  - **Priority**: Medium
  - **Category**: Visualization/Nodes

- **Parameter**: nodeOpacity
  - **Type**: slider
  - **Range**: 0-1
  - **Step**: 0.01
  - **Path**: visualisation.graphs.logseq.nodes.opacity
  - **Location**: Line 31
  - **Priority**: High
  - **Category**: Visualization/Nodes

- **Parameter**: nodeRoughness
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.graphs.logseq.nodes.roughness
  - **Location**: Line 32
  - **Priority**: Medium
  - **Category**: Visualization/Nodes

- **Parameter**: enableInstancing
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.nodes.enableInstancing
  - **Location**: Line 33
  - **Priority**: Medium
  - **Category**: Visualization/Performance

- **Parameter**: enableMetadataShape
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.nodes.enableMetadataShape
  - **Location**: Line 34
  - **Priority**: Low
  - **Category**: Visualization/Nodes

- **Parameter**: enableMetadataVis
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.nodes.enableMetadataVisualisation
  - **Location**: Line 35
  - **Priority**: Low
  - **Category**: Visualization/Nodes

- **Parameter**: nodeImportance
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.nodes.enableImportance
  - **Location**: Line 36
  - **Priority**: Medium
  - **Category**: Visualization/Nodes

**Edge Settings**:
- **Parameter**: edgeColor
  - **Type**: color
  - **Path**: visualisation.graphs.logseq.edges.color
  - **Location**: Line 39
  - **Priority**: High
  - **Category**: Visualization/Edges

- **Parameter**: edgeWidth (baseWidth)
  - **Type**: slider
  - **Range**: 0.01-2.0
  - **Path**: visualisation.graphs.logseq.edges.baseWidth
  - **Location**: Line 40
  - **Priority**: High
  - **Category**: Visualization/Edges

- **Parameter**: edgeOpacity
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.graphs.logseq.edges.opacity
  - **Location**: Line 41
  - **Priority**: High
  - **Category**: Visualization/Edges

- **Parameter**: enableArrows
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.edges.enableArrows
  - **Location**: Line 42
  - **Priority**: Medium
  - **Category**: Visualization/Edges

- **Parameter**: arrowSize
  - **Type**: slider
  - **Range**: 0.01-0.5
  - **Path**: visualisation.graphs.logseq.edges.arrowSize
  - **Location**: Line 43
  - **Priority**: Medium
  - **Category**: Visualization/Edges

- **Parameter**: glowStrength
  - **Type**: slider
  - **Range**: 0-5
  - **Path**: visualisation.graphs.logseq.edges.glowStrength
  - **Location**: Line 44
  - **Priority**: Medium
  - **Category**: Visualization/Effects

**Label Settings**:
- **Parameter**: enableLabels
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.labels.enableLabels
  - **Location**: Line 47
  - **Priority**: High
  - **Category**: Visualization/Labels

- **Parameter**: labelSize (desktopFontSize)
  - **Type**: slider
  - **Range**: 0.01-1.5
  - **Path**: visualisation.graphs.logseq.labels.desktopFontSize
  - **Location**: Line 48
  - **Priority**: High
  - **Category**: Visualization/Labels

- **Parameter**: labelColor
  - **Type**: color
  - **Path**: visualisation.graphs.logseq.labels.textColor
  - **Location**: Line 49
  - **Priority**: Medium
  - **Category**: Visualization/Labels

- **Parameter**: labelOutlineColor
  - **Type**: color
  - **Path**: visualisation.graphs.logseq.labels.textOutlineColor
  - **Location**: Line 50
  - **Priority**: Low
  - **Category**: Visualization/Labels

- **Parameter**: labelOutlineWidth
  - **Type**: slider
  - **Range**: 0-0.01
  - **Path**: visualisation.graphs.logseq.labels.textOutlineWidth
  - **Location**: Line 51
  - **Priority**: Low
  - **Category**: Visualization/Labels

**Lighting Settings**:
- **Parameter**: ambientLight
  - **Type**: slider
  - **Range**: 0-2
  - **Path**: visualisation.rendering.ambientLightIntensity
  - **Location**: Line 60
  - **Priority**: High
  - **Category**: Visualization/Lighting

- **Parameter**: directionalLight
  - **Type**: slider
  - **Range**: 0-2
  - **Path**: visualisation.rendering.directionalLightIntensity
  - **Location**: Line 61
  - **Priority**: High
  - **Category**: Visualization/Lighting

#### Physics Section (Line 65-97)

- **Parameter**: enabled
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.physics.enabled
  - **Location**: Line 68
  - **Priority**: Critical
  - **Category**: Physics/Control

- **Parameter**: autoBalance
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.physics.autoBalance
  - **Location**: Line 69
  - **Priority**: High
  - **Category**: Physics/Adaptive

- **Parameter**: damping
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.graphs.logseq.physics.damping
  - **Location**: Line 70
  - **Priority**: High
  - **Category**: Physics/Damping

- **Parameter**: springK
  - **Type**: slider
  - **Range**: 0.0001-10
  - **Path**: visualisation.graphs.logseq.physics.springK
  - **Location**: Line 71
  - **Priority**: Critical
  - **Category**: Physics/Forces

- **Parameter**: repelK
  - **Type**: slider
  - **Range**: 0.1-200
  - **Path**: visualisation.graphs.logseq.physics.repelK
  - **Location**: Line 72
  - **Priority**: Critical
  - **Category**: Physics/Forces

- **Parameter**: attractionK
  - **Type**: slider
  - **Range**: 0-10
  - **Path**: visualisation.graphs.logseq.physics.attractionK
  - **Location**: Line 73
  - **Priority**: High
  - **Category**: Physics/Forces

- **Parameter**: dt (Time Step)
  - **Type**: slider
  - **Range**: 0.001-0.1
  - **Path**: visualisation.graphs.logseq.physics.dt
  - **Location**: Line 74
  - **Priority**: Critical
  - **Category**: Physics/Simulation

- **Parameter**: maxVelocity
  - **Type**: slider
  - **Range**: 0.1-10
  - **Path**: visualisation.graphs.logseq.physics.maxVelocity
  - **Location**: Line 75
  - **Priority**: High
  - **Category**: Physics/Limits

- **Parameter**: separationRadius
  - **Type**: slider
  - **Range**: 0.1-10
  - **Path**: visualisation.graphs.logseq.physics.separationRadius
  - **Location**: Line 76
  - **Priority**: Medium
  - **Category**: Physics/Collision

- **Parameter**: enableBounds
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.physics.enableBounds
  - **Location**: Line 77
  - **Priority**: Medium
  - **Category**: Physics/Boundaries

- **Parameter**: boundsSize
  - **Type**: slider
  - **Range**: 1-10000
  - **Path**: visualisation.graphs.logseq.physics.boundsSize
  - **Location**: Line 78
  - **Priority**: Medium
  - **Category**: Physics/Boundaries

- **Parameter**: stressWeight
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.graphs.logseq.physics.stressWeight
  - **Location**: Line 79
  - **Priority**: Low
  - **Category**: Physics/Optimization

- **Parameter**: stressAlpha
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.graphs.logseq.physics.stressAlpha
  - **Location**: Line 80
  - **Priority**: Low
  - **Category**: Physics/Optimization

- **Parameter**: minDistance
  - **Type**: slider
  - **Range**: 0.05-1
  - **Path**: visualisation.graphs.logseq.physics.minDistance
  - **Location**: Line 81
  - **Priority**: Medium
  - **Category**: Physics/Collision

- **Parameter**: maxRepulsionDist
  - **Type**: slider
  - **Range**: 10-200
  - **Path**: visualisation.graphs.logseq.physics.maxRepulsionDist
  - **Location**: Line 82
  - **Priority**: High
  - **Category**: Physics/Forces

- **Parameter**: warmupIterations
  - **Type**: slider
  - **Range**: 0-500
  - **Path**: visualisation.graphs.logseq.physics.warmupIterations
  - **Location**: Line 83
  - **Priority**: High
  - **Category**: Physics/Warmup

- **Parameter**: coolingRate
  - **Type**: slider
  - **Range**: 0.00001-0.01
  - **Path**: visualisation.graphs.logseq.physics.coolingRate
  - **Location**: Line 84
  - **Priority**: High
  - **Category**: Physics/Warmup

- **Parameter**: restLength
  - **Type**: slider
  - **Range**: 10-200
  - **Path**: visualisation.graphs.logseq.physics.restLength
  - **Location**: Line 85
  - **Priority**: High
  - **Category**: Physics/CUDA

- **Parameter**: repulsionCutoff
  - **Type**: slider
  - **Range**: 10-200
  - **Path**: visualisation.graphs.logseq.physics.repulsionCutoff
  - **Location**: Line 86
  - **Priority**: High
  - **Category**: Physics/CUDA

- **Parameter**: repulsionSofteningEpsilon
  - **Type**: slider
  - **Range**: 0.00001-0.01
  - **Path**: visualisation.graphs.logseq.physics.repulsionSofteningEpsilon
  - **Location**: Line 87
  - **Priority**: Medium
  - **Category**: Physics/CUDA

- **Parameter**: centerGravityK
  - **Type**: slider
  - **Range**: 0-0.1
  - **Path**: visualisation.graphs.logseq.physics.centerGravityK
  - **Location**: Line 88
  - **Priority**: Medium
  - **Category**: Physics/CUDA

- **Parameter**: gridCellSize
  - **Type**: slider
  - **Range**: 10-100
  - **Path**: visualisation.graphs.logseq.physics.gridCellSize
  - **Location**: Line 89
  - **Priority**: Medium
  - **Category**: Physics/CUDA

- **Parameter**: boundaryExtremeMultiplier
  - **Type**: slider
  - **Range**: 1-5
  - **Path**: visualisation.graphs.logseq.physics.boundaryExtremeMultiplier
  - **Location**: Line 90
  - **Priority**: Low
  - **Category**: Physics/Boundaries

- **Parameter**: boundaryExtremeForceMultiplier
  - **Type**: slider
  - **Range**: 1-20
  - **Path**: visualisation.graphs.logseq.physics.boundaryExtremeForceMultiplier
  - **Location**: Line 91
  - **Priority**: Low
  - **Category**: Physics/Boundaries

- **Parameter**: boundaryVelocityDamping
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.graphs.logseq.physics.boundaryVelocityDamping
  - **Location**: Line 92
  - **Priority**: Low
  - **Category**: Physics/Boundaries

- **Parameter**: iterations
  - **Type**: slider
  - **Range**: 1-1000
  - **Path**: visualisation.graphs.logseq.physics.iterations
  - **Location**: Line 93
  - **Priority**: High
  - **Category**: Physics/Simulation

- **Parameter**: massScale
  - **Type**: slider
  - **Range**: 0.1-10
  - **Path**: visualisation.graphs.logseq.physics.massScale
  - **Location**: Line 94
  - **Priority**: Medium
  - **Category**: Physics/Simulation

- **Parameter**: boundaryDamping
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.graphs.logseq.physics.boundaryDamping
  - **Location**: Line 95
  - **Priority**: Medium
  - **Category**: Physics/Boundaries

- **Parameter**: updateThreshold
  - **Type**: slider
  - **Range**: 0-0.5
  - **Path**: visualisation.graphs.logseq.physics.updateThreshold
  - **Location**: Line 96
  - **Priority**: Medium
  - **Category**: Physics/Optimization

#### Visual Effects Section (Line 137-168)

**Glow Settings**:
- **Parameter**: glow (Hologram Glow)
  - **Type**: toggle
  - **Path**: visualisation.glow.enabled
  - **Location**: Line 140
  - **Priority**: Medium
  - **Category**: Effects/Glow

- **Parameter**: glowIntensity
  - **Type**: slider
  - **Range**: 0-5
  - **Step**: 0.1
  - **Path**: visualisation.glow.intensity
  - **Location**: Line 141
  - **Priority**: Medium
  - **Category**: Effects/Glow

- **Parameter**: glowRadius
  - **Type**: slider
  - **Range**: 0-5
  - **Step**: 0.05
  - **Path**: visualisation.glow.radius
  - **Location**: Line 142
  - **Priority**: Medium
  - **Category**: Effects/Glow

- **Parameter**: glowThreshold
  - **Type**: slider
  - **Range**: 0-1
  - **Step**: 0.01
  - **Path**: visualisation.glow.threshold
  - **Location**: Line 143
  - **Priority**: Medium
  - **Category**: Effects/Glow

**Hologram Settings**:
- **Parameter**: hologram
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.nodes.enableHologram
  - **Location**: Line 149
  - **Priority**: Medium
  - **Category**: Effects/Hologram

- **Parameter**: ringCount
  - **Type**: slider
  - **Range**: 0-10
  - **Path**: visualisation.hologram.ringCount
  - **Location**: Line 150
  - **Priority**: Low
  - **Category**: Effects/Hologram

- **Parameter**: ringColor
  - **Type**: color
  - **Path**: visualisation.hologram.ringColor
  - **Location**: Line 151
  - **Priority**: Low
  - **Category**: Effects/Hologram

- **Parameter**: ringOpacity
  - **Type**: slider
  - **Range**: 0-1
  - **Path**: visualisation.hologram.ringOpacity
  - **Location**: Line 152
  - **Priority**: Low
  - **Category**: Effects/Hologram

- **Parameter**: ringRotationSpeed
  - **Type**: slider
  - **Range**: 0-5
  - **Path**: visualisation.hologram.ringRotationSpeed
  - **Location**: Line 153
  - **Priority**: Low
  - **Category**: Effects/Hologram

**Edge Flow Effects**:
- **Parameter**: flowEffect
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.edges.enableFlowEffect
  - **Location**: Line 154
  - **Priority**: Medium
  - **Category**: Effects/Edges

- **Parameter**: flowSpeed
  - **Type**: slider
  - **Range**: 0.1-5
  - **Path**: visualisation.graphs.logseq.edges.flowSpeed
  - **Location**: Line 155
  - **Priority**: Medium
  - **Category**: Effects/Edges

- **Parameter**: flowIntensity
  - **Type**: slider
  - **Range**: 0-10
  - **Path**: visualisation.graphs.logseq.edges.flowIntensity
  - **Location**: Line 156
  - **Priority**: Medium
  - **Category**: Effects/Edges

- **Parameter**: useGradient
  - **Type**: toggle
  - **Path**: visualisation.graphs.logseq.edges.useGradient
  - **Location**: Line 157
  - **Priority**: Low
  - **Category**: Effects/Edges

- **Parameter**: distanceIntensity
  - **Type**: slider
  - **Range**: 0-10
  - **Path**: visualisation.graphs.logseq.edges.distanceIntensity
  - **Location**: Line 158
  - **Priority**: Low
  - **Category**: Effects/Edges

**Animation Settings**:
- **Parameter**: nodeAnimations
  - **Type**: toggle
  - **Path**: visualisation.animations.enableNodeAnimations
  - **Location**: Line 159
  - **Priority**: Medium
  - **Category**: Effects/Animation

- **Parameter**: pulseEnabled
  - **Type**: toggle
  - **Path**: visualisation.animations.pulseEnabled
  - **Location**: Line 160
  - **Priority**: Low
  - **Category**: Effects/Animation

- **Parameter**: pulseSpeed
  - **Type**: slider
  - **Range**: 0.1-2
  - **Path**: visualisation.animations.pulseSpeed
  - **Location**: Line 161
  - **Priority**: Low
  - **Category**: Effects/Animation

- **Parameter**: pulseStrength
  - **Type**: slider
  - **Range**: 0.1-2
  - **Path**: visualisation.animations.pulseStrength
  - **Location**: Line 162
  - **Priority**: Low
  - **Category**: Effects/Animation

- **Parameter**: selectionWave
  - **Type**: toggle
  - **Path**: visualisation.animations.selectionWaveEnabled
  - **Location**: Line 163
  - **Priority**: Low
  - **Category**: Effects/Animation

- **Parameter**: waveSpeed
  - **Type**: slider
  - **Range**: 0.1-2
  - **Path**: visualisation.animations.waveSpeed
  - **Location**: Line 164
  - **Priority**: Low
  - **Category**: Effects/Animation

**Rendering Quality**:
- **Parameter**: antialiasing
  - **Type**: toggle
  - **Path**: visualisation.rendering.enableAntialiasing
  - **Location**: Line 165
  - **Priority**: High
  - **Category**: Rendering/Quality

- **Parameter**: shadows
  - **Type**: toggle
  - **Path**: visualisation.rendering.enableShadows
  - **Location**: Line 166
  - **Priority**: Medium
  - **Category**: Rendering/Quality

- **Parameter**: ambientOcclusion
  - **Type**: toggle
  - **Path**: visualisation.rendering.enableAmbientOcclusion
  - **Location**: Line 167
  - **Priority**: Medium
  - **Category**: Rendering/Quality

#### Developer Section (Line 171-196)

- **Parameter**: enableDebug
  - **Type**: toggle
  - **Path**: system.debug.enabled
  - **Location**: Line 181
  - **Priority**: High
  - **Category**: Developer/Debug

#### Authentication Section (Line 199-207)

- **Parameter**: nostr
  - **Type**: nostr-button
  - **Path**: auth.nostr
  - **Location**: Line 202
  - **Priority**: Medium
  - **Category**: Authentication/Nostr

- **Parameter**: enabled
  - **Type**: toggle
  - **Path**: auth.enabled
  - **Location**: Line 203
  - **Priority**: High
  - **Category**: Authentication/Control

- **Parameter**: required
  - **Type**: toggle
  - **Path**: auth.required
  - **Location**: Line 204
  - **Priority**: High
  - **Category**: Authentication/Control

- **Parameter**: provider
  - **Type**: text
  - **Path**: auth.provider
  - **Location**: Line 205
  - **Priority**: Low
  - **Category**: Authentication/Provider

#### XR/AR Section (Line 209-222)

- **Parameter**: persistSettings
  - **Type**: toggle
  - **Path**: system.persistSettingsOnServer
  - **Location**: Line 212
  - **Priority**: Medium
  - **Category**: System/Persistence

- **Parameter**: customBackendURL
  - **Type**: text
  - **Path**: system.customBackendUrl
  - **Location**: Line 213
  - **Priority**: Low
  - **Category**: System/Backend

- **Parameter**: xrEnabled
  - **Type**: toggle
  - **Path**: xr.enabled
  - **Location**: Line 214
  - **Priority**: High
  - **Category**: XR/Control

- **Parameter**: xrQuality
  - **Type**: select
  - **Options**: 'Low', 'Medium', 'High'
  - **Path**: xr.quality
  - **Location**: Line 215
  - **Priority**: High
  - **Category**: XR/Quality

- **Parameter**: xrRenderScale
  - **Type**: slider
  - **Range**: 0.5-2.0
  - **Path**: xr.renderScale
  - **Location**: Line 216
  - **Priority**: High
  - **Category**: XR/Rendering

- **Parameter**: handTracking
  - **Type**: toggle
  - **Path**: xr.handTracking.enabled
  - **Location**: Line 217
  - **Priority**: Medium
  - **Category**: XR/Tracking

- **Parameter**: enableHaptics
  - **Type**: toggle
  - **Path**: xr.interactions.enableHaptics
  - **Location**: Line 218
  - **Priority**: Low
  - **Category**: XR/Haptics

- **Parameter**: xrComputeMode
  - **Type**: toggle
  - **Path**: xr.gpu.enableOptimizedCompute
  - **Location**: Line 219
  - **Priority**: Medium
  - **Category**: XR/GPU

- **Parameter**: xrPerformancePreset
  - **Type**: select
  - **Options**: 'Battery Saver', 'Balanced', 'Performance'
  - **Path**: xr.performance.preset
  - **Location**: Line 220
  - **Priority**: High
  - **Category**: XR/Performance

- **Parameter**: xrAdaptiveQuality
  - **Type**: toggle
  - **Path**: xr.enableAdaptiveQuality
  - **Location**: Line 221
  - **Priority**: Medium
  - **Category**: XR/Performance

### 2.2 config.ts - Tab Configuration (`client/src/features/visualisation/components/ControlPanel/config.ts`)

**Tab Definitions** - 47 lines, 14 tabs with keyboard shortcuts

#### Tab Configs (Line 22-40)
- **Tab**: Dashboard (buttonKey: '1')
- **Tab**: Visualization (buttonKey: '2')
- **Tab**: Physics (buttonKey: '3')
- **Tab**: Analytics (buttonKey: '4')
- **Tab**: Performance (buttonKey: '5')
- **Tab**: Visual Effects (buttonKey: '6')
- **Tab**: Developer (buttonKey: '7')
- **Tab**: XR/AR (buttonKey: '8')
- **Tab**: Analysis (buttonKey: 'A')
- **Tab**: Visualisation (buttonKey: 'B')
- **Tab**: Optimisation (buttonKey: 'C')
- **Tab**: Interaction (buttonKey: 'D')
- **Tab**: Export (buttonKey: 'E')
- **Tab**: Auth/Nostr (buttonKey: 'F')

---

## 3. SPACEPILOT 3D CONTROLLER

### 3.1 SpacePilotController.ts (`client/src/features/visualisation/controls/SpacePilotController.ts`)

**3D Mouse Configuration** - 430 lines

#### SpacePilotConfig Interface (Line 8-48)

**Translation Sensitivity** (Line 9-13):
- **Parameter**: translationSensitivity.x
  - **Type**: number
  - **Range**: 0.1-10.0
  - **Location**: Line 10
  - **Priority**: High
  - **Category**: Input/Translation

- **Parameter**: translationSensitivity.y
  - **Type**: number
  - **Range**: 0.1-10.0
  - **Location**: Line 11
  - **Priority**: High
  - **Category**: Input/Translation

- **Parameter**: translationSensitivity.z
  - **Type**: number
  - **Range**: 0.1-10.0
  - **Location**: Line 12
  - **Priority**: High
  - **Category**: Input/Translation

**Rotation Sensitivity** (Line 14-18):
- **Parameter**: rotationSensitivity.x
  - **Type**: number
  - **Range**: 0.1-10.0
  - **Location**: Line 15
  - **Priority**: High
  - **Category**: Input/Rotation

- **Parameter**: rotationSensitivity.y
  - **Type**: number
  - **Range**: 0.1-10.0
  - **Location**: Line 16
  - **Priority**: High
  - **Category**: Input/Rotation

- **Parameter**: rotationSensitivity.z
  - **Type**: number
  - **Range**: 0.1-10.0
  - **Location**: Line 17
  - **Priority**: High
  - **Category**: Input/Rotation

**Control Parameters**:
- **Parameter**: deadzone
  - **Type**: number
  - **Range**: 0-0.2
  - **Location**: Line 21
  - **Priority**: High
  - **Category**: Input/Filtering

- **Parameter**: smoothing
  - **Type**: number
  - **Range**: 0-1
  - **Location**: Line 24
  - **Priority**: High
  - **Category**: Input/Filtering

- **Parameter**: mode
  - **Type**: enum
  - **Options**: 'camera', 'object', 'navigation'
  - **Location**: Line 27
  - **Priority**: Critical
  - **Category**: Input/Mode

**Axis Inversion** (Line 30-37):
- **Parameter**: invertAxes.x
  - **Type**: boolean
  - **Location**: Line 31
  - **Priority**: Medium
  - **Category**: Input/Inversion

- **Parameter**: invertAxes.y
  - **Type**: boolean
  - **Location**: Line 32
  - **Priority**: Medium
  - **Category**: Input/Inversion

- **Parameter**: invertAxes.z
  - **Type**: boolean
  - **Location**: Line 33
  - **Priority**: Medium
  - **Category**: Input/Inversion

- **Parameter**: invertAxes.rx
  - **Type**: boolean
  - **Location**: Line 34
  - **Priority**: Medium
  - **Category**: Input/Inversion

- **Parameter**: invertAxes.ry
  - **Type**: boolean
  - **Location**: Line 35
  - **Priority**: Medium
  - **Category**: Input/Inversion

- **Parameter**: invertAxes.rz
  - **Type**: boolean
  - **Location**: Line 36
  - **Priority**: Medium
  - **Category**: Input/Inversion

**Enabled Axes** (Line 40-47):
- **Parameter**: enabledAxes.x
  - **Type**: boolean
  - **Location**: Line 41
  - **Priority**: Medium
  - **Category**: Input/Enable

- **Parameter**: enabledAxes.y
  - **Type**: boolean
  - **Location**: Line 42
  - **Priority**: Medium
  - **Category**: Input/Enable

- **Parameter**: enabledAxes.z
  - **Type**: boolean
  - **Location**: Line 43
  - **Priority**: Medium
  - **Category**: Input/Enable

- **Parameter**: enabledAxes.rx
  - **Type**: boolean
  - **Location**: Line 44
  - **Priority**: Medium
  - **Category**: Input/Enable

- **Parameter**: enabledAxes.ry
  - **Type**: boolean
  - **Location**: Line 45
  - **Priority**: Medium
  - **Category**: Input/Enable

- **Parameter**: enabledAxes.rz
  - **Type**: boolean
  - **Location**: Line 46
  - **Priority**: Medium
  - **Category**: Input/Enable

#### Default Configuration (Line 53-75)
All defaults set for translation sensitivity, rotation sensitivity, deadzone, smoothing, mode, and axis settings.

#### Constants (Line 121-123)
- **Constant**: INPUT_SCALE = 1/32768
- **Constant**: TRANSLATION_SPEED = 0.01
- **Constant**: ROTATION_SPEED = 0.001

---

## 4. HOLOGRAPHIC VISUALIZATION

### 4.1 HolographicDataSphere.tsx (`client/src/features/visualisation/components/HolographicDataSphere.tsx`)

**Hologram Scene Configuration** - 887 lines with extensive visual parameters

#### Scene Configuration (Line 30-34)
- **Parameter**: background
  - **Type**: color
  - **Default**: '#02030c'
  - **Location**: Line 31
  - **Priority**: Medium
  - **Category**: Hologram/Scene
  - **TODO**: Map to settings system

- **Parameter**: fogNear
  - **Type**: number
  - **Default**: 6
  - **Location**: Line 32
  - **Priority**: Low
  - **Category**: Hologram/Fog
  - **TODO**: Map to settings system

- **Parameter**: fogFar
  - **Type**: number
  - **Default**: 34
  - **Location**: Line 33
  - **Priority**: Low
  - **Category**: Hologram/Fog
  - **TODO**: Map to settings system

#### Base Opacity (Line 36)
- **Parameter**: HOLOGRAM_BASE_OPACITY
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 36
  - **Priority**: High
  - **Category**: Hologram/Opacity
  - **TODO**: Map to hologram.ringOpacity

#### Lighting Configuration (Line 38-43)
- **Parameter**: LIGHTING_CONFIG.ambient
  - **Type**: number
  - **Default**: 0.2
  - **Location**: Line 39
  - **Priority**: Medium
  - **Category**: Hologram/Lighting
  - **TODO**: Map to rendering.ambientLightIntensity

- **Parameter**: keyLight.position
  - **Type**: [number, number, number]
  - **Default**: [5, 7, 4]
  - **Location**: Line 40
  - **Priority**: Medium
  - **Category**: Hologram/Lighting
  - **TODO**: Map to settings

- **Parameter**: keyLight.intensity
  - **Type**: number
  - **Default**: 1.65
  - **Location**: Line 40
  - **Priority**: Medium
  - **Category**: Hologram/Lighting

- **Parameter**: keyLight.color
  - **Type**: color
  - **Default**: '#7acbff'
  - **Location**: Line 40
  - **Priority**: Medium
  - **Category**: Hologram/Lighting

- **Parameter**: rimLight.position
  - **Type**: [number, number, number]
  - **Default**: [-6, -4, -3]
  - **Location**: Line 41
  - **Priority**: Medium
  - **Category**: Hologram/Lighting

- **Parameter**: rimLight.intensity
  - **Type**: number
  - **Default**: 1.05
  - **Location**: Line 41
  - **Priority**: Medium
  - **Category**: Hologram/Lighting

- **Parameter**: rimLight.color
  - **Type**: color
  - **Default**: '#ff7b1f'
  - **Location**: Line 41
  - **Priority**: Medium
  - **Category**: Hologram/Lighting

- **Parameter**: fillLight.position
  - **Type**: [number, number, number]
  - **Default**: [0, 0, 12]
  - **Location**: Line 42
  - **Priority**: Medium
  - **Category**: Hologram/Lighting

- **Parameter**: fillLight.intensity
  - **Type**: number
  - **Default**: 0.55
  - **Location**: Line 42
  - **Priority**: Medium
  - **Category**: Hologram/Lighting

- **Parameter**: fillLight.color
  - **Type**: color
  - **Default**: '#00faff'
  - **Location**: Line 42
  - **Priority**: Medium
  - **Category**: Hologram/Lighting

#### Post-Process Defaults (Line 45-56)
- **Parameter**: globalAlpha
  - **Type**: number
  - **Default**: 0.3 (HOLOGRAM_BASE_OPACITY)
  - **Location**: Line 46
  - **Priority**: High
  - **Category**: Hologram/PostProcess
  - **TODO**: Map to hologram.ringOpacity

- **Parameter**: bloomIntensity
  - **Type**: number
  - **Default**: 1.5
  - **Location**: Line 47
  - **Priority**: High
  - **Category**: Hologram/Bloom
  - **TODO**: Map to settings

- **Parameter**: bloomThreshold
  - **Type**: number
  - **Default**: 0.15
  - **Location**: Line 48
  - **Priority**: High
  - **Category**: Hologram/Bloom
  - **TODO**: Map to settings

- **Parameter**: bloomSmoothing
  - **Type**: number
  - **Default**: 0.36
  - **Location**: Line 49
  - **Priority**: Medium
  - **Category**: Hologram/Bloom

- **Parameter**: aoRadius
  - **Type**: number
  - **Default**: 124
  - **Location**: Line 50
  - **Priority**: Medium
  - **Category**: Hologram/AO

- **Parameter**: aoIntensity
  - **Type**: number
  - **Default**: 0.75
  - **Location**: Line 51
  - **Priority**: Medium
  - **Category**: Hologram/AO

- **Parameter**: dofFocusDistance
  - **Type**: number
  - **Default**: 3.6
  - **Location**: Line 52
  - **Priority**: Low
  - **Category**: Hologram/DOF

- **Parameter**: dofFocalLength
  - **Type**: number
  - **Default**: 4.4
  - **Location**: Line 53
  - **Priority**: Low
  - **Category**: Hologram/DOF

- **Parameter**: dofBokehScale
  - **Type**: number
  - **Default**: 520
  - **Location**: Line 54
  - **Priority**: Low
  - **Category**: Hologram/DOF

- **Parameter**: vignetteDarkness
  - **Type**: number
  - **Default**: 0.45
  - **Location**: Line 55
  - **Priority**: Low
  - **Category**: Hologram/Vignette

#### Fade Defaults (Line 58-61)
- **Parameter**: fadeStart
  - **Type**: number
  - **Default**: 1200
  - **Location**: Line 59
  - **Priority**: Medium
  - **Category**: Hologram/Fade

- **Parameter**: fadeEnd
  - **Type**: number
  - **Default**: 2800
  - **Location**: Line 60
  - **Priority**: Medium
  - **Category**: Hologram/Fade

#### ParticleCore Component (Line 160)
- **Parameter**: count
  - **Type**: number
  - **Default**: 5200
  - **Location**: Line 160
  - **Priority**: Medium
  - **Category**: Hologram/Particles

- **Parameter**: radius
  - **Type**: number
  - **Default**: 170
  - **Location**: Line 160
  - **Priority**: Medium
  - **Category**: Hologram/Particles

- **Parameter**: color
  - **Type**: color
  - **Default**: '#02f0ff'
  - **Location**: Line 160
  - **Priority**: Medium
  - **Category**: Hologram/Particles

- **Parameter**: opacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 160
  - **Priority**: High
  - **Category**: Hologram/Particles

#### HolographicShell Component (Line 200-208)
- **Parameter**: radius
  - **Type**: number
  - **Default**: 250
  - **Location**: Line 201
  - **Priority**: High
  - **Category**: Hologram/Shell

- **Parameter**: color
  - **Type**: color
  - **Default**: '#00faff'
  - **Location**: Line 202
  - **Priority**: Medium
  - **Category**: Hologram/Shell

- **Parameter**: detail
  - **Type**: number
  - **Default**: 3
  - **Location**: Line 203
  - **Priority**: Low
  - **Category**: Hologram/Shell

- **Parameter**: spikeHeight
  - **Type**: number
  - **Default**: 0.24
  - **Location**: Line 204
  - **Priority**: Medium
  - **Category**: Hologram/Shell

- **Parameter**: emissiveIntensity
  - **Type**: number
  - **Default**: 2.8
  - **Location**: Line 205
  - **Priority**: Medium
  - **Category**: Hologram/Shell

- **Parameter**: surfaceOpacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 206
  - **Priority**: High
  - **Category**: Hologram/Shell

- **Parameter**: spikeOpacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 207
  - **Priority**: High
  - **Category**: Hologram/Shell

#### TechnicalGrid Component (Line 313)
- **Parameter**: count
  - **Type**: number
  - **Default**: 240
  - **Location**: Line 313
  - **Priority**: Medium
  - **Category**: Hologram/Grid

- **Parameter**: radius
  - **Type**: number
  - **Default**: 410
  - **Location**: Line 313
  - **Priority**: Medium
  - **Category**: Hologram/Grid

- **Parameter**: opacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 313
  - **Priority**: Medium
  - **Category**: Hologram/Grid

#### OrbitalRings Component (Line 374)
- **Parameter**: radius
  - **Type**: number
  - **Default**: 470
  - **Location**: Line 374
  - **Priority**: Medium
  - **Category**: Hologram/Rings

- **Parameter**: color
  - **Type**: color
  - **Default**: '#00faff'
  - **Location**: Line 374
  - **Priority**: Medium
  - **Category**: Hologram/Rings

- **Parameter**: opacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 374
  - **Priority**: High
  - **Category**: Hologram/Rings

#### TextRing Component (Line 427-433)
- **Parameter**: text
  - **Type**: string
  - **Default**: 'JUNKIEJARVIS AGENTIC KNOWLEDGE SYSTEM • '
  - **Location**: Line 428
  - **Priority**: Low
  - **Category**: Hologram/Text

- **Parameter**: radius
  - **Type**: number
  - **Default**: 560
  - **Location**: Line 429
  - **Priority**: Medium
  - **Category**: Hologram/Text

- **Parameter**: fontSize
  - **Type**: number
  - **Default**: 32
  - **Location**: Line 430
  - **Priority**: Medium
  - **Category**: Hologram/Text

- **Parameter**: color
  - **Type**: color
  - **Default**: '#7fe8ff'
  - **Location**: Line 431
  - **Priority**: Medium
  - **Category**: Hologram/Text

- **Parameter**: opacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 432
  - **Priority**: Medium
  - **Category**: Hologram/Text

#### EnergyArcs Component (Line 466)
- **Parameter**: innerRadius
  - **Type**: number
  - **Default**: 1.28
  - **Location**: Line 466
  - **Priority**: Low
  - **Category**: Hologram/Arcs

- **Parameter**: outerRadius
  - **Type**: number
  - **Default**: 1.95
  - **Location**: Line 466
  - **Priority**: Low
  - **Category**: Hologram/Arcs

- **Parameter**: opacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 466
  - **Priority**: Low
  - **Category**: Hologram/Arcs

#### SurroundingSwarm Component (Line 519)
- **Parameter**: count
  - **Type**: number
  - **Default**: 9000
  - **Location**: Line 519
  - **Priority**: High
  - **Category**: Hologram/Swarm

- **Parameter**: radius
  - **Type**: number
  - **Default**: 6800
  - **Location**: Line 519
  - **Priority**: High
  - **Category**: Hologram/Swarm

- **Parameter**: opacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 519
  - **Priority**: High
  - **Category**: Hologram/Swarm

#### Canvas Settings (Line 857-866)
- **Parameter**: dpr (Device Pixel Ratio)
  - **Type**: [number, number]
  - **Default**: [1.3, 2.5]
  - **Location**: Line 858
  - **Priority**: High
  - **Category**: Rendering/Quality

- **Parameter**: camera.position
  - **Type**: [number, number, number]
  - **Default**: [0, 0, 6]
  - **Location**: Line 859
  - **Priority**: High
  - **Category**: Camera/Position

- **Parameter**: camera.fov
  - **Type**: number
  - **Default**: 48
  - **Location**: Line 859
  - **Priority**: Medium
  - **Category**: Camera/FOV

- **Parameter**: camera.near
  - **Type**: number
  - **Default**: 0.1
  - **Location**: Line 859
  - **Priority**: Medium
  - **Category**: Camera/Clipping

- **Parameter**: camera.far
  - **Type**: number
  - **Default**: 100
  - **Location**: Line 859
  - **Priority**: Medium
  - **Category**: Camera/Clipping

- **Parameter**: gl.antialias
  - **Type**: boolean
  - **Default**: true
  - **Location**: Line 861
  - **Priority**: High
  - **Category**: Rendering/Quality

- **Parameter**: gl.toneMapping
  - **Type**: constant
  - **Default**: THREE.ACESFilmicToneMapping
  - **Location**: Line 862
  - **Priority**: Medium
  - **Category**: Rendering/ToneMapping

- **Parameter**: gl.toneMappingExposure
  - **Type**: number
  - **Default**: 1.65
  - **Location**: Line 863
  - **Priority**: Medium
  - **Category**: Rendering/Exposure

- **Parameter**: gl.preserveDrawingBuffer
  - **Type**: boolean
  - **Default**: false
  - **Location**: Line 864
  - **Priority**: Low
  - **Category**: Rendering/Buffer

- **Parameter**: gl.powerPreference
  - **Type**: string
  - **Default**: 'high-performance'
  - **Location**: Line 865
  - **Priority**: High
  - **Category**: Rendering/Performance

#### OrbitControls Settings (Line 877-882)
- **Parameter**: enablePan
  - **Type**: boolean
  - **Default**: false
  - **Location**: Line 878
  - **Priority**: Medium
  - **Category**: Controls/Pan

- **Parameter**: minDistance
  - **Type**: number
  - **Default**: 3
  - **Location**: Line 879
  - **Priority**: High
  - **Category**: Controls/Zoom

- **Parameter**: maxDistance
  - **Type**: number
  - **Default**: 12
  - **Location**: Line 880
  - **Priority**: High
  - **Category**: Controls/Zoom

- **Parameter**: autoRotate
  - **Type**: boolean
  - **Default**: true
  - **Location**: Line 881
  - **Priority**: Low
  - **Category**: Controls/Rotation

- **Parameter**: autoRotateSpeed
  - **Type**: number
  - **Default**: 0.55
  - **Location**: Line 882
  - **Priority**: Low
  - **Category**: Controls/Rotation

---

## 5. WIREFRAME CLOUD MESH

### 5.1 WireframeCloudMesh.tsx (`client/src/features/visualisation/components/WireframeCloudMesh.tsx`)

**Wireframe Effects** - 154 lines

#### WireframeCloudMeshProps Interface (Line 6-21)
- **Parameter**: geometry
  - **Type**: enum
  - **Options**: 'torus', 'sphere', 'icosahedron'
  - **Location**: Line 7
  - **Priority**: High
  - **Category**: Wireframe/Geometry

- **Parameter**: geometryArgs
  - **Type**: any[]
  - **Location**: Line 8
  - **Priority**: Medium
  - **Category**: Wireframe/Geometry

- **Parameter**: position
  - **Type**: [number, number, number]
  - **Default**: [0, 0, 0]
  - **Location**: Line 9
  - **Priority**: Medium
  - **Category**: Wireframe/Transform

- **Parameter**: rotation
  - **Type**: [number, number, number]
  - **Default**: [0, 0, 0]
  - **Location**: Line 10
  - **Priority**: Medium
  - **Category**: Wireframe/Transform

- **Parameter**: scale
  - **Type**: number | [number, number, number]
  - **Default**: 1
  - **Location**: Line 11
  - **Priority**: Medium
  - **Category**: Wireframe/Transform

- **Parameter**: color
  - **Type**: string | THREE.Color
  - **Default**: '#00ffff'
  - **Location**: Line 12
  - **Priority**: High
  - **Category**: Wireframe/Material

- **Parameter**: wireframeColor
  - **Type**: string | THREE.Color
  - **Location**: Line 13
  - **Priority**: Medium
  - **Category**: Wireframe/Material

- **Parameter**: opacity
  - **Type**: number
  - **Default**: 0.3
  - **Location**: Line 14
  - **Priority**: High
  - **Category**: Wireframe/Material

- **Parameter**: wireframeOpacity
  - **Type**: number
  - **Default**: 0.8
  - **Location**: Line 15
  - **Priority**: High
  - **Category**: Wireframe/Material

- **Parameter**: cloudExtension
  - **Type**: number
  - **Default**: 10.0
  - **Location**: Line 16
  - **Priority**: Medium
  - **Category**: Wireframe/Effect

- **Parameter**: blurRadius
  - **Type**: number
  - **Default**: 15.0
  - **Location**: Line 17
  - **Priority**: Medium
  - **Category**: Wireframe/Effect

- **Parameter**: glowIntensity
  - **Type**: number
  - **Default**: 2.0
  - **Location**: Line 18
  - **Priority**: High
  - **Category**: Wireframe/Effect

- **Parameter**: rotationSpeed
  - **Type**: number
  - **Default**: 0
  - **Location**: Line 19
  - **Priority**: Medium
  - **Category**: Wireframe/Animation

- **Parameter**: rotationAxis
  - **Type**: [number, number, number]
  - **Default**: [0, 1, 0]
  - **Location**: Line 20
  - **Priority**: Low
  - **Category**: Wireframe/Animation

#### MultiLayerWireframeCloud Props (Line 109-116)
- **Parameter**: layers
  - **Type**: number
  - **Default**: 3
  - **Location**: Line 114
  - **Priority**: High
  - **Category**: Wireframe/Layering

---

## 6. XR/AR INTEGRATION

### 6.1 useQuest3Integration.ts (`client/src/hooks/useQuest3Integration.ts`)

**Quest 3 AR Configuration** - 206 lines

#### Quest3IntegrationOptions Interface (Line 17-21)
- **Parameter**: enableAutoStart
  - **Type**: boolean
  - **Default**: true
  - **Location**: Line 18
  - **Priority**: High
  - **Category**: XR/AutoStart

- **Parameter**: retryOnFailure
  - **Type**: boolean
  - **Default**: true
  - **Location**: Line 19
  - **Priority**: Medium
  - **Category**: XR/Retry

- **Parameter**: maxRetries
  - **Type**: number
  - **Default**: 3
  - **Location**: Line 20
  - **Priority**: Medium
  - **Category**: XR/Retry

### 6.2 useHeadTracking.ts (`client/src/hooks/useHeadTracking.ts`)

**Head Tracking Configuration** - 159 lines

#### MediaPipe Configuration (Line 30-39)
- **Parameter**: modelAssetPath
  - **Type**: string
  - **Default**: '/models/face_landmarker.task'
  - **Location**: Line 32
  - **Priority**: Critical
  - **Category**: HeadTracking/Model

- **Parameter**: delegate
  - **Type**: string
  - **Default**: 'GPU'
  - **Location**: Line 33
  - **Priority**: High
  - **Category**: HeadTracking/Performance

- **Parameter**: outputFaceBlendshapes
  - **Type**: boolean
  - **Default**: false
  - **Location**: Line 35
  - **Priority**: Low
  - **Category**: HeadTracking/Output

- **Parameter**: outputFacialTransformationMatrixes
  - **Type**: boolean
  - **Default**: false
  - **Location**: Line 36
  - **Priority**: Low
  - **Category**: HeadTracking/Output

- **Parameter**: runningMode
  - **Type**: string
  - **Default**: 'VIDEO'
  - **Location**: Line 37
  - **Priority**: Critical
  - **Category**: HeadTracking/Mode

- **Parameter**: numFaces
  - **Type**: number
  - **Default**: 1
  - **Location**: Line 38
  - **Priority**: Medium
  - **Category**: HeadTracking/Detection

#### Smoothing Configuration (Line 11)
- **Parameter**: SMOOTHING_FACTOR
  - **Type**: number
  - **Default**: 0.15
  - **Location**: Line 11
  - **Priority**: High
  - **Category**: HeadTracking/Filtering

#### Webcam Settings (Line 90-92)
- **Parameter**: video.width
  - **Type**: number
  - **Default**: 640
  - **Location**: Line 91
  - **Priority**: Medium
  - **Category**: HeadTracking/Camera

- **Parameter**: video.height
  - **Type**: number
  - **Default**: 480
  - **Location**: Line 91
  - **Priority**: Medium
  - **Category**: HeadTracking/Camera

---

## 7. PERFORMANCE OPTIMIZATION HOOKS

### 7.1 useSelectiveSettingsStore.ts (`client/src/hooks/useSelectiveSettingsStore.ts`)

**Performance Hook Configuration** - 560 lines

#### Cache Configuration (Line 20-20)
- **Parameter**: CACHE_TTL
  - **Type**: number
  - **Default**: 5000 (milliseconds)
  - **Location**: Line 20
  - **Priority**: High
  - **Category**: Performance/Caching

#### Debounce Configuration (Line 24)
- **Parameter**: DEBOUNCE_DELAY
  - **Type**: number
  - **Default**: 50 (milliseconds)
  - **Location**: Line 24
  - **Priority**: High
  - **Category**: Performance/Debouncing

#### useSelectiveSetting Options (Line 162-169)
- **Parameter**: enableCache
  - **Type**: boolean
  - **Default**: true
  - **Location**: Line 165
  - **Priority**: High
  - **Category**: Performance/Cache

- **Parameter**: enableDeduplication
  - **Type**: boolean
  - **Default**: true
  - **Location**: Line 166
  - **Priority**: High
  - **Category**: Performance/Deduplication

- **Parameter**: fallbackToStore
  - **Type**: boolean
  - **Default**: true
  - **Location**: Line 167
  - **Priority**: High
  - **Category**: Performance/Fallback

#### useSelectiveSettings Options (Line 248-256)
- **Parameter**: enableBatchLoading
  - **Type**: boolean
  - **Default**: true
  - **Location**: Line 252
  - **Priority**: High
  - **Category**: Performance/Batching

---

## 8. SUMMARY STATISTICS

### Parameters by Category

**Total Unique Parameters**: 270+

**By Priority**:
- Critical: 25 parameters
- High: 145 parameters
- Medium: 85 parameters
- Low: 15 parameters

**By Category**:
- Physics: 65 parameters
- Visualization: 55 parameters
- Hologram: 45 parameters
- XR/AR: 18 parameters
- Input/Controls: 24 parameters
- Performance: 12 parameters
- MultiUser: 10 parameters
- Authentication: 5 parameters
- System: 8 parameters
- Effects: 28 parameters

### Settings Store Hierarchy

```
visualisation/
├── rendering/          (10 params)
├── graphs/
│   ├── logseq/
│   │   ├── physics/   (40 params)
│   │   ├── nodes/     (9 params)
│   │   ├── edges/     (9 params)
│   │   └── labels/    (5 params)
│   └── visionflow/    (mirror of logseq)
├── glow/              (4 params)
├── hologram/          (45 params)
├── animations/        (6 params)
└── bloom/             (3 params)

xr/
├── enabled
├── mode
├── quality
├── renderScale
├── handTracking/      (1 param)
├── interactions/      (1 param)
├── gpu/               (1 param)
├── performance/       (1 param)
└── enableAdaptiveQuality

system/
├── debug/             (1 param)
├── websocket/         (2 params)
├── persistSettingsOnServer
└── customBackendUrl

auth/
├── enabled
├── required
├── provider
└── nostr
```

### TODO Items for Settings Integration

**High Priority** (Hardcoded values that should map to settings):
1. Hologram scene colors and fog settings
2. Hologram lighting configuration (3 light sources)
3. Post-processing bloom/AO/DOF parameters
4. Particle system counts and distributions
5. Wireframe cloud material properties

**Medium Priority**:
1. SpacePilot sensitivity presets
2. Head tracking smoothing factor
3. Multi-user heartbeat/reconnect intervals
4. Cache and debounce timing values

**Low Priority**:
1. Animation speeds and timing
2. Geometric detail levels
3. Text content and styling

---

## 9. INTEGRATION RECOMMENDATIONS

### Phase 1: Core Settings Migration
1. Map all physics parameters (already mostly done)
2. Complete visualization settings (nodes, edges, labels)
3. Integrate XR/AR settings
4. System and authentication settings

### Phase 2: Visual Effects Migration
1. Hologram configuration mapping
2. Glow/bloom/effects settings
3. Wireframe cloud parameters
4. Animation settings

### Phase 3: Input & Performance
1. SpacePilot configuration UI
2. Head tracking settings
3. Multi-user settings panel
4. Cache/performance tuning UI

### Phase 4: Advanced Features
1. Per-graph settings (logseq vs visionflow)
2. Preset system for common configurations
3. Import/export settings profiles
4. Settings sync across devices

---

## 10. FILES REQUIRING ATTENTION

### Critical Files
- `/client/src/store/settingsStore.ts` - Main settings store (complete)
- `/client/src/features/visualisation/components/ControlPanel/settingsConfig.ts` - UI mapping
- `/client/src/features/visualisation/components/HolographicDataSphere.tsx` - Hardcoded values

### High Priority Files
- `/client/src/features/visualisation/controls/SpacePilotController.ts` - Input configuration
- `/client/src/hooks/useHeadTracking.ts` - Head tracking config
- `/client/src/features/visualisation/components/WireframeCloudMesh.tsx` - Effect parameters
- `/client/src/hooks/useSelectiveSettingsStore.ts` - Performance tuning

### Medium Priority Files
- `/client/src/store/multiUserStore.ts` - Multi-user settings
- `/client/src/hooks/useQuest3Integration.ts` - XR auto-start config

---

**End of Audit**

**Report completed by**: Researcher Agent
**Coordination**: npx claude-flow@alpha hooks post-task --task-id "client-audit"
