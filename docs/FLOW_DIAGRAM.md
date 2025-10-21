# System Flow Diagrams

## 1. Application Startup Sequence

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RUST SERVER STARTUP                           │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Initialize Logging  │
                    │  env_logger::init()  │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Create AppState    │
                    │                      │
                    │  ┌───────────────┐   │
                    │  │ GraphStateActor│   │
                    │  │  - Nodes       │   │
                    │  │  - Edges       │   │
                    │  │  - Physics     │   │
                    │  └───────────────┘   │
                    │                      │
                    │  ┌───────────────┐   │
                    │  │SettingsActor  │   │
                    │  │  - DB conn    │   │
                    │  │  - Cache      │   │
                    │  └───────────────┘   │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Register Routes     │
                    │                      │
                    │  REST:               │
                    │  /graph/data         │
                    │  /graph/state        │
                    │  /api/settings       │
                    │                      │
                    │  WebSocket:          │
                    │  /ws/realtime        │
                    │  /ws/settings        │
                    │  /ws/graph/logseq    │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Start Server       │
                    │   0.0.0.0:3000       │
                    │   ✓ Ready            │
                    └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      CLIENT APP STARTUP                              │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    App.tsx Load      │
                    │    React Mount       │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  AppInitializer      │
                    └──────────────────────┘
                               │
                               ├──────────────────────────────────────┐
                               │                                      │
                               ▼                                      ▼
                    ┌──────────────────────┐            ┌──────────────────────┐
                    │ settingsStore.       │            │ graphDataManager.    │
                    │ loadSettings()       │            │ fetchInitialData()   │
                    │                      │            │                      │
                    │ GET /api/settings    │            │ GET /graph/data      │
                    │      │               │            │      │               │
                    │      ▼               │            │      ▼               │
                    │ Build tree           │            │ Process nodes        │
                    │ initialized=true     │            │ Send to worker       │
                    └──────────────────────┘            └──────────────────────┘
                               │                                      │
                               └──────────────┬───────────────────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ WebSocketService.    │
                                   │ connect()            │
                                   │                      │
                                   │ WS /ws/realtime      │
                                   │      │               │
                                   │      ▼               │
                                   │ Subscribe channels   │
                                   │ - graph              │
                                   │ - settings           │
                                   │ - realtime           │
                                   └──────────────────────┘
                                              │
                                              ▼
                                   ┌──────────────────────┐
                                   │ Render UI            │
                                   │ - MainLayout         │
                                   │ - GraphVisualization │
                                   │ - ControlPanel       │
                                   │                      │
                                   │ ✓ App Ready          │
                                   └──────────────────────┘
```

---

## 2. Settings Control Panel Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                  USER OPENS CONTROL PANEL                            │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  ControlPanel.tsx    │
                    │  Mounts Component    │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Render Tabs         │
                    │                      │
                    │  [Appearance]        │
                    │  [Effects]           │
                    │  [Physics] ← active  │
                    │  [Rendering]         │
                    │  [XR]                │
                    └──────────────────────┘
                               │
                               ▼ User clicks "Physics"
                    ┌──────────────────────┐
                    │  setActiveTab(...)   │
                    │  sectionId="physics" │
                    └──────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│            <SettingsTabContent sectionId="physics" />                │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────────┐
                    │  const sectionConfig =           │
                    │    SETTINGS_CONFIG[sectionId]    │
                    └──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
     ┌─────────────────────┐      ┌─────────────────────┐
     │  sectionConfig      │      │  sectionConfig      │
     │  = undefined        │      │  = { title, fields }│
     │                     │      │                     │
     │  ❌ INVALID ID      │      │  ✅ VALID ID        │
     └─────────────────────┘      └─────────────────────┘
                │                             │
                ▼                             ▼
     ┌─────────────────────┐      ┌─────────────────────┐
     │  return (           │      │  ensureLoaded(      │
     │    <div>            │      │    [paths...]       │
     │      No settings    │      │  )                  │
     │      available      │      │                     │
     │    </div>           │      │  GET /api/settings/ │
     │  )                  │      │  {path} (if needed) │
     └─────────────────────┘      └─────────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │  Render Fields      │
                                   │                     │
                                   │  for each field:    │
                                   │    - Toggle         │
                                   │    - Slider         │
                                   │    - Color Picker   │
                                   │                     │
                                   │  getValueFromPath() │
                                   │  settings.visual... │
                                   └─────────────────────┘
```

---

## 3. Settings Update Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│              USER TOGGLES "PHYSICS ENABLED" SETTING                  │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  onClick Handler         │
                    │  updateSettingByPath(    │
                    │    "visualisation.graphs │
                    │     .logseq.physics      │
                    │     .enabled",           │
                    │    false                 │
                    │  )                       │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  settingsStore.          │
                    │  updateSettings()        │
                    │                          │
                    │  Uses Immer to create    │
                    │  immutable update        │
                    └──────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
     ┌─────────────────────┐      ┌─────────────────────┐
     │  Update Local State │      │  Persist to Server  │
     │                     │      │                     │
     │  set({ settings })  │      │  PUT /api/settings/ │
     │                     │      │  visualisation.     │
     │  ✓ UI updates       │      │  graphs.logseq.     │
     │    immediately      │      │  physics.enabled    │
     └─────────────────────┘      │                     │
                                  │  { value: false }   │
                                  └─────────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │  Server Receives    │
                                   │                     │
                                   │  SettingsActor      │
                                   │  .send(SetSetting)  │
                                   └─────────────────────┘
                                              │
                                ┌─────────────┴─────────────┐
                                │                           │
                                ▼                           ▼
                     ┌─────────────────────┐     ┌─────────────────────┐
                     │  Update SQLite DB   │     │  Broadcast to       │
                     │                     │     │  Other Clients      │
                     │  INSERT OR REPLACE  │     │                     │
                     │  settings (...)     │     │  WebSocket:         │
                     │                     │     │  {                  │
                     │  ✓ Persisted        │     │    type: "settings  │
                     └─────────────────────┘     │          Delta",    │
                                                 │    path: "...",     │
                                                 │    value: false     │
                                                 │  }                  │
                                                 └─────────────────────┘
                                                            │
                                                            ▼
                                                 ┌─────────────────────┐
                                                 │  Other Clients      │
                                                 │  Update UI          │
                                                 │                     │
                                                 │  ✓ Synchronized     │
                                                 └─────────────────────┘
```

---

## 4. Binary WebSocket Position Updates

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHYSICS ENGINE (SERVER)                           │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ 60 FPS (16ms interval)
                    ┌──────────────────────────┐
                    │  GraphStateActor         │
                    │  .step_physics()         │
                    │                          │
                    │  for each node:          │
                    │    - Apply forces        │
                    │    - Update velocity     │
                    │    - Update position     │
                    │                          │
                    │  Check if settled:       │
                    │    KE < threshold?       │
                    └──────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
     ┌─────────────────────┐      ┌─────────────────────┐
     │  Still Moving       │      │  Settled            │
     │  KE > 0.001         │      │  KE < 0.001         │
     └─────────────────────┘      │                     │
                │                 │  ✓ Stop updates     │
                ▼                 └─────────────────────┘
     ┌─────────────────────┐
     │  Create Binary Msg  │
     │                     │
     │  Header (8 bytes):  │
     │  ┌───────────────┐  │
     │  │ Type: 0x01    │  │
     │  │ Flags: 0x01   │  │  ← Include velocity
     │  │ Count: 1000   │  │
     │  │ Timestamp     │  │
     │  └───────────────┘  │
     │                     │
     │  Node Data (×1000): │
     │  ┌───────────────┐  │
     │  │ ID: 1         │  │
     │  │ X: 45.23      │  │
     │  │ Y: -12.67     │  │
     │  │ Z: 8.91       │  │
     │  │ VX: 0.05      │  │
     │  │ VY: -0.02     │  │
     │  │ VZ: 0.01      │  │
     │  └───────────────┘  │
     │  ... (999 more)     │
     │                     │
     │  Total: 32,008 bytes│
     └─────────────────────┘
                │
                ▼
     ┌─────────────────────┐
     │  Optional: Compress │
     │  zlib (40% smaller) │
     │                     │
     │  12,803 bytes       │
     └─────────────────────┘
                │
                ▼
     ┌─────────────────────┐
     │  WebSocket Broadcast│
     │                     │
     │  for client in      │
     │    subscribed:      │
     │    send_binary(msg) │
     └─────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT                                      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  WebSocket.onmessage     │
                    │  (event.data)            │
                    │                          │
                    │  instanceof ArrayBuffer? │
                    │  ✓ Yes → Binary          │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  graphDataManager.       │
                    │  updateNodePositions()   │
                    │                          │
                    │  Parse binary:           │
                    │  - Read header           │
                    │  - Extract count         │
                    │  - For each node:        │
                    │    * Parse ID, x, y, z   │
                    │    * Parse vx, vy, vz    │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  graphWorkerProxy.       │
                    │  processBinaryData()     │
                    │                          │
                    │  Send to Web Worker:     │
                    │  postMessage(            │
                    │    ArrayBuffer           │
                    │  )                       │
                    └──────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       WEB WORKER                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  Update Position Buffer  │
                    │                          │
                    │  positionBuffer[i*3+0]=x │
                    │  positionBuffer[i*3+1]=y │
                    │  positionBuffer[i*3+2]=z │
                    │                          │
                    │  Float32Array            │
                    │  (3 × 1000 × 4 bytes)    │
                    │  = 12 KB                 │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  postMessage back to     │
                    │  main thread             │
                    │                          │
                    │  {                       │
                    │    type: 'positionUpdate'│
                    │    data: Float32Array    │
                    │  }                       │
                    └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      RENDERING (MAIN THREAD)                         │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  GraphVisualization      │
                    │  onPositionUpdate()      │
                    │                          │
                    │  for i in 0..nodeCount:  │
                    │    x = positions[i*3+0]  │
                    │    y = positions[i*3+1]  │
                    │    z = positions[i*3+2]  │
                    │                          │
                    │    matrix.makeTranslation│
                    │      (x, y, z)           │
                    │                          │
                    │    instancedMesh.        │
                    │      setMatrixAt(i, m)   │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  instancedMesh.          │
                    │  instanceMatrix.         │
                    │  needsUpdate = true      │
                    │                          │
                    │  ✓ GPU updates on        │
                    │    next frame            │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  requestAnimationFrame   │
                    │                          │
                    │  renderer.render(        │
                    │    scene, camera         │
                    │  )                       │
                    │                          │
                    │  ✓ 60 FPS smooth         │
                    └──────────────────────────┘
```

---

## 5. Error Case: Invalid Section ID

```
┌─────────────────────────────────────────────────────────────────────┐
│              BUG: INCORRECT SECTION ID                               │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────┐
                    │  Parent Component        │
                    │                          │
                    │  const tabs = [          │
                    │    {                     │
                    │      id: "visualisation",│  ← ❌ TYPO!
                    │      label: "Appearance" │     Not in SETTINGS_CONFIG
                    │    },                    │
                    │    {                     │
                    │      id: "physics",      │  ← ✅ OK
                    │      label: "Physics"    │
                    │    }                     │
                    │  ]                       │
                    └──────────────────────────┘
                               │
                               ▼ User clicks "Appearance"
                    ┌──────────────────────────┐
                    │  setActiveSection(       │
                    │    "visualisation"       │
                    │  )                       │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  <SettingsTabContent     │
                    │    sectionId=            │
                    │      "visualisation"     │
                    │  />                      │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  SETTINGS_CONFIG lookup: │
                    │                          │
                    │  const sectionConfig =   │
                    │    SETTINGS_CONFIG[      │
                    │      "visualisation"     │
                    │    ]                     │
                    │                          │
                    │  Result: undefined       │  ← ❌
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  if (!sectionConfig) {   │
                    │    return (              │
                    │      <div>               │
                    │        No settings       │
                    │        available for     │
                    │        this section      │
                    │      </div>              │
                    │    )                     │
                    │  }                       │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  USER SEES ERROR:        │
                    │                          │
                    │  ╔══════════════════════╗│
                    │  ║  No settings         ││
                    │  ║  available for       ││
                    │  ║  this section        ││
                    │  ╚══════════════════════╝│
                    └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         THE FIX                                      │
└─────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────┐
                    │  Change tab config:      │
                    │                          │
                    │  const tabs = [          │
                    │    {                     │
                    │      id: "appearance",   │  ← ✅ FIXED!
                    │      label: "Appearance" │     Valid section ID
                    │    },                    │
                    │    {                     │
                    │      id: "physics",      │  ← ✅ OK
                    │      label: "Physics"    │
                    │    }                     │
                    │  ]                       │
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  SETTINGS_CONFIG[        │
                    │    "appearance"          │
                    │  ]                       │
                    │                          │
                    │  Result: {               │
                    │    title: "Appearance",  │
                    │    fields: [             │
                    │      {                   │
                    │        key: "nodeColor", │
                    │        path: "...",      │
                    │        type: "color"     │
                    │      },                  │
                    │      ...                 │
                    │    ]                     │
                    │  }                       │  ← ✅ Found!
                    └──────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │  Render all fields       │
                    │                          │
                    │  ✓ Settings panel works! │
                    └──────────────────────────┘
```

---

**Last Updated**: 2025-10-21
