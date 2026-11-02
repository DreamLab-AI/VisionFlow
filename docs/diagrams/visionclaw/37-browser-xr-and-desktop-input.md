---
id: VC-37
title: Browser XR surface and desktop spatial input
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/XR-client.md
adrs: [ADR-2032, ADR-2081]
sources:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/adr/ADR-2081-remove-dead-browser-xr-surface.md
  - ../project/client/src/services/platformManager.ts
  - ../project/client/src/contexts/ApplicationModeContext.tsx
  - ../project/client/src/services/SpaceDriverService.ts
  - ../project/client/src/features/visualisation/hooks/useSpacePilot.ts
  - ../project/client/src/features/control-center/status/useSpacePilot.ts
  - ../project/client/src/features/visualisation/controls/SpacePilotController.ts
  - ../project/client/src/features/visualisation/components/SpacePilotSimpleIntegration.tsx
  - ../project/client/src/features/settings/config/settings.ts
  - ../project/client/src/hooks/useHeadTracking.ts
  - ../project/client/src/features/visualisation/components/HeadTrackedParallaxController.tsx
  - ../project/client/src/features/graph/components/GraphCanvas.tsx
  - ../project/client/src/features/graph/hooks/useCameraAutoFit.ts
  - ../project/client/src/features/graph/components/GraphManager.tsx
  - ../project/client/src/features/graph/components/InstancedLabels.tsx
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/client/src/features/visualisation/components/EmbeddingCloudLayer.tsx
  - ../project/client/vite.config.ts
  - ../project/client/package.json
  - ../project/xr-client/project.godot
  - ../project/client/src/app/App.tsx
  - ../project/client/src/services/remoteLogger.ts
  - ../project/client/src/features/graph/contexts/NodePositionContext.tsx
  - ../project/client/src/features/graph/hooks/useGraphEventHandlers.ts
  - ../project/client/src/features/graph/hooks/dragHeartbeat.ts
  - ../project/client/src/features/graph/hooks/__tests__/dragHeartbeat.test.ts
  - ../project/src/handlers/socket_flow_handler/types.rs
verified_commit: {visionclaw: 86cdd76355823a51bb64d503c005fe277726e724}
---

Re-stamped at `86cdd7635` (2026-11-02, estate audit wave 3, [report](../../estate-review/2026-11-02-w3-vc-k.md)). The re-read found that two modules this topic drew as live are imported by nothing at this pin: the platform store with its WebXR capability probe, and the `useSpacePilot` hook with `SpacePilotController`. The mounted SpaceMouse path is `SpacePilotSimpleIntegration`, and the agent overlay that read the shared node positions is retired. VC-37.1 to VC-37.10 are redrawn to match. VC-37.11 adds the desktop drag-hold heartbeat (ADR-2169). The whole hold lifecycle across the server is [VC-52.6](52-drag-holds-and-observation-gates.md).

## VC-37.1 Browser XR capability probe — what platformManager would do, and that nothing calls it

```mermaid
sequenceDiagram
    autonumber
    participant NB as no importer at this pin
    participant PM as usePlatformStore.initialize<br/>client/src/services/platformManager.ts:85
    participant DP as detectPlatform<br/>client/src/services/platformManager.ts:154
    participant NX as navigator.xr (WebXR Device API)
    participant W as window

    Note over NB,PM: UNREACHABLE: no file under client/ imports platformManager.ts,<br/>so neither initialize() nor the PlatformManager adapter that wraps it<br/>(platformManager.ts:348-349, singleton :394) ever runs in the bundle
    NB-->>PM: initialize() - never called
    PM->>DP: detectPlatform() platformManager.ts:89
    DP->>DP: read navigator.userAgent
    alt UA contains "Quest 3"
        DP-->>DP: platform=quest3, xrDeviceType=quest
    else UA contains "Quest 2"
        DP-->>DP: platform=quest2, xrDeviceType=quest
    else UA contains "Quest"
        DP-->>DP: platform=quest, xrDeviceType=quest
    else UA contains "Pico" or "PICO"
        DP-->>DP: platform=pico, xrDeviceType=pico
    else Android/webOS/iPhone/iPad/iPod/BlackBerry/IEMobile/Opera Mini
        DP-->>DP: platform=mobile, xrDeviceType=mobile-xr
    else
        DP-->>DP: platform=desktop, xrDeviceType=desktop-xr
    end
    Note over DP: platformManager.ts:160-184. performanceTier low/medium/high<br/>is derived per branch at platformManager.ts:187-222
    alt navigator.xr present
        PM->>NX: isSessionSupported('immersive-vr')
        NX-->>PM: vrSupported
        PM->>NX: isSessionSupported('immersive-ar')
        NX-->>PM: arSupported
        PM->>PM: capabilities.xrSupported = vrSupported or arSupported
        Note over PM: platformManager.ts:92-113
    else navigator.xr absent
        PM-->>PM: capabilities untouched, isWebXRSupported stays false
    end
    opt navigator.xr present
        PM->>PM: capabilities.handTrackingSupported = isQuest()
        Note over PM: DIVERGENCE hand-tracking support is INFERRED from the<br/>user-agent, never queried. No WebXR feature descriptor is<br/>requested. platformManager.ts:116-130, isQuest() :254-257
    end
    PM->>W: addEventListener('resize', detectPlatform) platformManager.ts:133-137
    PM->>PM: isWebXRSupported = !!navigator.xr - initialized = true
    Note over PM: platformManager.ts:140-145
    Note over NB,NX: ADR-2081 deleted the XR session shell and kept this probe because<br/>remoteLogger reports it (ADR-2081-remove-dead-browser-xr-surface.md:36-39).<br/>At this pin remoteLogger runs its own probe and is itself imported by nothing - see VC-37.2.<br/>The immersive client is the Godot + OpenXR app - see VC-36.
```

**What it shows.** The probe's logic, which is unchanged, and the fact that matters more: the module that holds it has no importer, so in the running browser client none of this executes. **Why it is this way.** ADR-2081 removed the XR session state and kept the capability probe as telemetry input; the code that fed telemetry was never wired to this store.

**Drift:** ADR-2081 keeps the probe "because `remoteLogger` reports those as telemetry" (docs/adr/ADR-2081-remove-dead-browser-xr-surface.md:36-39), but `remoteLogger` computes `webXRSupported` from `navigator` itself (client/src/services/remoteLogger.ts:273) and calls `isSessionSupported` on its own (client/src/services/remoteLogger.ts:281-290); it never reads the platform store. No file under `client/` imports `platformManager.ts`, so the retained probe is unreachable at this pin.

**Debt:** the whole platform store, its compatibility adapter and its singleton (client/src/services/platformManager.ts:348-349, client/src/services/platformManager.ts:394) are dead code by import graph; deleting them, or wiring them where telemetry actually reads, would settle the ADR-2081 rationale.

## VC-37.2 The XR-mode surface is gone, and the retained probe is unreachable

```mermaid
flowchart TB
    subgraph kept["Retained in source - but platformManager.ts has no importer"]
        K1["isWebXRSupported = !!navigator.xr<br/>platformManager.ts:79, :140-144, getter :364"]
        K2["capabilities.xrSupported / vrSupported / arSupported<br/>from isSessionSupported platformManager.ts:95-97"]
        K3["capabilities.handTrackingSupported<br/>platformManager.ts:13, :119-124"]
        K4["xrDeviceType quest | pico | desktop-xr | mobile-xr | none<br/>platformManager.ts:8, :34, :157"]
    end
    subgraph tel["remoteLogger - its own probe, also imported by nothing"]
        T1["logXRInfo: webXRSupported = 'xr' in navigator<br/>client/src/services/remoteLogger.ts:271, :273"]
        T2["its own isSessionSupported vr and ar calls<br/>client/src/services/remoteLogger.ts:281-290"]
        T3["scheduled 1 s after module load<br/>client/src/services/remoteLogger.ts:319-324"]
    end
    subgraph gone["Deleted by ADR-2081 - modelled being IN a session"]
        G1["XRSessionState type"]
        G2["isXRMode and xrSessionState store fields"]
        G3["setXRMode and setXRSessionState"]
        G4["xrmodechange and xrsessionstatechange events"]
        G5["ApplicationMode 'xr' member and its layout branch"]
        G6["isXRMode prop chain GraphManager to InstancedLabels"]
        G7["buildLabelLines vrMode parameter"]
        G8["@react-three/xr dependency"]
    end
    K1 -.->|"never read by"| T1
    K2 -.->|"never read by"| T2
    T3 --> T1
    K3 --> K4
    gone --> R1
    R1["CONFIRMED at this pin: no isXRMode, XRSessionState, setXRMode,<br/>xrmodechange, vrMode, useApplicationMode consumer or @react-three/xr<br/>remains under client/src or in client/package.json"]
    kept --> R2
    tel --> R2
    R2["DRIFT: ADR-2081 kept the store probe because remoteLogger reports it<br/>ADR-2081-remove-dead-browser-xr-surface.md:36-39.<br/>remoteLogger never reads the store, and neither module is imported,<br/>so no capability probe and no XR telemetry runs in the browser bundle."]
```

**What it shows.** The deleted session surface stays deleted. The retained half is not wired: the store and the telemetry module are two separate probes, the edge between them that ADR-2081 assumed does not exist (the dotted never-read edges), and nothing imports either module. **Why it is this way.** The deletion was checked by grepping for the removed names; nobody checked that the kept module had a caller.

**Debt:** `remoteLogger.ts` schedules `logXRInfo` at module load (client/src/services/remoteLogger.ts:319-324), but no file under `client/` imports it, so the XR telemetry it describes is never sent from this build. Whether that is intended is not recorded.

## VC-37.3 ApplicationMode transitions and the mobile guard

```mermaid
stateDiagram-v2
    [*] --> desktop
    desktop --> mobile: resize and innerWidth below 768
    mobile --> desktop: resize and not mobile
    note right of mobile
        ApplicationMode is desktop | mobile only
        client/src/contexts/ApplicationModeContext.tsx:6
        resize handler and the 768 px test
        ApplicationModeContext.tsx:46-66, :48
        it also runs once on mount, so a narrow window starts mobile
    end note
    note left of desktop
        useState default 'desktop' ApplicationModeContext.tsx:42
        isMobileView tracked separately ApplicationModeContext.tsx:44
        layoutSettings identical for every mode, memoised on mode
        ApplicationModeContext.tsx:74-97
        The provider is mounted at client/src/app/App.tsx:171
        and useApplicationMode has no consumer ApplicationModeContext.tsx:114
    end note
```

**What it shows.** Two states and a width test. **Why it is this way.** ADR-2081 removed the `'xr'` member; what remains is a responsive-width switch.

**Debt:** the provider is mounted (client/src/app/App.tsx:171) but `useApplicationMode` (client/src/contexts/ApplicationModeContext.tsx:114) has no consumer at this pin, and every branch of `getLayoutSettings` returns the same three `true` flags (client/src/contexts/ApplicationModeContext.tsx:74-95), so the mode and its layout settings are computed on every resize and read by nothing.

## VC-37.4 SpaceMouse / SpacePilot — WebHID acquisition and secure-context gate

```mermaid
sequenceDiagram
    autonumber
    participant MOD as module load of SpaceDriverService
    participant IN as SpaceDriverService.initialize<br/>client/src/services/SpaceDriverService.ts:63
    participant HID as navigator.hid (WebHID)
    participant DV as HIDDevice
    participant ST as Status surface hook<br/>client/src/features/control-center/status/useSpacePilot.ts:33
    participant UI as SpacePilotSimpleIntegration<br/>mounted at GraphCanvas.tsx:500

    MOD->>IN: SpaceDriver.initialize() on import, once (SpaceDriverService.ts:318, :320-323)
    IN->>IN: return if isInitialized (SpaceDriverService.ts:64)
    IN->>HID: feature-detect navigator.hid (SpaceDriverService.ts:68)
    alt navigator.hid absent
        IN-->>IN: warn 4 causes - not HTTPS/localhost, browser lacks WebHID,<br/>flag disabled, insecure context (SpaceDriverService.ts:69-73)
        opt window.isSecureContext === false
            IN-->>IN: warn remediation - use http://localhost:3000, use HTTPS,<br/>or chrome://flags insecure-origins-treated-as-secure
            Note right of IN: SpaceDriverService.ts:76-84
        end
        IN-->>IN: dispatchEvent('webhid-unavailable', {isSecureContext, hostname, protocol})
        Note right of IN: SpaceDriverService.ts:87-93 - returns without setting isInitialized
    else navigator.hid present
        IN->>HID: getDevices() (SpaceDriverService.ts:100)
        HID-->>IN: already-paired devices
        IN->>IN: filter on SUPPORTED_VENDOR_IDS (SpaceDriverService.ts:101)
        Note over IN: VENDOR_ID_LOGITECH 0x046d SpaceDriverService.ts:11<br/>VENDOR_ID_3DCONNEXION 0x256f SpaceDriverService.ts:12<br/>SUPPORTED_VENDOR_IDS SpaceDriverService.ts:13
        opt at least one supported paired device
            IN->>DV: openDevice(first candidate) (SpaceDriverService.ts:109-113)
            Note right of IN: DEBT the loop means to try each device until one opens<br/>(SpaceDriverService.ts:107), but openDevice swallows its own<br/>failure (SpaceDriverService.ts:178-180), so opened=true after<br/>the FIRST candidate whatever happened (SpaceDriverService.ts:115-116)
            IN->>DV: addEventListener('inputreport', handleInputReport) (SpaceDriverService.ts:170)
        end
        IN->>HID: addEventListener disconnect and connect (SpaceDriverService.ts:129, :132-140)
        IN->>IN: isInitialized = true (SpaceDriverService.ts:142)
    end
    ST->>IN: connect button calls SpaceDriver.scan() (client/src/features/control-center/status/useSpacePilot.ts:82-88)
    IN->>HID: requestDevice(REQUEST_PARAMS) (SpaceDriverService.ts:289-291)
    Note over IN,HID: DEVICE_FILTERS derived from the vendor ids<br/>SpaceDriverService.ts:14, REQUEST_PARAMS :15
    alt user picks nothing
        HID-->>IN: [] - return false (SpaceDriverService.ts:293-296)
    else device chosen
        IN->>DV: openDevice(devices[0]) (SpaceDriverService.ts:298)
    end
    IN-->>ST: 'connect' event, status chip shows the device (client/src/features/control-center/status/useSpacePilot.ts:54-57)
    IN-->>UI: 'connect' event sets isConnected (SpacePilotSimpleIntegration.tsx:95-97)
    Note over UI: TENSION the status hook seeds from SpaceDriver.isConnected() when it subscribes<br/>(client/src/features/control-center/status/useSpacePilot.ts:73) - the camera integration does not,<br/>so a device opened by the import-time initialize before the canvas mounts<br/>leaves isConnected false and useFrame returns early (SpacePilotSimpleIntegration.tsx:40, :166)
```

**What it shows.** Who actually acquires the device at this pin: the service initialises itself when first imported, the Status surface's hook offers the pairing button, and the camera integration mounted in `GraphCanvas` only listens. **Why it is this way.** WebHID needs a secure context and a user gesture for `requestDevice`; already-paired devices can be opened silently through `getDevices`.

**Debt:** the "try each paired device until one opens" loop stops at the first candidate, because `openDevice` catches and logs its own error instead of rethrowing (client/src/services/SpaceDriverService.ts:107-120, client/src/services/SpaceDriverService.ts:178-180); with two paired devices and the first unplugged, the second is never tried.

**Tension:** the Status hook seeds its state from `SpaceDriver.isConnected()` (client/src/features/control-center/status/useSpacePilot.ts:73) but `SpacePilotSimpleIntegration` starts with `isConnected` false and only flips it on a later `connect` event (client/src/features/visualisation/components/SpacePilotSimpleIntegration.tsx:40, client/src/features/visualisation/components/SpacePilotSimpleIntegration.tsx:95-97). Whether the import-time open beats the canvas mount is a runtime race this source trace cannot settle.

## VC-37.5 SpacePilot HID report decode and the mounted camera drive

```mermaid
sequenceDiagram
    autonumber
    participant DV as HIDDevice inputreport
    participant HR as handleInputReport<br/>client/src/services/SpaceDriverService.ts:219
    participant HT as handleTranslation<br/>client/src/services/SpaceDriverService.ts:249
    participant HRo as handleRotation<br/>client/src/services/SpaceDriverService.ts:260
    participant HB as handleButtons<br/>client/src/services/SpaceDriverService.ts:272
    participant SI as SpacePilotSimpleIntegration listeners<br/>client/src/features/visualisation/components/SpacePilotSimpleIntegration.tsx:150
    participant CAM as three camera via useThree<br/>SpacePilotSimpleIntegration.tsx:12

    DV-->>HR: inputreport event
    HR->>HR: values = new Int16Array(evt.data.buffer)
    Note over HR: log on reports 1-3 then every 1000th<br/>SpaceDriverService.ts:220-223
    alt reportId === REPORT_ID_TRANSLATION (1)
        alt values.length >= 6
            HR->>HT: handleTranslation(values)
            HR->>HRo: handleRotation(Int16Array[values3, values4, values5])
            Note right of HR: SpaceNavigator USB packs all 6 axes into report 1<br/>(12 bytes). SpaceDriverService.ts:228-231
        else 6 bytes
            HR->>HT: handleTranslation(values)
            Note right of HR: SpacePilot sends translation only in report 1<br/>SpaceDriverService.ts:233-234
        end
    else reportId === REPORT_ID_ROTATION (2)
        HR->>HRo: handleRotation(values)
    else reportId === REPORT_ID_BUTTONS (3)
        HR->>HB: handleButtons(new Uint16Array(buffer)[0])
    else unknown
        HR-->>HR: warn 'Unknown report ID' - BREAK
        Note right of HR: SpaceDriverService.ts:243-244
    end
    Note over HR: REPORT_ID_TRANSLATION=1, REPORT_ID_ROTATION=2,<br/>REPORT_ID_BUTTONS=3 SpaceDriverService.ts:18-20
    HT->>SI: CustomEvent 'translate' {x: v0, y: v1, z: v2}
    HRo->>SI: CustomEvent 'rotate' {rx: -v0, ry: -v1, rz: v2}
    Note over HRo: INVARIANT rx and ry are NEGATED, rz is not -<br/>SpaceDriverService.ts:262-266
    HB->>HB: walk 16 bits of buttonBits
    HB->>SI: CustomEvent 'buttons' {buttons: ['[1]','[A]', ...]}
    Note over HB: label = (i+1).toString(16).toUpperCase() so button 10<br/>reads [A]. SpaceDriverService.ts:273-284
    SI->>SI: scale by 1/90, zero below deadzone 0.02 per axis<br/>SpacePilotSimpleIntegration.tsx:112-136
    Note over SI: listeners for connect, disconnect, translate, rotate, buttons<br/>SpacePilotSimpleIntegration.tsx:150-154, torn down at :156-162
    loop useFrame while isConnected
        SI->>SI: exponential smoothing 0.85 SpacePilotSimpleIntegration.tsx:169-177
        SI->>SI: active motion disables OrbitControls, re-enabled only on the next<br/>mouse or wheel event SpacePilotSimpleIntegration.tsx:190-206, :76-92
        SI->>CAM: camera.position and camera.quaternion moved in camera space<br/>SpacePilotSimpleIntegration.tsx:209-225
    end
    Note over SI,CAM: button [1] re-targets OrbitControls at the origin SpacePilotSimpleIntegration.tsx:138-148.<br/>The tuning is a hard-coded literal SpacePilotSimpleIntegration.tsx:29-38 - see VC-37.10
```

**What it shows.** The HID decode is unchanged and sound; what consumes it at this pin is `SpacePilotSimpleIntegration`, mounted inside the canvas (client/src/features/graph/components/GraphCanvas.tsx:500), which drives the camera directly with fixed tuning. **Why it is this way.** The integration hands off with OrbitControls lazily so releasing the puck does not snap the view level (client/src/features/visualisation/components/SpacePilotSimpleIntegration.tsx:61-71).

**Debt:** `client/src/features/visualisation/hooks/useSpacePilot.ts` (the hook with camera, object and navigation modes, client/src/features/visualisation/hooks/useSpacePilot.ts:39) has no importer at this pin, and `SpacePilotController` is imported only by that hook (client/src/features/visualisation/hooks/useSpacePilot.ts:4), so the controller has no production importer. The earlier drawing of this panel routed every event through them.

## VC-37.6 Head-tracked parallax — MediaPipe face landmarks to camera frustum

```mermaid
sequenceDiagram
    autonumber
    participant C as HeadTrackedParallaxController<br/>client/src/features/visualisation/components/HeadTrackedParallaxController.tsx:13
    participant H as useHeadTracking<br/>client/src/hooks/useHeadTracking.ts:13
    participant FR as FilesetResolver.forVisionTasks<br/>client/src/hooks/useHeadTracking.ts:27
    participant FL as FaceLandmarker<br/>@mediapipe/tasks-vision
    participant V as HTMLVideoElement webcam
    participant CAM as three PerspectiveCamera via useThree<br/>HeadTrackedParallaxController.tsx:14

    Note over C: mounted unconditionally in the canvas GraphCanvas.tsx:503,<br/>tracking starts only when settings enable it HeadTrackedParallaxController.tsx:21-23
    C->>H: useHeadTracking()
    H->>H: initialize() - skip if faceLandmarker already set
    H->>FR: forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm')
    H->>FL: createFromOptions({modelAssetPath '/models/face_landmarker.task', delegate 'GPU',<br/>runningMode 'VIDEO', numFaces 1, no blendshapes, no transform matrices})
    Note over H,FL: useHeadTracking.ts:30-39
    alt init throws
        H-->>C: error 'Failed to load head tracking model. Please check your network connection.'
        Note right of H: useHeadTracking.ts:41-44 - isTracking stays false
    end
    loop requestAnimationFrame predictWebcam
        H->>V: read currentTime
        alt no video, no landmarker, or no srcObject
            H-->>H: cancelAnimationFrame - BREAK
            Note right of H: useHeadTracking.ts:47-50
        end
        opt currentTime !== lastVideoTime
            H->>FL: detectForVideo(video, Date.now())
            FL-->>H: faceLandmarks
            opt at least one face
                H->>H: take nose tip landmark index 1
                H->>H: lerp into smoothedPosition
                Note right of H: SMOOTHING_FACTOR = 0.15 useHeadTracking.ts:11<br/>smoothedPosition is a THREE.Vector2 ref useHeadTracking.ts:21
            end
        end
    end
    H-->>C: {isEnabled, setIsEnabled, isTracking, headPosition, error}
    C->>C: cameraMode from settings.visualisation.interaction.headTrackedParallax.cameraMode<br/>default 'asymmetricFrustum' HeadTrackedParallaxController.tsx:19
    loop useFrame
        alt isTracking and headPosition and camera is PerspectiveCamera
            alt cameraMode === 'asymmetricFrustum'
                C->>CAM: setViewOffset(...) then updateProjectionMatrix()
                Note right of C: HeadTrackedParallaxController.tsx:46-54
            else
                C->>CAM: projectionMatrix.multiply(NUDGE_MATRIX.makeTranslation(offsetX, offsetY, 0))
                Note right of C: HeadTrackedParallaxController.tsx:57-62 - the module-level<br/>NUDGE_MATRIX :11 replaced a per-frame Matrix4 plus Vector3 allocation (:60-61)
            end
        else not tracking
            opt camera.view set
                C->>CAM: clearViewOffset() then updateProjectionMatrix()
                Note right of C: restores the plain frustum<br/>HeadTrackedParallaxController.tsx:66-68
            end
        end
    end
    Note over C,CAM: This is the desktop stand-in for stereo parallax. It shares the<br/>R3F camera with the graph render path - see VC-31 - and touches<br/>no SharedArrayBuffer of its own.
```

**What it shows.** The webcam path, unchanged since the previous stamp. **Why it is this way.** An off-axis frustum gives a window-like parallax on a flat screen without stereo hardware.

**Tension:** the non-frustum mode multiplies a translation into `camera.projectionMatrix` on every tracked frame without first rebuilding it (client/src/features/visualisation/components/HeadTrackedParallaxController.tsx:62); the only other client callers of `updateProjectionMatrix` run when the canvas size changes (client/src/features/graph/components/GraphCanvas.tsx:181-191) and on an event-driven camera fit, once on first data, on an explicit request or when the memory cloud changes (client/src/features/graph/hooks/useCameraAutoFit.ts:66, client/src/features/graph/hooks/useCameraAutoFit.ts:146-160), so between those events the offsets would compound frame on frame, and stopping tracking in that mode does not restore the matrix because the reset branch runs only when `camera.view` is set (client/src/features/visualisation/components/HeadTrackedParallaxController.tsx:66-68). Not observed at runtime.

## VC-37.7 How the desktop spatial inputs share state with the R3F path

```mermaid
flowchart LR
    subgraph inputs["Desktop spatial inputs"]
        SP["SpaceDriver WebHID singleton<br/>client/src/services/SpaceDriverService.ts:318"]
        HT["useHeadTracking MediaPipe<br/>client/src/hooks/useHeadTracking.ts:13"]
    end
    subgraph r3f["Shared R3F context (useThree)"]
        CAM["PerspectiveCamera"]
    end
    subgraph graph["Graph render path (VC-31)"]
        SAB["worker SharedArrayBuffer positions<br/>getPositionsSync graphWorkerProxy.ts:301"]
        GMx["GraphManager.tsx useFrame copies the buffer once per frame<br/>into the drawn array GraphManager.tsx:409, :420-424"]
        ILx["InstancedLabels reads the drawn array through nodePositionsRef<br/>GraphManager.tsx:719-722"]
        NPC["NodePositionContext module singleton<br/>sharedNodePositions and sharedNodeIdToIndexMap<br/>client/src/features/graph/contexts/NodePositionContext.tsx:8, :11"]
        ECL["EmbeddingCloudLayer reads both to frame the memory cloud<br/>EmbeddingCloudLayer.tsx:49, :85-90"]
    end
    SP --> SIx["SpacePilotSimpleIntegration useThree camera<br/>SpacePilotSimpleIntegration.tsx:12, mounted GraphCanvas.tsx:500"] --> CAM
    HT --> HPCx["HeadTrackedParallaxController useThree camera, size<br/>HeadTrackedParallaxController.tsx:14, mounted GraphCanvas.tsx:503"] --> CAM
    SAB --> GMx
    CAM --> GMx
    GMx --> ILx
    GMx --> NPC
    NPC --> ECL
    ILx --> NOTE1
    NOTE1["RESOLVED ADR-2081: the isXRMode prop chain from the platform store<br/>through GraphManager into InstancedLabels is deleted, along with the<br/>constant-false vrMode parameter of buildLabelLines. Label layout takes<br/>no XR input."]
    inputs --> NOTE2
    NOTE2["INVARIANT both spatial inputs mutate the SAME R3F camera and<br/>write nothing into the position buffers. Position data flows<br/>one way - websocket to worker buffer to drawn copy to renderer<br/>(VC-32, VC-31) - so an input device cannot desynchronise graph state."]
    NPC --> NOTE3
    NOTE3["DEBT: NodePositionContext is named a context but is a module-level<br/>mutable singleton, not a React context. GraphManager publishes the drawn<br/>copy on every frame (GraphManager.tsx:424) and the id map on every<br/>node-set change (GraphManager.tsx:81), and the memory cloud layer reads<br/>them directly. Nothing clears them on unmount, so a reader after a<br/>remount sees the last frame. NodePositionContext.tsx:13-19"]
```

**What it shows.** Who writes the camera and who shares the node positions at this pin. The agent overlay that used to read the singleton (`BotsVisualization.tsx`) was retired in VisionClaw `923f9ae83`; its one reader now is the memory cloud layer, which takes the graph's bounds from it. **Why it is this way.** The drawn array is a per-frame copy so drag and layout transitions write the copy, never the buffer the worker rewrites (client/src/features/graph/components/GraphManager.tsx:420-421).

## VC-37.8 Browser path versus the shipped Godot path

```mermaid
flowchart TB
    subgraph br["Browser client (client/)"]
        B1["Three.js / R3F, WebGL2 or WebGPU<br/>see VC-31"]
        B2["no capability probe in the bundle - platformManager.ts<br/>and remoteLogger.ts are imported by nothing, see VC-37.1"]
        B3["No requestSession, no XRButton, no XR session state<br/>(RESOLVED ADR-2081 - the shell is deleted)"]
        B4["Spatial input = WebHID SpaceMouse via SpacePilotSimpleIntegration<br/>+ MediaPipe head parallax"]
        B5["Binary V3/V5 over /wss to SAB<br/>see VC-32"]
    end
    subgraph gd["Godot XR client (xr-client/)"]
        G1["Godot 4.6.1 + gdext, OpenXR<br/>gl_compatibility on desktop project.godot:56"]
        G1M["TENSION: rendering_method.mobile is 'mobile', so an Android or Quest export<br/>would take the Mobile renderer, not Compatibility, unless overridden<br/>project.godot:57"]
        G1D["DEBT: config/features metadata is still pinned at 4.3 while the verified runtime is 4.6.1,<br/>to be re-pinned in the editor with the Quest renderer decision<br/>project.godot:3-7"]
        G1 -.-> G1D
        G1 -.-> G1M
        G2["Real stereo submission on SteamVR<br/>ADR-2032"]
        G3["OpenXR action map, 13 interaction profiles<br/>xr-client/openxr_action_map.tres"]
        G4["Hand ray + pinch, head/eye gaze<br/>see VC-36"]
        G5["Same binary V3/V5 wire, decoded in Rust<br/>see VC-36.5"]
    end
    B5 --- SHARED["Shared server surface: /wss binary position stream,<br/>/api REST. Identical V3 52-byte record and V5 envelope."]
    G5 --- SHARED
    B2 --> DIV
    G2 --> DIV
    DIV["RESOLVED ADR-2081: docs/BASELINE-architecture.md:271-272 describes the<br/>React client as 'consuming the binary WebSocket position stream and the<br/>/api REST surface' - which is what the code is. The XR-mode<br/>shell and the unimported @react-three/xr dependency are deleted, so no<br/>reader can mistake the browser client for an immersive one. The immersive<br/>client is the Godot app alone. Babylon.js was already removed<br/>(client/vite.config.ts:47)."]
```

**What it shows.** The two clients side by side: the browser one draws and accepts desktop spatial input, the Godot one is the only immersive client. **Why it is this way.** ADR-2032 put immersive XR in Godot; ADR-2081 removed the browser's unreachable XR shell.

**Tension:** the project forces `gl_compatibility` (xr-client/project.godot:56) but sets `rendering_method.mobile="mobile"` beside it (xr-client/project.godot:57), so the Quest renderer is not the one the desktop runtime verified; the header defers that decision to the editor re-pin (xr-client/project.godot:3-7).

## VC-37.9 SpaceDriver hot-plug — disconnect and reconnect lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant NAV as navigator.hid
    participant SD as SpaceDriver singleton<br/>client/src/services/SpaceDriverService.ts:318
    participant DV as open HIDDevice
    participant SI as SpacePilotSimpleIntegration<br/>client/src/features/visualisation/components/SpacePilotSimpleIntegration.tsx:150
    participant ST as Status surface hook<br/>client/src/features/control-center/status/useSpacePilot.ts:68

    Note over SD,NAV: handleDisconnect bound in the constructor SpaceDriverService.ts:59,<br/>both listeners registered by initialize() SpaceDriverService.ts:129, :132
    NAV-->>SD: 'disconnect' event (unplug, power off)
    SD->>SD: handleDisconnect(evt)<br/>SpaceDriverService.ts:210
    alt evt.device === this.device
        SD->>SD: disconnect()
        SD->>DV: removeEventListener('inputreport') then close()<br/>SpaceDriverService.ts:187-189
        SD-->>SI: dispatchEvent(new Event('disconnect'))<br/>SpaceDriverService.ts:195
        SI->>SI: isConnected false, sync OrbitControls to the camera, re-enable them<br/>SpacePilotSimpleIntegration.tsx:99-110
        SD-->>ST: same event, chip cleared (client/src/features/control-center/status/useSpacePilot.ts:58-62)
    else evt.device is a different, already-idle device
        SD-->>SD: ignored - not the active device (SpaceDriverService.ts:211)
    end
    NAV-->>SD: 'connect' event (device replugged or newly paired)
    Note right of SD: inline closure, not handleDisconnect -<br/>SpaceDriverService.ts:132-140
    alt evt.device.vendorId in SUPPORTED_VENDOR_IDS
        SD->>SD: openDevice(evt.device)
        opt this.device already set
            SD->>DV: disconnect() the previous device first<br/>SpaceDriverService.ts:154-156 - replaces, does not stack
        end
        SD->>DV: device.open(), addEventListener('inputreport')<br/>SpaceDriverService.ts:159, :170
        SD-->>SI: dispatchEvent(new CustomEvent('connect', {device}))<br/>SpaceDriverService.ts:175-177
        SI->>SI: isConnected true SpacePilotSimpleIntegration.tsx:95-97
        SD-->>ST: same event, chip shows the device name (client/src/features/control-center/status/useSpacePilot.ts:54-57)
    else unsupported vendor
        SD-->>SD: connect event ignored - no open() call
    end
    Note over SD,ST: INVARIANT only one device is held at a time -<br/>openDevice() always disconnects the current device first,<br/>SpaceDriverService.ts:154-156. Reconnection is event-driven, not polled.
    Note over SD: DEBT destroy() removes only the disconnect listener SpaceDriverService.ts:307-313 -<br/>the connect closure cannot be removed, so destroy then initialize stacks a second one
```

**What it shows.** Unplug and replug at this pin, with the two live listeners: the camera integration and the Status surface. **Why it is this way.** WebHID reports hot-plug on `navigator.hid`; the service keeps one device and replaces it rather than stacking.

**Debt:** the `connect` handler is an anonymous closure (client/src/services/SpaceDriverService.ts:132-140) and `destroy()` removes only the `disconnect` listener (client/src/services/SpaceDriverService.ts:307-313); nothing in the client calls `destroy()` at this pin, so this is latent.

## VC-37.10 Spatial-input settings — the SpacePilot settings reach nothing that is mounted

```mermaid
flowchart TB
    subgraph store["settingsStore"]
        SS["settings.visualisation.spacePilot, typed SpacePilotConfig<br/>client/src/features/settings/config/settings.ts:596"]
        HS["settings.visualisation.interaction.headTrackedParallax<br/>{enabled, sensitivity, cameraMode}<br/>HeadTrackedParallaxController.tsx:17-19"]
    end
    subgraph dead["Unimported: visualisation/hooks/useSpacePilot.ts and SpacePilotController"]
        DEF["defaultSpacePilotConfig deadzone 0.1, smoothing 0.8<br/>SpacePilotController.ts:88-110"]
        MERGE["mergedConfig = default, then store, then userConfig<br/>client/src/features/visualisation/hooks/useSpacePilot.ts:71-76"]
        UPD["setMode and updateConfig write the store back<br/>client/src/features/visualisation/hooks/useSpacePilot.ts:188-202, :205-217"]
    end
    subgraph live["Mounted: SpacePilotSimpleIntegration"]
        LIT["hard-coded config: translationSpeed 5, Y 2.5, rotationSpeed 0.02,<br/>deadzone 0.02, smoothing 0.85, invertRX true<br/>SpacePilotSimpleIntegration.tsx:29-38"]
    end
    subgraph hp["Head-parallax read path (no write back)"]
        HREAD["trackingEnabled, sensitivity, cameraMode read every render<br/>HeadTrackedParallaxController.tsx:17-19"]
        HAPPLY["asymmetricFrustum: virtualScreenScale = 1 + sensitivity*0.5<br/>HeadTrackedParallaxController.tsx:39<br/>else: offsetX/Y = headPosition * sensitivity * ±0.5<br/>HeadTrackedParallaxController.tsx:57-58"]
    end
    DEF --> MERGE
    SS --> MERGE
    UPD --> SS
    LIT --> CAMX["camera"]
    SS -.->|"not read by"| LIT
    HS --> HREAD --> HAPPLY
    note1["DRIFT: the SpacePilot settings round trip exists only in the unimported hook.<br/>The mounted integration reads no settings, so a stored deadzone, sensitivity,<br/>inversion or mode changes nothing on screen. Head-parallax settings are live<br/>and read-only from the controller's side."]
    SS --> note1
    LIT --> note1
```

**What it shows.** Which settings reach the camera. Head-parallax settings do. SpacePilot settings are typed in the settings schema and would be read and written by the hook, but the hook is not imported, and the integration that is mounted uses literals. **Why it is this way.** The simpler integration replaced the hook-and-controller path in the canvas without the settings being re-pointed at it.

**Drift:** `settings.visualisation.spacePilot` (client/src/features/settings/config/settings.ts:596) is read and written only by `client/src/features/visualisation/hooks/useSpacePilot.ts` (client/src/features/visualisation/hooks/useSpacePilot.ts:58-59, client/src/features/visualisation/hooks/useSpacePilot.ts:211-215), which has no importer, while the mounted `SpacePilotSimpleIntegration` uses a fixed literal (client/src/features/visualisation/components/SpacePilotSimpleIntegration.tsx:29-38). The earlier drawing of this panel showed the hook's round trip as the live path.

## VC-37.11 A still desktop drag keeps its server hold alive with a heartbeat

```mermaid
flowchart TD
    START["Drag start sends nodeDragStart with the start position<br/>useGraphEventHandlers.ts:159"] --> BEAT["Any old heartbeat stopped, a new one started<br/>useGraphEventHandlers.ts:171,172"]
    BEAT --> TICK["Every 500 ms resend the current position as nodeDragUpdate<br/>dragHeartbeat.ts:9,15<br/>useGraphEventHandlers.ts:175,176"]
    TICK --> SERVER["Server releases a hold after 2000 ms with no update<br/>types.rs:23"]
    END["Drag end sends nodeDragEnd<br/>useGraphEventHandlers.ts:278"] --> STOP["Heartbeat cleared, also on unmount<br/>useGraphEventHandlers.ts:296,85<br/>dragHeartbeat.ts:16"]
    TEST["Test pins heartbeat below half the timeout<br/>dragHeartbeat.test.ts:5,17"] -.-> DEBT["DEBT: the 2000 ms server timeout is restated<br/>as a client test constant, not shared<br/>dragHeartbeat.test.ts:5"]
    SERVER -.-> DEBT
```

**What it shows.** The desktop only sends an update when the pointer moves, so a held-still node would be dropped without the timer. **Why it is this way.** ADR-2169 made a drag a hold with a 2 s socket timeout (2026-11-01). The heartbeat (`db1ab342d`) is the client half of that contract.
