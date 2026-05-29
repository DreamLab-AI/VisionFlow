# VisionFlow XR — Unity 6 + Meta Quest Technical Design Document

**Revision 1.0 · May 2025**  
**DreamLab-AI / VisionFlow**

---

## 1. Executive Summary

VisionFlow is a distributed, governed AI-agent mesh where shared semantic knowledge graphs are the first-class data structure. This document covers everything remaining to ship the **Unity 6 / Meta Quest XR layer** on top of the core C# services already implemented.

The six files already merged (`VisionFlowManager`, `NostrIdentity`, `GraphDataModels`, `MCPBridgeClient`, `SolidPodClient`, `NodeInspectorUI` / `GraphVisualizer`) provide a solid foundation. **Six major work-streams** remain before the XR layer is production-ready:

1. Replace crypto placeholders with real secp256k1 / Schnorr (NBitcoin.Secp256k1)
2. Implement XRI 3.x interaction layer (hover, select, grab, ray-cast)
3. Build the Governance approval UI (Nostr kind 31402)
4. Performance & spatial optimisation (LOD, culling, edge batching)
5. Nostr relay event feed (live agent activity)
6. Scene setup, asmdef hygiene, CI/CD for Quest

---

## 2. Current Architecture

### 2.1 Dependency Graph

Runtime dependencies between existing C# modules and external services:

| Layer | C# Class | External Service |
|-------|----------|------------------|
| Bootstrap | `VisionFlowManager` | Unity scene lifecycle |
| Identity | `NostrIdentity` | PlayerPrefs / Secure Keystore |
| Network | `MCPBridgeClient` | VisionClaw `ws://…/v1/graph` |
| Storage | `SolidPodClient` | Agentbox Solid Pod (LDP) |
| Graph Render | `GraphVisualizer` + `NodeView` | URP / LineRenderer |
| UI | `NodeInspectorUI` | UGUI World Space Canvas |

### 2.2 Thread Model

- `MCPBridgeClient` runs its receive loop on a background thread (`Task` / async/await).
- All Unity API calls are deferred via a `Queue<Action>` drained in `VisionFlowManager.Update()` each frame.
- `SolidPodClient` uses `HttpClient` on the calling thread (typically async from a MonoBehaviour).
- **No Unity API is ever called from a background thread.**

### 2.3 Key Data Flows

- **Graph stream:** VisionClaw CUDA → WebSocket JSON-RPC → `MCPBridgeClient` → Queue → `GraphVisualizer` → `NodeView` / `LineRenderer`
- **Provenance:** User XR Select → `NodeView` → `NodeInspectorUI.Show()` → `SolidPodClient.WriteProvenanceBeadAsync()`
- **Governance:** Governance tap → `VisionFlowManager.RequestGovernanceApproval()` → `MCPBridgeClient.SendAsync("nostr.publish", kind 31402)`

---

## 3. What Is Already Implemented

| File | Responsibility | Status |
|------|----------------|--------|
| `VisionFlowManager.cs` | Singleton bootstrap; wires all services; retry connect loop; main-thread event dispatch | **DONE** |
| `NostrIdentity.cs` | secp256k1 keypair (PlayerPrefs); NIP-98 header; BIP-340 Schnorr via `SchnorrHelper` + NBitcoin.Secp256k1 | **DONE** |
| `GraphDataModels.cs` | JSON-RPC 2.0 envelope; `GraphNode` / `GraphEdge` / `GraphSnapshot` / `GraphDelta` DTOs; `VisionFlowJson` settings | **DONE** |
| `MCPBridgeClient.cs` | WebSocket client; reconnect loop; `graph.subscribe`; `DispatchMessage` routing; main-thread queue | **DONE** |
| `SolidPodClient.cs` | LDP GET / PUT / PATCH / DELETE; NIP-98 auth; provenance bead (kind 30001); WebID profile | **DONE** |
| `GraphVisualizer.cs` | 3-D node spawn / update / remove; community gradient; OWL relation colours; edge `LineRenderer` | **DONE** |
| `NodeInspectorUI.cs` | Floating World Space panel; gaze-follow; OWL metadata labels; provenance bead write button | **DONE** |

---

## 4. Remaining Work — Detailed Breakdown

### 4.1 Crypto — Real Schnorr Signatures

**Priority: HIGH.** Blocker for production Nostr event publishing.

Currently `NostrIdentity` uses HMAC-SHA256 as a placeholder — structurally valid NIP-98 headers but **not cryptographically Nostr-compliant**.

**Tasks**

- Add `NBitcoin.Secp256k1.dll` to `Assets/Plugins/` (netstandard2.1 build from NuGet `NBitcoin.Secp256k1` ≥ 1.0.6)
- Replace `DerivePublicKey()` placeholder with `Context.Instance` → `TryCreateECPrivKey` → `CreateXOnlyPubKey().ToBytes().ToHex()`
- Replace `Sign()` placeholder with `key.SignBIP340(msgBytes, out var sig)` → `sig.ToBytes().ToHex()`
- Add unit test: sign known message → verify with matching pubkey
- Consider migrating PlayerPrefs storage to Android Keystore / iOS Keychain via Unity.Android.Security / Apple.Security plugins

**New file**

- `Assets/VisionFlow/Crypto/SchnorrHelper.cs` — thin wrapper around NBitcoin.Secp256k1

---

### 4.2 XRI Interaction Layer

**Priority: HIGH.** Nothing in the current codebase wires XRI events to `NodeView`. Without this, users cannot select nodes in the headset.

**Package requirements**

- `com.unity.xr.interaction.toolkit` ≥ 3.0 (XRI 3.x)
- `com.unity.inputsystem` ≥ 1.8
- `com.unity.xr.openxr` ≥ 1.10 + Meta OpenXR Feature Group

**Tasks**

- Create `NodeInteractable.cs` — extends `XRSimpleInteractable`; expose `OnHoverEntered`, `OnSelectEntered`; forward to `NodeInspectorUI.Show()`
- Attach `NodeInteractable` to node prefab alongside `NodeView`; add XR Collider (`SphereCollider`, radius = 0.06)
- Create `GraphRaycaster.cs` — caches `XRRayInteractor` from left/right controllers; hover highlights via `NodeView.SetHighlight(bool)`
- Implement `NodeView.SetHighlight()` — lerp emission on hover; pulse scale on select
- Add grab support (`XRGrabInteractable` on graph root) to reposition the whole graph in space
- Hand-tracking fallback: if Meta Hand Tracking enabled, map pinch gesture to Select

**New files**

- `Assets/VisionFlow/Interaction/NodeInteractable.cs`
- `Assets/VisionFlow/Interaction/GraphRaycaster.cs`
- `Assets/VisionFlow/Interaction/GestureMapper.cs` (hand-tracking pinch → select)

---

### 4.3 Governance Approval Panel (kind 31402)

**Priority: HIGH.** `VisionFlowManager.RequestGovernanceApproval()` exists but there is no UI for human-in-the-loop judgment — the core VisionFlow value proposition.

**Design**

World Space Canvas panel when the bridge receives inbound kind **31402** governance-request from another agent. Human reviews agent DID, action verb, and context, then taps **Approve** or **Reject**. Decision published as kind **31403**.

**Tasks**

- Extend `MCPBridgeClient.DispatchMessage()` to route `nostr.event` with `kind == 31402` to `OnGovernanceRequest`
- Create `GovernanceRequestUI.cs` — display agent DID, action, context; Approve / Reject buttons
- Approve → `VisionFlowManager.PublishGovernanceDecision(eventId, "approved")`
- Reject → same with `"rejected"` + optional free-text reason
- Wire `PublishGovernanceDecision()` — kind 31403 with `e`-tag referencing original request
- Queue management: sequential display; badge count on persistent HUD

**New files**

- `Assets/VisionFlow/UI/GovernanceRequestUI.cs`
- `Assets/VisionFlow/UI/GovernanceHUDBadge.cs`

---

### 4.4 Spatial Performance & Optimisation

**Priority: MEDIUM.** VisionClaw can stream tens of thousands of nodes. Naïve per-node GameObjects will tank Quest 3 frame rate (target **90 Hz**).

**Tasks**

- LOD on `NodeView`: impostor billboard sprites beyond 3 m; full sphere within 1 m gaze cone
- Frustum + occlusion culling: `NodeView.OnBecameVisible` / `OnBecameInvisible` to disable Renderer
- Edge batching: replace per-edge `LineRenderer` with single `GL.Lines` or `Graphics.DrawMeshInstanced` from `GraphVisualizer`
- Fix empty `LateUpdate` in `GraphVisualizer` — refresh edge positions; JobSystem `IJobParallelFor` beyond ~2k edges
- Node pool: pre-allocate 512 `NodeView` GameObjects instead of Instantiate/Destroy per delta
- Delta tick-rate: expose `tick_rate_hz` (currently 10) as runtime Debug slider

**New files**

- `Assets/VisionFlow/Graph/NodePool.cs`
- `Assets/VisionFlow/Graph/EdgeBatcher.cs`

---

### 4.5 Nostr Live Event Feed

**Priority: MEDIUM.** Bridge currently handles `graph.snapshot` / `graph.delta` only. Live Nostr events should appear as a spatial timeline.

**Tasks**

- Extend `MCPBridgeClient`: `OnNostrEvent(NostrEvent)`; route `nostr.event` notifications
- Create `NostrEvent.cs` DTO: id, pubkey, kind, tags, content, created_at, sig
- Create `EventFeedUI.cs`: vertical World Space scroll-list; colour by kind (30001 bead, 31402 request, 31403 decision)
- On connect: `nostr.subscribe` RPC with kinds filter in `SendSubscribeAsync`
- Tap event → highlight relevant `NodeView` when tags reference node id

**New files**

- `Assets/VisionFlow/Network/NostrEvent.cs`
- `Assets/VisionFlow/UI/EventFeedUI.cs`
- `Assets/VisionFlow/UI/EventFeedItem.cs`

---

### 4.6 Scene Setup, Assembly Definitions & CI/CD

**Priority: HIGH** (required for team builds). No `.asmdef` files today — everything compiles into `Assembly-CSharp`.

**Assembly definitions**

| Assembly | References |
|----------|------------|
| `VisionFlow.Crypto` | Newtonsoft.Json, NBitcoin.Secp256k1 |
| `VisionFlow.Graph` | VisionFlow.Crypto, Newtonsoft.Json |
| `VisionFlow.Network` | VisionFlow.Crypto, VisionFlow.Graph, Newtonsoft.Json |
| `VisionFlow.UI` | VisionFlow.Graph, VisionFlow.Network, TextMeshPro, XRI |
| `VisionFlow.Interaction` | VisionFlow.Graph, VisionFlow.UI, XRI |
| `VisionFlow.Tests` | VisionFlow.Crypto, NUnit (Editor only) |

**Quest scene setup checklist**

- XR Plug-in Management: OpenXR on Android; Meta Quest Feature Group
- Android: Min API 29; Target API 33; IL2CPP; ARM64; Vulkan only
- URP: MSAA 4x; disable HDR; Foveated Rendering (Fixed Foveation Level 2)
- Main Camera → XR Origin (VR); Tracking Origin Mode = Floor
- `VisionFlowManager` on DontDestroyOnLoad empty GO; wire `GraphVisualizer` and `NodeInspectorUI`
- Quest Hand Tracking in Meta XR SDK → `GestureMapper.cs`

**CI/CD (GitHub Actions)**

- `game-ci/unity-builder@v4` with Unity 6000.0.x, Android target
- On `main` push: build APK → upload via MQDH CLI
- Run `VisionFlow.Tests` (EditMode) in pipeline

See [§7 Appendix](#7-appendix-cicd-template-github-actions) for workflow template.

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NBitcoin.Secp256k1 IL2CPP stripping strips native lib | MED | HIGH | `link.xml` preserve rules; test on device early |
| WebSocket wakes suspended Quest (OS background limit) | HIGH | MED | `OnApplicationPause` → `Bridge.Dispose()`; reconnect on resume |
| 10k+ nodes saturate Quest 3 GPU (draw calls) | HIGH | HIGH | Ship EdgeBatcher + NodePool before public demo; node count circuit-breaker |
| NIP-98 clock skew rejected by relay (±10 min) | LOW | MED | NTP on startup; adjusted offset in `BuildNip98Header()` |
| PlayerPrefs private key exposed on rooted device | MED | HIGH | Migrate to Android Keystore before v1.0; document in `NostrIdentity.cs` |
| MCPBridge large messages exceed 64 KB buffer | MED | MED | Multi-frame reassembly in `ReceiveLoopAsync`; or negotiate fragment size with VisionClaw |

---

## 6. Milestone Plan

| M# | Name | Deliverables | Target |
|----|------|--------------|--------|
| **M1** | Crypto + Build | Real Schnorr; `.asmdef` files; Quest IL2CPP build green in CI | Week 1–2 |
| **M2** | Interaction | XRI `NodeInteractable`; `GraphRaycaster`; hover highlight; graph grab; hand-tracking pinch | Week 3–4 |
| **M3** | Governance UI | `GovernanceRequestUI`; 31403 decision publish; HUD badge; queue management | Week 5 |
| **M4** | Performance | `NodePool`; `EdgeBatcher`; LOD; frustum culling; `LateUpdate` edge fix; 90 Hz on Quest 3 with 5k nodes | Week 6–7 |
| **M5** | Nostr Feed | `OnNostrEvent`; `EventFeedUI`; kind-colour coding; tap-to-highlight | Week 8 |
| **M6** | Hardening | Android Keystore migration; NTP clock sync; buffer reassembly; end-to-end integration test | Week 9–10 |

---

## 7. Appendix: CI/CD Template (GitHub Actions)

Place at `.github/workflows/quest-build.yml` in the repository root.

```yaml
name: Quest Build
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { lfs: true }

      - uses: game-ci/unity-builder@v4
        env:
          UNITY_LICENSE: ${{ secrets.UNITY_LICENSE }}
          UNITY_EMAIL:   ${{ secrets.UNITY_EMAIL }}
          UNITY_PASSWORD: ${{ secrets.UNITY_PASSWORD }}
        with:
          unityVersion: 6000.0.x
          targetPlatform: Android
          buildName: VisionFlowXR

      - name: Upload to MQDH
        run: |
          mqdh apk upload \
            --apk build/Android/VisionFlowXR.apk \
            --app-id ${{ secrets.META_APP_ID }} \
            --token ${{ secrets.META_ACCESS_TOKEN }}
```

---

## 8. Open Questions

1. Should VisionClaw WebSocket use `wss://` TLS in production? (currently `ws://` — fine for local dev)
2. Which Meta OpenXR features are required: Mixed Reality passthrough? Spatial Anchors for persistent graph position?
3. Maximum expected node count from a VisionClaw session? Drives whether JobSystem edge batching is required at M4.
4. Is Agentbox Solid Pod on-device, LAN, or cloud? Impacts `SolidPodClient` timeouts and offline fallback.
5. Should `NodeInspectorUI.labelDid` show the viewing user's DID or the selected node's agent DID? (Currently shows viewer — likely a bug.)
6. `GraphVisualizer.LateUpdate` body is empty (edge positions not refreshed when nodes move). Intentional until CUDA layout stabilises?

---

*Generated from the VisionFlow XR TDD source (docx builder). Maintained as Markdown for version control and agent context.*