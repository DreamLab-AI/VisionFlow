---
title: Mobile sensing companion and acceptance boundaries
status: in-progress
date: 2026-09-05
type: explanation
---

# Mobile sensing companion and acceptance boundaries

RuView's mobile companion could make physical sensing available to field operators. [ADR-034](../../../RuView/docs/adr/ADR-034-expo-mobile-app.md) proposes five screens on iOS, Android and web, with local-server sensing, offline simulation and platform RSSI. The current source contains those screens, stores and services, but its displayed geometry, MAT records and connection state do not establish a verified physical observation journey. No VisionClaw mobile sensing consumer is established by this review. See the [sensing context](sensing-extension.md) and [closeout roadmap](closeout/README.md).

The [receipt](evidence/ruview-mobile-snapshot.json) records twenty selected source hashes, all 25 test-file hashes, six Maestro file sizes and five passing isolated WebSocket-service assertions. TypeScript was erased from the unchanged service body; imports were replaced with fake stores, sockets and timers. No network, browser, native build, Jest suite, typecheck, device or clinical validation ran. The local mobile Jest/tsc executables were absent; no dependencies were installed.

## Architectural and implementation dispositions

| ADR phase | What exists | Gap and disposition |
|---|---|---|
| 1: core infrastructure | Expo 55/RN 0.83 package declarations, entry/config files, theme and five-tab navigation | Source presence is not a three-platform build. Navigation catches lazy import failures and supplies a placeholder screen, so successful navigation alone cannot prove the real screen loaded |
| 2: state and services | Pose, MAT and persisted settings stores; WebSocket, Axios REST, simulation and platform RSSI services | The schema and fallback behaviour differ from the ADR. Establish one consumed server contract, source provenance and settings hydration before connection |
| 3: shared components | Banners, gauges, grid, error boundary and themed components are present | Render tests must distinguish unknown, stale and synthetic values; a connected transport is not a live-hardware attestation |
| 4: screens | Live, Vitals, Zones, MAT and Settings source files exist | Web Live uses direct Three.js, MAT seeds a local scenario, and several visual values are derived proxies. Resolve these meanings before asserting the original screen responsibilities |
| 5: testing | 25 `.test.*` files and six Maestro YAML files exist | All 25 contain a placeholder suite; all six YAML files are zero bytes. Their existence supplies no acceptance coverage |

The original 63-source/107-total file counts are planning descriptions rather than acceptance evidence. The package declares the expected main technology families, but the Android RSSI source imports `@react-native-wifi-reborn` while the manifest declares `react-native-wifi-reborn`; module resolution and permission handling need native-build verification. iOS emits a fixed synthetic network at −60 dBm, contrary to the empty-result promise. Web emits fixed synthetic networks. No real platform scan was performed.

## Transport, source and lifecycle

`ws.service.ts` remaps HTTP port 3000 to WebSocket port 3001 at `/ws/sensing`; other ports use `/api/v1/stream/pose`. This differs from a universal `wsUrl/ws/sensing` contract and needs testing against the selected server/proxy profile.

The source constants are ten retries and a 500 ms simulation tick, rather than the ADR's five attempts and 10 Hz. `scheduleReconnect()` starts simulation on the first failure. Each timer calls `connect()`, which resets the attempt counter, so consecutive failed connections schedule one-second retries in the probe rather than escalating through 1/2/4/8/16 seconds. Normal code-1000 closure does not schedule recovery. Resolve retry, fallback and restart policies together; do not infer the five-second acceptance criterion from a counter constant.

`onmessage` parses JSON and casts it to `SensingFrame` without runtime shape or protocol validation. Arbitrary JSON reaches a subscribed listener in the probe. Socket opening marks the store connected; `isSimulated` is derived only from connection status, not the frame's source. Server-generated simulation therefore needs an independent source label. Existing frames/history are not cleared by a connection-status change.

`usePoseStream()` subscribes and calls `connect()` per mounted consumer; cleanup only unsubscribes. Several screens use it, so verify subscription ownership and duplicate store dispatch as tabs accumulate. Calling `connect()` with a changed URL keeps an existing open/connecting socket; the Settings save handler explicitly disconnects first, which avoids that specific path. Async persistence hydration still needs its own acceptance case. REST base URL assignment is found in Settings save, while the service starts with an empty base URL; startup restore and the draft connection-test target require verification.

## Meaning of the five screens

| Screen | Observed behaviour | Required distinction |
|---|---|---|
| Live | Native wraps bundled HTML and a JavaScript bridge. Web directly constructs Three.js bodies, capped at three, from `BASE_POSE`, classification, estimated count and animation; it is not the ADR's iframe path | Illustrative body animation versus received model keypoints; platform renderer parity and data age |
| Vitals | Breathing uses `breathing_bpm` or scales breathing-band power to 0–30 BPM. Heart uses `hr_proxy_bpm` or combines motion/breathing powers; it is labelled HR PROXY | Preserve proxy labelling, distinguish absent data from derived numbers and validate actual measurement fields before any physiological claim |
| Zones | Reads signal-field values and positions nested in classification. Its `collectPositions()` returns an array, so `??` fallbacks to alternative position keys cannot run after an empty first result | Agreed frame schema, coordinate frame, zone mapping and missing-position state; the ADR's top-level `persons` injection is not this consumer contract |
| MAT | Bundled dashboard starts a training event and three local survivor detections; screen also seeds event/zones on readiness. Frame updates move local markers using band-power deltas and random jitter | Local demonstration records versus authoritative backend tracks, confirmed counts, lifecycle events and alert provenance |
| Settings | Persists URL, RSSI flag, theme and alert sound. Default URL matches the ADR, but default theme is `system`, with no separate persisted `wsUrl`/simulation-mode field | Actual settings schema, hydration, validation, selected endpoint and restart behaviour |

The MAT dashboard's local detection helper generates IDs, positions and sometimes depth/rates, then emits survivor messages to React Native. Its `processFrame()` moves existing records; it does not consume ADR-026 tracking events. The native store upserts those messages and retains survivor/alert arrays without a bounded-history or source-reset contract. Socket status and demo records can coexist, so a connected banner cannot qualify those records as field observations. The [backend tracker integration gap](sensing-extension.md#survivor-tracking-and-operational-integration) independently remains open.

## Acceptance register

All 31 original criteria remain open for the scope stated below. Existing source or the isolated service probe narrows the work but does not replace platform acceptance.

| Original ID | Evidence required to close |
|---|---|
| B-1 | Reproducible iOS, Android and web builds; starting a development server alone is insufficient |
| B-2 | Actual specified iOS simulator launch and screen execution |
| B-3 | Actual Android emulator launch, including native RSSI module resolution |
| B-4 | Real Chrome, Safari and Firefox rendering and interaction results |
| B-5 | Strict TypeScript check on the selected dependency lock |
| W-1 | Selected sensing server handshake plus schema-valid received frame; connected status alone is insufficient |
| W-2 | Measured dispatch latency to the current `lastFrame` consumer, with workload and clock |
| W-3 | Corrected or explicitly revised retry policy and timed failure assertions |
| W-4 | Agreed immediate-versus-delayed simulation policy, source labels and timed fallback test |
| W-5 | Server restart, normal/abnormal close, late callbacks and settings-change recovery |
| S-1 | All five actual screen implementations render a known real-server scenario |
| S-2 | Explicitly synthetic scenario on all five screens with consistent labels |
| S-3 | Gauge animation/performance plus distinction between measured, missing and proxy inputs |
| S-4 | Native and web renderer evidence using known keypoints, or explicit acceptance of illustrative geometry |
| S-5 | Three-position fixture in the agreed schema produces three correctly located markers |
| S-6 | Authoritative survivor/alert fixture and selected backend integration; no local training seed counted as live |
| S-7 | Banner text/colour across actual statuses plus server simulation, unknown and stale data |
| P-1 | Persist/kill/restart with hydration and connection target observed |
| P-2 | Cleared-storage default test; current source declares the expected localhost URL |
| P-3 | URL rejection/acceptance plus confirmation that Test Connection uses the intended target |
| N-1 | Navigate all five loaded screens and prove placeholders did not mask import failures |
| N-2 | Platform visual inspection with agreed default/system/light/dark behaviour |
| N-3 | Thousand-frame mount/tab/unmount exercise measuring subscriptions, arrays and memory |
| N-4 | Deliberate child error proves fallback behaviour and recovery |
| R-1 | Android permission and actual scan evidence with resolved native dependency |
| R-2 | Decide empty versus explicitly synthetic iOS behaviour, then test it |
| R-3 | Web synthetic scan test with clear provenance |
| T-1 | Replace placeholder suites with meaningful assertions and run `npm test` |
| T-2 | Populate six empty Maestro files and execute five-screen flows |
| T-3 | Execute populated offline/recovery flow, including source distinctions |
| T-4 | Same typecheck evidence as B-5, rather than a second implied gate |

## Future scope and closeout sequence

The five future-work items remain separate proposed increments: on-device ONNX needs a model/export and measured device inference; push alerts need a delivery/authority contract and background test; watch support needs an actual supported platform and companion bridge; BLE needs firmware plus a selected native transport; multi-server viewing needs per-server identity, time and merge semantics. None is established by the current singleton WebSocket service or package declarations. External platform promises and performance/cost estimates in the ADR were not verified here.

ADR-019 supplies related UI concepts, not proof of mobile parity. The “Consumed” relationships to ADR-021/026/029/031/032 require actual field, identity and transport evidence: a REST/WebSocket client does not prove vital-sign validity, tracker ingestion, multistatic fusion or authenticated sensor admission. In particular, the earlier [firmware admission assessment](sensing-extension.md#secure-sensing-claims-and-model-admission) prevents treating the mesh-security reference as a trust guarantee.

CP-01/05/06/08/09 should first settle server/schema, source labels and demo-versus-operational records; then correct or revise lifecycle/rendering/platform contracts; replace placeholder gates; run native/web scenarios; and finally evaluate selected hardware and estate consumption. Mobile, sensing-server and delivery maintainers are proposed accountable roles. Disaster-response and healthcare benefits remain unvalidated aspirations, with no field or clinical acceptance claimed by this documentation review.
