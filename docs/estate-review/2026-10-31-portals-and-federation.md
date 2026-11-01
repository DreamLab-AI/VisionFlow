---
title: Portal interaction and proposed data-space federation
status: in-progress
date: 2026-10-31
type: explanation
---

# Portal interaction and proposed data-space federation

Portal entry is implemented; data-space federation remains proposed. Fresh headless tests reproduce a scale-dependent walk-through threshold and an API-level resize that precedes the comfort dim. Source inspection also finds missing portal draw-call accounting and a mismatch between the XR connection handshake and the release server's upgrade requirements. These findings do not establish a headset comfort or frame-rate failure.

This continues the [spatial-volume and concept-link review](2026-10-31-xr-volumes-and-code-links.md). [Source receipts](evidence/2026-10-31/portal-federation-sources.json) pin VisionClaw to `75b72416bac176ca8fde715cac3bd7c77657b3e7` and its agentbox checkout to `9dfc27b16e82e9786535a55e790710493dd51fdf`. Inspected files match those revisions. ADR-2161 and ADR-2166 were read in full; the proposed sections of their governing documents and relevant dependency surfaces were traced. This is not a complete audit of agentbox ADR-2162, the forum spatial-room chain or ADR-2167 visual extras.

## Executed portal evidence

An isolated project on the authorised HP host ran Godot `4.6.1.stable.official.14d19694e`, with Compatibility configured, the headless dummy renderer and a private `XDG_DATA_HOME`. Production scripts, scene and shaders were copied unchanged. [Remote hashes](evidence/2026-10-31/portal-remote-hashes.json) match the [fixture manifest](evidence/2026-10-31/portal-fixture-manifest.json). No native extension, live server, headset or real shader renderer was loaded.

The portal suite has 48 tests. **41 unmodified test bodies passed, with 19,288 assertions**. Seven tests instantiate GraphScene and require native-extension classes; those were deliberately removed from the isolated test copy, named in the manifest and not counted as passing. The retained [test copy](evidence/2026-10-31/portal-isolated-tests.gd), [raw output](evidence/2026-10-31/portal-gut-suite.txt), [plain output](evidence/2026-10-31/portal-gut-suite-plain.txt) and [one-script guard](evidence/2026-10-31/portal-gut-guard.txt) make this boundary explicit. Plain logs remove ANSI colour escapes and trailing whitespace. These results do not refresh the ADR's historical full-scene or two-engine receipts.

An additional [seven-case probe](evidence/2026-10-31/portal-boundary-probe.gd) calls the real portal, arrangement and volume scripts. [Results](evidence/2026-10-31/portal-boundary-probe.json):

| Input | Observed result | Assessment |
|---|---|---|
| Unit-scale arrangement, head 0.2 m outside a portal | Helper returns about 0.2; walk-through remains disarmed | Control matches the 0.3 m contract. |
| Half-scale arrangement, same physical 0.2 m gap | Helper returns about 0.4; walk-through re-arms | Reproduced early re-arming relative to the documented metre threshold. |
| Unit-scale arrangement, head 0.4 m outside | Helper returns about 0.4; walk-through re-arms | Control matches the contract. |
| Double-scale arrangement, same physical 0.4 m gap | Helper returns about 0.2; walk-through remains disarmed | Reproduced late re-arming. |
| Eight portal boxes | Eight membranes | Documented-capacity control. Existing tests also check box separation at eight portals with 1, 2.5 and 4 m boxes. |
| Nine portal boxes admitted by the arrangement API | Nine portal ids, eight membranes | An unguarded extension boundary beyond the documented eight-portal capacity, not a defect in the current three-volume scene. |
| While entered with reduced motion on, call `set_runtime_bounds` on another portal, changing 1 m to 2 m | Bounds change at fade alpha 0; the following tick starts the dim at alpha 0 | An API-level counterexample to an unconditional “resizes happen in the dark” claim. The scene blocks box grips while entered; this probe does not show a reachable HUD resize or a visible headset frame. |

`distance_to_box` maps the world point through the inverse global transform, then returns the local excess vector's length (`space_portals.gd:595-599`). `update_walkthrough` compares that directly with `REARM_DISTANCE` (`:617`). Two-hand manipulation is permitted while entered and scales the arrangement (`graph_scene.gd:2899-3000`), so scale is a supported state. For uniform scale `s`, the effective physical re-arm threshold is `0.3 × s` metres. Inside/outside detection at zero is unaffected by positive uniform scale; the nonzero hysteresis distance is the failing contract. This pass did not test arbitrary shear or nonuniform scaling.

For resizing, `set_runtime_bounds` applies bounds immediately (`spatial_volume.gd:248-251`), while `SpacePortals.tick` detects a changed signature afterwards and starts a dim (`space_portals.gd:550-556`). Acceptance should either stage the bounds mutation until full dark or explicitly limit the promise to repositioning after a caller has already resized the box. The preceding audit's cached-AABB growth counterexample also applies to the entered room's inherited fitting code; a static-content E7 test is not evidence for all live extent changes.

## Source-level integration gaps

**Portal draw-call accounting.** `_tick_frame_budget` sums node/edge layers, volume frames/handles, agent instances, links and hulls (`graph_scene.gd:3048-3083`). It does not include `SpacePortals.membranes` or the fade quad. Both are created as additional mesh instances (`space_portals.gd:198-245`). `FrameBudgetPass.tick` forwards the supplied call count to its allocator (`frame_budget_pass.gd:129-138`), so this source path undercounts portal-related calls. The renderer's total primitives are sampled separately: this is not evidence that their triangles are also uncounted. No live draw-call or GPU-time measurement was taken.

**Release upgrade mismatch.** The XR transport calls `connect_async_with_config` with a URL string (`transport.rs:95`), then requests topology and sends a NIP-98 `authenticate` text message only after upgrade (`:99-118`). `BackendConnection._dial` supplies the URL and signing secret, not upgrade headers (`backend_connection.gd:316-322`). The release server requires Origin before upgrade (`http_handler.rs:67-140`) and accepts signed WebSocket subprotocol authentication or a legacy bearer during upgrade (`:145-193`); an unauthenticated release upgrade is refused (`:314-326`). Supplying a secret for a later text frame cannot satisfy those earlier checks. ADR-2161 already calls out the missing Origin; its prerequisite work must also align the authentication phase. This is a source-confirmed mismatch, not a fresh request to a deployed release endpoint. Proxy header injection and permissive development settings were not tested.

**Visibility control, not a reproduced default-open defect.** The initial-load builder contains an obsolete “Default OFF” comment (`types.rs:255-256`). Its actual helper delegates to `visionclaw_domain::utils::visibility_filter::filter_enabled`; absent or unrecognised configuration enables filtering, while explicit `0/false/off/no` disables it. All **18 tests** in that production, framework-free module passed in a fresh `rustc --test` build ([receipt](evidence/2026-10-31/federation-visibility-unit.txt)). A comment or a private/public filter alone cannot prove the proposed selector/ACL/revocation behaviour.

## ADR-2166, section by section

| Section | Disposition at the pinned source |
|---|---|
| Metadata and Context | Accepted/partial/inactive and the explicit headset acceptance boundary are appropriate to the available evidence. The historical second-eye concern was not re-experimented here. |
| D1: entered state | Implemented as bounded room-scale presentation, with frame/handles hidden and the room positioned from the user's floor. Existing headless tests pass. Universal E7 containment remains qualified by the prior dynamic-extent counterexample. |
| D2: portal boxes and separation | Real boxes are laid out outside the room. Existing maximum-count and swap separation tests pass. Registration permits more than eight, while membrane storage caps at eight; keep that extension boundary explicit. |
| D3: membrane | One MultiMesh, per-instance tint/focus and reduced-motion uniform exist. Retained tests cover shape and forbidden sampling constructs. Headless resource loading is not real shader compilation or stereo evidence. |
| D4: action and walk-through | Enter, swap, dwell and disabled-default logic pass retained tests. Re-arming measures local units rather than world metres after arrangement scaling. Seven scene-level tests, including HUD wiring, were excluded rather than claimed as fresh passes. |
| D5: restoration | Snapshot/restore, slots, moved flags, modes, bounds and arrangement-transform tests pass; Front is refused while layout-locked. No physical-controller journey was run. |
| D6: transitions | Retained tests cover dim/snap, room loss during both fade phases, swaps/leaves and eased entry. Programmatic bounds changes precede the dim; subsequent ring repositioning is staged. A comfort claim needs those operations distinguished. |
| D7: refusals | Native-state, unknown, hidden, closed, offline, repeated-entry and in-flight controls pass at the portal API. Scene/native lifecycle acceptance remains outside this fixture. |
| D8: default and drag | The ring remains the default. Source retains Codescape capability refusal; its scene-level test was excluded. No new drag or 90 Hz claim follows from the logic suite. |
| D9: budget hook | Source wires entered weights through `set_volume_weights`. Unit weight tests pass. The owner already accepted nearly inert weights at production density; this pass does not relabel that decision as a new defect. Membrane/fade call accounting remains separate. |
| Consequences | Room-scale inspection and exact restoration are supported at the fixture level. Portal readability at about 5.3 m, comfort and production-density rendering require user/device evidence. Two-hand scaling makes the hysteresis mismatch relevant. |
| Verification and restamps | Historical 4.6.1/4.8 and merged-tree results remain historical. Fresh coverage here is 41/48 logic tests on 4.6.1 plus seven audit cases, with no shader-renderer or headset acceptance. The R11 weighting caveat and later owner decision are retained. |

## ADR-2161, section by section

The [production bridge/resolver probe](evidence/2026-10-31/federation-prerequisite-probe.cjs) invokes modules without starting an HTTP listener. Four cases confirm that a proposed dataspace URN does not cross, the corresponding agentbox resolver returns `404 unknown-kind`, the advertised grammar has no dataspace, and an existing bead crosses successfully ([results](evidence/2026-10-31/federation-prerequisite-probe.json)). The resolver currently advertises 307/404 and explicitly disclaims 410. ADR-2161's context that it still advertises 410 is stale; the missing erasure prerequisite remains real.

| Section | Disposition at the pinned source |
|---|---|
| Metadata and Context | Proposed/none/inactive remains supported for this feature. Existing agent-event attribution and identifier translation are narrower cross-boundary paths, not peer data-space federation. Update the resolver-advertisement context when the owner next revises the ADR. |
| D1: selector and identifier | `src/uri/mod.rs` has no Dataspace kind or typed mint/parse. No selector-bound space implementation was found in the searched backend/XR source. The proposed grammar is not a shipped endpoint. |
| D2: signed catalogue | Signature, pinned endpoint, monotonic revision, withdrawal and persistent watermark rules are specified. No catalogue producer/consumer was found in those sources. None of the replay or same-second-withdrawal guarantees is established by existing Nostr signing elsewhere. |
| D3: pinned peers and rotation | The agentbox dependency describes the peer projection and separate node key. This pass inspects the consuming contract, not a running mesh service. Peer rotation and old-key retraction remain activation work. |
| D4: viewer-owned direct read | Current upgrade and visibility controls exist, but they do not implement a dataspace connection binding, selector ∩ visibility ∩ ACL filtering, mutation refusal or a 10 s re-evaluation loop. Fix both Origin and upgrade authentication before release native-client acceptance. |
| D5: link ownership and isolation | Primary authored links and disposable received references are correctly separated in the proposed data-authority document. Existing concept links use stream-generation pairs; they do not establish `(node key, dataspace, generation)` isolation or authorised-both-end rendering. |
| D6: control-only envelopes | Three allowed envelope types, sender authority, durable sequence checks and write-before-act are specified, not verified implementations here. A live generic envelope library would not by itself establish these application checks. |
| D7: erasure and leases | Current agentbox resolver has no tombstone lookup or 410 response. VisionClaw has no matching `/v1/uri` route in the searched Rust source. Identify the authoritative home resolver and store before claiming immediate erasure, then test display removal, received-reference purge, seven-day expiry and eight-day tombstone retention separately. |
| D8: crossing kind | The consumed `federation-kinds.json` lacks dataspace; the production JS bridge refuses it. VisionClaw embeds that file and still has no Dataspace kind. The required cross-repository order has not been fulfilled. |
| D9: exclusions | Codescape federation, cross-node writes, ontology alignment and remote presence remain excluded from v1. Portal entry or the local code-to-concept join does not implement any of them. |
| Cross-operator evolution | Additional policy, peer agreements and parked ADR activation remain future work. No automatic discovery or peering follows from a same-operator pin design. |
| Consequences and costs | Socket count, re-evaluation, durable replay state, leases, signer availability and the security-profile entry need implementation and operational budgets. No local portal test measures those costs. |
| EDD D1 | No fresh catalogue/restart test: unpinned key, endpoint mismatch, lower revision and replay after withdrawal remain required. |
| EDD D2 | No dataspace ACL/count non-disclosure test. Existing private-node tests are narrower. |
| EDD D3 | No federated viewing session or before/after durable-store diff. |
| EDD D4 | No retraction, sender forgery, offline lease or authored-versus-received lifetime test. |
| EDD D5 | No federated-read server mode to attack with a raw client. Client capability refusal is insufficient. |
| EDD D6 | Local connection tests do not establish overlapping-space ACL isolation or both-end authorisation. |
| EDD D7 | No live ACL/selector/visibility revocation timing receipt. |
| Activation prerequisites | Keep `public`/`dids` independent of forum membership; add the room-admission/forum chain only for `space:`. None of the missing signer, identifier, stream, ACL or erasure gates is closed by portal rendering. |
| Verification | The feature-level “none” classification remains appropriate; the [scoped source-search receipt](evidence/2026-10-31/federation-source-search.json) records the inspected search roots. The audit supplies present-source prerequisites and counterexamples, not an activation restamp. |

One design question should be resolved before implementing link-notice authority: does “subject/object under the node's key” admit individual graph URNs or only dataspace URNs? Current KG URNs carry an owner key, concept URNs carry a domain, and execution URNs carry a content hash; those are not necessarily a federation node key (`src/uri/mod.rs:11-26`). Define the authoritative binding instead of treating a string prefix or a viewer's identity as proof of the home node. This is an unresolved contract question, not a reproduced exploit. The proposed protocol entry also calls the initial load “REST” (`PROTOCOL-registry.md:351`), while the current XR path requests topology over its WebSocket (`transport.rs:99`). Specify the actual carrier for the new `urn` field before using a REST-only change as evidence that the XR consumer receives it.

## Comparison and user utility

Matrix distinguishes space organisation from room joining: spaces form a room hierarchy, while room membership and restricted join conditions are explicit protocol rules. Its parent-link validation also rejects unsupported claims of belonging to a space. This is a useful comparison for keeping “shown in a room”, “authorised to read a dataspace” and “trusted peer” separate in VisionFlow's interface and tests. Matrix is not evidence that VisionClaw's proposed revocation or graph projection is implemented. [Matrix specification](https://spec.matrix.org/latest/client-server-api/#spaces), [restricted rooms](https://spec.matrix.org/latest/client-server-api/#restricted-rooms).

ShapesXR's dated 2022 release describes 1:1 alignment, floor levelling, a physical-room reference, teleportation and a passthrough comfort treatment. Those are concrete interaction alternatives to evaluate alongside a room-scale graph and dim/snap transition; the vendor's comfort statement is not an independent comparative result. [ShapesXR release](https://www.shapesxr.com/post/update-1-1-mode-real-world-reference-passthrough-material-teleportation-tool).

For a useful evaluation, ask a reviewer to inspect a concept, enter its code volume, return to the exact earlier layout, then identify which remote source they may read and why. Repeat at half, unit and double arrangement scale, with an offline source and a revoked permission. Record wrong-space entry, re-arm distance, successful return, task accuracy, time and reported discomfort. Compare the ring, explicit Enter/Leave and optional walk-through before treating portals as a user benefit. Federation acceptance additionally needs a denied viewer, an overlapping space with different ACLs, and a restart after withdrawal. These are proposed tests, not measured outcomes.

## Reproduction and remaining work

See the fixture manifest for copied paths, retained tests and all seven exclusions. Put its copied files and the repository's `addons/gut` in a temporary project; install `portal-isolated-tests.gd` as `tests/unit/test_space_portals.gd` and the audit probe at project root. Configure Compatibility, use a private `XDG_DATA_HOME`, import with `godot --headless --editor --import --quit`, then run `--script addons/gut/gut_cmdln.gd -gdir=res://tests/unit -gexit -glog=1` and `--script res://portal-boundary-probe.gd`. Run `tests/gut_guard.sh` against both plain logs. The [execution receipt](evidence/2026-10-31/portal-execution-receipt.json) records the engine and commands.

Run the prerequisite probe with `node federation-prerequisite-probe.cjs <agentbox-root> <output.json>`. The visibility tests use `rustc --edition=2021 --test <project>/crates/visionclaw-domain/src/utils/visibility_filter.rs -o <temporary-binary>` and execute that fresh binary.

Remaining work includes the seven native scene tests, real shader/stereo/headset acceptance, scaled hysteresis and staged-resize fixes, portal call accounting, release handshake integration, all seven federation EDD cases, ADR-2157/2160/2162 and the forum spatial-room chain. The whole-estate ADR, diagram and user-utility audit remains active.

## Documentation checks

The index now contains 169 topics and 1,560 diagrams. The [strict citation pass](evidence/2026-10-31/portal-citations.txt) and [revision verification](evidence/2026-10-31/portal-verification.txt) pass. All twelve diagrams in touched topics VC-44, VC-46, VC-47 and VC-48 render; the six new diagrams were visually inspected. This structural corpus check does not substitute for semantic review of the remaining topics.

## Ledger follow-up — 2026-11-01

VisionClaw commit `5e1bd90a434818b066f9c0d769940e5f61640804` adds a scoped
re-verification annex to ADR-2166 and advances its metadata pin to `ecd78b6cb`.
The annex was read and its cited `hud.gd` diff checked: the two changed lines
make the flow-edge key swatch gold and label its slow breathing. This does not
change portal admission, weighting or entry behaviour, and supplies no new
headset acceptance. The section assessment and open findings above remain;
only this added annex and metadata are reconciled here.

Later in this batch, incoming grab/client work moved VisionClaw main to `434afb6b7`
and changed ADR-2166 again. The narrow metadata annex above does not reverify
those later changes. The completion ledger deliberately leaves that assessment
stale until the new obligations and runtime evidence receive their own review.

## Grab-amendment refresh — 2026-11-01

The [grab lifetime assessment](2026-11-01-xr-grab-boundaries.md) now reviews those
annexes at `531df5ef7`. Only `graph_scene.gd` and the HUD swatch changed among
ADR-2166's eight governed paths. The new review traces per-hand admission,
release on portal transitions and the added locks, including their remaining
boundary gaps. This restores the ADR-2166 document binding. All prior portal
findings and headset limits remain; no historical portal test is presented as
a fresh run. The new held-node fit exclusion also needs an explicit E7 exception.
