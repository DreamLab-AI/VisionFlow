---
title: Action, wire and rendered state
status: in-progress
date: 2026-09-04
type: explanation
---

# Action, wire and rendered state

VisionClaw's spatial client turns graph structure and agent activity into navigable positions, work beams and query-result planes. The source supports more of this vision than the current XR governing document describes: hover motion and query execution are implemented. It also exposes unresolved contracts between authenticated activity, current work state and what the renderer shows. The [source/test receipt](evidence/xr-render-snapshot.json) records this pass; the browser renderer and full GPU pipeline still need separate investigation.

## From authenticated socket to attributed activity

The [agent-events handler](../../../project/src/agent_events/ingest.rs) validates a session token before upgrade, with an explicitly compiled development exception. Its actor retains the session public key, but calls `process_frame(text)` without that identity. The processor parses the canonical notification and records structural provenance before publishing to the hub. An attributed envelope is therefore not, by this path alone, proof that the authenticated sender owns the claimed agent identity. This is not a claim of anonymous production access: transport authentication and event attribution are different boundaries.

The [beam actor](../../../project/src/actors/agent_beam_actor.rs) projects events into binary actions and coalesces bursts. It retains a pending frame when the coordinator mailbox is full, clears on successful handoff and caps the backlog in favour of recent actions. This is a bounded visual activity stream, not a durable audit log. Complete-system acceptance must name the audit authority and show how dropped visual events affect the displayed state.

## Wire compatibility and freshness

The graph record remains 52 bytes. The [XR decoder](../../../project/xr-client/rust/src/binary_protocol.rs) accepts V3 and the V5 envelope, rejects unknown position versions/misaligned payloads, and sanitises invalid numeric records. It skips the V5 sequence bytes without retaining an ordering watermark. The envelope supplies ordering information, but this consumer does not enforce it. Ordered delivery on an individual socket does not by itself prove that concurrent full/delta producers and reconnects can never supply stale state.

Agent actions use a different framing rule on the same graph socket: a `0x23` tag, a `u16` count, then length-prefixed events, not a six-byte generic header. The decoder accepts complete events preceding a truncated event and returns the accepted prefix. This deliberate tolerant visual parse must not be described as atomic frame rejection. The separate presence codec reserves `0x44`; source searches of the server/client integration still find no live use of its encode/decode functions. Its staged activation remains appropriate.

## Rendered work state

[RenderStore](../../../project/xr-client/rust/src/render_store.rs) masks node flags into the registry key and stores target, action, timestamp and task. Every action sets status to `WORKING`; the action timestamp is retained but not compared for freshness. A JSON state update can set idle/done, and a later-arriving old action can set working again. No expiry based on that timestamp appears in this store. This is a source-level stale-state risk; an end-to-end reproduction remains open.

The store implements local hover positions for active agents and stride-16 beam packing. Beam targets are remapped through the fold plan and must be drawn; absent or coincident endpoints produce no beam. The agent endpoint is read directly from the local position store, so the older statement that both endpoints are fold-remapped overstates the implementation. A beam is visual evidence of received activity and render eligibility, not proof of currently authorised work or successful application.

## Query and headset boundaries

`query_builder.gd` sets `EXECUTE_ENABLED = true`. [GraphScene](../../../project/xr-client/scripts/graph_scene.gd) posts the built pattern to `/api/graph/query/pattern`, passes authentication headers, and builds result planes from successful bindings. Failure paths emit warnings; this pass has not established an adequate user-visible error state or server-side query correctness. The old governing-document claim that Execute is a disabled stub is removed.

[OpenXR boot](../../../project/xr-client/scripts/xr_boot.gd) checks optional eye-gaze support, warns when unavailable, and defers the graph scene transition. A missing or uninitialisable OpenXR runtime instead shows an error and stops. These source branches do not prove every headset boots. Spatial voice routing maths are present; this pass does not establish media transport, Quest performance or current desktop deployment.

## Evidence and closeout

All **218 Rust library tests pass**. They cover codecs, render maths, interaction/state, signing and routing helpers. Godot-facing classes are excluded by `cfg(test)`, so these results do not certify scene wiring, shaders, frame rate, network orchestration or headset interaction.

CP-04/06 acceptance requires a session-bound event identity or an explicit trusted-relay delegation contract; full/delta and action/state freshness rules; tests for stale, duplicate and truncated frames; clear stale/disconnected work indicators; and one authenticated action traced through ingestion, storage/audit receipt, binary delivery and a rendered beam. CP-06/08 also needs query success/denial/error evidence, live pose/co-presence integration before changing its staged status, and separately measured desktop and target-headset results. The [estate roadmap](closeout/README.md) retains these as open conditions.

## GPU packaging and runtime boundary

Agentbox [ADR-2006](../../../project/agentbox/docs/adr/ADR-2006-nixgl-gpu-wrap.md) records a CUDA library-discovery remedy. The current [wrapper](../../../project/agentbox/lib/gpu-wrap.nix) retains `--suffix LD_LIBRARY_PATH` and the flake's `local-cuda` gate, but also sets default GLX vendor, EGL vendor file and Vulkan ICD values. The flake wraps selected ffmpeg, QGIS and Blender binaries, plus the enabled 3DGS package binaries. Graphics configuration is therefore present even though the ADR excludes a presentation path; its explicit graphics review trigger has been reached. Setting those defaults is not proof of a functioning viewport or headset.

The wrapper preserves pre-existing values for its three `--set-default` variables. The [backend descriptor](../../../project/agentbox/lib/gpu-backend.nix) separately supplies supervisor environment assignments, including LD_LIBRARY_PATH and VK_ICD_FILENAMES. These are different configuration mechanisms; final inheritance and emitted service configuration need their own checks. The descriptor also distinguishes inference sidecars from local CUDA packages. A working sidecar does not certify the main image's wrapped binaries, and a wrapped Blender result does not certify the XR runtime or browser container.

The [receipt](evidence/gpu-runtime-snapshot.json) records an attempted existing backend test: exit 77, Nix unavailable, zero tests passed. Inspection finds that its expressions supply only the backend argument to a dispatcher now requiring both backend and toolchain flag; it also resolves a registry `nixpkgs` flake rather than explicitly binding this repository's lock. Those are source-level test-maintenance findings, not a newly executed evaluation failure. The historical hardware results in ADR-2006 remain historical evidence.

CP-01/06/08 needs separate acceptance for package selection, generated wrappers, actual driver/library resolution, compute workload and presentation. Bind each receipt to image, lock, driver, selected device, effective environment and application path. Cover inherited/absent vendor settings, missing driver files, backend-none and inference-only selection, supervisor versus CLI launch, and visible fallback/failure. Exercise loader resolution before asserting universal ABI isolation from path suffixing. Update the backend test interface and bind its dependencies before using it as an acceptance gate. Graphics scope should be explicitly adopted or separated in a follow-up decision; no current GPU deployment or rendering result is certified by this source review.

## Visibility defaults and output coverage

VisionClaw's current initial graph response and binary position stream both apply caller-based visibility filtering. The [initial response](../../../project/src/handlers/socket_flow_handler/types.rs) keeps public nodes or nodes whose owner string matches the session public key, then includes edges only when both endpoints remain. Its sync-generation check prevents an older in-flight response replacing a newer session sync. The [position path](../../../project/src/handlers/socket_flow_handler/position_updates.rs) projects visibility onto flagged wire IDs, computes a caller-specific drop set and filters before encoding a full-state position frame.

The flag parser defaults on, recognises explicit falsy strings and caches its result on first use. Some nearby comments still say default-off; executable parsing takes precedence. Node public status comes from metadata value equal to true ignoring case, while ownership is an exact string match. Missing public/owner data denies private visibility. This establishes a useful restrictive default, but says nothing by itself about who can author or change those metadata fields.

All six [domain filter tests](evidence/visibility-snapshot.json) pass: anonymous and non-owner filtering, owner access, public-only passthrough, empty drop sets and unmatched drop IDs. They exercise the pure set operation, not metadata integrity, initial/stream handler orchestration or browser state. Current source supersedes older claims that initial snapshots omit filtering; no earlier memory or historical review should be treated as current implementation evidence.

CP-02/04/06/08 requires authenticated ingest/update authority for visibility and owner metadata, canonical owner identity, and a complete output-path matrix including queries, labels, edges, analytics and alternate transports. Test public-to-private and owner changes, reauthentication and in-flight delivery with real client state. Omitting a node from future position frames does not by itself prove its label or cached geometry disappears from every client surface. Separate stopping future disclosure from attempting to revoke data already delivered. ADR-2003 retains the scoped default-on implementation declaration; complete private-data handling requires these additional receipts.

## Simulation layout and force authority

The [reproducible extracted-struct probe](evidence/simparams-layout-probe.py) compiles the current Rust declaration and the CUDA declaration as ordinary host C++, preserving their original size assertions. [All 53 field offsets agree](evidence/simparams-layout-probe.json), with size 212 and alignment 4 on these host compilers. The test also swaps dt and damping only in a temporary C++ fixture: the original size assertion still passes while the two offsets differ. Total size guards detect growth, but do not establish field order or type identity. This corrects ADR-2028's claim that any mismatch fails the build.

ADR-2028 retains complete/live for its scoped flat, size-locked representation. The stronger compatibility guarantee remains open. No CUDA compilation, driver load, device copy, shipped PTX comparison or client consumption ran. Current host parity is useful evidence, not a release certificate. Appending fields preserves old offsets; it does not by itself make a shorter allocation, old device module or raw-copy consumer safe with the new size.

The [force source review](evidence/wire-force-boundaries.json) confirms the final physics-step wrapper rebuilds flags, derives constraint enablement from num_constraints and overwrites the converter result immediately before execute. Constraints.apply remains a no-op. An actor parameter mirror also copies converter-derived flags; this is not evidence of a conflicting executed tick. The authority claim must name the dispatch path, with a caller inventory for direct execute, configuration updates and any future backend. Keep the deferred array representation separate from correctness of today's scalar mapping.

CP-01/06/08 requires a versioned field/type/offset and feature-bit manifest tested by the actual host/device toolchains, negative same-size drift fixtures in CI, and loaded module identity matched to the host binary. For force acceptance, exercise zero → nonzero → zero constraint residency, runtime SSSP changes, scalar zero/positive boundaries and configuration changes through the actual actor/device path. Record the final uploaded word and observed force result; historical resident counts are not current receipts. Define coordinated release and rollback for every raw-copy consumer before growing the ABI.

## Wire identifier overflow coverage

Server, XR Rust and browser TypeScript agree on a 26-bit ID mask in the [inspected sources](evidence/wire-force-boundaries.json). Five typed setters log overflow and mask in release. The untyped encoder branch instead has only debug_assert followed by the unchanged ID, and to_wire_id_v2 is an identity function. A release build therefore does not gain the typed-setter warning by passing through that branch. This is source coverage evidence; no over-range live ID or rendered collision was observed.

ADR-2024 retains its scoped ephemeral wire-ID decision. CP-01/02/06/08 requires allocator and encoder boundary cases for each node class in debug and release, including the untyped branch, plus a defined reject/remap policy before any overflow can alias an existing ID or type bit. Check per-generation durable-to-wire mappings, reconnect/full/delta consistency and retirement of stale mappings in every client. Do not treat compact IDs as durable authority, or infer deployed capacity safety from agreement on a mask.

## PTX build acceptance and loaded artefact identity

The [isolated build probe](evidence/ptx-build-probe.py) executes the unchanged PTX phase extracted from the current build script, stopping before native compilation/linking. Synthetic nvcc output and temporary fallback files expose distinctions that ADR-2030 previously compressed into a compatibility guarantee. [Six cases](evidence/ptx-build-probe.json) establish the following:

| Fixture | PTX phase result | Closeout implication |
|---|---|---|
| nvcc absent, fallback files present | Panic before fallback | Missing executable differs from compiler failure |
| nvcc runs and fails, fallback present | All nine fallback modules processed | Fallback is reached on unsuccessful process exit |
| Successful fake compiler writes NOT PTX | Phase succeeds | Nonempty gate is not syntax or kernel validation |
| Successful fake compiler writes empty file | Panic | Empty-content rejection works |
| Existing .version 9.0 with newline | Unchanged content, nine downgrade warnings | Warning does not prove a version change |
| Invented future .version 9.10 | Rewritten to .version 9.00 | Fixed-width splice is not version-token parsing |

The last case is a synthetic future-format boundary, not evidence that an installed toolkit emits that version. None of these cases compiles CUDA, runs the native phase or loads a driver module. A header rewrite changes declared ISA; it does not prove all instructions are supported by the target driver. ADR-2030 becomes partial against its promised missing-toolchain fallback and build-time compatibility guarantee, while preserving the implemented rewrite and historical live declaration.

Runtime selection is another boundary. Current ptx_loader source uses a different, parsed header rewrite after loading. Its validate_ptx checks for three substrings (.version, .target and .entry), not full assembly semantics or the required symbol set. Ordinary selection prefers a baked path; Docker selection first tries precompiled candidates. Those candidates include an environment path, modification-time-sorted build outputs and source/image copies. File recency is not proof of source revision, ABI identity or release provenance. The constructor subsequently invokes Module::from_ptx, where driver rejection can still occur. This is source inspection, not an executed runtime selection or JIT receipt.

CP-01/06/08 requires separate tests for absent/failed compiler, invalid/missing/empty output, fallback provenance, native-versus-stub linking, exact version parsing and driver capability. Record source, toolchain, architecture, original/rewritten content hashes and the actually selected module for each release. Match required symbols and SimParams layout to the host, then run representative kernels and recovery on the intended driver. Define selection and rollback policy without relying on newest modification time. Keep the build-phase and runtime evidence separate; neither nonempty output nor a downgrade warning proves compatibility.

## XR control coverage and hierarchy semantics

The [current source and extracted-test receipt](evidence/xr-decision-probe.json) qualifies three remaining XR decisions. Desktop project settings still select XRBoot, gl_compatibility, zero 3D MSAA and disabled HDR 2D. The mobile renderer setting differs. Boot requires an available, initialised OpenXR interface before advancing. These source facts retain ADR-2032's scoped configuration decision; they do not establish today's driver version, stereo output, frame budget, exported package or Android build availability. Historical headset evidence remains historical.

HUD construction is less uniform than ADR-2033 claims. Eleven Button/CheckButton construction sites occur in hud.gd, but only the tab, action-button helper and type-toggle helper set press mode. Query Execute/Clear, swarm teleport, Join Room, Mute, Reconnect and scroll arrows omit that assignment. The count describes source constructors, not the number of rendered controls. ADR-2033 is partial against its every-control rule. No controller jitter or missed click was reproduced; runtime action-mode inspection and pointer-to-dispatch tests are still needed.

ADR-2035's predicate accepts hierarchical alongside explicit subclass labels. Its existing directed_hierarchy_relation_accepts_only_class_subsumption test still asserts that hierarchical is rejected, citing domain-membership reuse. The [probe](evidence/xr-decision-probe.py) compiles the unchanged predicate and that existing test with rustc: the test fails on hierarchical. This is a real test/implementation conflict in an extracted boundary, not a result from the full actor suite. The ADR's consequence claiming the source doc-comment is stale is itself stale: the present comment already agrees with acceptance.

The label decision retains scoped complete/live implementation, but current ingest provenance remains an acceptance question. A predicate match cannot establish that all collapsed edges represent subclass relations. Rank computation is shortest-depth multi-source BFS, with a seed for a wholly rootless hierarchy. It is a layout projection rather than proof that input is a DAG. Mixed rooted/disconnected cyclic components, edge direction, invalid indices and producer label changes belong in the acceptance set; no new rank-result assertion or GPU layout was executed here.

CP-01/02/06/08 requires an exported-client manifest and stereo/frame/input receipts on the intended headset/runtime. Inventory every ray-driven control and verify press semantics, disabled controls, drag-off, reconnect and duplicate-dispatch handling. Reconcile the hierarchy test with a ratified producer-label contract backed by actual ingest fixtures, then check rank buffers and resulting layout. Preserve mobile and desktop as separately accepted targets. A Rust library or extracted predicate test does not cover Godot scene integration.

## GPU supervision and context delivery

The [current topology receipt](evidence/crate-supervision-snapshot.json) confirms GPUManagerActor creates Resource, Physics, Analytics and GraphAnalytics supervisors. Context delivery differs from ADR-2007's broadcast-only description: the manager registers supervisor addresses with ResourceSupervisor, which sends SetSharedGPUContext directly to each and then publishes to the bus for additional subscribers. A search of current src finds the GetContextBus definition/handler/re-export, but no external subscription caller for these supervisors. Presence of the bus does not establish adoption by the four-way topology.

ResourceSupervisor discards results from these direct try_send calls, logs context as sent, and clears pending graph data after distribution. A successful send would establish queueing, not child readiness; a discarded failure supplies neither. No mailbox failure was injected here. The bus publishes availability events rather than retaining a last-value register, so a later subscriber needs its own bootstrap contract. Its subscriber_count increases on subscribe without tracking drops; has_subscribers consults the actual sender receiver count separately. Those metrics must not be treated as equivalent health signals.

Physics uses an AllForOne restart path because its children share GPU state. The implementation drops stored child addresses and creates replacements. That alone is insufficient evidence that all old children have terminated when other addresses or activity may remain. Independent actor supervision also does not establish isolation from a shared device/context failure. ADR-2007 is partial against broadcast-only distribution and guaranteed failure isolation, while the four-supervisor structure remains implemented.

CP-01/06/08 requires context-generation identity, acknowledged delivery/readiness, retry or reconciliation after failed sends, and late-subscriber/restart bootstrap. Inject child failure, supervisor failure, mailbox saturation and invalidated context; verify old actors stop, replacements receive current graph/residency state and unrelated services behave according to the declared isolation policy. Distinguish actor restart, device recovery and healthy results. No crash or device recovery ran in this pass.
