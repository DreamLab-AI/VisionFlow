---
title: Codescape presence lifecycle and concurrent XR connections
status: in-progress
date: 2026-10-31
type: explanation
---

# Codescape presence lifecycle and concurrent XR connections

Codescape's presence endpoint relays avatars, but replacement does not immediately revoke the old membership's ability to relay. Its bounded outbound queue also drops roster transitions alongside poses. Separately, the current XR scene has two concurrent content connections and gates Codescape drag locally. The baseline's “no avatars” statement and ADR-2159's “nothing built” verification are stale.

Following the [extraction and command-policy review](2026-10-31-codescape-boundaries.md), this continuation assesses inherited ADR-2143 and ADR-2159 section by section. It reads Codescape at `58108c0357bd0dbf6588f8b6adf6e044f4c04556`, its consumed checkout at `e3b76f63e801c6a8460110b93e88a0b420e9b0b4`, and VisionClaw at `75b72416bac176ca8fde715cac3bd7c77657b3e7`. The [source receipt](evidence/2026-10-31/codescape-presence-sources.json) verifies that inspected files match their commits and that the presence, authentication, proxy and baseline files are byte-identical between Codescape checkouts. This is source and local-test evidence, not a current deployment observation.

## Reproduced presence findings

The [Rust probe](evidence/2026-10-31/codescape-presence-probe.rs) links the freshly built production library. Its [twelve cases](evidence/2026-10-31/codescape-presence-probe.json) include four counterexamples arising from two underlying defects, seven passing controls and one documented configuration boundary. The [runner](evidence/2026-10-31/codescape-presence-run.cjs) builds offline, selects the runtime library artefacts, runs assertions and removes the executable. It does not start a service or contact a model.

| Finding | Production behaviour and probe | Practical boundary |
|---|---|---|
| Replaced membership can still relay | `xr_presence.rs:564` removes the matching roster entry and queues Close. `Membership::relay` at `:668` checks only its stored room and avatar, then broadcasts without checking its current local-id membership. The old handle's pose reaches both the observer and its replacement, carrying the removed local id. | Revocation and socket shutdown are separate events. The route can select inbound data before consuming Close. This is a deterministic library counterexample, not a measured live-socket race or proof of permanent connection survival. |
| Full queue loses join | The probe fills the observer's 64-message queue with poses admitted by the real rate limiter at synthetic 60 ms intervals over 3.78 seconds. Joining another member drops its announcement; after draining, the observer receives the new member's pose without its identity mapping. | The 20-pose/s limit does not prevent a slow reader's queue filling. The initial roster is not refreshed when a later event is lost. |
| Full queue loses leave | `Inner::send_all` uses `try_send` for text and binary alike (`:521-525`). Dropping a member removes it from the hub but cannot enqueue AvatarLeft for the saturated observer. | The server roster is correct while the consumer has no removal notification. This probe does not measure how long any particular UI retains an avatar. |
| Full queue loses replacement Close | A saturated old receiver contains only poses; `try_send(Close)` fails. Removing its sender eventually closes the receiver. The still-held old membership nevertheless relays to two recipients. | Channel termination remains a cleanup path. The finding is missing immediate revocation and reliable control delivery, not an immortal socket. |

Passing controls establish that another avatar or room cannot be substituted into an existing handle's pose, dropping the old handle does not remove the replacement, the replacement relays normally, and the rate limiter refuses its twenty-first pose in a fixed window and resets after one second. Existing tests check replacement notification and safe drop, but do not attempt `old.relay` after replacement. Their queue-bound test checks capacity, not whether a lost control message leaves client state inconsistent.

Before relay, the route decodes finite poses and applies its rate limit before calling the hub (`routes/presence.rs:175-181`). It multiplexes inbound frames and outbound notifications with `tokio::select!` (`:170,198`). The code supports the window described above; the audit does not claim a particular scheduler ordering in a running deployment.

On the inspected XR consumer, Joined and AvatarJoined populate `local_to_avatar`; a pose with an unmapped id falls back to its decimal representation (`xr-client/rust/src/presence.rs:591-636`). That establishes a mapping gap, not a tested visual artefact. More importantly, the default multi-volume scene derives presence from the VisionClaw base (`graph_scene.gd:2424-2428`). It does not join Codescape presence merely because the Codescape graph is visible. The server findings therefore concern consumers of that endpoint and should not be described as an observed defect in the current default headset session.

Before accepting reliable shared presence, Codescape needs an assertion that a replaced local id cannot relay, and an explicit policy for roster transitions under backpressure. Candidate policies include a separate control path or closing/resynchronising a reader whose roster event cannot be delivered. Acceptance should test join, leave and replacement while the pose queue is full, and verify the resulting consumer roster. These are proposed repair criteria; product code is unchanged.

## Authentication and proxy boundaries

`build_verifier(true)` refuses startup because NIP-98 is not implemented; open mode returns no principal for HTTP or WebSocket envelopes (`auth.rs:124-146`). The presence challenge's signature is deliberately ignored (`routes/presence.rs:98`), and a claimed DID is a label. ADR-2143 A4 records the owner's open-auth dev-LAN waiver. No new authentication bypass is inferred from exercising that documented mode.

A separate token guard protects the ingest listener. The probe confirms that missing and wrong tokens fail and the fixture token passes. Proxy checks confirm that disabled trust ignores forwarded headers, a configured IP peer restriction ignores unmatched peers, an unknown peer does not match that restriction, and trusted X-Forwarded-For selects the last supplied address. Conversely, enabling trust with an empty peer list trusts any peer; this is an explicit configuration contract (`proxy.rs:267-280`). Host Compose sets both trust and its peer restriction (`docker-compose.unified.yml:514-518`). Its nginx location appends the remote address and overwrites forwarded host (`nginx.dev.conf:74-86`). DNS refresh, live peer identity and multi-proxy deployments were not exercised.

OriginPolicy permits absent Origin and configured/default loopback clients, while rejecting foreign and null origins in the probe. It is a browser-origin filter, not client authentication. The source configuration publishes the client port on loopback by default, leaves ingest off the published ports, and provides the separate dev-nginx route. No live port or mount inspection was performed, so configuration is not promoted to an isolation acceptance result.

## ADR-2143 section dispositions

ADR-2143 remains accepted, partially implemented and inactive. Its history includes superseding extraction and XR changes. These dispositions preserve that distinction rather than advancing the owner's activation state.

| Section | Assessment against source and evidence |
|---|---|
| Metadata and Context | Historical separate-workspace rationale remains intelligible. The current standalone repository and host gitlink supersede the original in-tree location. Metadata's partial/inactive status is consistent with outstanding deployment and performance gates, even though some implementation prose lags. |
| Decision 1–2: workspace and crate boundaries | Five workspace crates remain separated by role. Extraction is now governed by Codescape ADR-0001; the previous pass verified dependency isolation, including its negative control. The old `crates/codescape/` path is historical. |
| Decision 3: ports | Config and Compose distinguish client 4100 from bearer-protected ingest 4101. Non-publication is not proof of network reachability, egress enforcement or live configuration. |
| Decision 4 and 6: wire and field mapping | The preceding model/collaboration suite checks shared wire fixtures and flags. This pass does not rerun captured live XR wire tests or validate every desktop renderer mapping. The current presence codec is implemented; no new position tag or flag is inferred. |
| Decision 5: interned ids | Previous fresh model tests cover the persisted interner guards. Presence's transient local ids are a separate namespace; their replacement defect does not establish failure of the graph interner. |
| Decision 7: GPU | Runtime PTX replaced build-time coupling in A5. The preceding pass reproduced retained loader-valid artefacts after a missing-source build path announced CPU fallback. Historical GPU conformance and current loader acceptance are different evidence. |
| Decision 8: four isolation layers | Build checks are executable. A2 itself rejects egress/mount isolation as established invariants. Current endpoint policy also has the explicit-endpoint exception documented in the previous pass. Source mounts and allowlists alone do not close the operational isolation gate. |
| Decision 9: voice and dispatch | Typed commands and bounded Jev selection exist. Previous probes show exact-one confidence can select a view command. Full audio, floor contention, signed dispatch and P6 were not established by these library checks. |
| Invariants I1, I6, I8, I10 | The previous pass freshly verified dependency, own-workspace, boot-refusal and no-VisionClaw-git checks. The inherited I10 “not enforced yet” row is stale relative to `tests/isolation.rs:215`. The extraction pack is the current source of these checks. |
| Invariants I2–I5 | Wire/flag/ack/id checks are covered by the preceding model suite, with its stated consumer limits. Historical fixture paths and A2 verdicts should remain dated rather than masquerading as current paths. |
| I7 and I7a | ABI/provenance checks exist; opt-in device conformance is correctly treated as a required run, not an always-enforced invariant. No new GPU acceptance is claimed. |
| I8a and I9 | I8a remains an explicit rejected isolation claim. I9's client import graph has historical tests in the close-out; the TypeScript suite was not rerun in this pass. |
| Consequences | Independent dependency resolution is supported; live rendering and frame-rate benefits remain bounded by their dated receipts. Forbidden knowledge routes and server ownership do not authenticate presence members. |
| Alternatives | Historical architectural tradeoffs are preserved. This pass does not validate complexity estimates, historic repository size or every rejected layout technique. |
| Phase plan P0–P2 | Fresh pure/build checks from the prior pass supplement dated close-out evidence. GPU numerical conformance does not close step time, settle time, mount isolation or edit-to-visible acceptance. |
| Phase plan P3–P6 | P3 ingest and scale, P4 complete XR/hover behaviour, P5 live audio/floor and P6 authenticated publication/dispatch retain distinct gates. P4 already explicitly records the presence shim; VC-39's contrary reading was wrong even at its old pin and is corrected. |
| A1: layout thresholds and ABI | Defines historical numeric acceptance thresholds. No GPU rerun here; the recorded latency/settling misses are not erased by passing numerical overlap. |
| A2: cross-vendor audit | Its distinction between unit checks and deployed guarantees remains necessary. Fixes to hostile counts, indices and test fixtures have source/tests; “four tested layers” in the earlier decision remains too broad without A2's qualifications. |
| A3: token readability and opt-in | Historical capability/readability change is separate from this pass's successful token comparison checks. Later sidecar-menu selection supersedes the old environment invocation, as the record itself acknowledges. No container permission test was run. |
| A4: dev nginx and trust | Source and proxy controls agree with the declared trust model. Open LAN publication is an explicit dev waiver. The handshake-only description is now incomplete: the hub carries roster events and poses. The old XR mode toggle is superseded by concurrent connections. |
| A5: extraction | The prior pass assessed the operative ADR-0001 and both checkouts. This inherited amendment still has stale paths and I10 wording; separate-repository status does not settle its original operational acceptance gaps. |
| Verification and October 19–20 rechecks | These are historical observations at their named pins, not current service evidence. Their partial/inactive disposition is retained. |
| A6: capability gating and switch hygiene | Capability gating survives but belongs to connections and volumes. The exclusive switch is gone. Codescape drag is now gated before transmission, superseding A6's explicit ungated-drag limitation. Palette equivalence was not freshly re-audited here. |
| October 30 re-verifications | Jev and estate mounts are explicit changes; an allowlisted proxy is not an independent locality guarantee. Annotation metadata is still separate from authorising or reading the knowledge graph. Full code-to-concept link correctness remains a later audit. |
| October 31 close-out | Historical P0/P1/P2 receipts remain valid only for their recorded revisions/environments. The prior pass found seven ignored tests in the raw log, not six CUDA-device tests. Partial/inactive still reflects outstanding scope. |

## ADR-2159 section dispositions

| Section | Assessment |
|---|---|
| Metadata and Context | Accepted/none/inactive and the old global-switch description are historical. Current source contradicts “none” and “No BackendConnection exists”; this review records drift without declaring headset activation. |
| Decision 1: per connection | `BackendConnection.make` creates its own client; scene construction instantiates VisionClaw and Codescape separately (`backend_connection.gd:57-70`, `graph_scene.gd:1282-1293`). Pins, HTTP ownership, retry state and generation are instance fields. Federated node/dataspace isolation is outside this local two-source check. |
| Decision 2: per volume | Three sources bind to two connections. Focused controls consult that volume's connection; drag start/update/end gate before client sends. Source GUT cases assert no exclusive switch, fold refusal and no Codescape drag. They were inspected, not executed here. |
| Decision 3: presence | The scene uses VisionClaw's base for its room socket. Codescape graph visibility does not cause a Codescape presence join. Later bound-room/federated admission is not established by this path. |
| Decision 4: budget | `frame_budget_pass.gd:125-137` allocates once and splits volume caps; `graph_scene.gd:1506-1532` clears hidden/offline instances and packs visible volumes. Source does not establish measured 90 Hz performance. |
| Decision 5: preflight | `two_clients.rs` exists and drives the production transport and freshness/ack logic against loopback servers. It substitutes a test-thread pump for Godot and excludes topology-receiving ticks from its normal pump timing gate. Render-store loading and headset work are explicitly outside its measurement. The attempted execution did not reach assertions because the linker blocked on filesystem writeback; details are below. |
| Rejected alternatives | The implemented arrangement follows concurrent connections, avoiding the proposed exclusive switch and shared node-id store. This pass does not independently benchmark a multi-process alternative. |
| Consequences | Independent reconnect/pin/request state is present. Cost on the headset, source-loss behaviour in the engine and complete per-volume control coverage still require engine/runtime acceptance. Cross-connection concept links are a separate ADR-2164 review. |
| Activation prerequisites | Source is wired into the scene, so “nothing built until preflight” is no longer a current-state description. This pass does not reconstruct commit ordering to prove that historical build sequencing was followed. E8 remains necessary; a transport test cannot substitute for it. |
| Verification and dated rereads | The pinned assertions were historical. They need a new dated implementation assessment distinguishing the transport preflight, GUT integration and headset measurement, rather than a blanket promotion to live. |

## Verification and coverage

Fresh focused Codescape tests passed **32 cases**: nine presence, twelve proxy, five origin and six authentication tests. [Raw output](evidence/2026-10-31/codescape-presence-suite.txt). Twelve additional audit cases passed their assertions, including the reproduced failure behaviours. These extend the preceding 189-test receipt rather than replacing its scope.

The Rust XR transport attempt, `cargo test --offline --locked --test two_clients -- --nocapture` from `project/xr-client/rust`, **did not complete**. Its [raw output](evidence/2026-10-31/xr-two-clients-suite.txt) reaches compilation only. The [attempt receipt](evidence/2026-10-31/xr-two-clients-attempt.json) records linker workers waiting in `balance_dirty_pages_ratelimited`, container I/O pressure above 90%, and interruption of only this audit's process group (exit 130). No test assertion result was obtained and the older cached test binary was not used. This is an environment-limited verification attempt, not evidence of a transport regression. Re-run it when filesystem writeback recovers. Godot is unavailable on this container's PATH, so no GUT, actual gdext instance, visual rendering or headset performance result is claimed.

[VC-43](../diagrams/visionclaw/43-codescape-presence-lifecycle.md) maps admission, replacement and queue loss. [VC-44](../diagrams/visionclaw/44-xr-concurrent-backend-connections.md) maps the current two-connection scene. VC-39 corrects a false statement about the historical P4 row; VC-40 explicitly retains its old switch diagrams as history and links to the current arrangement. Other historical claims are not restamped. Product code and owner ADR statuses remain unchanged.

All 16 diagrams in the four affected topics rendered with no citation warnings. The [full citation gate](evidence/2026-10-31/codescape-presence-citations.txt) passed for 165 topics and 1,548 diagrams: 24,025 revision-bound references, zero working-tree fallbacks or unresolved references, and nine explicitly allowlisted generated-file references. The [verification check](evidence/2026-10-31/codescape-presence-verification.txt) is fresh for all 39 declared revisions. These structural checks do not establish semantic review of the whole corpus.

This closes a bounded reading pass over ADR-2143 and ADR-2159, with explicit unresolved acceptance items. It does not close Codescape's full client/runtime audit or the wider estate queue. Remaining work includes reliable presence repair, actual engine and headset acceptance, P3/P5/P6, mounted-data and network isolation, current GPU/deployment evidence, and the spatial-room/federation/link decisions and their consumers.
