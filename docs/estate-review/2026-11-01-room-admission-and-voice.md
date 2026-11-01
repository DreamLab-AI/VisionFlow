---
title: Room admission and deferred voice
status: in-progress
date: 2026-11-01
type: explanation
---

# Room admission and deferred voice

Room membership enforcement remains proposed. The production presence actor accepts 33 members, and the current parser rejects the proposed NIP-29 room identifier. The identifier's published golden vector is correct. The more serious design finding is that ADR-2160's 150-second voice revocation argument omits tokens refreshed by the SFU itself. Voice remains explicitly deferred by the operator; this audit does not activate it.

This continues the [portal and federation review](2026-10-31-portals-and-federation.md). ADR-2157 and ADR-2160 were read in full, with their governing identifier, authority, protocol and XR documentation and the current presence/voice source. The [source receipt](evidence/2026-11-01/room-source-receipt.json) pins VisionClaw to `75b72416bac176ca8fde715cac3bd7c77657b3e7`. The forum dependency chain and agentbox federation signer still require their own complete assessments. No production code, owner ADR status, server or voice configuration was changed.

## Executed evidence and its limits

The [reproduction script](evidence/2026-11-01/room-audit-run.sh) builds isolated copies of the shared presence crate, the actual Actix presence actor and the XR voice-state module. Production source is copied unchanged. The actor's existing tests and voice core compile under `cfg(test)`, without the Godot wrapper, full CUDA backend, WebSocket handler, real identity verifier, microphone or SFU. Dependency locks and the fixture manifest are retained beside the script. Offline reproduction requires the recorded dependencies in the Cargo cache.

**88 presence-library tests passed** (40 unit, 24 adversarial, 9 aggregate/validator integration and 15 property tests); **29 actor/voice tests passed** (27 existing tests plus two audit tests). The [library output](evidence/2026-11-01/room-presence-suite.txt), [actor output](evidence/2026-11-01/room-actor-voice-suite.txt) and [exit receipt](evidence/2026-11-01/room-test-exits.txt) record those results; text logs have trailing blank lines removed. An initial attempt using the shared target directory waited on another build and was cancelled; its [lock log](evidence/2026-11-01/room-shared-target-lock.txt) is not a test result. The isolated runner avoids that lock.

Two [audit tests](evidence/2026-11-01/room-audit.rs) emit six observations:

| Case | Result | Interpretation |
|---|---|---|
| Send 33 distinct identities through the actual actor's `JoinRoom` handler | All 33 appear in `ListMembers` | The proposed default cap of 32 is absent. This bypasses the network handshake in the fixture and is not an authentication bypass. |
| Join the same DID again | `DuplicateMember` | Existing duplicate control works. |
| Leave and rejoin | Member count returns to 33 | Removal/re-entry control works. |
| Hash the specified domain, raw home key, big-endian length and `general` with RustCrypto SHA-256 | Exact published `nip29-e5488c85b92647026b5d54e422122a6f`; domain is 24 bytes | Positive control for the proposal, not evidence of a production constructor. |
| Parse that proposed room URN with the production parser | Rejected | Parser support has not landed. |
| Parse equivalent upper/lower-case legacy hexadecimal suffixes | Distinct room strings and wire hashes | Existing legacy syntax does not canonicalise case; migration must choose and test its intended behaviour. This does not defeat a membership gate. |

`presence_handler.rs:210-305` checks the challenge, verifies the DID, checks replay state, parses the room and sends `JoinRoom`. `presence_actor.rs:664-686` delegates to `PresenceRoom::join`, whose duplicate check is its admission restriction (`room.rs:42-62`). The current port module exposes `IdentityVerifier` and `Broadcaster`, with no membership port. The [scoped search](evidence/2026-11-01/room-source-search.json) found none of the proposed admission, mint, revocation or room-notification symbols in `src`, `crates` and `xr-client/rust/src`. These are source observations, not a deployed-session test.

`SpatialVoiceRouterCore` stores track positions, mute flags and the listener transform (`webrtc_audio.rs:43-131`). Its tests establish state management and concurrent attachment. They do not exercise HRTF, decode audio or deliver a media track. Configuration fields and `AudioRouter`'s participant-id setter likewise do not establish a working token issuer or call.

## Voice deadline: a missing issuer in the argument

ADR-2160 D3 assumes every refresh passes through `RoomAdmission`; D6 concludes that no reusable credential survives 150 seconds after home removal. Current official LiveKit documentation says the server refreshes connected clients' tokens for at least ten minutes, and self-hosted removal does not revoke cached tokens. This is a dependency-contract conflict, rather than evidence that deferred VisionClaw voice is running. [LiveKit token lifecycle](https://docs.livekit.io/frontends/reference/tokens-grants/).

At upstream commit `9a26932591fc466b06664c634983a462c0a8af15`, `rtcSessionWorker` refreshes on session start and every five minutes. `refreshToken` signs with the SFU key, preserves grants and chooses the greater of ten minutes or the original remaining lifetime. It has no VisionClaw admission call. [Pinned source, lines 777–789 and 1159–1187](https://github.com/livekit/livekit/blob/9a26932591fc466b06664c634983a462c0a8af15/pkg/service/roommanager.go#L1159).

A source-derived counterexample is: join at t=0 with a 60-second application token; receive the SFU's token expiring at t=600; remove membership at t=1 and successfully kick at t=10; attempt a new join at t=160 with the cached SFU token. Its expiry alone cannot enforce the promised t=151 cutoff. A reconciler may kick again, but that is different from making rejoin impossible. This is an analytical counterexample, not an executed SFU exploit or a measured revocation delay.

The [deployment search](evidence/2026-11-01/room-deployment-search.json) found no LiveKit image declaration in tracked compose/Dockerfile paths. It does not establish absence of an externally managed SFU or identify a deployed version. Before lifting the deferral, pin the chosen server and SDK, enumerate every issuer/refresh path, and test rejection of a captured SFU-refreshed token after removal. Revise the design around enforcement at the SFU admission boundary, or a separately assessed hosting/revocation contract. A lock around VisionClaw's mint alone cannot govern another issuer.

## ADR-2157 section dispositions

| Section | Assessment and remaining acceptance |
|---|---|
| Status and context | Proposed, no implementation, inactive is consistent with current source. Existing nonce authentication establishes identity, not NIP-29 membership. The old draft pin is historical. |
| D1: identifier bytes | Golden vector passes. Test lower-case grammar, invalid group lengths, raw-key decoding and key/group changes when the real constructor lands. |
| D2: single mint and registry | No constructor or binding registry found; production parsing refuses the new form. Test a syntactically valid but unregistered URN, and persistence/recovery of the authoritative binding. |
| D3: pinned home authority | Distinguishing the signing key from a discovered administrator key is sound. No inspected adapter enforces this proposed chain; wrong-key and relay-host rename tests remain. |
| D4: freshness and revocation | L=60, S=10 and C=10 are proposed requirements. No executable lease/recheck mechanism was found. Test future dating, both clock extremes, lost updates and the join-registration race before claiming D=90. Its separate voice-bound claim inherits the issuer gap above. |
| D5: revisions and watermarks | Persisted revision/content/heartbeat ordering is not implemented here. Specify canonical encoding precisely, then test equivocation, equal-revision heartbeats, rollback, restart and exact-event rehydration. A fresh signature must not authorise stale membership. |
| D6: fork versus rename | Key/group identity is consistent with the vector. A changed key requires explicit new-room admission; transport-host rename must preserve identity. No migration path was exercised. |
| D7: legacy rooms and activation flag | Legacy parsing exists; the proposed cap and `bound_only` flag do not. Keep presence-only legacy semantics explicit and test default-room migration before P2. |
| D8: admission port and post-registration check | Port absent. Tests must schedule removal between initial approval and registration, then prove cleanup and no authorised output after failed recheck. |
| D9: atomic cap | The actual actor accepts 33 members. Add a cap inside the join mutation; competing joins at 31/32 and failed-join cleanup require tests. |
| D10: persistent authenticated relay adapter | Required NIP-42 read grant and pinned-home subscription are not evidenced by the current presence path. Prove the bridge grant can read the needed membership projection without becoming a member or gaining write authority. |
| D11: room notifications | Proposed `room_kicked`/`room_meta` symbols absent from searched code. Specify client handling, reason disclosure and ordering relative to stream closure. |
| D12: cache versus durable watermark | Source has no proposed durable anti-replay store. Test cold start with unavailable relay, missing/corrupt watermark and crash after accepting a revision. |
| D13: cohort freshness | Depends on the home's coherent projection; local signer liveness cannot establish it. Forum lag and signing-stop tests remain in that repository's assessment. |
| D14: scope | Enterprise delegation and Codescape are excluded. Existing Codescape membership defects remain separate; sharing a codec does not adopt this policy. |
| Rejected alternatives | The rejected legacy hash, team-membership URN, first-use discovery trust, move alias, fail-open cache, new binary opcode and NIP-42 presence handshake each address a different concern. Their alternatives are specified, but only legacy identity/room handling currently runs. |
| Consequences | Registry, watermark durability, relay credentials, admission/recheck scheduling and caps need operational ownership. The simple legacy join remains a distinct current mode. |
| Activation prerequisites | Forum ADR-2030/2031, the complete revocation/race suite and the P2 migration remain open. Voice is independently deferred. |
| Verification | Keep implementation `none`. The passing vector and present-state probes are new audit evidence, not feature acceptance or an owner-status restamp. |

## ADR-2160 section dispositions

| Section | Assessment and remaining acceptance |
|---|---|
| Status, context and operator deferral | Preserve proposed/none/inactive and the explicit prohibition on voice activation. The inspected module manages state; its header's transport/routing language is broader than executable media behaviour. Research claims about other SFUs were not independently re-audited here. |
| D1: room mapping | No fixed LiveKit room-name mapping/vector found. Define encoding and collision behaviour, including length and character constraints. |
| D2: one credential path | No VisionClaw mint found. The complete issuer inventory must include the dependency path discussed above. Unbound and unregistered room refusal remain required. |
| D3: token shape and SFU custody | No grants/custody implementation to test. Resolve the refresh conflict above before treating the TTL as a system-wide bound. Test role changes, cross-room use and per-SFU isolation at the selected version. |
| D4: discovery | No validated 30312-to-config endpoint binding in this path. Test wrong signer, wrong group, stale revision, URL mismatch and withdrawal after a call starts. |
| D5: spatialisation | Existing state tests do not establish HRTF, pose-to-audio timing, missing-speaker behaviour or media delivery. Require real client/SDK integration and measured headset acceptance. |
| D6: voluntary leave | Persisted 25-second grace is proposed only. Test restart before/after the deadline, reconnect cancellation and conservative recovery. |
| D6: removal, reconciler and deadline | No server-API client or reconciler found. The analytical counterexample above invalidates the current justification for an unconditional 150-second cutoff. A passing application-mint test would be insufficient. |
| D6: serialisation and partition | Test mint/removal races across every participating issuer. API failure needs fail-closed minting, recovery cleanup and explicit operator acceptance of the connected-participant exception. None was exercised live. |
| D7: signalling | Keeping SDP/ICE off public relay events remains a design requirement; no end-to-end transport/privacy capture was performed. |
| D8: server choice | LiveKit remains the proposed dependency. Replacing it with a Rust SFU would add room, forwarding and operational work and does not itself repair authority semantics. |
| D9: scope | VisionClaw only. Do not infer Codescape voice acceptance from these state tests. |
| Rejected alternatives | The rejection of expiry-only removal is sound but incomplete without issuer analysis. The stated Nostr challenge, custom SFU, relay signalling, immediate voluntary drop and audience alternatives remain design trade-offs, not tested comparisons. |
| Consequences and counterexamples | Keep all E10 blockers: unauthorised mint, excess grants/TTL, kick/grace confusion, restart, partition, reused token and discovery mismatch. Add a captured dependency-refreshed token and repeat join after the claimed bound. |
| Revisit/follow-on | Hosting, licensing, call independence and E2EE remain review triggers. Their future adoption requires a new source and acceptance review. |
| Activation prerequisites | Recorded lifting of the deferral; ADR-2157 and `space:` federation; forum 2030→2031→2032; pinned SFU tests; partition-risk sign-off. No prerequisite is closed by this audit. |
| Verification | `none` remains accurate. Fresh state tests narrow uncertainty about existing code while leaving media, authority and revocation acceptance open. |

## User utility and next acceptance work

For a user, “leave this room”, “remove this member” and “ban this member” need visibly different, dependable outcomes. A published numerical promise should cover reconnection credentials as well as the currently connected session. The existing LiveKit ecosystem supplies media and client integration; VisionClaw's proposed value is aligning those sessions with one space membership and shared spatial context. That value depends on the missing authority integration, rather than on configuration fields or a voice-state map.

After an explicit lifting of the deferral, run one pinned end-to-end scenario with two clients: admitted join, spoken audio at tracked positions, voluntary reconnect, removal while connected, replay of every captured credential, server restart and API partition/recovery. Record home-removal time, last audible sample, each rejoin decision and each issuing component. Until then, implement and accept membership independently, without coupling non-voice spaces to the voice rollout.

[VC-49 admission diagrams](../diagrams/visionclaw/49-room-admission.md) and [VC-50 voice boundary diagrams](../diagrams/visionclaw/50-deferred-space-voice.md) distinguish current code from proposed controls. This pass adds two complete ADR reading dispositions to the estate review. It does not complete the whole-repository, all-ADR or all-diagram audit.

## Documentation verification

All 22 diagrams in the four touched topics render with strict citations; six new diagrams received visual inspection. The whole-corpus citation pass covers 171 topics and 1,566 diagrams: 24,235 references verified at declared revisions, zero working-tree fallbacks, zero unresolved references and nine explicitly allowlisted generated-input references. The final verification gate recognises all 39 declared revisions. This validates structure and citation resolution, not the semantics of every estate diagram.

[Gate output](evidence/2026-11-01/room-final-gate.txt), [citation receipt](evidence/2026-11-01/room-corpus-citations.txt), [122 relative-link checks](evidence/2026-11-01/room-link-check.json) and [retained render hashes](evidence/2026-11-01/room-artifact-retention.json) accompany the report. Rust formatting, shell syntax and diff-whitespace checks passed. Prose and invisible-character checks found no high-severity prose flags or suspicious invisible characters in this report. Existing historical topic pins were retained; only the new topics claim this pass's source revision.
