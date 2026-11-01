---
title: Forum spaces, replication and rollback boundaries
status: in-progress
date: 2026-11-01
type: explanation
---

# Forum spaces, replication and rollback boundaries

Forum source still advertises NIP-29 without a group engine. Current storage retains late older and same-second events; its timestamp-based deletion can also erase a higher revision required by the proposed federation design. A further design gap needs resolution before implementation: ADR-2030's off switch would disable read gating while retaining group data under today's open group-read rules.

This pass reads ADR-2030, ADR-2031 and ADR-2032 in full against their governing documentation and current source at forum commit `c39f41f104b114d6a3fa29013daaf6fa58b45080`. It continues the [room admission and deferred-voice audit](2026-11-01-room-admission-and-voice.md). The [source receipt](evidence/2026-11-01/forum-source-receipt.json) records hashes and confirms the inspected files match that revision. The forum checkout is unchanged. Proposed/none/inactive statuses and the operator's voice deferral remain intact; neither local tests nor a protocol advertisement establish deployment acceptance.

## What was executed

**69 tests passed**: 61 existing tests in unchanged copies of the relay's `filter.rs` and `nip42.rs`, plus eight audit tests producing twelve observations. The fixture links the production `nostr-bbs-core`, verifies signatures from both real metadata builders, and runs extracted production storage/schema/query SQL against an in-memory SQLite database. A declaration-only `NostrRelayDO` permits compilation of the pure filter module; no Worker methods, environment, network or live database are substituted and claimed to work. The existing tests exercise the copied pure module functions.

[Reproduction script](evidence/2026-11-01/forum-spaces-run.sh), [audit tests](evidence/2026-11-01/forum-spaces-audit.rs), [SQL extraction](evidence/2026-11-01/forum-storage-sql.json), [test output](evidence/2026-11-01/forum-spaces-suite.txt) and [exit receipt](evidence/2026-11-01/forum-test-exits.txt) make the boundary explicit. The runner checks the extracted statements against the retained SQL and uses a locked, offline dependency graph. Offline repetition requires those dependencies in the Cargo cache. Logs have trailing blank lines removed.

| Probe | Observed result | What it establishes |
|---|---|---|
| Anonymous group read predicates in both auth modes | Kinds 9 and 39000–39003 pass `protected_read_permitted`; none is a zone channel | Current pure read gates contain no group membership protection. Anonymous kind-4 correspondence is denied as a control. |
| Both published metadata builders with the same key/group | Both signatures verify; `create_group_metadata` produces `h` and no `d`; `build_group_metadata` produces `d` and no `h` | Inconsistent public APIs. The current handler requires `h` for these admin kinds, so the standard-shaped builder fails that gate by source trace. |
| Newer row followed by older row, kinds 10312 and 39000 | Both survive and both are returned by the extracted untagged read query | The gap extends beyond retention to query output. Worker delivery was traced, not executed. |
| Same-second ids `ff` and `00`, kinds 10312 and 39000 | Both survive | No single-winner tie resolution in the current replacement statements. |
| Revision 9 at t=100, then revision 8 at t=200 | Revision 9 is deleted for 39002, 30312 and `30078 / visionclaw:dataspace:one` | The proposed revision-aware families need storage changes before consumption. This is a SQL prerequisite test, not proof these events pass current admission. |
| Newer event at address `a`, with another event at `b` | New `a` replaces old `a`; `b` survives; repeated id adds no row | Positive controls for timestamp replacement, address isolation and id deduplication. |
| Two legacy-builder group addresses, fixture timestamps 100 and 101 | Both addresses are empty; the second replaces the first | The old `h` metadata builder can collapse different groups into one address under the real storage SQL. Fixture timestamps and SQL rows were supplied directly; this is not a network exploit. |
| `/public/%2e%2e/admin` resolved by the URL library | Becomes `/admin`, despite having no literal `.` or `..` segment | Raw path inspection is insufficient to enforce ADR-2032's intended path restriction. No production descriptor validator exists to test. |

Read requests follow `handle_req` → `query_events` → `authorize_event` → `send_event` (`nip_handlers.rs:1865-1895`). Storage orders by timestamp and does not collapse addresses (`storage.rs:312-325`). The shared authorisation path checks correspondence, calendar, kanban and channel zones, then returns `Deliver` (`nip_handlers.rs:2264-2372`); `h`/`d` group membership is absent. These observations explain the pure-predicate result without claiming a live socket test.

The [source search](evidence/2026-11-01/forum-source-search.json) finds no proposed engine, switches, publisher or claim namespace in the searched crate source. Its one numerical match is `10312` inside a DID test-vector hex string, not a handler. No `PeerConnector` implementation was found. The current alarm clears idle state (`relay_do/mod.rs:371-378`); it does not run the proposed membership heartbeat/recheck schedule.

## Findings that change the implementation order

**Rollback must preserve confidentiality.** ADR-2030 D12 says disabling groups stops gating immediately and serves retained events under today's rules. After a future private/hidden group has stored messages or membership lists, this would deliberately restore the unprotected path demonstrated above. The switch itself is absent, so this is a contradiction in the proposed rollback contract, not a reproduced feature-toggle exploit. Separate stopping new group activity from removing access controls: retain a deny/authorisation boundary for previously protected data, close existing subscriptions, and test REQ, COUNT, live broadcast, restart and an on→off transition. Do not use a feature rollback as an implicit declassification decision.

**Replacement requires a migration and an admission policy.** Both branches delete only rows with earlier timestamps and then insert by unique event id. They neither reject incoming older versions nor select a same-second winner. Revision tags are opaque JSON, with no indexed revision column. A later, lower-revision event can destroy a higher-revision snapshot before a consumer sees it. Implement the ADR-2031 revision families, ordinary-event ordering, equal-revision equivocation checks and durable per-kind watermarks together with transactional race tests. Current generic event handling is not a partial implementation of that design.

**The licence dependency invariant needs a feature scope.** The [default dependency tree](evidence/2026-11-01/forum-core-default-dependency-tree.txt) contains no AGPL-labelled normal dependency; the [tree with `did`](evidence/2026-11-01/forum-core-did-dependency-tree.txt) includes `solid-pod-rs 0.5.0-alpha.18`, whose manifest declares AGPL-3.0-only. The core manifest explicitly documents that optional boundary (`Cargo.toml:25-29`), whereas ADR-2030 D9 and the proposed baseline say core never depends on an AGPL crate. A default-feature CI command cannot prove that unqualified invariant. Reconcile the intended feature contract and test each supported graph. This is a manifest/CI consistency finding, not a legal conclusion about distribution obligations.

**Descriptor references need canonical parsing.** Define whether paths are validated before or after percent decoding and normalisation, then validate the path actually fetched. Include encoded dot segments, encoded separators, redirects and approved resource endpoints in the eventual validator tests. A same-origin reference is not itself authority to read every endpoint on that origin. The observed URL case stays on the same host; it does not demonstrate arbitrary-host fetching or an existing server traversal vulnerability.

**Voice timing remains a shared unresolved contract.** ADR-2031 D4 repeats the 150-second rejoin cutoff challenged by the preceding [LiveKit issuer analysis](2026-11-01-room-admission-and-voice.md#voice-deadline-a-missing-issuer-in-the-argument). Update the forum, VisionClaw ADRs and governing documents together when that argument is repaired. The independent 90-second presence requirement still needs its own clock, projection-failure and restart acceptance evidence.

## ADR-2030 section dispositions

| Section | Disposition and acceptance still required |
|---|---|
| Status and context | Proposed/none/inactive matches the source. NIP-11 still lists 29 (`nip11.rs:207`); the handler retains its `h`/admin TODO. Cohort membership and NIP-29 state remain separate mechanisms. |
| D1: pure group engine | No `nostr-bbs-groups` crate. Admission-order folding, roles, flags, joins/leaves, invite policy, timeline validation and signed snapshots all remain work. Prove native and wasm behaviour, not just builders. |
| D2: refuse client state | Current admin clients with `h` can reach the legacy gate; standard `d` alone fails it. Test signer authority independently of syntactic validity and the caller's admin role. |
| D3: cohort projection | No group projection found. Exercise every listed mutation: 9000 with/without cohort, 9001, 9005, 9009, 9021 and 9022. Verify routing/refusal, coherent `proj`, local denial on lag and signing cessation while network/signer stay healthy. |
| D4: read/write visibility | Current group read protection is absent. Add a flag matrix covering `private`, `hidden`, `restricted` and `closed`, including hidden 39000/39003 metadata and counts. The listed private 39002 test alone cannot prove all hidden-state behaviour. |
| D5: scoped bridge read grant | No grant mechanism found in this path. Test read without membership/write, grant revocation, open subscriptions and separately authorised bridge posting. |
| D6: dialect amendment | Proposed forum/room split needs implementation and a deployment-profile matrix. A generic kind-9 handler cannot demonstrate both dialects or deliberate rejection of `h` in room mode. ADR-2029's full separate assessment remains open. |
| D7: encryption boundary and deferral | Keep encrypted zones outside projection until the recorded decision changes. Relay-gated group messages must not be described as end-to-end encrypted. ADR-2016/2028 implementation is outside this pass. |
| D8: phase-zero fixes | All three remain open: advertisement, incompatible metadata API and feature-aware dependency gate. The old builder's deletion would affect a published API; correct/deprecate it with explicit compatibility handling. |
| D9: licence/provenance | Engine absent. No engine provenance/header gate to assess. Resolve the optional `did` dependency conflict before asserting the absolute core invariant. Preserve origin notices and the own/derived distinction if code is later imported. |
| D10: source disclosure | Proposed exact-build source link is stronger than a generic repository URL. No new deployment/source-link acceptance was performed; a field-existence test cannot prove exact deployed source. |
| D11: activation order | Forum comes after the earlier signer/transport work and before membership-dependent spaces. Public/DID federation remains independent. Applicable forum tests are still prerequisites, not completed gates. |
| D12: disabled rollout and reversal | Switch absent. Its written reversal rule conflicts with private-data protection; resolve the rollback finding above before implementation. Run on, off, on→off and off→on tests, including retained data and subscriptions. |
| Consequences | Cohort projection, authoritative signing, storage and schedules add new failure modes and cost. Bridge writes need separate membership. Zone-to-group binding and visibility derivation remain explicitly undecided. |
| Rejected alternatives | Whole-engine port, third dialect, duplicate member authority, AGPL engine in core and bridge membership solely for read access are distinct trade-offs. Partial code reuse does not establish protocol compliance or provenance. |
| Revisit triggers | Protocol changes, encrypted-group semantics, relicensing and upstream licence changes remain open triggers; this pass does not authorise any of them. |
| Verification and adoption list | Existing pure tests establish present behaviour only. Preserve `none`; add rollback and hidden-metadata cases to the complete admission, ordering, grant, projection, provenance and source-link suite already required. |

## ADR-2031 section dispositions

| Section | Disposition and acceptance still required |
|---|---|
| Status and context | Proposed/none/inactive remains supported. Mesh planning is not group replication; the outbound connector is still a trait. The SQL defects described by the ADR are now reproduced locally. |
| D1: pinned home identity | No group pin registry/discovery comparison implemented. Test host rename, key mismatch, explicit fork and refusal to adopt a discovered key. Reconcile current NIP-29 reference encoding as discussed below. |
| D2: sole writer and revision sequence | Durable admission order, snapshot coalescing, state-change revisions and identical-content heartbeats remain absent. Test clock rollback and same-second bursts separately from arrival order. Define canonical digest encoding. |
| D3: verbatim mirrors | No live mirror path or persistent watermark. Test forged home, replay after restart, per-kind independence, equal-revision equivocation and exact-event rehydration without treating it as new state. |
| D4: clock and live-session removal | Current alarm cannot provide the required schedule. Test L/S/C boundaries, withheld updates, registration/removal race and all joined sessions. Its voice exception inherits the unresolved issuer argument. |
| D5: signed transport | Existing envelope transport is a different path. Implement authenticated event transport without re-signing mirrored state; NIP-77 remains a later phase. |
| D6: key custody | No group signer is present. Keep home signing, admin contact, peer AUTH and agentbox node signing roles separate; test key substitution and exposure boundaries. No secrets were read. |
| D7: replacement and persistence prerequisites | Exact SQL counterexamples cover both ordinary branches and all three proposed revision-aware families. Migration, durable dedup/watermarks, connector, old-state admission exemption and group-aware peer gate remain open. Preserve ordinary ordering for other 30078 namespaces. |
| D8: fork, no move | Deliberate non-adoption of migration is clear. Test no inherited membership/history authority and inert `moved_to`; future moves require a separate two-home protocol. |
| D9: federation advertisement | Homed/mirrored discovery must remain absent until prerequisites pass. NIP-11 schema shape alone would not prove a working peer. |
| D10: coherent projection | A signed fresh event is insufficient if its source cohort is stale. Required healthy-signer/failed-projector test has not run; remote expiry must be measured from the authoritative removal. |
| D11: estate order | Keep public/DID federation independent and membership activation downstream. Clarify that the repeated step-3 30312 wording is conditional on the separately lifted voice deferral. |
| D12: separate federation switch | Switches absent. Federation-on/groups-off must refuse startup. Also specify the fate of already mirrored private data when federation is disabled; local-only mode cannot silently re-home it. |
| Consequences | Home ordering is a single authority per group. Alarm costs, key compromise/fork handling, local-whitelist versus remote-member admission and revision-aware consumer compatibility remain material limits. |
| Rejected alternatives | Hub authority, multi-writer state, mirror re-signing, gift-wrap replication, redirect aliases, timestamp-only ordering, timestamp bumping and refreshing stale projections are all rejected explicitly. Tests must establish the chosen alternative, not just absence of these options. |
| Archived rationale | Archive summaries are historical design lineage, not current implementation evidence. No new claim of archived-ADR activation or complete archive review is made. |
| Revisit triggers | Key compromise, protocol changes, moves, clock constants and scope changes still require coordinated review. |
| Verification and adoption list | No two-node, Worker restart, signing cadence or live revocation test ran. The full listed forged-state, timing, storage, coherence and nine-day-state tests remain activation requirements. |

## ADR-2032 section dispositions

| Section | Disposition and acceptance still required |
|---|---|
| Status and context | Proposed/none/inactive matches the absence of descriptor/publisher handlers. Existing generic storage of a kind is not implementation of its space semantics. |
| D1: descriptor and fallback placement | No validator or fallback vectors. Test finite/bounded dimensions, malformed/repeated tags, admin authority, deterministic fallback and rejection before signing. |
| D2: closed references | No production parser. The encoded-dot result exposes an unspecified canonicalisation boundary; include it in both engine rejection and client ignore tests. Unknown tags must not invalidate otherwise valid space content. |
| D3: space, volume, frames and revisions | Local placement and fit tests in VisionClaw do not prove forum descriptors. Validate positive usable bounds, degenerate axes and highest verified revision; do not infer a shared global frame. |
| D4: NIP-53 media publisher | Explicitly deferred with voice. No 30312 publisher or status table. Define role/status mapping and test configured URL, home key, group, revision and withdrawal. Its step-3 dependency is conditional, not permission to enable voice. |
| D5: coarse presence | No 10312-specific path. Publication authority and freshness are explicitly undecided; storage tests expose ordering prerequisites. The current standard's one-room scope needs an explicit product decision for multi-volume users. |
| D6: advisory claims | Namespace, author membership, expiry and arbitration are proposed only. Backdating and refresh priority loss are acknowledged weaknesses; claims must never grant access. Test removal, stale authority, independent homes and both admission/read expiry. |
| D7: registry-only kinds | Namespace entries exist in the baseline, but source has no claim/descriptor handler. Registry documentation is not deployed adoption. |
| D8: in-world apps | Remains excluded. No unlicensed assets/code were imported and no application sandbox was built or asserted. |
| Consequences | Cross-client placement needs signed source and deterministic interpretation. Metadata size, reference expansion, 10312 authority/staleness and the permanent legacy `visionflow:claim` namespace remain explicit costs or unresolved choices. |
| Rejected alternatives | Private kinds, separate placement event, activity-derived placement, free-form URLs and claims-based access are rejected. The relative-path option still needs a complete parser/fetch contract. |
| Revisit triggers | Duplicate claims, NIP-53 changes, dataspace grammar changes and in-world apps remain triggers. |
| Verification and adoption list | None of the proposed engine/client acceptance cases has run. Current SQL and URL evidence identify prerequisites; they do not substitute for descriptor interoperability, real claim lifecycle or publisher tests. |

## Protocol comparison and user utility

This comparison uses NIPs commit `400d157c53264b8e58c7b3cdbd21ef546fb7a157`, retrieved on 1 November 2026. NIP-29 specifies relay-signed `d`-addressed metadata; distinguishes private reading, restricted writing, hidden metadata and closed joining; and permits partial/absent member lists. Its current referencing section uses a metadata `naddr`, whereas ADR-2031 still names `host'id`. VisionFlow therefore needs a declared interoperability profile: its complete membership projection, revision order and freshness lease cannot be inferred from generic NIP-29 support. [Pinned NIP-29](https://github.com/nostr-protocol/nips/blob/400d157c53264b8e58c7b3cdbd21ef546fb7a157/29.md).

NIP-01 expects the latest replaceable version in responses and the lowest id on timestamp ties. Merely retaining older rows is not, by itself, the complete issue: the current query also returns them without address reduction. NIP-53 describes 10312 as replaceable presence in one room, with periodic updates and age filtering. The proposal leaves that freshness policy undecided, so fine-grained multi-volume poses and coarse social presence should not be presented as equivalent. [Pinned NIP-01](https://github.com/nostr-protocol/nips/blob/400d157c53264b8e58c7b3cdbd21ef546fb7a157/01.md), [pinned NIP-53](https://github.com/nostr-protocol/nips/blob/400d157c53264b8e58c7b3cdbd21ef546fb7a157/53.md).

| Comparable project | What the primary source says | Useful comparison for VisionFlow |
|---|---|---|
| [Groups](https://github.com/max21dev/groups) | README lists group chat/moderation, public/private group creation and a mixture of completed and unfinished operations, including private joining | Use concrete join, leave, kick and metadata journeys as interoperability probes; a supported-kind checklist is insufficient. This review did not run the client. |
| [Nostrord](https://nostrord.com/) | Product site presents group administration, threads, remote signers and relay-dependent live rooms | These are established user-facing expectations to compare against. VisionFlow's proposed difference is linking cohort authority to spatial data and work, which still needs end-to-end acceptance. Site claims are not independently validated here. |
| [Chachi](https://github.com/purrgrammer/chachi) | NIP-29 group client; repository archived on 23 April 2026 | A historical interoperability reference, not evidence of a currently maintained dependency to adopt. |

Measure the intended user benefit through a two-client scenario: one admin admits a person through the cohort; both clients agree on membership and placement; the person reads the permitted material; removal closes access within the stated bound; a restart or feature rollback does not reveal private history. Add an ordinary NIP-29 client to establish which features interoperate and which require the estate profile. This would test the proposed advantage over standalone group chat without claiming the currently missing integration already exists.

[NF-13](../diagrams/nostr-rust-forum/13-group-admission-and-replacement.md) and [NF-14](../diagrams/nostr-rust-forum/14-proposed-group-replication-and-spaces.md) map the current code and proposed boundaries. The full estate, remaining forum decisions, agentbox signer and all historical diagram semantics remain under review.

## Documentation checks

All 32 diagrams in the four touched topics pass rendering and strict citation checks; the six new diagrams were visually inspected. The whole-corpus pass now covers 173 topics and 1,572 diagrams, with 24,299 references verified at their declared revisions, no working-tree fallbacks, no unresolved references and nine explicitly allowlisted generated-input references. The final verification gate recognises all 39 declared revisions. These checks do not establish semantic completeness of the whole estate audit.

[Full citation receipt](evidence/2026-11-01/forum-corpus-citations.txt), [final gate](evidence/2026-11-01/forum-final-gate.txt), [relative-link check](evidence/2026-11-01/forum-link-check.json), [structured probe results](evidence/2026-11-01/forum-probe-results.json) and [retained render hashes](evidence/2026-11-01/forum-artifact-retention.json) accompany the report. Rust formatting, shell syntax and diff-whitespace checks passed. Prose and invisible-character checks found no high-severity prose flags or suspicious invisible characters in this report. Historical topic pins remain historical; only the new topics claim this pass's revision.
