---
id: NF-32
title: The register at de31aaa - successors for the kit #99 and #100 drift fixes, KF-31 to KF-35 and kanban agent intents
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/security/known-findings.md
adrs: [ADR-2007, ADR-2017, ADR-2028, ADR-2033, ADR-2034, ADR-2035]
sources:
  - ../nostr-rust-forum/docs/archive/adr/ADR-100-key-lifecycle.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/SETUP.md
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0008_whitelist.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0009_events.sql
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/filter.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/cron.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admins.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/acl.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/src/ssrf.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/mock.rs
  - ../nostr-rust-forum/crates/nostr-bbs-upstream-canary/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/tests/upstream_vectors/mod.rs
  - ../nostr-rust-forum/docs/consumer-surface-map.md
  - ../nostr-rust-forum/.github/workflows/ci.yml
  - ../nostr-rust-forum/docs/security/known-findings.md
  - ../nostr-rust-forum/docs/security/advisory-exceptions.md
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/onboarding_modal.rs
  - ../nostr-rust-forum/.github/workflows/audit.yml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/feature_gate.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/git_panel.rs
  - ../nostr-rust-forum/benchmarks/js-vs-wasm/bench.mjs
  - ../nostr-rust-forum/benchmarks/js-vs-wasm/package.json
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/wasm_bridge.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/tests/upstream_vectors/all_fixtures.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/mesh.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/moderation.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay/receipt.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/thread.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/lookup.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/signer_admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/deletion_sync.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/indexed_db.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/store.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/keyset.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/key_owner.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/bookmarks_modal.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/pod_browser.rs
  - ../nostr-rust-forum/CONTRIBUTING.md
  - ../nostr-rust-forum/SECURITY.md
  - ../nostr-rust-forum/CHANGELOG.md
  - ../nostr-rust-forum/docs/architecture.md
  - ../nostr-rust-forum/docs/archive/adr/ADR-099-revocable-device-keys.md
  - ../nostr-rust-forum/docs/adr/ADR-2029-room-profile-deployment-type.md
  - ../nostr-rust-forum/docs/deployment-profiles.md
  - ../nostr-rust-forum/crates/nostr-bbs-core/README.md
  - ../nostr-rust-forum/crates/nostr-bbs-core/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/did.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/groups.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/nip11.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/http.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/delegation.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/transport.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/envelope.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/section.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/pod_client.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/kanban.rs
  - ../nostr-rust-forum/docs/adr/ADR-2035-kanban-agent-columns-close-the-dispatch-loop-through-a-governed-case.md
  - ../nostr-rust-forum/docs/adr/ADR-2017-sealed-original-history-migration.md
  - ../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md
  - ../nostr-rust-forum/docs/adr/ADR-2034-absence-is-reported-only-after-an-authenticated-answer.md
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/event.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/board.rs
verified_commit: {nostr-rust-forum: de31aaa6e6c6b672cc310a2659cf33fd4887602c}
---

## For developers

Pass C over kit commits 4a06539 to de31aaa. NF-10 is the register, already at the fifteen-panel cap, so its superseded panels NF-10.4, NF-10.11, NF-10.15 and NF-10.16 get their successors here (NF-32.1, NF-32.2, NF-32.4, NF-32.5), and the new grant-paging findings KF-31 to KF-35 are drawn in NF-32.3 (`../nostr-rust-forum/docs/security/known-findings.md:77-78`). Five markers sit in panels on `resolved-ratchet.json`, which cannot be frozen, and no longer hold at de31aaa: NF-03.2 (the nip42 console message), NF-04.10 N5 (the pod-worker feature comment), NF-26.8 DRIFT3 (the ADR-2017 note), NF-10.6 DD9 (BASELINE's verdict anchor, now one line stale) and DW-01.5 (the website kit pin). Those topics keep their old stamps, NF-10 included (its four superseded panels are frozen at that stamp); NF-32.1 and NF-32.6 record the current text. NF-32.7 draws the kanban agent intent's new `actor` tag under the proposed ADR-2035.

## For the business

The forum's own records caught up with its code: the security table now shows the cold-link fix as closed and lists five new, open limits of the zone-key paging, one of which (KF-34) must be fixed before the group key plane can be switched on. A kanban card sent to an agent now names the agent explicitly, the first built piece of a proposed design in which boards hand work to agents and return it for human review.

## NF-32.1 Anomaly register refined - the pod-worker feature comment now matches the deleted shims

```mermaid
flowchart TB
    O4["O4 members-vs-whitelist admin split - CLOSED in both resolvers. The relay unions the static set and whitelist<br/>(nostr-bbs-relay-worker/src/auth.rs:185, nostr-bbs-relay-worker/src/auth.rs:194). CORRECTED: the auth worker does too - static set<br/>nostr-bbs-auth-worker/src/admin.rs:62-69, RELAY_DB whitelist nostr-bbs-auth-worker/src/admin.rs:71-83, otherwise false nostr-bbs-auth-worker/src/admin.rs:85,<br/>as its module doc says nostr-bbs-auth-worker/src/admin.rs:4-5; members is written but read by neither (NF-08.12). A doc comment on<br/>query_is_admin still names members nostr-bbs-relay-worker/src/auth.rs:174. See NF-08.7."]
    O6["O6 NIP-07 silent no-op DM subscription - CORRECTED. The subscription registers unconditionally<br/>(nostr-bbs-forum-client/src/dm/mod.rs:337). The real limit is per-event: a NIP-04-only extension fails gift-wrap<br/>unwrap at nostr-bbs-forum-client/src/dm/mod.rs:1229 and it IS surfaced via state.error at dm/mod.rs:1256. See NF-30.1."]
    O7["O7 NIP05_USERNAME_HOST hardcoded - NOW SHARED, no longer a settings-local constant. settings.rs:35 aliases the onboarding<br/>modal constant nostr-bbs-forum-client/src/pages/settings.rs:35, which bakes NOSTR_BBS_NIP05_DOMAIN at build time and falls back to example.test<br/>nostr-bbs-forum-client/src/components/onboarding_modal.rs:59; settings.rs:343 uses it. A display fallback only. See NF-05.7."]
    O9["O9 the relay does NOT implement MeshTransport - VisionFlow keeps this reading against the retired kit register, which said it did.<br/>The one MeshTransport impl is the mesh crate's generic RelayTransport nostr-bbs-mesh/src/transport.rs:527, and the only MeshSocket<br/>is the mock nostr-bbs-mesh/src/mock.rs:249. The relay declares the dependency nostr-bbs-relay-worker/Cargo.toml:27 but imports only<br/>MeshConfig, MeshError and PeerManager nostr-bbs-relay-worker/src/mesh.rs:52-54, leaves the outbound PeerConnector a trait with no impl<br/>nostr-bbs-relay-worker/src/mesh.rs:143, and nothing outside mesh.rs uses the module beyond mod mesh nostr-bbs-relay-worker/src/lib.rs:25.<br/>DOC-DRIFT: the module doc and the manifest comment describe a wired transport nostr-bbs-relay-worker/src/mesh.rs:5 nostr-bbs-relay-worker/Cargo.toml:25-26. See NF-09.8."]
    O10["O10 wasm_bridge has no JS consumers - REFINED. The one in-repo JS consumer of its exports nostr-bbs-core/src/wasm_bridge.rs:228<br/>is the benchmark, which imports @dreamlab/nostr-core-wasm benchmarks/js-vs-wasm/bench.mjs:39 while its package.json<br/>declares @nostr-bbs/nostr-core-wasm at the absent crates/nostr-core/pkg benchmarks/js-vs-wasm/package.json:12, so it cannot<br/>resolve. The upstream-vector suite returns early and passes when fixtures are absent<br/>(nostr-bbs-core/tests/upstream_vectors/all_fixtures.rs:18) and no workflow runs sync-fixtures.sh. DRIFT: the loader doc says<br/>such tests are ignore-attributed nostr-bbs-core/tests/upstream_vectors/mod.rs:11, and none is. See NF-09.3 and NF-09.7."]
    O10b["O10 pod-worker deferral shims - DELETED: key_provisioning.rs and nip05_endpoint.rs are gone and the removal is recorded at<br/>nostr-bbs-pod-worker/src/lib.rs:37-40; only mod export remains nostr-bbs-pod-worker/src/lib.rs:41, behind solid-pod-rs-phase1<br/>nostr-bbs-pod-worker/Cargo.toml:25. Kit PR #100: the feature comment now says export.rs is the only module it gates and the other two<br/>shims were removed at closeout nostr-bbs-pod-worker/Cargo.toml:20-22, and the consumer surface map lists only export.rs with a dated note<br/>docs/consumer-surface-map.md:46-50. NF-04.10, on the ratchet, keeps its 4a73a7a reading."]
    O11c["O11 broker_decisions is write-only - CORRECTED. A read endpoint exists:<br/>GET /api/governance/decisions with pagination and a case filter,<br/>nostr-bbs-auth-worker/src/governance_api.rs:619. See NF-06.9."]
```

**Drift:** the upstream-vector loader says missing-fixture tests are ignore-attributed (`../nostr-rust-forum/crates/nostr-bbs-core/tests/upstream_vectors/mod.rs:11`); they return early and pass (O10).

## NF-32.2 The security-gate ledger at de31aaa

```mermaid
flowchart TB
    GATE["deepsec-gate --diff scans the BLAST RADIUS of a change, not only its lines,<br/>so a branch routinely surfaces defects it did not introduce<br/>docs/security/known-findings.md:7-9"]
    RULE["a finding reaches the table only WITH an owner and a disposition -<br/>we saw it and moved on is not a disposition<br/>docs/security/known-findings.md:4-5"]
    NOTSUP["INVARIANT: the table is NOT a suppression list - nothing here is waived<br/>and no entry makes a gate green<br/>docs/security/known-findings.md:13-14"]
    SEP["accepted risk in THIRD-PARTY dependencies is the separate, narrower list<br/>docs/security/advisory-exceptions.md:7, owned by the security maintainers<br/>with a dated review docs/security/advisory-exceptions.md:4-5"]

    GATE --> RULE --> NOTSUP --> SEP

    N1["Debt KF-1 HIGH_BUG: the NIP-40 expiry sweep and the serve path disagree on a malformed expiration<br/>tag, and the reap side wins docs/security/known-findings.md:22 - relay maintainers.<br/>MECHANISM FIXED at HEAD, row not yet retired: admission refuses a malformed or past expiration<br/>(expiration_admission filter.rs:375, called before any side effect nip_handlers.rs:1110) and the sweep's<br/>SQL now skips non-numeric values cron.rs:419, so an accepted event can no longer be silently swept"]
    N2["Debt KF-2 and KF-2a: the governance READ endpoints verify a NIP-98 signature with no membership<br/>check, so authenticated is effectively anonymous, and the admin-gated reviewer aggregate can be<br/>reconstructed from the un-gated decisions read docs/security/known-findings.md:23<br/>docs/security/known-findings.md:24 - an operator decision, taken once for both"]
    N3["DRIFT KF-3: the table still says ensure_schema runs ~75 DDL statements on EVERY request before the CORS short-circuit<br/>docs/security/known-findings.md:26, but the relay now latches it once per isolate, after the preflight and before the WebSocket branch,<br/>nostr-bbs-relay-worker/src/lib.rs:224 nostr-bbs-relay-worker/src/lib.rs:202; the auth worker still bootstraps on every request nostr-bbs-auth-worker/src/lib.rs:164 - see NF-08.5"]
    N4["Debt KF-4 and KF-9: the write gate resolves a device key to its owner, but the moderation gates<br/>that follow key on the RAW signing pubkey, and a gift wrap has no author to resolve at all<br/>docs/security/known-findings.md:27 docs/security/known-findings.md:44 - see NF-03.4 and NF-03.5"]
    N5["Debt KF-5: decision_id is a 64-bit truncation of the event id and is already a primary key, so<br/>changing it is a migration docs/security/known-findings.md:28"]
    N6["Debt KF-6: the self-review and original-signer guards compare pubkeys case-SENSITIVELY while<br/>NIP-98 returns the pubkey verbatim docs/security/known-findings.md:29 - the same class the registry<br/>closed in NF-06.15, left open in the core aggregate"]
    N7["Former debt KF-8, now FIXED in the table: KIND_GOVERNANCE_AUDIT_LOG, which duplicated KIND_PANEL_RETIRED (31405), is deleted<br/>with its audit-rule validation docs/security/known-findings.md:43 - see NF-06.2 N2"]
    N8["KF-10 and probe blindness are recorded as a DOCUMENTED LIMITATION rather than an untriaged defect:<br/>the digest cannot leave the signed envelope without invalidating the signature both clients verify<br/>strictly docs/security/known-findings.md:33-42 docs/security/known-findings.md:45 - it is ADR-2011's<br/>own review_trigger, see NF-06.14"]
    N9["Debt KF-13 LOW: broker_decisions.superseded_by exists only through ensure_schema, not in any migration, so a database provisioned by migrations alone lacks it<br/>docs/security/known-findings.md:48; KF-11 supersession binding is FIXED and KF-12 appeal standing is PARTIALLY FIXED<br/>docs/security/known-findings.md:46 docs/security/known-findings.md:47"]
    N10["KF-14 to KF-28, opened after the 2026-11-22 deletion, key-scoping and cold-link work, open IN THE TABLE except KF-23<br/>docs/security/known-findings.md:82 docs/security/known-findings.md:96. KF-23 is Closed, with the kit #98 fix and the operator live check,<br/>docs/security/known-findings.md:91 and a dated note docs/security/known-findings.md:74-75. DRIFT, narrowed: the section preamble still says every row is open<br/>docs/security/known-findings.md:69; the note beneath qualifies it. Each is re-verified as O12 to O26 in NF-10.14 and NF-32.4.<br/>KF-29 now locates RELAY_URL at wrangler.toml line 49 docs/security/known-findings.md:106, where it is nostr-bbs-relay-worker/wrangler.toml:49;<br/>KF-30 open docs/security/known-findings.md:107. NEW at de31aaa: KF-31 to KF-35 from kit PR #95, all open, KF-34 blocking ZONE_GROUPS_ENABLED<br/>docs/security/known-findings.md:77-78 docs/security/known-findings.md:108-112 - see NF-32.3."]
```

## NF-32.3 KF-31 to KF-35 - the open limits of grant paging (kit PR #95)

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs:403
    K31["KF-31 MEDIUM: a failed D1 query is skipped, Err continue in query_events storage.rs:341,<br/>and the REQ still ends in EOSE nip_handlers.rs:1973 - an error reads as no events"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:288
    M31["client limits it: empty page 0 is Done unverified, no marker written"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:108
    K32["KF-32 LOW: page 0 plus 50 pages, 51,000 wraps per scope, then GiveUp<br/>every session on a cold device, no resume cursor"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1175
    K33["KF-33 LOW: anyone can address kind-1059 wraps to a member, admitted by recipient<br/>gift_wrap_recipient nip_handlers.rs:160"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/deletion_sync.rs:441
    G33["a full page of one created_at gives up deletion_sync.rs:442-443,<br/>so 1,001 junk wraps at one timestamp hide every older grant"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/tickets.rs:580
    K34["KF-34 MEDIUM, blocks ZONE_GROUPS_ENABLED: NoBinding is Unbound,<br/>handled as Unreadable store.rs:1280, unmarked and holding nothing"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:145
    W34["a warm stop below the marker never reaches it again,<br/>so a genuine ticket whose binding appears later is lost"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_backlog.rs:151
    K35["KF-35 LOW: a junk wrap to a NIP-07 member ends UnreadableBySigner<br/>and lowers the marker floor grant_backlog.rs:242-243"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/zone_crypto/grant_paging.rs:145
    W35["every later session walks all history above the floor"]
    %% Source: ../nostr-rust-forum/docs/security/known-findings.md:108
    T["All five open in the table, never Not Found on a page<br/>docs/security/known-findings.md:109-112"]
    K31 --> M31
    K33 --> G33
    K34 --> W34
    K35 --> W35
    M31 --> T
    K32 --> T
    G33 --> T
    W34 --> T
    W35 --> T
```

**Debt:** each limit ends in the keys-timed-out copy or an unverified walk, never a false absence, but KF-31 is a relay-wide blind spot shared by every REQ consumer (`../nostr-rust-forum/docs/security/known-findings.md:108`). **Open (KF-34 vs ADR-2028):** the proposed capped store for Unbound tickets is a prerequisite for `ZONE_GROUPS_ENABLED` (`../nostr-rust-forum/docs/adr/ADR-2028-zones-adopt-nip171-key-plane.md:153-160`).

## NF-32.4 Sealed originals, client and UX items after the grant backlog pages

```mermaid
flowchart TB
    O19["O19 / KF-19, NARROWED: a member cannot delete their own migrated post - refusal nip_handlers.rs:2966, the sealed flag is read<br/>before the tombstone flag nip_handlers.rs:3057. The purge DOES write (id, original author) tombstones admin_events.rs:283 through<br/>nostr-bbs-relay-worker/src/lib.rs:870; they guard re-insert at ingest, but the member gate never reaches them."]
    O21["O21 / KF-23 - CLOSED by kit #98, merged as fd31193: Lookup::decide_with_keys waits on the grant backlog once the relay has answered<br/>nostr-bbs-forum-client/src/utils/lookup.rs:134-151, and the thread page pages/thread.rs:645-656 pages/thread.rs:668 and the section page<br/>pages/section.rs:318 pages/section.rs:334 use it. The table now reads Closed docs/security/known-findings.md:91, and the governing doc's<br/>ADR-2034 section describes the zone-key wait with dated notes for #98 and #95 docs/BASELINE-architecture.md:324-327 docs/BASELINE-architecture.md:328-334.<br/>Since #95 the backlog it waits on is through only when no scope gave up and no wrap is in flight or retrying<br/>zone_crypto/grant_backlog.rs:324-334. See NF-30.9."]
    O21r["O21 residue, NARROWED by kit PR #95: the grant sync pages past the 1000-wrap filter zone_crypto/store.rs:1637-1643, page 0 plus at most 50<br/>zone_crypto/grant_paging.rs:108, so an account with more wraps than one page no longer misses its oldest grants. The code's 28.6 s figure now only<br/>sizes the 120 s cap zone_crypto/grant_backlog.rs:45-49. A member with no key settles too nostr-bbs-forum-client/src/utils/lookup.rs:161-163,<br/>so a sealed-original topic they cannot open is NotFound once keys settle utils/lookup.rs:147. Limits KF-31 to KF-33 in NF-32.3."]
    O22["O22 / KF-24: a write made while the socket is closed bypasses the AUTH deferral, which holds writes only on an open<br/>unauthenticated socket nostr-bbs-forum-client/src/relay.rs:785-804; it is buffered relay.rs:840 and flushed on reopen<br/>before AUTH relay.rs:518-526, where the relay refuses it."]
    O23["O23 / KF-25: deletion paging gives up at MAX_PAGES nostr-bbs-forum-client/src/stores/deletion_sync.rs:123, on a full page with<br/>no progress or on an empty page, keeping the old marker with no resume cursor deletion_sync.rs:30-33, next_page deletion_sync.rs:424."]
    O24["O24 / KF-26, accepted limit: zone-key rows written before per-identity scoping zone_crypto/key_owner.rs:13-19 and pre-#92<br/>bookmarks nostr-bbs-forum-client/src/components/bookmarks_modal.rs:8-11 stay visible to every account on the browser;<br/>same-origin storage is no boundary against the device zone_crypto/key_owner.rs:21-22."]
    O25["O25 / KF-27: the client cannot identify TL3 moderators nostr-bbs-forum-client/src/stores/tombstones.rs:19-21, and check-whitelist sets<br/>isAdmin from the D1 row alone nostr-bbs-relay-worker/src/whitelist.rs:180 while the relay's own check honours static<br/>ADMIN_PUBKEYS first nostr-bbs-relay-worker/src/auth.rs:185 - so T3 in NF-10.13 is narrower than relay admin standing."]
    O26["O26 / KF-28: the Pod Browser mesh card prints the probe's error string verbatim under Details<br/>nostr-bbs-forum-client/src/pages/pod_browser.rs:1056-1058; whether that is a stack trace depends on the failing call, not established."]
    O19 --> O21 --> O21r --> O22
    O23 --> O24 --> O25 --> O26
```

## NF-32.5 Resolved history R1 to R9, re-checked at de31aaa

```mermaid
flowchart TB
    R1["R1 licence prose claimed MIT/Apache on AGPL code - fixed. HOLDS for the application: workspace licence nostr-rust-forum/Cargo.toml:64,<br/>CONTRIBUTING.md:105-106. SUPERSEDED for one crate by a deliberate relicence: nostr-bbs-core is MIT OR Apache-2.0 from beta.18<br/>nostr-bbs-core/Cargo.toml:6 nostr-bbs-core/README.md:10-13, as the README states nostr-rust-forum/README.md:164."]
    R2["R2 version drift README rc11, SECURITY rc tables, NIP-11 serving 3.0.0. HOLDS for NIP-11: version is env CARGO_PKG_VERSION nip11.rs:214.<br/>Kit PR #100: SECURITY.md now lists 1.0.0-beta.20 for core, config, mesh and rate-limit on crates.io, and main, as current SECURITY.md:19,<br/>explaining that the workspace carries no single version SECURITY.md:21-24; core is 1.0.0-beta.20 nostr-bbs-core/Cargo.toml:3. The v3.0.0-rc<br/>headings stay, under Past Security Fixes SECURITY.md:26-28."]
    R3["R3 NIP-17 over-advertised - HOLDS: NIP-11 lists 59, not 17 nip11.rs:16, saying why nip11.rs:205-208,<br/>and the architecture doc says there is no kind-14 inbox routing docs/architecture.md:227.<br/>DRIFT for the client: the NIP-17 overlay is on main, dormant behind RECIPIENT_KEYS_ENABLED nostr-rust-forum/README.md:124<br/>nostr-bbs-forum-client/src/dm/mod.rs:1632, and DM sends publish a kind-10050 DM-relay list once per pubkey per session,<br/>ungated, nostr-bbs-forum-client/src/dm/mod.rs:610-611<br/>nostr-bbs-forum-client/src/dm/mod.rs:611 nostr-bbs-forum-client/src/dm/mod.rs:2012, so that sentence holds only for the relay. See NF-01."]
    R4["R4 NIP-26 deleted as superseded by ADR-099 device keys - HOLDS for core and the auth worker; CHANGELOG.md:2257-2258 records NIP-26 rejected.<br/>NOT GONE from the workspace: nostr-bbs-mesh carries its own NIP-26 verifier for mesh delegation nostr-bbs-mesh/src/delegation.rs:1,<br/>noting core has no nip26 module nostr-bbs-mesh/src/delegation.rs:22-28; the upstream-vector suite still wires the fixture all_fixtures.rs:59-60."]
    R5["R5 NIP-90 DVM module deleted - HOLDS: removal recorded nip11.rs:209-211 and 90 not advertised nip11.rs:16.<br/>Kit PR #100: the architecture note now says NIP-90 is not implemented and no DVM kind is handled specially docs/architecture.md:215-224,<br/>and the NIP matrix no longer carries a 90 row docs/architecture.md:205-206."]
    R6["R6 template drift - HOLDS: SETUP lists ZONE_CONFIG and DEVICE_KEYS_ENABLED for the relay SETUP.md:150 and explains the flag SETUP.md:175-178;<br/>the auth template binds RELAY_DB nostr-bbs-auth-worker/wrangler.toml:18; the relay template declares both nostr-bbs-relay-worker/wrangler.toml:26<br/>nostr-bbs-relay-worker/wrangler.toml:29. ADMIN_PUBKEYS stays undeclared - S2 in NF-10.7."]
    R7["R7 ad-hoc did:nostr format - MOSTLY HOLDS: production sites call did_nostr_uri nostr-bbs-core/src/did.rs:141 and keep format! only as the<br/>parse-failure fallback webauthn.rs:1114-1116 nostr-bbs-pod-worker/src/lib.rs:903-905. RESIDUE: the mesh AS2 projection formats it<br/>directly nostr-bbs-mesh/src/envelope.rs:395-396."]
    R8["R8 dual auth cors_headers with an example.com fallback - HOLDS: one builder, fail-closed, no Allow-Origin when EXPECTED_ORIGIN is<br/>unset or blank nostr-bbs-auth-worker/src/http.rs:19-30, imported by the worker nostr-bbs-auth-worker/src/lib.rs:38. See NF-02.1."]
    R9["R9 architecture doc coverage - HOLDS: rate-limit crate row docs/architecture.md:184, NIP-56/59/65/90 note docs/architecture.md:215,<br/>ADR-098/099 section docs/architecture.md:487; ADR-099 status Accepted docs/archive/adr/ADR-099-revocable-device-keys.md:4.<br/>Its NIP-90 sentence, the old R5 drift, is corrected."]
    R1 --> R2 --> R3 --> R4 --> R5
    R6 --> R7 --> R8 --> R9
```

## NF-32.6 Drift kit #99 and #100 closed where the ratchet blocks a freeze

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs:288
    A["nip42 console error now says RELAY_URL and RELAY_URL_ALIASES are unset, every NIP-42 AUTH is refused,<br/>no client can authenticate, every write and protected-kind read refused"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2135
    B["call-site comment says the same, nip_handlers.rs:2136-2137,<br/>matching KF-29 docs/security/known-findings.md:106 - NF-03.2 keeps its 4a73a7a reading"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:44
    C["MAX_TIMESTAMP_DRIFT now reads core's MAX_CREATED_AT_DRIFT_SECS<br/>nostr-bbs-core/src/event.rs:28 - NF-03.11 keeps its literal seven days"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2017-sealed-original-history-migration.md:265
    D["ADR-2017 now carries a dated note that kit #99 corrected both comments<br/>ADR-2017-sealed-original-history-migration.md:265-267 - NF-26.8 DRIFT3 keeps its 4a73a7a reading"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs:18
    E["both comments name tombstones fold_deletion<br/>nip_handlers.rs:2942"]
    %% Source: ../nostr-rust-forum/docs/BASELINE-architecture.md:165
    F["DRIFT: BASELINE places gift_wrap_recipient_verdict at line 2352;<br/>since the comment at nip_handlers.rs:2135 grew, the fn is at nip_handlers.rs:2353"]
    A --> B
    C
    D --> E
    F
```

**Drift:** the governing doc's verdict anchor is one line stale at de31aaa (`../nostr-rust-forum/docs/BASELINE-architecture.md:163-165`); NF-10.6 DD9, on the ratchet, still reads it as matching, true at 4a73a7a.

## NF-32.7 The kind-38000 agent intent names its agent (ADR-2035, proposed)

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/board.rs:920
    UI["Send to agent on a card builds the intent in the client"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/kanban.rs:802
    FN["create_agent_intent_signer, kind KIND_AGENT_INTENT 38000 kanban.rs:59"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/kanban.rs:812
    TAGS["tags p agent, actor agent, a card coordinate, board, card_d<br/>kanban.rs:811-815"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/kanban.rs:1174
    TEST["unit test: actor equals p, never the card d-tag kanban.rs:1191-1193"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:86
    RELAY["relay: the intent is a ban-gated kind, no other intent-specific check"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2035-kanban-agent-columns-close-the-dispatch-loop-through-a-governed-case.md:5
    ADR["ADR-2035 proposed, implementation partial, inactive<br/>ADR-2035-kanban-agent-columns-close-the-dispatch-loop-through-a-governed-case.md:6-7"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2035-kanban-agent-columns-close-the-dispatch-loop-through-a-governed-case.md:50
    NOT["OPEN: only the actor tag is built - agent column, registry,<br/>prompt tag and return path are not"]
    UI --> FN --> TAGS --> TEST
    TAGS --> RELAY
    ADR --> NOT
    TAGS -.-> ADR
```

**What it shows.** Kit PR #102 (`3e66d40`): before it, a consumer that fell back to the `a` coordinate read the card's own d-tag as the actor (`../nostr-rust-forum/docs/adr/ADR-2035-kanban-agent-columns-close-the-dispatch-loop-through-a-governed-case.md:23`). **Open:** the relay still holds no approval-column check, and ADR-2035 depends on the unmerged ADR-2024 (`../nostr-rust-forum/docs/adr/ADR-2035-kanban-agent-columns-close-the-dispatch-loop-through-a-governed-case.md:27`, `:47`).
