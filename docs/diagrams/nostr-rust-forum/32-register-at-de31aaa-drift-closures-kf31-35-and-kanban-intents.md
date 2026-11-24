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
  - ../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md
  - ../nostr-rust-forum/crates/nostr-bbs-governance-probe/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-governance-probe/src/main.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/export.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/tests/wac_proptests.rs
  - ../nostr-rust-forum/docs/adr/ADR-2011-operator-task-properties-set-the-escalation-boundary.md
  - ../nostr-rust-forum/docs/adr/ADR-2012-d1-ledger-becomes-a-chain-view.md
verified_commit: {nostr-rust-forum: de31aaa6e6c6b672cc310a2659cf33fd4887602c, dreamlab-ai-website: 5dc4c644517ff36219b7fe6cc1a70f20dc6a8d83}
aliases:
  NF-32.6: NF-32.8
---

## For developers

Pass C over kit commits 4a06539 to de31aaa. NF-10 is the register, already at the fifteen-panel cap, so its superseded panels NF-10.4, NF-10.11, NF-10.15 and NF-10.16 get their successors here (NF-32.1, NF-32.2, NF-32.4, NF-32.5), and the new grant-paging findings KF-31 to KF-35 are drawn in NF-32.3 (`../nostr-rust-forum/docs/security/known-findings.md:77-78`). The panels that were held on `resolved-ratchet.json` are now frozen at their own stamps, with successors here re-derived at de31aaa: NF-03.2 and NF-03.11 (NF-32.8, NF-32.9), NF-04.10 (NF-32.10), NF-26.8 (NF-32.11), NF-10.6 (NF-32.12) and NF-06.18 (NF-32.13, which reads the website at 5dc4c64); DW-01.5's successor is DW-01.9. NF-32.6, which held their current text while they could not be frozen, is folded into those successors (alias to NF-32.8). Their topics keep their old stamps, NF-10 included. NF-32.7 draws the kanban agent intent's new `actor` tag under the proposed ADR-2035.

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
    O10b["O10 pod-worker deferral shims - DELETED, and kit #100 fixed the feature comment: see NF-32.10,<br/>the successor to NF-04.10. Only mod export remains nostr-bbs-pod-worker/src/lib.rs:41"]
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

## NF-32.8 NIP-42 AUTH verdict table at de31aaa - the console error now says every AUTH is refused
```mermaid
sequenceDiagram
    autonumber
    participant C as Client socket
    participant DO as NostrRelayDO, handle_auth<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2124
    participant EV as evaluate_auth_event<br/>nostr-bbs-relay-worker/src/relay_do/nip42.rs:197

    DO-->>C: AUTH challenge at connect (per-session, unpredictable) - see NF-11.2
    C->>DO: ["AUTH", kind-22242 event]
    DO->>DO: allowed set = RELAY_URL plus RELAY_URL_ALIASES, canonicalised, explicit config only nip_handlers.rs:2131
    DO->>EV: event, expected_challenge, allowed_relay_urls, now, max_skew nip_handlers.rs:2141
    EV->>EV: kind must be 22242 nip42.rs:204
    EV->>EV: verify_event_strict - id + Schnorr nip42.rs:208
    EV->>EV: challenge tag must equal THIS session's issued value nip42.rs:214
    EV->>EV: empty allowed set REJECTS - invalid: relay url unconfigured nip42.rs:226
    EV->>EV: relay tag must match one allowed URL, canonicalised nip42.rs:229
    EV->>EV: created_at within +/- 600 s nip42.rs:239 nip42.rs:35
    EV-->>DO: AuthVerdict::Ok(pubkey) nip42.rs:243
    DO->>DO: mark session authed and persist it, then OK true nip_handlers.rs:2153 nip_handlers.rs:2157

    Note over EV: A MISSING session challenge can never match, so it rejects - there is no unchallenged path nip42.rs:212-217
    Note over EV: Since 6064622 the relay-tag check no longer fails OPEN. A blank RELAY_URL leaves the allowed set empty and every AUTH is refused, because a request-derived identity would let an attacker's Host poison the binding nip42.rs:220-227. The set is built by allowed_relay_urls nip42.rs:252
    Note over DO: DEBT the template still ships RELAY_URL and RELAY_URL_ALIASES blank nostr-bbs-relay-worker/wrangler.toml:49 nostr-bbs-relay-worker/wrangler.toml:53, so an unconfigured nip42 deployment refuses every AUTH and therefore every write. handle_auth logs a console error on each AUTH while it is blank nip_handlers.rs:2138 nip42.rs:284. The kit records it as open finding KF-29, docs/security/known-findings.md:106
    Note over DO: Kit #100 - the console error now says RELAY_URL and RELAY_URL_ALIASES are unset, so every NIP-42 AUTH is REFUSED, no client can authenticate and every write and protected-kind read is refused nip42.rs:288. The call-site comment says the same nip_handlers.rs:2135-2137, matching the empty-set rejection nip42.rs:226
    Note over EV: Since 4a73a7a the evaluate_auth_event doc comment says the caller builds the set from RELAY_URL and RELAY_URL_ALIASES only, never from the request host nip42.rs:187-193, and the helper comment says its one caller is the per-AUTH log, nothing at startup or in NIP-11 nip42.rs:274-277 nip_handlers.rs:2138
    Note over EV: canonical_relay_url absorbs scheme, case and a trailing slash but keeps host identity nip42.rs:163, asserted nip42.rs:432-443
```

**What it shows.** Successor to the frozen NF-03.2. Kit #100 rewrote the console error and its call-site comment, so the DOC-DRIFT is closed (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs:288`); the call grew one line, so the handler anchors after it sit one lower. KF-29, the blank template, stays open.

## NF-32.9 Structural validation limits at de31aaa - the drift cap comes from core
```mermaid
flowchart LR
    V["validate_event<br/>nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1728"]
    L1["id 64, pubkey 64, sig 128 hex chars<br/>nip_handlers.rs:1729"]
    L2["content cap - registration kinds 0 and 9024 get 8 KiB<br/>nip_handlers.rs:1733 nip_handlers.rs:1739 nip_handlers.rs:41"]
    L3["max 2000 tags nip_handlers.rs:1743 nip_handlers.rs:42"]
    L4["max 1024 bytes per tag value nip_handlers.rs:1748 nip_handlers.rs:43"]
    L5["created_at drift cap 7 days, now read from core: MAX_TIMESTAMP_DRIFT is nostr_bbs_core::event::MAX_CREATED_AT_DRIFT_SECS<br/>nip_handlers.rs:44, nostr-bbs-core/src/event.rs:28; sealed originals exempt from the PAST bound only (ADR-2017) nip_handlers.rs:1754 nip_handlers.rs:1794"]
    L6["max 20 subscriptions per socket, decided by subscription_cap_reached on REQ<br/>nip_handlers.rs:45 nip_handlers.rs:52-53 nip_handlers.rs:1862"]
    L7["#91: a REQ reusing an id the session holds is a REPLACEMENT and is never refused - an AUTH replay<br/>re-sends every held id, and at a full session it used to be refused wholesale nip_handlers.rs:47-51"]

    V --> L1 & L2 & L3 & L4 & L5
    V -.-> L6 --> L7

    N1["OPEN (NF-10 O20, KF-22): the cap is read and the borrow released at nip_handlers.rs:1862, the subscription<br/>is inserted only at nip_handlers.rs:1929-1931, after the D1 awaits at nip_handlers.rs:1907 and nip_handlers.rs:1908.<br/>Two REQs with different ids on one session can both pass, so the cap is a resource bound, not a security boundary"]
```

**What it shows.** Successor to the frozen NF-03.11: the seven-day drift limit is no longer a relay-local literal but core's constant (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:44`, `../nostr-rust-forum/crates/nostr-bbs-core/src/event.rs:28`), the value unchanged. The subscription-cap race (O20, KF-22) is unchanged.

## NF-32.10 The three wasm32-unreachable Phase-1 surfaces at de31aaa - the feature comment now matches the deleted shims
```mermaid
flowchart TB
    FEAT["solid-pod-rs-phase1 feature<br/>nostr-bbs-pod-worker/Cargo.toml:25"]
    F1["solid-pod-rs/provision-keys nostr-bbs-pod-worker/Cargo.toml:26"]
    F2["solid-pod-rs/nip05-endpoint nostr-bbs-pod-worker/Cargo.toml:27"]
    F3["solid-pod-rs/export-jsonld nostr-bbs-pod-worker/Cargo.toml:28"]
    EXPORT["export.rs is a bare re-export of solid_pod_rs::export<br/>nostr-bbs-pod-worker/src/export.rs:19"]
    STATE["Off by default: the workspace dep enables only the core feature<br/>nostr-rust-forum/Cargo.toml:188, pulled bare by the worker nostr-bbs-pod-worker/Cargo.toml:59<br/>the trio is opt-in only nostr-bbs-pod-worker/Cargo.toml:25"]

    FEAT --> F1 & F2 & F3
    FEAT -.-> EXPORT
    STATE --> FEAT

    N1["DIVERGENCE BASELINE-architecture: these three surfaces remain STRUCTURALLY unreachable from wasm32<br/>CF Workers. Legacy ADR-087 (portable cores) is deferred, so ADR-086's pod-federation fallback is<br/>degenerate - the pod returns the same data D1 already holds. Latent risk, not scheduled work<br/>docs/BASELINE-architecture.md:103-106"]
    N2["DIVERGENCE: the related WAC Turtle serializer bare-path IRI quirk (legacy ADR-088) has ZERO live<br/>impact because this worker writes JSON-LD ACLs and never round-trips Turtle docs/BASELINE-architecture.md:106-109 - see NF-04.4"]
    N3["The kit groups the three upstream features under ONE forum-facing alias so an operator flips a single<br/>switch rather than three docs/consumer-surface-map.md:54"]
    N4["Property tests cover the WAC document handling this whole topic rests on<br/>nostr-bbs-pod-worker/tests/wac_proptests.rs:1"]
    N5["Parked stubs: the key_provisioning and nip05_endpoint shims are deleted, recorded at nostr-bbs-pod-worker/src/lib.rs:37-40;<br/>mod export is the one staging shim left nostr-bbs-pod-worker/src/lib.rs:41, gated nostr-bbs-pod-worker/src/export.rs:17.<br/>Kit #100: the feature comment now says export.rs is the only module it gates and the other two shims were removed<br/>at closeout nostr-bbs-pod-worker/Cargo.toml:20-22, and that provision-keys and export-jsonld enable tokio-runtime,<br/>so the alias builds natively only, not for wasm32 nostr-bbs-pod-worker/Cargo.toml:23-24;<br/>the consumer surface map lists only export.rs with a dated note docs/consumer-surface-map.md:46-50"]
    STATE -.- TEN["TENSION ADR-2012 vs Cargo.toml: decision 4 speaks of the exact pin, bumped in lockstep with the host,<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:76-80, while the workspace dependency is a caret range<br/>at 0.5.0-alpha.24 by owner decision, nostr-rust-forum/Cargo.toml:184-188"]
```

**What it shows.** Successor to the frozen NF-04.10: kit #100 closed the N5 drift; the feature comment names `export.rs` as the only gated module and says the alias builds natively only (`../nostr-rust-forum/crates/nostr-bbs-pod-worker/Cargo.toml:20-24`). The comment grew three lines, so the feature anchors sit three lower.

## NF-32.11 Open items on sealed-original deletion at de31aaa - ADR-2017 carries the dated note
```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2966
    O19["OPEN NF-10 O19, KF-19: an author cannot delete their own migrated post<br/>the member gate refuses every sealed original"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs:870
    O19B["The purge's row-author tombstone exists<br/>but the classifier reads the sealed flag first"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:123
    DEAD["For a new kind 5 the client's author branch is reachable<br/>only when the author is also admin or TL3"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:3399
    O18["DEBT NF-10 O18, KF-18: a migrator who lost admin leaves<br/>envelopes, any current admin's envelope is removable"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:19
    O25["OPEN NF-10 O25, KF-27: the client cannot identify TL3<br/>moderators, so on a restored sealed original<br/>their deletions hide nothing client-side"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/whitelist.rs:180
    O25B["check-whitelist isAdmin reads whitelist.is_admin only<br/>whitelist.rs:135"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs:184
    O25C["The relay also honours static ADMIN_PUBKEYS admins<br/>auth.rs:184-189, which the client never sees as admin"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/tombstones.rs:14
    DRIFT1["Since nostr-rust-forum 4a73a7a the comment says<br/>since #94 the relay refuses such a kind 5 unless admin or TL3"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs:18
    DRIFT2["Since nostr-rust-forum 4a73a7a both comments name<br/>tombstones::fold_deletion, also relay_do/nip_handlers.rs:2942"]
    %% Source: ../nostr-rust-forum/docs/adr/ADR-2017-sealed-original-history-migration.md:265
    DRIFT3["Kit #100: ADR-2017 now carries a dated note that kit #99 corrected both comments<br/>ADR-2017-sealed-original-history-migration.md:265-267, its line numbers as of 4a73a7a"]
    O19 --> O19B --> DEAD
    O25 --> O25B --> O25C
    O18
    DRIFT1
    DRIFT2 --> DRIFT3
```

**What it shows.** Successor to the frozen NF-26.8: DRIFT3 is closed by ADR-2017's dated note (`../nostr-rust-forum/docs/adr/ADR-2017-sealed-original-history-migration.md:265-267`). The note cites `nip_handlers.rs:2941` as checked at 4a73a7a; at de31aaa the comment sits at `nip_handlers.rs:2942`, which is the note's own stated revision boundary, not a fresh drift.

## NF-32.12 Doc-drift register DD1 to DD10 at de31aaa - BASELINE's verdict anchor is one line stale
```mermaid
flowchart TB
    DD1["CLOSED by the README rewrite: the Known-gaps row calling NIP-42 AUTH scaffolded with auth_required false is gone<br/>(the README at this stamp carries no scaffolded or planned rows nostr-rust-forum/README.md:60). The code is unchanged: nip42 is the DEFAULT nip42.rs:134,<br/>the template ships AUTH_MODE nip42 nostr-bbs-relay-worker/wrangler.toml:39, and every EVENT passes the gate first nip_handlers.rs:1142.<br/>See NF-03.3."]
    DD2["CLOSED by the README rewrite: the row saying nostr-bbs-mesh is NOT a dependency of the relay-worker is gone and the README<br/>now draws mesh as a dotted configured-routing edge nostr-rust-forum/README.md:60. nostr-bbs-relay-worker/Cargo.toml:27 declares the dependency. See NF-09.8."]
    DD3["nostr-bbs-relay-worker/wrangler.toml:18 asserts there is no ADMIN_PUBKEYS reader in src/.<br/>nostr-bbs-relay-worker/src/auth.rs:185 reads it, and so does nostr-bbs-auth-worker/src/admin.rs:63.<br/>Only nostr-bbs-search-worker/wrangler.toml:41 declares the var. See NF-08.7."]
    DD4["nostr-rust-forum/Cargo.toml:72 workspace.metadata.ci.wasm-check-packages names three crates (config, preview-worker, jev) and is read by NOTHING -<br/>the wasm job checks the whole workspace .github/workflows/ci.yml:188. Inert metadata. See NF-01.2."]
    DD5["nostr-bbs-upstream-canary/src/lib.rs:10 names a five-NIP build matrix but only three smokes exist<br/>nostr-bbs-upstream-canary/src/lib.rs:35 nostr-bbs-upstream-canary/src/lib.rs:56 nostr-bbs-upstream-canary/src/lib.rs:89 - nip04, nip59 and nip98 are unexercised. The same doc calls the crate<br/>nostr-upstream-canary nostr-bbs-upstream-canary/src/lib.rs:14 while the package is nostr-bbs-upstream-canary. See NF-01.5."]
    DD6["The search-worker cron runs every five minutes and reindexes NOTHING - the handler body is a<br/>load_store warm touch nostr-bbs-search-worker/src/lib.rs:1008. See NF-07.1."]
    DD7["CLOSED in part: KIND_GOVERNANCE_AUDIT_LOG, which was numerically the SAME kind as KIND_PANEL_RETIRED, is deleted and a regression test<br/>keeps the six kinds pairwise distinct nostr-bbs-core/src/governance.rs:2222, KIND_PANEL_RETIRED stays nostr-bbs-core/src/governance.rs:32.<br/>Still open: only three of the six documented ACS types have Rust structs nostr-bbs-core/src/governance.rs:102 nostr-bbs-core/src/governance.rs:1227 nostr-bbs-core/src/governance.rs:1257.<br/>See NF-06.1 and NF-06.2."]
    DD8["docs/consumer-surface-map.md:21 lists the git panel as calling /.well-known/apps; the client fetches<br/>/apps/manifest.json nostr-bbs-forum-client/src/components/git_panel.rs:893 and no .well-known path exists in that crate. See NF-30.2."]
    DD9["DOC-DRIFT, one line: the governing doc's gift-wrap paragraph cites the recipient gate at nip_handlers.rs lines 1175 to 1185,<br/>gift_wrap_recipient at line 160 and the verdict at line 2352 docs/BASELINE-architecture.md:163-165. The first two still hold<br/>nip_handlers.rs:160 nip_handlers.rs:1175-1185, but the call-site comment at nip_handlers.rs:2135-2137 grew a line,<br/>so gift_wrap_recipient_verdict is at nip_handlers.rs:2353. See NF-10.1 I5."]
    DD10["Since 4a73a7a the client tombstone doc says that since the member gate the relay refuses a kind-5 naming a<br/>restored sealed original unless the signer is an admin or TL3+, and that kind-5s stored before the gate can still name one<br/>nostr-bbs-forum-client/src/stores/tombstones.rs:11-15, matching nip_handlers.rs:2966 nip_handlers.rs:3057. See NF-10.15."]
```

**What it shows.** Successor to the frozen NF-10.6. DD9 no longer holds as matching: BASELINE still places `gift_wrap_recipient_verdict` at line 2352 (`../nostr-rust-forum/docs/BASELINE-architecture.md:163-165`), the function is at `../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:2353`. The other entries are unchanged at de31aaa.

## NF-32.13 The M4 probe suite at de31aaa, with the website kit pin at 5dc4c64
```mermaid
flowchart TB
    SUITE["nostr-bbs-governance-probe - the ADR-2011 M4 suite: pure verdicts in the library,<br/>sockets in the binary nostr-bbs-governance-probe/src/lib.rs:1-8"]
    OWN["probes live on their own panel and never need the owner's key<br/>nostr-bbs-governance-probe/src/lib.rs:12-19, PANEL_D nostr-bbs-governance-probe/src/lib.rs:58"]
    TABLE["P01 to P11 - NIP-11 posture, routes, effective_tier stamped high, probe tag blinding,<br/>rationale refusal, system decider refusal, withdrawal nostr-bbs-governance-probe/src/lib.rs:23-35"]
    NOTRUN["NotRun never counts as a pass nostr-bbs-governance-probe/src/lib.rs:43-45"]
    BIN["the binary reads the signing key from a named variable and never prints it<br/>nostr-bbs-governance-probe/src/main.rs:10-11"]
    LIVE["ADR-2011 activation live: run 20261002t132144z passed 11 of 11 on the edge<br/>docs/adr/ADR-2011-operator-task-properties-set-the-escalation-boundary.md:193"]
    REC["ADR-2010 activation staged on the same run - the refusal path is live, the commit path is not<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:7, docs/adr/ADR-2010-durable-governance-outcome-receipts.md:153"]
    PIN["the kit pin is now de31aaa (website PR #101), a descendant of fd31193, 9c36b60, 8a5cf81 and ebcfaa2 and so of the correlation fix a5b809e<br/>../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md:30"]

    SUITE --> OWN --> TABLE --> NOTRUN
    SUITE --> BIN
    TABLE --> LIVE --> REC
    PIN --> REC

    N1["OPEN: the requesting agent holds the admin flag, and the relay tells a human from a system by the<br/>31403's own decided_by - so the boundary is as strong as the admin list; not exercised live, on purpose<br/>docs/adr/ADR-2011-operator-task-properties-set-the-escalation-boundary.md:201"]
    N2["OPEN: no human has yet decided a high-tier case; the owner's first such 31403 at projection-committed<br/>is what closes ADR-2010 docs/adr/ADR-2010-durable-governance-outcome-receipts.md:157"]
    N3["Since 13cbe6c ADR-2010 records the correlation fix as on the edge, pinned through 341c5d2<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:155"]
    N4["Probe cases persist as open broker_cases rows after withdrawal, identifiable by subject_kind m4-probe<br/>docs/adr/ADR-2011-operator-task-properties-set-the-escalation-boundary.md:204"]
    N5["The kit record pins de31aaa in its SHA column and CANONICAL_KIT_SHA (branch column main at de31aaa, no tag);<br/>the 9c36b60 paragraph still describes the env-driven Jev local-only switches that pin brought; see DW-05.11<br/>../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md:26, :66"]
```

**What it shows.** Successor to the frozen NF-06.18: the website pins kit de31aaa (`../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md:26`, `kit-compatibility-record.md:30`), still a descendant of the correlation fix a5b809e. NF-06 keeps its website key at e255b2b because the frozen NF-06.18 is its only website citation and a frozen panel pins only the forum repository.
