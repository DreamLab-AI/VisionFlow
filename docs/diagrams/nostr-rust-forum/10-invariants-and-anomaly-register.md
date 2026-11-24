---
id: NF-10
title: Invariants, the re-verified anomaly register and the doc-drift ledger
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2002, ADR-2003, ADR-2004, ADR-2005, ADR-2006, ADR-2007, ADR-2008, ADR-2009, ADR-2010, ADR-2014, ADR-2016, ADR-2017, ADR-2019, ADR-2029, ADR-2033, ADR-2034]
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
verified_commit: {nostr-rust-forum: 4a73a7a94bfb4d92fea03f04926cccd38c14e9af}
---

## NF-10.1 The compliance surface — BASELINE-architecture invariants

```mermaid
flowchart LR
    I1["1 nostr-bbs-core owns on-wasm32 Schnorr until the canary records Shape A<br/>nostr-bbs-upstream-canary/src/lib.rs:17 - see NF-01.5"]
    I2["2 FORUM_BASE is applied in exactly TWO places<br/>Router base nostr-bbs-forum-client/src/app.rs:929 | base_href nostr-bbs-forum-client/src/app.rs:54 - see NF-05.1"]
    I3["3 channel counts are derived, never accumulated<br/>count_for nostr-bbs-forum-client/src/stores/channels.rs:305 | dedup on insert channels.rs:629 - see NF-05.6"]
    I4["4 .acl and .meta sidecar access coerces to Control<br/>nostr-bbs-pod-worker/src/acl.rs:111 | shared policy nostr-bbs-pod-worker/src/acl.rs:37 - see NF-04.3"]
    I5["5 gift-wraps are recipient-gated, never author-gated<br/>gift_wrap_recipient nip_handlers.rs:160 | admission nip_handlers.rs:1175 | verdict nip_handlers.rs:2352<br/>other recipient routes: a configured INGRESS_RECIPIENTS key (config, flag-independent), a registered zone epoch key<br/>(ZONE_GROUPS_ENABLED), an inbox key judged by its owner's membership (RECIPIENT_KEYS_ENABLED); otherwise the recipient itself must be whitelisted - see NF-03.5"]
    I6["6 TENSION: BASELINE invariant 6 still says solid-pod-rs stays an EXACT pin docs/BASELINE-architecture.md:226,<br/>now with a note that it does not hold docs/BASELINE-architecture.md:228, and the kit section records the caret requirement<br/>docs/BASELINE-architecture.md:126; the manifest declares a caret range under ER-35 nostr-rust-forum/Cargo.toml:184,<br/>at 0.5.0-alpha.24 nostr-rust-forum/Cargo.toml:188, with no ledger record of the decision - see NF-01.4"]
    I7["7 a member's kind-5 is accepted only when the relay can vouch for every e target<br/>docs/BASELINE-architecture.md:232 | gate nip_handlers.rs:3351 called below TL3 nip_handlers.rs:1321 - see NF-10.13"]
    I8["8 a deep-linked page never reports absence from ChannelStore loading, only from an<br/>authenticated receipt or the deadline docs/BASELINE-architecture.md:237 | utils/lookup.rs:101;<br/>in an encrypted zone it also waits on zone keys utils/lookup.rs:134 - see NF-10.13 and NF-10.15"]

    I1 --> I2 --> I3 --> I7
    I4 --> I5 --> I6 --> I8

    N1["These eight are the compliance surface of BASELINE-architecture.md; 7 and 8 joined with ADR-2033 and<br/>ADR-2034. Each has at least one diagram in this area that shows the mechanism enforcing it, not just the rule."]
    N2["Any change to a crate topology, the absorption status, the pod tiers, ACL coercion or a kit version<br/>must confirm the touched invariant still holds and bump the governing doc in the SAME change."]
```

## NF-10.2 The compliance surface — IDENTITY invariants

```mermaid
flowchart LR
    K1["1 derive_subkey is a byte-for-byte cross-stack contract with agentbox's JS mirror derivation<br/>raw HMAC, not HKDF nostr-bbs-core/src/keys.rs:254 | vector nostr-bbs-core/src/keys.rs:479 | digest keys.rs:484 - see NF-09.3"]
    K2["2 derive_from_prf and derive_subkey are DISTINCT and never substitutable<br/>HKDF path nostr-bbs-core/src/keys.rs:198 | contrast documented nostr-bbs-core/src/keys.rs:225"]
    K3["3 DEVICE_KEYS_ENABLED enables only on the exact string true, checked in both workers<br/>shared rule nostr-bbs-core/src/feature_gate.rs:65 | auth delegates nostr-bbs-auth-worker/src/devices.rs:112 | asserted nostr-bbs-auth-worker/src/devices.rs:988 - see NF-02.7"]
    K4["4 gift-wrap admission keys on the recipient p tag, never the ephemeral author<br/>nip_handlers.rs:160 - see NF-03.5"]
    K5["5 TL3 is never auto-demoted; demotion stays time-driven on cron, never inline on admission<br/>nostr-bbs-relay-worker/src/trust.rs:334 - see NF-03.8"]
    K6["6 root possession recovers derived children<br/>nostr-bbs-core/src/keys.rs:241<br/>Child-only exposure is a separate case; see N1"]

    K1 --> K2 --> K3
    K4 --> K5 --> K6

    %% Source: ../nostr-rust-forum/docs/archive/adr/ADR-100-key-lifecycle.md:78
    %% Source: ../nostr-rust-forum/docs/archive/adr/ADR-100-key-lifecycle.md:85
    N1["Root exposure and isolated child exposure<br/>need different responses.<br/>A tag bump does not revoke old signatures<br/>or registry grants. See NF-20 and NF-21."]
    N2["K1 and K4 are the two invariants that cross a repo boundary - EXTERNAL: see AB-11 and ES-04 for the<br/>agentbox side of the derivation contract"]
```

## NF-10.3 Anomaly register — closed and still live

```mermaid
flowchart TB
    subgraph closed["Confirmed CLOSED at this commit"]
        O1["O1 TL1 promotion and demotion unwired - CLOSED.<br/>Reads tally then check_promotion nip_handlers.rs:2612; demotion is cron-driven. See NF-03.8."]
        O3["O3 KV ACL fast-path masks R2 delegation - CLOSED.<br/>R2 authoritative, KV miss-fallback only nostr-bbs-pod-worker/src/acl.rs:359, rationale acl.rs:281. See NF-04.4."]
        O5["O5 deleting a kind-40 destroyed the is_channel_creator lookup, locking a TL2 author out of their own<br/>channel metadata - CLOSED at 66ca8d8. The creator is persisted at creation, relay_do/storage.rs:235, and read<br/>first, nip_handlers.rs:3715, so the kind-5 path nip_handlers.rs:3512 no longer changes the answer.<br/>Residual: a failed write and a pre-existing channel fall back to the kind-40 event, nip_handlers.rs:3716. See NF-03.7."]
    end
    subgraph live["Confirmed STILL LIVE"]
        O2["O2 NIP-29 group metadata 39000-39002 accepted from admin CLIENTS rather than relay-key-signed.<br/>Acknowledged TODO nip_handlers.rs:1330, admin gate nip_handlers.rs:1337. Spec drift. See NF-03.4 step 11."]
        O11a["O11 KV and SESSIONS bind the SAME physical namespace id - nostr-bbs-auth-worker/wrangler.toml:28 and nostr-bbs-auth-worker/wrangler.toml:44.<br/>Now documented as deliberate, with disjoint key prefixes, and guarded by a unit test nostr-bbs-auth-worker/wrangler.toml:22<br/>nostr-bbs-auth-worker/src/admins.rs:347. See NF-08.4 and NF-02.2."]
        O11b["O11 /api/native-pod/provision ships placeholder vars - route nostr-bbs-auth-worker/src/lib.rs:637,<br/>NATIVE_POD_URL nostr-bbs-auth-worker/wrangler.toml:66. CORRECTED: it is NOT uncalled - the admin page posts<br/>to it nostr-bbs-forum-client/src/pages/admin.rs:837, a caller older than the previous stamp. See NF-02.2."]
        O8a["O8 qr_svg is defined TWICE with no shared helper -<br/>recovery_sheet.rs:66 and utils/devices.rs:246. See NF-05.4."]
        O8b["O8 a per-page kind-0 subscription still duplicates the app-root ProfileCache subscription -<br/>root sub app.rs:753, page sub settings.rs:457. It is now closed on unmount settings.rs:479 and generation-guarded;<br/>the 5 s timer only flags a slow relay and the subscription stays open settings.rs:488-496. See NF-05.6."]
        O8c["O8 provision_pod has THREE near-identical copies - signup.rs:176-204, settings.rs:2024-2052 and<br/>provision_pod_signer utils/pod_client.rs:49-78. The settings avatar path provisions again on a 403 or 404 settings.rs:658-659<br/>after upload_to_pod_signer has already provisioned and retried on either utils/pod_client.rs:38-40. See NF-05.8."]
    end
```

## NF-10.4 Anomaly register — refined or corrected by this pass
> **Superseded** by NF-32.1 at de31aaa (frozen at 4a73a7a).

```mermaid
flowchart TB
    O4["O4 members-vs-whitelist admin split - CLOSED in both resolvers. The relay unions the static set and whitelist<br/>(nostr-bbs-relay-worker/src/auth.rs:185, nostr-bbs-relay-worker/src/auth.rs:194). CORRECTED: the auth worker does too - static set<br/>nostr-bbs-auth-worker/src/admin.rs:62-69, RELAY_DB whitelist nostr-bbs-auth-worker/src/admin.rs:71-83, otherwise false nostr-bbs-auth-worker/src/admin.rs:85,<br/>as its module doc says nostr-bbs-auth-worker/src/admin.rs:4-5; members is written but read by neither (NF-08.12). A doc comment on<br/>query_is_admin still names members nostr-bbs-relay-worker/src/auth.rs:174. See NF-08.7."]
    O6["O6 NIP-07 silent no-op DM subscription - CORRECTED. The subscription registers unconditionally<br/>(nostr-bbs-forum-client/src/dm/mod.rs:337). The real limit is per-event: a NIP-04-only extension fails gift-wrap<br/>unwrap at nostr-bbs-forum-client/src/dm/mod.rs:1229 and it IS surfaced via state.error at dm/mod.rs:1256. See NF-30.1."]
    O7["O7 NIP05_USERNAME_HOST hardcoded - NOW SHARED, no longer a settings-local constant. settings.rs:35 aliases the onboarding<br/>modal constant nostr-bbs-forum-client/src/pages/settings.rs:35, which bakes NOSTR_BBS_NIP05_DOMAIN at build time and falls back to example.test<br/>nostr-bbs-forum-client/src/components/onboarding_modal.rs:59; settings.rs:343 uses it. A display fallback only. See NF-05.7."]
    O9["O9 the relay does NOT implement MeshTransport - VisionFlow keeps this reading against the retired kit register, which said it did.<br/>The one MeshTransport impl is the mesh crate's generic RelayTransport nostr-bbs-mesh/src/transport.rs:527, and the only MeshSocket<br/>is the mock nostr-bbs-mesh/src/mock.rs:249. The relay declares the dependency nostr-bbs-relay-worker/Cargo.toml:27 but imports only<br/>MeshConfig, MeshError and PeerManager nostr-bbs-relay-worker/src/mesh.rs:52-54, leaves the outbound PeerConnector a trait with no impl<br/>nostr-bbs-relay-worker/src/mesh.rs:143, and nothing outside mesh.rs uses the module beyond mod mesh nostr-bbs-relay-worker/src/lib.rs:25.<br/>DOC-DRIFT: the module doc and the manifest comment describe a wired transport nostr-bbs-relay-worker/src/mesh.rs:5 nostr-bbs-relay-worker/Cargo.toml:25-26. See NF-09.8."]
    O10["O10 wasm_bridge has no JS consumers - REFINED. The one in-repo JS consumer of its exports nostr-bbs-core/src/wasm_bridge.rs:228<br/>is the benchmark, which imports @dreamlab/nostr-core-wasm benchmarks/js-vs-wasm/bench.mjs:39 while its package.json<br/>declares @nostr-bbs/nostr-core-wasm at the absent crates/nostr-core/pkg benchmarks/js-vs-wasm/package.json:12, so it cannot<br/>resolve. The upstream-vector suite returns early and passes when fixtures are absent<br/>(nostr-bbs-core/tests/upstream_vectors/all_fixtures.rs:18) and no workflow runs sync-fixtures.sh. DRIFT: the loader doc says<br/>such tests are ignore-attributed nostr-bbs-core/tests/upstream_vectors/mod.rs:11, and none is. See NF-09.3 and NF-09.7."]
    O10b["O10 pod-worker deferral shims - DELETED: key_provisioning.rs and nip05_endpoint.rs are gone and the removal is recorded at<br/>nostr-bbs-pod-worker/src/lib.rs:37-40; only mod export remains nostr-bbs-pod-worker/src/lib.rs:41, behind solid-pod-rs-phase1<br/>nostr-bbs-pod-worker/Cargo.toml:22. DRIFT: the feature comment still names the deleted stub modules nostr-bbs-pod-worker/Cargo.toml:20-21. See NF-04.10."]
    O11c["O11 broker_decisions is write-only - CORRECTED. A read endpoint exists:<br/>GET /api/governance/decisions with pagination and a case filter,<br/>nostr-bbs-auth-worker/src/governance_api.rs:619. See NF-06.9."]
```

**Drift:** the upstream-vector loader says missing-fixture tests are ignore-attributed (`../nostr-rust-forum/crates/nostr-bbs-core/tests/upstream_vectors/mod.rs:11`); they return early and pass (O10).

## NF-10.6 New doc-drift found in this pass
> **Superseded** by NF-32.12 at de31aaa (frozen at 4a73a7a).

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
    DD9["Resolved at 4a73a7a: the governing doc's gift-wrap paragraph now cites the recipient gate at nip_handlers.rs lines 1175 to 1185,<br/>gift_wrap_recipient at line 160 and the verdict at line 2352 docs/BASELINE-architecture.md:163-165, which is where the code has them<br/>nip_handlers.rs:160 nip_handlers.rs:1175-1185 nip_handlers.rs:2352. See NF-10.1 I5."]
    DD10["Resolved at 4a73a7a: the client tombstone doc now says that since the member gate the relay refuses a kind-5 naming a<br/>restored sealed original unless the signer is an admin or TL3+, and that kind-5s stored before the gate can still name one<br/>nostr-bbs-forum-client/src/stores/tombstones.rs:11-15, matching nip_handlers.rs:2965 nip_handlers.rs:3056. See NF-10.15."]
```

## NF-10.7 Deployment-shaped divergences

```mermaid
flowchart TB
    S1["The relay events table was created only by a SETUP.md copy-paste - CLOSED in two steps. 1a26e51: whitelist got its first DDL, migration 0008_whitelist.sql:30,<br/>run first by ensure_schema nostr-bbs-relay-worker/src/lib.rs:1100 before the ALTER list nostr-bbs-relay-worker/src/lib.rs:1196. 66ca8d8: migration 0009_events.sql:23 creates events<br/>and ensure_schema creates it first, nostr-bbs-relay-worker/src/lib.rs:1071, so SETUP.md now points at the migration SETUP.md:100 instead of carrying the DDL.<br/>Residual: the auth schema is still pasted SQL SETUP.md:75. See NF-08.6."]
    S2["ADMIN_PUBKEYS is read by auth and relay but declared in neither template and named in no SETUP step<br/>SETUP.md:147. A by-the-book deployment has no static admin bootstrap. See NF-08.7."]
    S3["The preview worker's SSRF guard is denylist-only without PREVIEW_ALLOWED_HOSTS, which the template<br/>never sets nostr-bbs-preview-worker/wrangler.toml:13. The code says so itself - the Workers runtime exposes no<br/>resolve-then-pin primitive nostr-bbs-preview-worker/src/ssrf.rs:13 - and there is no wall-clock timeout.<br/>See NF-07.6."]
    S4["DEVICE_KEYS_ENABLED ships false in BOTH templates nostr-bbs-auth-worker/wrangler.toml:69<br/>nostr-bbs-relay-worker/wrangler.toml:29, so the whole ADR-099/100 device story is dormant by default and<br/>revocation has no effect at AUTH. See NF-02.7."]
    S8["NEW at af4f0d5, recorded as KF-30 at 4a73a7a docs/security/known-findings.md:101: the NIP-09 tombstone tables deleted_events and deleted_addresses exist only through ensure_schema<br/>nostr-bbs-relay-worker/src/lib.rs:746 nostr-bbs-relay-worker/src/lib.rs:755 nostr-bbs-relay-worker/src/lib.rs:1083 - no file in migrations/ creates them,<br/>the same shape as KF-13. A database provisioned by migrations alone lacks them until the fetch bootstrap runs, and the<br/>ingest lookup fails open meanwhile nip_handlers.rs:3476."]
    S5["MESH_ALLOWED_REMOTE_DIDS ships empty nostr-bbs-relay-worker/wrangler.toml:98, so the federated-kind gate at<br/>nip_handlers.rs:4719 is inert. Standalone is the only supported mode. See NF-03.13."]
    %% Source: ../nostr-rust-forum/.github/workflows/ci.yml:293
    %% Source: ../nostr-rust-forum/.github/workflows/ci.yml:296
    %% Source: ../nostr-rust-forum/.github/workflows/ci.yml:306
    %% Source: ../nostr-rust-forum/.github/workflows/ci.yml:20
    %% Source: ../nostr-rust-forum/.github/workflows/ci.yml:283
    S6["drift-gates runs parity and lint.<br/>ci-pass omits it from needs.<br/>External branch rules are separate;<br/>see dated report.<br/>Workflow path filters apply.<br/>No paired agentbox release ref is set.<br/>See NF-21 for source-check limits."]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/mod.rs:268
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:50
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:56
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/wallet/parent.rs:63
    S7["ADR-2019 parent-coin view needs<br/>SIDESTR_WALLET and an HTTPS backend.<br/>No backend means no request.<br/>Public-key address only; no parent spend.<br/>Backend-reported, not browser-validated.<br/>See NF-16 for parser and loading limits."]
```

## NF-10.8 ADR-2006 and ADR-2010 — what the closeout holds open

```mermaid
flowchart LR
    A2006["ADR-2006 trust demotion<br/>accepted, complete, live"]
    Q2006["Current sweep: TL2 may move directly to TL0; stable keyset paging and explicit<br/>commit/error outcomes replace OFFSET and ignored writes. D1 batches state and audit.<br/>Conditional audit now requires a changed row and all observed policy inputs; see NF-10.10."]
    A2010["ADR-2010 governance receipts<br/>proposed, partial, staged"]
    Q2010["Relay OK establishes acceptance only. The full signed / accepted / projection-committed /<br/>consumer-received / applied contract needs agreement with the authority consumer and the mutation<br/>owner, plus failure and restart evidence."]

    A2006 --> Q2006
    A2010 --> Q2010

    N1["DOC-DRIFT: the IDENTITY-keys-and-trust closeout still states OFFSET row-skipping and ignored write<br/>errors as CURRENT defects. Both are fixed - keyset paging nostr-bbs-relay-worker/src/trust_sweep.rs:22<br/>and confirmed-commit-only counters nostr-bbs-relay-worker/src/trust_sweep.rs:302 trust_sweep.rs:517. The governing doc needs<br/>the qualification retired; see NF-11.11 and NF-11.12."]
    N2["DOC-DRIFT: the relay-side receipt machine is REAL, not proposed - stages at<br/>nostr-bbs-core/src/governance.rs:987, applied at relay_do/receipts.rs:452, and handle_event logs<br/>accepted-but-not-applied from the returned receipt nip_handlers.rs:1701. ADR-2010 now records appeal and supersession as<br/>receipt-wired through the same apply_with_receipt machine docs/adr/ADR-2010-durable-governance-outcome-receipts.md:111,<br/>KF-11 and KF-12 docs/security/known-findings.md:46 docs/security/known-findings.md:47. Only the CONSUMER half is absent.<br/>See NF-11.9 and NF-11.10."]
    N3["EXTERNAL: both remaining halves belong to other repos - VisionClaw's elevation consumer see VC-24,<br/>agentbox's approvals pipeline see AB-14, the estate loop see ES-05"]
```

## NF-10.10 Trust conflicts cannot create an audit fact for an unchanged row

```mermaid
sequenceDiagram
    participant S as Sweep snapshot
    participant D as D1 transactional batch
    participant O as Outcome counters
    S->>D: UPDATE matching level, admin, timestamps and activity counters
    alt exact observed snapshot still matches
        D->>D: One changed row
        D->>D: INSERT audit SELECT WHERE changes equals 1
        D-->>O: Both statements changed one row
        O->>O: Count confirmed demotion
    else concurrent policy input changed or row disappeared
        D->>D: Zero changed rows
        D->>D: Conditional audit SELECT inserts zero rows
        D-->>O: Conflict, no false audit fact
    end
    Note over S,O: SQL failure aborts the batch. Local SQLite tests execute the<br/>production SQL constants, deployed D1 remains a separate receipt.
```

Execution update, 2026-09-07: `trust_sweep.rs` compares every observed policy input using null-safe timestamp/admin predicates. The audit INSERT is conditional inside the transaction; both result counts must equal one. Five exact-SQL trust tests cover concurrent level/activity/admin edits, missing rows, nullable snapshots, rollback and idempotency. The earlier false-audit probe remains in the [audit](../../estate-review/2026-09-07-federation-audit.md); it is not a current-code claim.

## NF-10.11 The security-gate ledger — findings owned rather than silenced
> **Superseded** by NF-32.2 at de31aaa (frozen at 4a73a7a).

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
    N10["KF-14 to KF-28, opened after the 2026-11-22 deletion, key-scoping and cold-link work, are all open IN THE TABLE<br/>docs/security/known-findings.md:65 docs/security/known-findings.md:74; DRIFT: the table still says every row is open<br/>docs/security/known-findings.md:69 and KF-23 Open docs/security/known-findings.md:85, but kit fd31193 closed KF-23 (O21 in NF-10.15). Each is re-verified against the code as O12 to O26<br/>in NF-10.14 and NF-10.15. KF-1 to KF-13 above are unchanged in the table at this stamp.<br/>NEW at 4a73a7a: KF-29, the blank RELAY_URL template that refuses every AUTH (NF-03.2), and KF-30, the tombstone tables<br/>made only by ensure_schema (S8 in NF-10.7), both open docs/security/known-findings.md:93 docs/security/known-findings.md:100<br/>docs/security/known-findings.md:101. KF-29 locates RELAY_URL at wrangler.toml line 47, its line at fd31193; it is now<br/>nostr-bbs-relay-worker/wrangler.toml:49."]
```

## NF-10.12 New invariant — DecisionOutcome gains Demote, still uncovered downstream

```mermaid
flowchart LR
    ENUM["DecisionOutcome adds Demote iri alongside Promote iri<br/>enum def governance.rs:1580 | Promote governance.rs:1591 | Demote governance.rs:1597"]
    STATE["Demote maps to CaseState.Decided, NOT a dedicated Demoted state - only<br/>Promoted predates this enum's first consumer governance.rs:1864-1869 | Promote maps Promoted governance.rs:1871"]
    GATE["Both variants stay ADMIN-ONLY: 31403 admission already established the<br/>signer is admin or a delegated reviewer response_admission nip_handlers.rs:1384"]
    RATIONALE["Since d5f6308: promote and demote on an ontology case now carry the<br/>SAME rationale rule as approve - outcome_requires_rationale names both<br/>governance.rs:792-795, so a scripted 31403 cannot skip what the UI enforces"]

    ENUM --> STATE --> GATE --> RATIONALE

    N1["OPEN: Demote is no longer brand new (ADR-2013 ontology promotion/demotion) but still has no confirmed<br/>consumer downstream - this repo is a producer.<br/>EXTERNAL: VisionClaw's elevation actor is the candidate consumer, see VC-24"]
```

## NF-10.13 New invariants at af4f0d5 — deletion, sealed originals, zone keys and absence

```mermaid
flowchart TB
    T1["INVARIANT: a tombstone is keyed (id, author), never the id alone - PRIMARY KEY nostr-bbs-relay-worker/src/lib.rs:750,<br/>matched on both nostr-bbs-relay-worker/src/lib.rs:777; ingest refuses a tombstoned event before save_event nip_handlers.rs:1607.<br/>Limits: the lookup fails open nip_handlers.rs:3476; retention and NIP-40 expiry record none cron.rs:454."]
    T2["INVARIANT: below TL3 a kind-5 is accepted only if every e target is canonical and is the signer's own stored event,<br/>a wrap to their recipient key, or an id with no row and an (id, signer) tombstone - gate nip_handlers.rs:3351, classes nip_handlers.rs:3047.<br/>One bad target refuses the whole event; a D1 failure refuses too. Admins skip the block nip_handlers.rs:1259, TL3+ skip the gate nip_handlers.rs:1320.<br/>Exceptions: pre-gate claims satisfy the no-row case nip_handlers.rs:3057 (O13); a kind-5 with no e target passes nip_handlers.rs:3353 (O14)."]
    T3["INVARIANT: the client honours a deletion of a restored sealed original only from its author or a pubkey the relay vouches<br/>is an admin nostr-bbs-forum-client/src/stores/tombstones.rs:123; for an ordinary post every tombstone counts tombstones.rs:121.<br/>Vouching is check-whitelist isAdmin nostr-bbs-forum-client/src/stores/signer_admin.rs:128, the D1 whitelist row alone<br/>nostr-bbs-relay-worker/src/whitelist.rs:139-141 - static ADMIN_PUBKEYS admins and TL3 moderators are not recognised (O25)."]
    T4["INVARIANT (client): no second key is accepted or minted for a zone epoch already held - the slot guard reads and writes<br/>in one IndexedDB transaction nostr-bbs-forum-client/src/stores/indexed_db.rs:443, and plan_mint waits for the sync and mints above<br/>the highest believable epoch nostr-bbs-forum-client/src/zone_crypto/keyset.rs:169. Limits: a post more than MAX_EPOCH_SKIP ahead is<br/>ignored keyset.rs:125; with ZONE_GROUPS_ENABLED on the pubkey joins the store id, so the slot guard cannot fire zone_crypto/mod.rs:206."]
    T5["INVARIANT: a deep-linked page reports not found only on a receipt - Lookup::decide nostr-bbs-forum-client/src/utils/lookup.rs:101;<br/>for a gated zone only an EOSE the REQ ledger calls authenticated settles it utils/lookup.rs:63, ledger relay/receipt.rs:71;<br/>no receipt by the 15 s deadline is TimedOut, not NotFound utils/lookup.rs:38. Since kit #98 an encrypted zone also waits on<br/>the member's zone keys before NotFound utils/lookup.rs:134-151 - O21 closed, NF-10.15."]
    T1 --> T2 --> T3
    T4 --> T5
```

## NF-10.14 Anomaly register — relay deletion items opened 2026-11-22 (O12 to O18, O20)

```mermaid
flowchart TB
    O12["O12 / KF-21: the member gate binds up to MAX_TAG_COUNT targets nip_handlers.rs:42 as one JSON value through json_each<br/>nostr-bbs-relay-worker/src/lib.rs:811; its tests run on in-memory SQLite nip_handlers.rs:7700-7709, never on D1. A D1 failure<br/>refuses the kind-5 nip_handlers.rs:3377, so the risk is refused legitimate deletions, not a hole."]
    O13["O13 / KF-17, NARROWED: process_deletion writes an (id, signer) tombstone for any well-formed target, row or not nip_handlers.rs:3535-3547.<br/>Members can no longer create new no-row claims - Unknown is refused nip_handlers.rs:3058 - but moderators skip the gate and still write<br/>them, and claims written before the gate, including the kind-5 backfill nostr-bbs-relay-worker/src/lib.rs:919, satisfy AlreadyDeleted<br/>nip_handlers.rs:3057, so such a signer below TL3 can re-name the id and every client folds it. Sealed originals are read first nip_handlers.rs:3056."]
    O14["O14 / KF-20: a kind-5 with no e target is accepted nip_handlers.rs:3353-3357, and an unexpired expiration tag on a kind-5 is<br/>admitted filter.rs:375; the retention sweep later reaps it cron.rs:419, a residue the client's deletion sync names<br/>nostr-bbs-forum-client/src/stores/deletion_sync.rs:35-39."]
    O15["O15 / KF-14: no admin route removes one tombstone - the only removal of deleted_events and deleted_addresses is the<br/>whole-database reset nostr-bbs-relay-worker/src/whitelist.rs:524-525. A moderator's or purge's tombstone is permanent."]
    O16["O16 / KF-15: GET /api/reports nostr-bbs-relay-worker/src/moderation.rs:142 lists reportedPubkey moderation.rs:211, which is<br/>the reporter's p-tag claim nip_handlers.rs:4767; the stored author and kind are read only by the resolve route moderation.rs:341."]
    O17["O17 / KF-16: the user purge tombstones and deletes by pubkey only nostr-bbs-relay-worker/src/user_admin.rs:227 user_admin.rs:233,<br/>via nostr-bbs-relay-worker/src/lib.rs:899 - events signed by the member's device keys and wraps addressed to them survive it."]
    O18["O18 / KF-18: a moderator's kind-5 on a sealed original removes an envelope only when its author is a CURRENT admin<br/>nip_handlers.rs:3448, inside delete_sealed_envelopes nip_handlers.rs:3406 - a demoted migrator's envelopes stay readable."]
    O20["O20 / KF-22: the 20-subscription cap is checked nip_handlers.rs:1858-1869 before the wrap_read_set await nip_handlers.rs:1907<br/>and the subscription is inserted after it nip_handlers.rs:1931, so concurrent REQs with distinct ids can exceed the cap.<br/>A REQ replacing a held id is exempt by design nip_handlers.rs:52-53."]
    O12 --> O13 --> O14 --> O15
    O16 --> O17 --> O18 --> O20
```

## NF-10.15 Anomaly register — sealed originals, client and UX items opened 2026-11-22 (O19, O21 to O26)
> **Superseded** by NF-32.4 at de31aaa (frozen at 4a73a7a).

```mermaid
flowchart TB
    O19["O19 / KF-19, NARROWED: a member cannot delete their own migrated post - refusal nip_handlers.rs:2965, the sealed flag is read<br/>before the tombstone flag nip_handlers.rs:3056. The purge DOES write (id, original author) tombstones admin_events.rs:283 through<br/>nostr-bbs-relay-worker/src/lib.rs:870; they guard re-insert at ingest, but the member gate never reaches them."]
    O21["O21 / KF-23 - CLOSED by kit #98, merged as fd31193. A cold deep link in an encrypted zone no longer says Topic Not Found while keys arrive:<br/>Lookup::decide_with_keys waits on the grant backlog once the relay has answered nostr-bbs-forum-client/src/utils/lookup.rs:134-151, the backlog<br/>settles on every grant EOSE with no wrap in flight zone_crypto/grant_backlog.rs:153-159, and both the thread page pages/thread.rs:645-656<br/>pages/thread.rs:668 and the section page pages/section.rs:318 pages/section.rs:334 use it. Operator observation 2026-11-22, same cold device:<br/>Fetching your keys from 9.2 s, topic at 28.4 s, no Not Found (before: Not Found 12.2 s to 28.6 s). DRIFT: KF-23 still reads Open<br/>docs/security/known-findings.md:85, and the governing doc's ADR-2034 section still calls the zone-key wait an open gap<br/>docs/BASELINE-architecture.md:319-320. See NF-30.8."]
    O21r["O21 residue, STILL OPEN: keys still take about 28 s cold - the code's own figure is 28.6 s to the first usable key<br/>zone_crypto/grant_backlog.rs:38-39. The grant filter takes at most 1000 wraps, shared with DMs zone_crypto/store.rs:1138-1148, so an<br/>account with more can miss its oldest grants; paging them is kit PR #95, a draft. A member with no key settles too<br/>nostr-bbs-forum-client/src/utils/lookup.rs:161-163, so a sealed-original topic they cannot open is NotFound once keys settle utils/lookup.rs:147."]
    O22["O22 / KF-24: a write made while the socket is closed bypasses the AUTH deferral, which holds writes only on an open<br/>unauthenticated socket nostr-bbs-forum-client/src/relay.rs:785-804; it is buffered relay.rs:840 and flushed on reopen<br/>before AUTH relay.rs:518-526, where the relay refuses it."]
    O23["O23 / KF-25: deletion paging gives up at MAX_PAGES nostr-bbs-forum-client/src/stores/deletion_sync.rs:123, on a full page with<br/>no progress or on an empty page, keeping the old marker with no resume cursor deletion_sync.rs:30-33, next_page deletion_sync.rs:424."]
    O24["O24 / KF-26, accepted limit: zone-key rows written before per-identity scoping zone_crypto/key_owner.rs:13-19 and pre-#92<br/>bookmarks nostr-bbs-forum-client/src/components/bookmarks_modal.rs:8-11 stay visible to every account on the browser;<br/>same-origin storage is no boundary against the device zone_crypto/key_owner.rs:21-22."]
    O25["O25 / KF-27: the client cannot identify TL3 moderators nostr-bbs-forum-client/src/stores/tombstones.rs:19-21, and check-whitelist sets<br/>isAdmin from the D1 row alone nostr-bbs-relay-worker/src/whitelist.rs:180 while the relay's own check honours static<br/>ADMIN_PUBKEYS first nostr-bbs-relay-worker/src/auth.rs:185 - so T3 in NF-10.13 is narrower than relay admin standing."]
    O26["O26 / KF-28: the Pod Browser mesh card prints the probe's error string verbatim under Details<br/>nostr-bbs-forum-client/src/pages/pod_browser.rs:1056-1058; whether that is a stack trace depends on the failing call, not established."]
    O19 --> O21 --> O21r --> O22
    O23 --> O24 --> O25 --> O26
```

## NF-10.16 Resolved history R1 to R9 — each re-checked at fd31193
> **Superseded** by NF-32.5 at de31aaa (frozen at 4a73a7a).

```mermaid
flowchart TB
    R1["R1 licence prose claimed MIT/Apache on AGPL code - fixed. HOLDS for the application: workspace licence nostr-rust-forum/Cargo.toml:64,<br/>CONTRIBUTING.md:105-106. SUPERSEDED for one crate by a deliberate relicence: nostr-bbs-core is MIT OR Apache-2.0 from beta.18<br/>nostr-bbs-core/Cargo.toml:6 nostr-bbs-core/README.md:10-13, as the README states nostr-rust-forum/README.md:154."]
    R2["R2 version drift README rc11, SECURITY rc tables, NIP-11 serving 3.0.0. HOLDS for NIP-11: version is env CARGO_PKG_VERSION nip11.rs:214.<br/>DRIFT again: SECURITY.md lists 1.0.0-beta.3 as current SECURITY.md:19 above v3.0.0-rc headings SECURITY.md:23, while core is<br/>1.0.0-beta.20 nostr-bbs-core/Cargo.toml:3 and the relay 1.0.0-beta.10 nostr-bbs-relay-worker/Cargo.toml:3."]
    R3["R3 NIP-17 over-advertised - HOLDS: NIP-11 lists 59, not 17 nip11.rs:16, saying why nip11.rs:205-208,<br/>and the architecture doc says there is no kind-14 inbox routing docs/architecture.md:221.<br/>DRIFT for the client: the NIP-17 overlay is on main, dormant behind RECIPIENT_KEYS_ENABLED nostr-rust-forum/README.md:124<br/>nostr-bbs-forum-client/src/dm/mod.rs:1632, and DM sends publish a kind-10050 DM-relay list once per pubkey per session,<br/>ungated, nostr-bbs-forum-client/src/dm/mod.rs:610-611<br/>nostr-bbs-forum-client/src/dm/mod.rs:611 nostr-bbs-forum-client/src/dm/mod.rs:2012, so that sentence holds only for the relay. See NF-01."]
    R4["R4 NIP-26 deleted as superseded by ADR-099 device keys - HOLDS for core and the auth worker; CHANGELOG.md:2139-2140 records NIP-26 rejected.<br/>NOT GONE from the workspace: nostr-bbs-mesh carries its own NIP-26 verifier for mesh delegation nostr-bbs-mesh/src/delegation.rs:1,<br/>noting core has no nip26 module nostr-bbs-mesh/src/delegation.rs:22-28; the upstream-vector suite still wires the fixture all_fixtures.rs:59-60."]
    R5["R5 NIP-90 DVM module deleted - HOLDS: removal recorded nip11.rs:209-211 and 90 not advertised nip11.rs:16.<br/>DOC-DRIFT: the architecture note still says data vending machines are validated at ingress docs/architecture.md:216-217."]
    R6["R6 template drift - HOLDS: SETUP lists ZONE_CONFIG and DEVICE_KEYS_ENABLED for the relay SETUP.md:150 and explains the flag SETUP.md:175-178;<br/>the auth template binds RELAY_DB nostr-bbs-auth-worker/wrangler.toml:18; the relay template declares both nostr-bbs-relay-worker/wrangler.toml:26<br/>nostr-bbs-relay-worker/wrangler.toml:29. ADMIN_PUBKEYS stays undeclared - S2 in NF-10.7."]
    R7["R7 ad-hoc did:nostr format - MOSTLY HOLDS: production sites call did_nostr_uri nostr-bbs-core/src/did.rs:141 and keep format! only as the<br/>parse-failure fallback webauthn.rs:1114-1116 nostr-bbs-pod-worker/src/lib.rs:903-905. RESIDUE: the mesh AS2 projection formats it<br/>directly nostr-bbs-mesh/src/envelope.rs:395-396."]
    R8["R8 dual auth cors_headers with an example.com fallback - HOLDS: one builder, fail-closed, no Allow-Origin when EXPECTED_ORIGIN is<br/>unset or blank nostr-bbs-auth-worker/src/http.rs:19-30, imported by the worker nostr-bbs-auth-worker/src/lib.rs:38. See NF-02.1."]
    R9["R9 architecture doc coverage - HOLDS: rate-limit crate row docs/architecture.md:184, NIP-56/59/65/90 note docs/architecture.md:216,<br/>ADR-098/099 section docs/architecture.md:481; ADR-099 status Accepted docs/archive/adr/ADR-099-revocable-device-keys.md:4.<br/>Its NIP-90 sentence is the R5 drift."]
    R1 --> R2 --> R3 --> R4 --> R5
    R6 --> R7 --> R8 --> R9
```

## NF-10.17 Open — the room-dialect sequence this corpus still owes (ADR-2029)

```mermaid
flowchart LR
    ADR["ADR-2029 room profile: proposed, implementation none<br/>docs/adr/ADR-2029-room-profile-deployment-type.md:5-6"]
    DESIGN["Three design-level diagrams only - wire shape, who reads what, choosing<br/>docs/deployment-profiles.md:41 docs/deployment-profiles.md:69 docs/deployment-profiles.md:105<br/>not citation-verified, so NOT copied into this corpus"]
    CODE["Code today: the h-scoped group builders are deprecated with no caller,<br/>citing ADR-2029 nostr-bbs-core/src/groups.rs:124"]
    OWED["OPEN: when ADR-2029 is implemented, this VisionFlow area gains a room-dialect topic -<br/>kind 34550 definition publish, kind 9 post and q reply, edit by the Anagram convention,<br/>kind 7 and 5 with h and k, the relay's dialect-selected ban gate and trust counters,<br/>each cited to the lines that implement them"]
    ADR --> DESIGN --> OWED
    CODE -.-> OWED
```
