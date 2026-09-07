---
id: NF-10
title: Invariants, the re-verified anomaly register and the doc-drift ledger
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2002, ADR-2003, ADR-2004, ADR-2005, ADR-2006, ADR-2007, ADR-2008, ADR-2009, ADR-2010]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs
  - ../nostr-rust-forum/docs/diagrams/00-anomaly-register.md
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/SETUP.md
  - ../nostr-rust-forum/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/acl.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/src/ssrf.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/stores/channels.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/dm/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/settings.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/relay_url.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/mock.rs
  - ../nostr-rust-forum/crates/nostr-bbs-upstream-canary/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/tests/upstream_vectors/mod.rs
  - ../nostr-rust-forum/docs/consumer-surface-map.md
  - ../nostr-rust-forum/.github/workflows/ci.yml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs
verified_commit: worktree-2026-09-07
---

## NF-10.1 The compliance surface — BASELINE-architecture invariants

```mermaid
flowchart LR
    I1["1 nostr-bbs-core owns on-wasm32 Schnorr until the canary records Shape A<br/>nostr-bbs-upstream-canary/src/lib.rs:17 - see NF-01.5"]
    I2["2 FORUM_BASE is applied in exactly TWO places<br/>Router base nostr-bbs-forum-client/src/app.rs:817 | base_href nostr-bbs-forum-client/src/app.rs:50 - see NF-05.1"]
    I3["3 channel counts are derived, never accumulated<br/>count_for nostr-bbs-forum-client/src/stores/channels.rs:159 | dedup on insert channels.rs:444 - see NF-05.6"]
    I4["4 .acl and .meta sidecar access coerces to Control<br/>nostr-bbs-pod-worker/src/acl.rs:85 | shared policy nostr-bbs-pod-worker/src/acl.rs:34 - see NF-04.3"]
    I5["5 gift-wraps are recipient-whitelist-gated, never author-gated<br/>gift_wrap_recipient nip_handlers.rs:114 | admission nip_handlers.rs:549 - see NF-03.5"]
    I6["6 solid-pod-rs stays an EXACT pin<br/>nostr-rust-forum/Cargo.toml:155 - see NF-01.4"]

    I1 --> I2 --> I3
    I4 --> I5 --> I6

    N1["These six are the compliance surface of BASELINE-architecture.md. Each has at least one diagram in<br/>this area that shows the mechanism enforcing it, not just the rule."]
    N2["Any change to a crate topology, the absorption status, the pod tiers, ACL coercion or a kit version<br/>must confirm the touched invariant still holds and bump the governing doc in the SAME change."]
```

## NF-10.2 The compliance surface — IDENTITY invariants

```mermaid
flowchart LR
    K1["1 derive_subkey is a byte-for-byte cross-stack contract with agentbox's JS mirror derivation<br/>raw HMAC, not HKDF nostr-bbs-core/src/keys.rs:253 | vector nostr-bbs-core/src/keys.rs:478 | digest keys.rs:483 - see NF-09.3"]
    K2["2 derive_from_prf and derive_subkey are DISTINCT and never substitutable<br/>HKDF path nostr-bbs-core/src/keys.rs:197 | contrast documented nostr-bbs-core/src/keys.rs:224"]
    K3["3 DEVICE_KEYS_ENABLED enables only on the exact string true, checked in both workers<br/>auth gate nostr-bbs-auth-worker/src/devices.rs:112 | asserted nostr-bbs-auth-worker/src/devices.rs:974 - see NF-02.7"]
    K4["4 gift-wrap admission keys on the recipient p tag, never the ephemeral author<br/>nip_handlers.rs:114 - see NF-03.5"]
    K5["5 TL3 is never auto-demoted; demotion stays time-driven on cron, never inline on admission<br/>nostr-bbs-relay-worker/src/trust.rs:334 - see NF-03.8"]
    K6["6 a derived subkey is ROOT-RECOVERABLE - domain separation, not compromise isolation<br/>nostr-bbs-core/src/keys.rs:240"]

    K1 --> K2 --> K3
    K4 --> K5 --> K6

    N1["K6 is load-bearing for the whole device-key model: rotation of the ROOT, not per-subkey secrecy, is<br/>the compromise boundary. Any design that assumes a compromised subkey is contained is wrong."]
    N2["K1 and K4 are the two invariants that cross a repo boundary - EXTERNAL: see AB-11 and ES-04 for the<br/>agentbox side of the derivation contract"]
```

## NF-10.3 Anomaly register — closed and still live

```mermaid
flowchart TB
    subgraph closed["Confirmed CLOSED at this commit"]
        O1["O1 TL1 promotion and demotion unwired - CLOSED.<br/>Reads tally then check_promotion nip_handlers.rs:889; demotion is cron-driven. See NF-03.8."]
        O3["O3 KV ACL fast-path masks R2 delegation - CLOSED.<br/>R2 authoritative, KV miss-fallback only nostr-bbs-pod-worker/src/acl.rs:311, rationale acl.rs:253. See NF-04.4."]
    end
    subgraph live["Confirmed STILL LIVE"]
        O2["O2 NIP-29 group metadata 39000-39002 accepted from admin CLIENTS rather than relay-key-signed.<br/>Acknowledged TODO nip_handlers.rs:700, admin gate nip_handlers.rs:707. Spec drift. See NF-03.4 step 11."]
        O5["O5 deleting a kind-40 destroys the is_channel_creator lookup, locking a TL2 author out of their own<br/>channel metadata - lookup nip_handlers.rs:660, deletion path nip_handlers.rs:896. See NF-03.7."]
        O11a["O11 KV and SESSIONS are the SAME physical namespace id -<br/>nostr-bbs-auth-worker/wrangler.toml:24 and nostr-bbs-auth-worker/wrangler.toml:39. See NF-08.4."]
        O11b["O11 /api/native-pod/provision has no in-repo caller and placeholder vars -<br/>route nostr-bbs-auth-worker/src/lib.rs:578, NATIVE_POD_URL nostr-bbs-auth-worker/wrangler.toml:52. See NF-02.2."]
    end
```

## NF-10.4 Anomaly register — refined or corrected by this pass

```mermaid
flowchart TB
    O4["O4 members-vs-whitelist admin split - PARTLY fixed, then broken differently.<br/>Both workers now union static plus both tables (nostr-bbs-auth-worker/src/admin.rs:57, nostr-bbs-relay-worker/src/auth.rs:179),<br/>but the relay queries members in its OWN D1 (nostr-bbs-relay-worker/src/auth.rs:192) where no migration creates it.<br/>See NF-08.7."]
    O6["O6 NIP-07 silent no-op DM subscription - CORRECTED. The subscription registers unconditionally<br/>(nostr-bbs-forum-client/src/dm/mod.rs:229). The real limit is per-event: a NIP-04-only extension fails gift-wrap<br/>unwrap at nostr-bbs-forum-client/src/dm/mod.rs:647 and it IS surfaced via state.error at dm/mod.rs:648. See NF-05.9."]
    O7["O7 NIP05_USERNAME_HOST hardcoded - NARROWER than filed. nostr-bbs-forum-client/src/pages/settings.rs:32 defines<br/>and settings.rs:334 uses it; signup.rs reads no such constant. A settings-only display fallback. See NF-05.7."]
    O9["O9 nostr-bbs-mesh has no impl AND no relay import - HALF holds. No production MeshSocket impl exists<br/>(only nostr-bbs-mesh/src/mock.rs:249), but the relay DOES declare the dependency nostr-bbs-relay-worker/Cargo.toml:26.<br/>See NF-09.8."]
    O10["O10 wasm_bridge has no JS consumers - CONFIRMED. The upstream-vector suite is soft-skipped, not<br/>failed, when fixtures are absent (nostr-bbs-core/tests/upstream_vectors/mod.rs:11) and no workflow runs<br/>sync-fixtures.sh. See NF-09.3."]
    O11c["O11 broker_decisions is write-only - CORRECTED. A read endpoint exists:<br/>GET /api/governance/decisions with pagination and a case filter,<br/>nostr-bbs-auth-worker/src/governance_api.rs:503. See NF-06.9."]
```

## NF-10.5 O8 duplications — each re-verified

```mermaid
flowchart LR
    D1["provision_pod TWICE<br/>nostr-bbs-forum-client/src/pages/signup.rs:174<br/>nostr-bbs-forum-client/src/pages/settings.rs:1752<br/>CONFIRMED - two independent implementations"]
    D2["qr_svg TWICE<br/>nostr-bbs-forum-client/src/components/recovery_sheet.rs:66<br/>nostr-bbs-forum-client/src/utils/devices.rs:236<br/>CONFIRMED"]
    D3["relay URL resolved TWICE<br/>inline nostr-bbs-forum-client/src/relay.rs:964<br/>shared helper nostr-bbs-forum-client/src/utils/relay_url.rs:121<br/>CONFIRMED - and the FALLBACKS DIVERGE<br/>nostr-bbs-forum-client/src/relay.rs:22 versus nostr-bbs-forum-client/src/utils/relay_url.rs:13"]
    D4["per-page kind-0 subscription duplicating the app-root ProfileCache<br/>root sub nostr-bbs-forum-client/src/app.rs:736<br/>page sub nostr-bbs-forum-client/src/pages/settings.rs:435<br/>CONFIRMED - the page sub self-cancels after 5 s settings.rs:450"]

    N1["D3 is the only one with a behavioural consequence rather than mere duplication: two different default<br/>relay URLs mean a deployment that forgets window.__ENV__ reaches a DIFFERENT relay depending on which<br/>code path asked first. See NF-05.7 N2."]
```

## NF-10.6 New doc-drift found in this pass

```mermaid
flowchart TB
    DD1["README.md:383 calls NIP-42 AUTH scaffolded with auth_required false. The code says otherwise:<br/>nip42 is the DEFAULT nip42.rs:62, the template ships AUTH_MODE nip42 nostr-bbs-relay-worker/wrangler.toml:31,<br/>and every EVENT passes the gate first nip_handlers.rs:535. The row understates a shipped feature.<br/>See NF-03.3."]
    DD2["README.md:384 says nostr-bbs-mesh is NOT a dependency of the relay-worker.<br/>nostr-bbs-relay-worker/Cargo.toml:26 declares it. See NF-09.8."]
    DD3["nostr-bbs-relay-worker/wrangler.toml:10 asserts there is no ADMIN_PUBKEYS reader in src/.<br/>nostr-bbs-relay-worker/src/auth.rs:183 reads it, and so does nostr-bbs-auth-worker/src/admin.rs:71.<br/>Only nostr-bbs-search-worker/wrangler.toml:33 declares the var. See NF-08.7."]
    DD4["nostr-rust-forum/Cargo.toml:49 workspace.metadata.ci.wasm-check-packages names two crates and is read by NOTHING -<br/>the wasm job checks the whole workspace .github/workflows/ci.yml:174. Inert metadata. See NF-01.2."]
    DD5["nostr-bbs-upstream-canary/src/lib.rs:10 names a five-NIP build matrix but only three smokes exist<br/>nostr-bbs-upstream-canary/src/lib.rs:35 nostr-bbs-upstream-canary/src/lib.rs:56 nostr-bbs-upstream-canary/src/lib.rs:89 - nip04, nip59 and nip98 are unexercised. The same doc calls the crate<br/>nostr-upstream-canary nostr-bbs-upstream-canary/src/lib.rs:14 while the package is nostr-bbs-upstream-canary. See NF-01.5."]
    DD6["The search-worker cron runs every five minutes and reindexes NOTHING - the handler body is a<br/>load_store warm touch nostr-bbs-search-worker/src/lib.rs:708. See NF-07.1."]
    DD7["nostr-bbs-core/src/governance.rs:295 KIND_GOVERNANCE_AUDIT_LOG is numerically the SAME kind as<br/>KIND_PANEL_RETIRED nostr-bbs-core/src/governance.rs:32, and only three of the six documented ACS types have<br/>Rust structs. See NF-06.1 and NF-06.2."]
    DD8["docs/consumer-surface-map.md:20 lists the git panel as calling /.well-known/apps; the client fetches<br/>/apps/manifest.json and no .well-known path exists in that crate. See NF-05.10."]
```

## NF-10.7 Deployment-shaped divergences

```mermaid
flowchart TB
    S1["The relay's two most load-bearing tables - events and whitelist - are created by a SETUP.md<br/>copy-paste SETUP.md:69 SETUP.md:82, not by a repo migration, while every other table is created<br/>idempotently in code. See NF-08.6."]
    S2["ADMIN_PUBKEYS is read by auth and relay but declared in neither template and named in no SETUP step<br/>SETUP.md:119. A by-the-book deployment has no static admin bootstrap. See NF-08.7."]
    S3["The preview worker's SSRF guard is denylist-only without PREVIEW_ALLOWED_HOSTS, which the template<br/>never sets nostr-bbs-preview-worker/wrangler.toml:13. The code says so itself - the Workers runtime exposes no<br/>resolve-then-pin primitive nostr-bbs-preview-worker/src/ssrf.rs:13 - and there is no wall-clock timeout.<br/>See NF-07.6."]
    S4["DEVICE_KEYS_ENABLED ships false in BOTH templates nostr-bbs-auth-worker/wrangler.toml:55<br/>nostr-bbs-relay-worker/wrangler.toml:21, so the whole ADR-099/100 device story is dormant by default and<br/>revocation has no effect at AUTH. See NF-02.7."]
    S5["MESH_ALLOWED_REMOTE_DIDS ships empty nostr-bbs-relay-worker/wrangler.toml:55, so the federated-kind gate at<br/>nip_handlers.rs:584 is inert. Standalone is the only supported mode. See NF-03.13."]
    S6["Neither anti-drift-lint.sh nor identity-vector-parity.mjs is invoked by any workflow, so the<br/>ADR-2003 cross-stack parity proof is manual on the JS side. See NF-09.3."]
```

## NF-10.8 ADR-2006 and ADR-2010 — what the closeout holds open

```mermaid
flowchart LR
    A2006["ADR-2006 trust demotion<br/>accepted, complete, live"]
    Q2006["Current sweep: TL2 may move directly to TL0; stable keyset paging and explicit<br/>commit/error outcomes replace OFFSET and ignored writes. D1 batches state and audit.<br/>Conditional audit now requires a changed row and all observed policy inputs; see NF-10.10."]
    A2010["ADR-2010 governance receipts<br/>proposed, partial, INACTIVE"]
    Q2010["Relay OK establishes acceptance only. The full signed / accepted / projection-committed /<br/>consumer-received / applied contract needs agreement with the authority consumer and the mutation<br/>owner, plus failure and restart evidence."]

    A2006 --> Q2006
    A2010 --> Q2010

    N1["DOC-DRIFT: the IDENTITY-keys-and-trust closeout still states OFFSET row-skipping and ignored write<br/>errors as CURRENT defects. Both are fixed - keyset paging nostr-bbs-relay-worker/src/trust_sweep.rs:432<br/>and confirmed-commit-only counters nostr-bbs-relay-worker/src/trust_sweep.rs:487. The governing doc needs<br/>the qualification retired; see NF-11.11 and NF-11.12."]
    N2["DOC-DRIFT: the relay-side receipt machine is REAL, not proposed - stages at<br/>nostr-bbs-relay-worker/src/relay_do/receipts.rs:76, applied at receipts.rs:334, and handle_event logs<br/>accepted-but-not-applied from the returned receipt nip_handlers.rs:947. Only the CONSUMER half is absent.<br/>See NF-11.9 and NF-11.10."]
    N3["EXTERNAL: both remaining halves belong to other repos - VisionClaw's elevation consumer see VC-24,<br/>agentbox's approvals pipeline see AB-14, the estate loop see ES-05"]
```

## NF-10.9 Coverage map

```mermaid
flowchart LR
    NF01["NF-01 workspace, five workers, entry points, canary"]
    NF02["NF-02 auth-worker: passkey, NIP-98, membership, REST"]
    NF03["NF-03 relay-worker: AUTH, admission, trust, federation"]
    NF04["NF-04 pod-worker: LDP, WAC, delegation, quota, payments"]
    NF05["NF-05 both Leptos clients"]
    NF06["NF-06 Agent Control Surface 31400-31405"]
    NF07["NF-07 search, preview, rate-limit, ascii"]
    NF08["NF-08 config, zones, stores, admin authority"]
    NF09["NF-09 CI, deploy, fixtures, benchmarks, e2e, setup-skill"]
    NF10["NF-10 invariants and anomalies (this file)"]

    NF01 --> NF02 & NF03 & NF04 & NF07 & NF05
    NF03 --> NF06
    NF02 --> NF06
    NF08 --> NF02 & NF03 & NF04 & NF05
    NF09 --> NF01
    NF10 --> NF01 & NF02 & NF03 & NF04 & NF05 & NF06 & NF07 & NF08 & NF09

    N1["Crates with no dedicated topic are covered inside another: nostr-bbs-core across NF-01, NF-02, NF-06<br/>and NF-09; nostr-bbs-config in NF-08 and NF-09; nostr-bbs-mesh in NF-03.13 and NF-09.8;<br/>nostr-bbs-rate-limit and nostr-bbs-ascii in NF-07.8; nostr-bbs-setup-skill in NF-09.9;<br/>nostr-bbs-upstream-canary in NF-01.5 and NF-09.8."]
    N2["EXTERNAL edges out of this repo: did:nostr and key derivation to AB-11 and ES-04; the ACS consumers<br/>to VC-24 and AB-14 with the estate loop at ES-05; Solid pods to the solid-pod-rs area (SP-*) and ES-08;<br/>the downstream kit consumer to the dreamlab-ai-website area (DW-*)."]
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
