---
id: NF-04
title: pod-worker — Solid LDP surface, WAC evaluation, delegation, quota and payments
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2009, ADR-2007, ADR-2012]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/acl.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/payments.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/pay_ledger.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/deposit_address.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/quota.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/patch.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/content_negotiation.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/conditional.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/container.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/contexts.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/notifications.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/remote_storage.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/git.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/webid.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/did.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/auth.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/export.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/tests/wac_proptests.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/cors.rs
  - ../nostr-rust-forum/docs/consumer-surface-map.md
  - ../nostr-rust-forum/docs/adr/ADR-2012-d1-ledger-becomes-a-chain-view.md
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
  - ../nostr-rust-forum/Cargo.toml
verified_commit: 72463fbde35ac4c68539b1f65a08ff03b9941201
---

## NF-04.1 Route surface

```mermaid
flowchart TB
    F["fetch nostr-bbs-pod-worker/src/lib.rs:422"]
    OPT["Options preflight lib.rs:427"]
    HEALTH["/health lib.rs:437"]
    subgraph disc["Discovery documents"]
        WF["/.well-known/webfinger lib.rs:472"]
        SOL["/.well-known/solid lib.rs:495"]
        N05["/.well-known/nostr.json lib.rs:506"]
        DID["/.well-known/did/nostr/{pk} lib.rs:562"]
        WL["/.well-known/webledgers/webledgers.json lib.rs:583"]
    end
    PODS["POST /.pods - authenticated provisioning alias lib.rs:594"]
    PAY["/pay/ routes - HTTP 402 Web Ledgers lib.rs:652 lib.rs:689"]
    GITG["_git request guard lib.rs:743"]
    LDP["/pods/{pubkey}/... LDP resource verbs<br/>parsed at lib.rs:94<br/>GET/HEAD lib.rs:991 PUT lib.rs:1217 POST lib.rs:1293<br/>PATCH lib.rs:1429 DELETE lib.rs:1521"]
    PROV["/.provision lib.rs:124 and /.deprovision lib.rs:129"]

    F --> OPT --> HEALTH
    F --> disc
    F --> PODS & PAY & GITG & LDP & PROV

    N1["CORS uses the canonical POD_CORS_HEADERS constant from core so the DPoP, Updates-Via, WAC and<br/>payment headers cannot drift per worker nostr-bbs-pod-worker/src/lib.rs:170, constant<br/>nostr-bbs-core/src/cors.rs:46"]
    N2["Every LDP response advertises resource-sidecar notification discovery via Updates-Via<br/>nostr-bbs-pod-worker/src/lib.rs:277"]
    N3["A 401 emits a JSS-compatible Solid challenge - WWW-Authenticate with DPoP and Bearer realms<br/>nostr-bbs-pod-worker/src/lib.rs:348"]
```

## NF-04.2 A resource request, end to end

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant W as pod-worker fetch<br/>nostr-bbs-pod-worker/src/lib.rs:422
    participant A as NIP-98 verify<br/>nostr-bbs-pod-worker/src/auth.rs:13
    participant ACL as find_effective_acl<br/>nostr-bbs-pod-worker/src/acl.rs:285
    participant R2 as PODS bucket
    participant Q as quota

    C->>W: METHOD /pods/{pubkey}/{path}
    W->>W: parse owner + resource path lib.rs:94
    W->>W: git request guard lib.rs:743
    W->>A: Authorization: Nostr <base64 event> lib.rs:766
    W->>W: required_mode = coerce_required_mode_for_acl lib.rs:966
    W->>ACL: resolve the effective ACL lib.rs:967
    ACL->>R2: sidecar walk, most specific first acl.rs:295
    ACL->>ACL: KV miss-fallback only acl.rs:311
    W->>W: evaluate_access against the resolved doc acl.rs:56
    alt GET or HEAD
        W->>W: content negotiation lib.rs:1062, preconditions lib.rs:1081
    else PUT
        W->>Q: atomic check_and_reserve_d1 lib.rs:1258
        W->>W: notify_change subscribers lib.rs:1279
    else POST or PATCH
        W->>Q: reservation_account then check_and_reserve_with_limit_d1 lib.rs:1315 lib.rs:1320
    end
    W-->>C: response with the WAC allow header

    Note over W: The quota reservation is ATOMIC in D1 - reserve-then-write, so a concurrent PUT cannot both pass a stale check nostr-bbs-pod-worker/src/quota.rs:110
    Note over W: Teardown releases the owner's quota best-effort - a stale quota row must not fail the deprovision nostr-bbs-pod-worker/src/lib.rs:922
    Note over A: Replay protection shares the auth-worker's D1 through REPLAY_DB nostr-bbs-pod-worker/wrangler.toml:25 - see NF-02.5 and NF-08.4
```

## NF-04.3 The sidecar escalation guard — ADR-2009

```mermaid
flowchart TB
    REQ["incoming method + path"]
    BASE["base = method_to_mode(method)<br/>nostr-bbs-pod-worker/src/acl.rs:86"]
    COERCE["coerce_required_mode_for_acl<br/>nostr-bbs-pod-worker/src/acl.rs:85"]
    SHARED["delegates to the SHARED upstream policy effective_acl_target<br/>re-exported nostr-bbs-pod-worker/src/acl.rs:34"]
    CTRL["a .acl or .meta path collapses to AccessMode::Control<br/>nostr-bbs-pod-worker/src/acl.rs:46"]
    NORM["a normal path keeps its method-derived mode"]

    REQ --> BASE --> COERCE --> SHARED
    SHARED --> CTRL
    SHARED --> NORM

    N1["INVARIANT ADR-2009: a sidecar governs another resource's AUTHORISATION GRAPH, so WAC section 4.3.5<br/>requires acl:Control for ANY access - read AND write alike nostr-bbs-pod-worker/src/acl.rs:73-74"]
    N2["This closes BOTH holes at once: the write escalation - a mere acl:Write holder seizing acl:Control by<br/>overwriting a sidecar - and the read-side disclosure P2-1, an acl:Read holder reading the graph<br/>nostr-bbs-pod-worker/src/acl.rs:80-82"]
    N3["The kit no longer re-derives the rule; it delegates to the shared upstream policy, so the forum and<br/>the pod server cannot drift on what counts as a sidecar nostr-bbs-pod-worker/src/acl.rs:75-77.<br/>EXTERNAL: the policy lives in solid-pod-rs - see the solid-pod-rs area (SP-*)"]
    N4["Never widen this. Widening reopens both holes - BASELINE-architecture invariant 4."]
```

## NF-04.4 ACL resolution order — R2 authoritative, KV fallback only

```mermaid
flowchart TB
    START["find_effective_acl<br/>nostr-bbs-pod-worker/src/acl.rs:285"]
    SEQ["acl_probe_sequence - ordered, most specific first<br/>nostr-bbs-pod-worker/src/acl.rs:180"]
    OWN["1 the resource's OWN flat sidecar {path}.acl, inherited = false<br/>nostr-bbs-pod-worker/src/acl.rs:190"]
    CONT["2 each enclosing container sidecar {dir}/.acl, inherited = true<br/>nostr-bbs-pod-worker/src/acl.rs:209"]
    LEG["2b legacy flat ancestor form {dir}.acl, also inherited<br/>nostr-bbs-pod-worker/src/acl.rs:217"]
    HIT["first parseable R2 hit WINS acl.rs:295 acl.rs:329"]
    KV["3 KV acl:{owner_pubkey} - reached ONLY when the R2 walk resolved nothing<br/>nostr-bbs-pod-worker/src/acl.rs:311, parser acl.rs:339"]
    NONE["None - deny"]

    START --> SEQ --> OWN --> CONT --> LEG --> HIT
    HIT -.->|"no hit anywhere"| KV --> NONE

    N1["INVARIANT: KV can NEVER shadow a more-specific R2 sidecar, so it can never mask a delegation grant<br/>nostr-bbs-pod-worker/src/acl.rs:253-273. This is the R11 / O3 fix, re-verified live."]
    N2["The inherited flag matters: for an ANCESTOR container only acl:default rules may apply per WAC 4.2,<br/>which the upstream evaluator gates on AclDocument::inherited<br/>nostr-bbs-pod-worker/src/acl.rs:161-164; the resource's own sidecar is non-inherited so its<br/>acl:accessTo applies directly acl.rs:166-167"]
    N3["Kit-specific hardening on top of upstream: a stricter 64 KiB ACL document cap<br/>nostr-bbs-pod-worker/src/acl.rs:93, parser at acl.rs:111"]
    N4["resolve_effective_acl acl.rs:360 exists so the ordering is unit-testable WITHOUT the worker R2/KV<br/>runtime types - the ordering rule is provable in a native test"]
```

## NF-04.5 Container delegation — the owner is never coerced out

```mermaid
sequenceDiagram
    autonumber
    participant O as Pod owner
    participant W as pod-worker
    participant B as build_delegation_acl<br/>nostr-bbs-pod-worker/src/acl.rs:402
    participant R2 as container sidecar

    O->>W: PUT a delegation to {container}/.acl
    W->>W: sidecar access coerces to Control - only the owner can do this, see NF-04.3
    W->>B: owner_did, agent_did, container_path, modes
    B->>B: normalise the container path, keeping a trailing slash so acl:accessTo names the container acl.rs:411
    B->>B: emit owner Control plus agent modes MINUS Control acl.rs:394
    B->>B: modes are deduped into canonical Read, Write, Append, Control order acl.rs:423
    B-->>W: AclDocument AST
    W->>R2: serialise to the canonical wire shape
    W->>W: preserves_owner_control assertion acl.rs:123

    Note over B: INVARIANT: the delegate never gets Control, so it can neither re-delegate nor seize the container nostr-bbs-pod-worker/src/acl.rs:394-396
    Note over B: DIDs are written verbatim as did:nostr hex form - callers validate the shape upstream nostr-bbs-pod-worker/src/acl.rs:397-398
    Note over R2: The emitted doc round-trips cleanly through this crate's own parser - that round-trip is what makes the grant readable by the next request's walk-up
```

## NF-04.7 The LDP mechanics around each verb

```mermaid
classDiagram
    class ContentNegotiation {
        parse_accept : content_negotiation.rs:27
        negotiate : content_negotiation.rs:86
        is_rdf_type : content_negotiation.rs:146
        ensure_solid_context : content_negotiation.rs:157
        JSONLD TURTLE NTRIPLES HTML : content_negotiation.rs:7
    }
    class Conditional {
        check_preconditions ETag : conditional.rs:14
        parse_range : conditional.rs:39
    }
    class Container {
        is_container : container.rs:10
        list_container : container.rs:16
        resolve_slug : container.rs:73
    }
    class Patch {
        apply_patches : patch.rs:15
    }
    class Notifications {
        subscribe : notifications.rs:25
        unsubscribe : notifications.rs:53
        notify_change : notifications.rs:94
    }
    class Contexts {
        foaf solid acl did-v1 bundled inline : contexts.rs:15
        context_for_iri : contexts.rs:38
    }
    class Quota {
        INBOX_QUOTA 5 MiB : quota.rs:23
        reservation_account : quota.rs:26
        check_and_reserve_d1 : quota.rs:110
        get/set_quota_d1 : quota.rs:167
        legacy KV path : quota.rs:216
    }

    note for Contexts "JSON-LD contexts are compiled INTO the worker<br/>with include_str! rather than fetched - a Worker<br/>cannot depend on an outbound context fetch at<br/>request time contexts.rs:15"
    note for Quota "Two quota backends coexist - the D1 path is<br/>authoritative and atomic quota.rs:110 while a<br/>legacy KV path survives quota.rs:216, the same<br/>R2-authoritative / KV-legacy shape as the ACL resolver"
    note for Patch "apply_patches is the PATCH verb's whole<br/>surface patch.rs:15, reached from lib.rs:1422"
```

## NF-04.8 Discovery, identity and the git boundary

```mermaid
flowchart LR
    WFR["webfinger_response remote_storage.rs:10<br/>parse_webfinger_resource remote_storage.rs:38"]
    SOLD["solid_discovery remote_storage.rs:58"]
    NJ["nostr_json - the pod-resident NIP-05 doc remote_storage.rs:82"]
    WID["webid re-export of solid_pod_rs::webid nostr-bbs-pod-worker/src/webid.rs:17"]
    DIDM["did re-export - render_did_document_tier3, verify_webid_tag<br/>nostr-bbs-pod-worker/src/did.rs:11<br/>renders from the hex identifier only, so documents carry the fe70102 Multikey did.rs:7"]
    GITG["is_git_request git.rs:44 | is_dot_git_path git.rs:53<br/>git_dir_forbidden git.rs:58 | git_not_implemented git.rs:76"]

    N1["INVARIANT: .git paths are FORBIDDEN, not merely unimplemented - the two outcomes are distinct<br/>responses nostr-bbs-pod-worker/src/git.rs:58 versus nostr-bbs-pod-worker/src/git.rs:76. The CF-Workers<br/>tier is non-git BY DESIGN; the git-capable pod is the agentbox native tier."]
    N2["EXTERNAL: the forum client's git control panel talks to the NATIVE server's /_git/* REST API, never<br/>to this worker - see NF-05.10 and the solid-pod-rs area (SP-*)"]
    N3["This worker serves a pod-resident /.well-known/nostr.json nostr-bbs-pod-worker/src/lib.rs:506, while<br/>the AUTH worker owns the central NIP-05 registry - see NF-02.2"]
```

## NF-04.9 HTTP 402 payments

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant W as pod-worker
    participant PAY as payments::handle_pay_route<br/>nostr-bbs-pod-worker/src/payments.rs:157
    participant D1 as payment schema
    participant CH as chain

    W->>W: PAY_ENABLED and PAY_COST_SATS from env nostr-bbs-pod-worker/wrangler.toml:31 wrangler.toml:32
    C->>W: request a paid resource
    W->>PAY: /pay/ route dispatch nostr-bbs-pod-worker/src/lib.rs:689
    PAY->>D1: ensure_payment_schema over pay_ledger SCHEMA payments.rs:45 pay_ledger.rs:130
    PAY->>CH: deposit credits a TXO only via qualifying_output - it must pay the pod's<br/>deposit address, keyed with its chain pay_ledger.rs:564 pay_ledger.rs:618
    PAY->>PAY: .address derives the frozen deposit address deposit_address.rs:120 pay_ledger.rs:802
    PAY->>D1: recover_orphaned_jobs pay_ledger.rs:1273
    PAY-->>C: 402 with the Web Ledgers payment body

    Note over PAY: The discovery/config vocabulary is re-exported from solid_pod_rs::payments rather than reimplemented nostr-bbs-pod-worker/src/payments.rs:34 - EXTERNAL: see the solid-pod-rs area (SP-*)
    Note over W: The shipped template has PAY_ENABLED "false" nostr-bbs-pod-worker/wrangler.toml:31<br/>DRIFT: production sets it "true" - ADR-2012's 2026-10-02 amendment withdraws the<br/>dormant assumption ADR-2012-d1-ledger-becomes-a-chain-view.md:35-40
    Note over PAY: The orphaned-job recovery path exists because a chain confirmation can outlive a Worker invocation pay_ledger.rs:1273
```

## NF-04.10 The three wasm32-unreachable Phase-1 surfaces

```mermaid
flowchart TB
    FEAT["solid-pod-rs-phase1 feature<br/>nostr-bbs-pod-worker/Cargo.toml:22"]
    F1["solid-pod-rs/provision-keys nostr-bbs-pod-worker/Cargo.toml:23"]
    F2["solid-pod-rs/nip05-endpoint nostr-bbs-pod-worker/Cargo.toml:24"]
    F3["solid-pod-rs/export-jsonld nostr-bbs-pod-worker/Cargo.toml:25"]
    EXPORT["export.rs is a bare re-export of solid_pod_rs::export<br/>nostr-bbs-pod-worker/src/export.rs:19"]
    STATE["Off by default: the workspace dep enables only the core feature<br/>nostr-rust-forum/Cargo.toml:181, pulled bare by the worker nostr-bbs-pod-worker/Cargo.toml:56<br/>the trio is opt-in only nostr-bbs-pod-worker/Cargo.toml:22"]

    FEAT --> F1 & F2 & F3
    FEAT -.-> EXPORT
    STATE --> FEAT

    N1["DIVERGENCE BASELINE-architecture: these three surfaces remain STRUCTURALLY unreachable from wasm32<br/>CF Workers. Legacy ADR-087 (portable cores) is deferred, so ADR-086's pod-federation fallback is<br/>degenerate - the pod returns the same data D1 already holds. Latent risk, not scheduled work."]
    N2["DIVERGENCE: the related WAC Turtle serializer bare-path IRI quirk (legacy ADR-088) has ZERO live<br/>impact because this worker writes JSON-LD ACLs and never round-trips Turtle - see NF-04.4"]
    N3["The kit groups the three upstream features under ONE forum-facing alias so an operator flips a single<br/>switch rather than three docs/consumer-surface-map.md:52"]
    N4["Property tests cover the WAC document handling this whole topic rests on<br/>nostr-bbs-pod-worker/tests/wac_proptests.rs:1"]
```

## NF-04.11 The journalled D1 ledger today, and the chain view ADR-2012 is moving to

```mermaid
flowchart TB
    subgraph live["LIVE at this revision - D1 is a journalled cache, credits carry evidence"]
        JOURNAL["pay_credits journal - every credit row names a chain + outpoint<br/>or the NIP-98 request id that released a job-hold, CHECK-enforced<br/>nostr-bbs-pod-worker/src/pay_ledger.rs:110"]
        SCHEMA["idempotent SCHEMA run at worker startup<br/>pay_ledger.rs:130 payments.rs:45"]
        DEP["deposit - a TXO is credited once, only via qualifying_output,<br/>keyed with its chain pay_ledger.rs:618 pay_ledger.rs:564"]
        APPLY["SQL_APPLY_CREDIT - the ONE statement that creates or raises a<br/>balance, reading account and amount from an unapplied journal row<br/>pay_ledger.rs:168"]
        DEB["debit - the surviving spend-side primitive<br/>pay_ledger.rs:434"]
        BAL["read_balance returns the cached D1 number<br/>pay_ledger.rs:412"]
        REC["recover_orphaned_jobs pay_ledger.rs:1273"]
        ADDR["derive_deposit_address frozen as live, ported to rust-bitcoin<br/>nostr-bbs-pod-worker/src/deposit_address.rs:120"]
    end
    SCHEMA --> JOURNAL
    DEP --> JOURNAL --> APPLY --> BAL
    DEB --> BAL
    REC -.-> JOURNAL

    PROP["STILL PROPOSED - the full view: decision_status proposed,<br/>implementation_status partial<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:6-7"]
    P1["D1 demoted to a height-stamped derived view folded from sidestr UTXOs,<br/>staleness bound an error - not yet built<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:48-52"]
    P2["DONE in the amendment: credit_atomic and D1PaymentStore (with its<br/>write_ledger) are REMOVED - every credit is a journalled row<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:61-68"]
    P3["solid-pod-rs moves in lockstep with the host, closing the pin<br/>skew as a P1 exit criterion - pending<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:76-80"]

    BAL -.->|"proposed successor"| PROP
    PROP --> P1 & P2 & P3

    N1["INVARIANT ADR-2012 D6 (structural): tests/pay_credit_guard.rs scans every workspace source<br/>that can reach the shared D1 and fails unless the only balance-raising statement is<br/>SQL_APPLY_CREDIT - no statement sets or adds balance_sats from a bound parameter<br/>ADR-2012-d1-ledger-becomes-a-chain-view.md:109-117"]
    N2["DRIFT the amendment withdraws: the routes were triaged as dormant, but production sets<br/>PAY_ENABLED true - the three evidence-free value paths named in the amendment are the<br/>defect this journal closes ADR-2012-d1-ledger-becomes-a-chain-view.md:35-44"]
    N3["EXTERNAL: ADR-2012 D5 freezes derive_subkey and its JS-parity vector as a Published Language<br/>for the settlement domain nostr-bbs-core/src/keys.rs:251-265, ADR-2012-d1-ledger-becomes-a-chain-view.md:81-89"]
```
