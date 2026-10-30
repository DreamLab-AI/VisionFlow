---
id: SP-01
title: Workspace composition, crate dependency graph and the feature-flag matrix
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md, ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md]
adrs: [ADR-2001, ADR-2004, ADR-2005]
sources:
  - ../solid-pod-rs/Cargo.toml
  - ../solid-pod-rs/deny.toml
  - ../solid-pod-rs/crates/solid-pod-rs/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/blocktrail.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-forge/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-git/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-idp/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/resolver.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/nip98.rs
  - ../solid-pod-rs/crates/solid-pod-rs/docs/examples-index.md
  - ../solid-pod-rs/crates/solid-pod-rs/examples/embed_in_actix.rs
  - ../solid-pod-rs/crates/solid-pod-rs/examples/custom_storage.rs
  - ../solid-pod-rs/crates/solid-pod-rs/examples/nip98_client.rs
  - ../solid-pod-rs/crates/solid-pod-rs/examples/notifications_consumer.rs
  - ../solid-pod-rs/crates/solid-pod-rs/examples/webhook_receiver.rs
  - ../solid-pod-rs/crates/solid-pod-rs/examples/wac_admin.rs
  - ../solid-pod-rs/crates/solid-pod-rs/examples/oidc_client.rs
verified_commit: 76ab7aab3e882f32508b2461b3f51e4c09bdcfa1
---

## SP-01.1 Eight workspace members and the library-first split

```mermaid
flowchart TD
    WS["[workspace] members<br/>../solid-pod-rs/Cargo.toml:2"]
    CORE["solid-pod-rs (core library)<br/>../solid-pod-rs/Cargo.toml:3"]
    SRV["solid-pod-rs-server (actix-web binary)<br/>../solid-pod-rs/Cargo.toml:4"]
    AP["solid-pod-rs-activitypub<br/>../solid-pod-rs/Cargo.toml:5"]
    GIT["solid-pod-rs-git<br/>../solid-pod-rs/Cargo.toml:6"]
    FORGE["solid-pod-rs-forge<br/>../solid-pod-rs/Cargo.toml:7"]
    IDP["solid-pod-rs-idp<br/>../solid-pod-rs/Cargo.toml:8"]
    NOSTR["solid-pod-rs-nostr<br/>../solid-pod-rs/Cargo.toml:9"]
    DIDKEY["solid-pod-rs-didkey<br/>../solid-pod-rs/Cargo.toml:10"]

    WS --> CORE
    WS --> SRV
    WS --> AP
    WS --> GIT
    WS --> FORGE
    WS --> IDP
    WS --> NOSTR
    WS --> DIDKEY

    SRV --> CORE
    AP --> CORE
    GIT --> CORE
    FORGE --> CORE
    IDP --> CORE
    NOSTR --> CORE
    DIDKEY --> CORE

    N1["INVARIANT: dependency direction is one-way — every sibling depends on the core,<br/>the core depends on no sibling. README 'Architecture' asserts the same shape."]
    CORE -.-> N1

    N2["EXTERNAL: workspace.package.license = AGPL-3.0-only<br/>../solid-pod-rs/Cargo.toml:17 — every crate above inherits it, so a consumer<br/>that statically links CORE or a sibling (e.g. VisionClaw's embedded pod, VC-26)<br/>inherits the same obligation."]
    CORE -.-> N2
```

## SP-01.3 Core library module surface

```mermaid
classDiagram
    class AlwaysCompiled {
        auth  solid-pod-rs/src/lib.rs:134
        bitcoin_tx  solid-pod-rs/src/lib.rs:139
        blocktrail  solid-pod-rs/src/lib.rs:140
        config  solid-pod-rs/src/lib.rs:141
        error  solid-pod-rs/src/lib.rs:142
        interop  solid-pod-rs/src/lib.rs:143
        ldp  solid-pod-rs/src/lib.rs:144
        mashlib  solid-pod-rs/src/lib.rs:145
        metrics  solid-pod-rs/src/lib.rs:146
        mrc20  solid-pod-rs/src/lib.rs:147
        multitenant  solid-pod-rs/src/lib.rs:148
        payments  solid-pod-rs/src/lib.rs:149
        provenance  solid-pod-rs/src/lib.rs:150
        rdf_input  solid-pod-rs/src/lib.rs:151
        security  solid-pod-rs/src/lib.rs:152
        trading  solid-pod-rs/src/lib.rs:153
        wac  solid-pod-rs/src/lib.rs:154
        webid  solid-pod-rs/src/lib.rs:155
    }
    class FeatureGated {
        did_nostr_types  solid-pod-rs/src/lib.rs:166
        notifications  solid-pod-rs/src/lib.rs:177
        provision  solid-pod-rs/src/lib.rs:179
        quota  solid-pod-rs/src/lib.rs:181
        storage  solid-pod-rs/src/lib.rs:183
        oidc  solid-pod-rs/src/lib.rs:186
        export  solid-pod-rs/src/lib.rs:202
        handlers  solid-pod-rs/src/lib.rs:209
    }
    AlwaysCompiled <.. FeatureGated
    note for AlwaysCompiled "The always-compiled half is pure logic plus the crypto and protocol primitives,\nwhich is what makes the core wasm/edge surface in SP-01.4 possible: storage,\nnotifications, provision, quota and oidc — everything that needs I/O or a\nruntime — sit behind a feature. blocktrail (new in 0.5.0-alpha.12) is always compiled for the shared\nBlocktrail type and TXO URIs, but its walk and per-link verification sit behind mrc20\n(solid-pod-rs/src/blocktrail.rs:896)."
```

## SP-01.4 `core` — the no-IO wasm/edge subset

```mermaid
flowchart TD
    CORE_F["feature core = std + js-sys + did-nostr-types<br/>crates/solid-pod-rs/Cargo.toml:121"]
    STD["feature std<br/>crates/solid-pod-rs/Cargo.toml:122"]
    TOKIO["feature tokio-runtime = tokio + tungstenite + futures-util<br/>crates/solid-pod-rs/Cargo.toml:132"]
    FSB["fs-backend = tokio-runtime + notify + cap-std<br/>crates/solid-pod-rs/Cargo.toml:134"]
    MEMB["memory-backend = tokio-runtime<br/>crates/solid-pod-rs/Cargo.toml:135"]

    CORE_F --> STD
    CORE_F -. "excludes" .-> TOKIO
    TOKIO --> FSB
    TOKIO --> MEMB

    E1["EXTERNAL: nostr-rust-forum's nostr-bbs-pod-worker compiles wasm32 against<br/>default-features = false, features = core — the WAC evaluator, WebID parser,<br/>NIP-98 structural verify and LDP parsers with no tokio and no reqwest.<br/>See the nostr-rust-forum area."]
    CORE_F -.-> E1

    E2["INVARIANT: classify_policy_read is runtime-free so the edge tier can adopt it<br/>solid-pod-rs/src/wac/resolver.rs:265"]
    CORE_F -.-> E2

    E3["Since alpha.10 core carries k256 through did-nostr-types, pure Rust and<br/>wasm32-safe, for the did:nostr full-point encoder and point validation; since alpha.18 it sits on the<br/>RustCrypto 0.14 line (k256 0.14, crates/solid-pod-rs/Cargo.toml:73), whose getrandom 0.4 needs wasm_js on wasm32<br/>crates/solid-pod-rs/Cargo.toml:290, did-nostr-types crates/solid-pod-rs/Cargo.toml:208"]
    CORE_F -.-> E3
```

## SP-01.5 Core feature graph — auth and identity flags

```mermaid
flowchart LR
    NIP98S["nip98-schnorr = k256<br/>crates/solid-pod-rs/Cargo.toml:137"]
    NIP98R["nip98-replay = lru + tokio-runtime<br/>crates/solid-pod-rs/Cargo.toml:144"]
    LWS["lws-cid = k256<br/>crates/solid-pod-rs/Cargo.toml:149"]
    LWSP["lws-cid-p256<br/>crates/solid-pod-rs/Cargo.toml:150"]
    LWSE["lws-cid-eddsa<br/>crates/solid-pod-rs/Cargo.toml:151"]
    LWSF["lws-cid-full<br/>crates/solid-pod-rs/Cargo.toml:152"]
    OIDCF["oidc = openidconnect + jsonwebtoken + reqwest<br/>crates/solid-pod-rs/Cargo.toml:136"]
    DPOPR["dpop-replay-cache = oidc + lru<br/>crates/solid-pod-rs/Cargo.toml:177"]
    DPOPT["dpop-symmetric-test (TEST ONLY)<br/>crates/solid-pod-rs/Cargo.toml:183"]

    LWS --> LWSP
    LWS --> LWSE
    LWSP --> LWSF
    LWSE --> LWSF
    OIDCF --> DPOPR

    G1["INVARIANT: without nip98-schnorr the verifier is a fail-CLOSED stub,<br/>never a structural-only accept<br/>solid-pod-rs/src/auth/nip98.rs:376"]
    NIP98S -.-> G1
    G2["DIVERGENCE: dpop-symmetric-test compiles the HS256 oct arm; a symmetric DPoP alg<br/>is an RFC 9449 alg-confusion vector production must reject<br/>crates/solid-pod-rs/Cargo.toml:179"]
    DPOPT -.-> G2
```

## SP-01.6 Core feature graph — JSS-parity umbrella

```mermaid
flowchart TD
    JSS["jss-v04 (parent flag, no-op alone)<br/>crates/solid-pod-rs/Cargo.toml:160"]
    SP["security-primitives<br/>crates/solid-pod-rs/Cargo.toml:164"]
    LEG["legacy-notifications (solid-0.1 WS adapter)<br/>crates/solid-pod-rs/Cargo.toml:169"]
    ORG["acl-origin (WAC acl:origin gate)<br/>crates/solid-pod-rs/Cargo.toml:173"]
    CFGL["config-loader = serde_norway + toml<br/>crates/solid-pod-rs/Cargo.toml:192"]
    WHS["webhook-signing = ed25519-dalek + httpdate<br/>crates/solid-pod-rs/Cargo.toml:196"]
    RL["rate-limit = lru + parking_lot<br/>crates/solid-pod-rs/Cargo.toml:222"]
    QU["quota = jss-v04 + config-loader<br/>crates/solid-pod-rs/Cargo.toml:230"]
    DNT["did-nostr-types = k256, full-point Multikey<br/>crates/solid-pod-rs/Cargo.toml:208"]
    MR["mrc20 = dep:bitcoin — BIP-341 taproot and BIP-340 signing<br/>moved to rust-bitcoin 0.32 in alpha.11 (ADR-2008); links<br/>libsecp256k1 (C), so wasm32 mrc20 needs a C toolchain<br/>crates/solid-pod-rs/Cargo.toml:215"]
    DN["did-nostr<br/>crates/solid-pod-rs/Cargo.toml:216"]

    JSS --> SP
    JSS --> LEG
    JSS --> ORG
    JSS --> CFGL
    JSS --> WHS
    JSS --> RL
    JSS --> QU
    CFGL --> QU
    DNT --> DN
    SP --> DN

    O1["DIVERGENCE: acl-origin is NOT in the default set, so a default build ignores<br/>the Origin header entirely — see SP-04.8"]
    ORG -.-> O1
```

## SP-01.7 Default feature set versus the empty server default

```mermaid
flowchart LR
    LIBD["core lib default = std, fs-backend,<br/>memory-backend, tokio-runtime, notifications<br/>crates/solid-pod-rs/Cargo.toml:112"]
    SRVD["server default = [] (EMPTY)<br/>crates/solid-pod-rs-server/Cargo.toml:127"]

    LIBD --> R1["a default library build has FS + memory storage,<br/>the tokio runtime and the notifications stack"]
    SRVD --> R2["a default server build has NO tls, NO git, NO forge,<br/>NO quota, NO did-nostr, NO export, NO nip05"]

    D1["INVARIANT (ADR-2004): the empty server default means git_mark_write compiles to<br/>the no-op shim, so a default build records ZERO provenance marks<br/>solid-pod-rs-server/src/lib.rs:4110"]
    SRVD -.-> D1
```

## SP-01.8 Server feature pass-throughs and the git/forge chain

```mermaid
flowchart TD
    TLS["tls = rustls + actix-web/rustls-0_23<br/>crates/solid-pod-rs-server/Cargo.toml:136"]
    RLP["rate-limit -> solid-pod-rs/rate-limit<br/>crates/solid-pod-rs-server/Cargo.toml:139"]
    QUP["quota -> solid-pod-rs/quota<br/>crates/solid-pod-rs-server/Cargo.toml:140"]
    DNP["did-nostr -> solid-pod-rs/did-nostr<br/>crates/solid-pod-rs-server/Cargo.toml:141"]
    SPP["security-primitives<br/>crates/solid-pod-rs-server/Cargo.toml:142"]
    PK["provision-keys (library surface only)<br/>crates/solid-pod-rs-server/Cargo.toml:150"]
    N05["nip05-endpoint<br/>crates/solid-pod-rs-server/Cargo.toml:154"]
    EXJ["export-jsonld<br/>crates/solid-pod-rs-server/Cargo.toml:155"]
    GITF["git = solid-pod-rs-git + solid-pod-rs/git-auto-init<br/>crates/solid-pod-rs-server/Cargo.toml:160"]
    FRG["forge = solid-pod-rs-forge + git<br/>crates/solid-pod-rs-server/Cargo.toml:165"]
    FA["forge-anchoring = forge + forge/anchoring + mrc20<br/>crates/solid-pod-rs-server/Cargo.toml:166"]
    FAN["forge-announce = forge + forge/announce<br/>crates/solid-pod-rs-server/Cargo.toml:167"]
    FMI["forge-mirror = forge + forge/mirror — started at boot only when FORGE_MIRROR_RELAYS and FORGE_MIRROR_FOLLOW name something<br/>crates/solid-pod-rs-server/Cargo.toml:170"]
    FFU["forge-full = forge + anchoring + announce + mirror, for builds that run the forge (ADR-2010)<br/>crates/solid-pod-rs-server/Cargo.toml:172"]
    APF["activitypub = solid-pod-rs-activitypub, default-off<br/>crates/solid-pod-rs-server/Cargo.toml:130"]
    INS["install = solid-pod-rs/nip98-schnorr<br/>crates/solid-pod-rs-server/Cargo.toml:178"]

    GITF --> FRG
    FRG --> FA
    FRG --> FAN
    FRG --> FMI
    FA --> FFU
    FAN --> FFU
    FMI --> FFU

    K1["INVARIANT: provision-keys mints an nsec, so no HTTP route exposes it —<br/>it stays a library surface by design<br/>crates/solid-pod-rs-server/Cargo.toml:150"]
    PK -.-> K1
    K2["The cargo defaults do not change: a library user who depends on the forge crate still gets<br/>nothing optional. Only consumers that build with forge-full ship the forge and mirror<br/>(ADR-2010, accepted, implementation partial, activation inactive).<br/>crates/solid-pod-rs-server/Cargo.toml:172"]
    FFU -.-> K2
```

## SP-01.9 Sibling crate feature surfaces

```mermaid
classDiagram
    class forge {
        +default = []  crates/solid-pod-rs-forge/Cargo.toml:55
        +anchoring = solid-pod-rs/mrc20 + k256  crates/solid-pod-rs-forge/Cargo.toml:57
        +announce = solid-pod-rs-nostr + k256 + relay publisher  crates/solid-pod-rs-forge/Cargo.toml:61
        +mirror = solid-pod-rs-nostr + k256 + relay sockets (kind-30618 follow)  crates/solid-pod-rs-forge/Cargo.toml:65
    }
    class git {
        +default = []  crates/solid-pod-rs-git/Cargo.toml:47
        +with-git-binary (e2e CGI tests)  crates/solid-pod-rs-git/Cargo.toml:48
        +git-auto-init  crates/solid-pod-rs-git/Cargo.toml:53
    }
    class idp {
        +default = []  crates/solid-pod-rs-idp/Cargo.toml:68
        +axum-binder = axum  crates/solid-pod-rs-idp/Cargo.toml:71
        +passkey = webauthn-rs + dashmap  crates/solid-pod-rs-idp/Cargo.toml:75
        +schnorr-sso = dashmap + k256  crates/solid-pod-rs-idp/Cargo.toml:79
    }
    note for forge "Every sibling ships default = [] — nothing beyond the core is on unless asked for."
```

## SP-01.10 The one trait every consumer bolts onto

```mermaid
classDiagram
    class Storage {
        <<trait>>
        +get(path) (Bytes, ResourceMeta)  solid-pod-rs/src/storage/mod.rs:75
        +put(path, body, content_type)  solid-pod-rs/src/storage/mod.rs:80
        +delete(path)  solid-pod-rs/src/storage/mod.rs:88
        +list(container) Vec~String~  solid-pod-rs/src/storage/mod.rs:94
        +head(path) ResourceMeta  solid-pod-rs/src/storage/mod.rs:97
        +exists(path) bool  solid-pod-rs/src/storage/mod.rs:100
        +create_container(path)  solid-pod-rs/src/storage/mod.rs:108
        +watch(path) Receiver~StorageEvent~  solid-pod-rs/src/storage/mod.rs:127
    }
    class ResourceMeta {
        +etag  solid-pod-rs/src/storage/mod.rs:27
    }
    class StorageEvent {
        solid-pod-rs/src/storage/mod.rs:58
    }
    Storage ..> ResourceMeta
    Storage ..> StorageEvent
    note for Storage "EXTERNAL: VisionClaw's embedded pod and the agentbox pod bridge both wire an\nArc&lt;dyn Storage&gt; of their own choosing behind this trait — see VC-26 and ES-08."
```

## SP-01.11 Supply-chain gates the workspace applies to itself

```mermaid
flowchart LR
    ADV["[advisories] yanked = deny<br/>../solid-pod-rs/deny.toml:37"]
    LIC["[licenses] allow-list<br/>../solid-pod-rs/deny.toml:73"]
    BANS["[bans] wildcards = deny<br/>../solid-pod-rs/deny.toml:124"]
    MULTI["multiple-versions = warn<br/>../solid-pod-rs/deny.toml:123"]
    SRC["[sources] unknown-registry = deny<br/>../solid-pod-rs/deny.toml:148"]
    GITS["unknown-git = deny, allow-git = []<br/>../solid-pod-rs/deny.toml:149"]
    REG["allow-registry = crates.io only<br/>../solid-pod-rs/deny.toml:150"]

    DEBT["DEBT: the advisory ignore list gained RUSTSEC-2026-0387 on 2026-10-29: crypto-bigint 0.5.5 survives only via openidconnect to p256 0.13, so the oidc feature is exposed and unassessed until openidconnect moves to the 0.14 line<br/>../solid-pod-rs/deny.toml:57-63"]
    ADV -.-> DEBT
    ADV --> OUT["cargo-deny CI job — see SP-09.4"]
    LIC --> OUT
    BANS --> OUT
    MULTI --> OUT
    SRC --> OUT
    GITS --> OUT
    REG --> OUT

    N["INVARIANT: allow-git is empty — every dependency resolves from the crates.io<br/>index, so a git-patched dependency cannot enter a release build."]
    GITS -.-> N
    X["EXCEPTION: the licences gate carries one scoped exception — hex_lit (a dep of<br/>rust-bitcoin per ADR-2008) is allowed under MITNFA, MIT plus a<br/>no-false-attribution clause, crate-scoped not workspace-wide.<br/>../solid-pod-rs/deny.toml:109"]
    LIC -.-> X
```

## SP-01.12 The `examples/` integration surface — the library-first contract

```mermaid
flowchart TD
    subgraph SERVER["Server-side — embed the pod"]
        E1["embed_in_actix — mount the pod as a sub-scope of a larger app<br/>crates/solid-pod-rs/examples/embed_in_actix.rs:1"]
        E2["custom_storage — implement Storage over a BTreeMap<br/>crates/solid-pod-rs/examples/custom_storage.rs:1"]
        E3["webhook_receiver — an Axum sink for WebhookChannel2023 POSTs<br/>crates/solid-pod-rs/examples/webhook_receiver.rs:1"]
    end
    subgraph CLIENT["Client-side — talk to a pod"]
        E4["nip98_client — sign, PUT a Turtle resource, read it back<br/>crates/solid-pod-rs/examples/nip98_client.rs:1"]
        E5["notifications_consumer — subscribe over WebSocketChannel2023<br/>crates/solid-pod-rs/examples/notifications_consumer.rs:1"]
        E6["oidc_client — discovery, registration, DPoP, token verify<br/>crates/solid-pod-rs/examples/oidc_client.rs:1"]
    end
    subgraph ADMIN["Administration"]
        E7["wac_admin — grant / show / check against an FsBackend root<br/>crates/solid-pod-rs/examples/wac_admin.rs:1"]
    end
    IDX["docs/examples-index.md — the documented map<br/>crates/solid-pod-rs/docs/examples-index.md:1"]

    IDX --> SERVER
    IDX --> CLIENT
    IDX --> ADMIN

    N["These seven ARE the integration contract. README leads with 'As a library',<br/>and each example maps to one boundary in SP-01.10 and SP-08.16: Storage (E2),<br/>LDP+NIP-98 (E1, E4), notifications (E3, E5), OIDC (E6), WAC (E7)."]
    IDX -.-> N
```

## SP-01.13 Example registration and the compilation gate

```mermaid
flowchart LR
    D1["[[example]] embed_in_actix<br/>crates/solid-pod-rs/Cargo.toml:334"]
    D2["custom_storage<br/>crates/solid-pod-rs/Cargo.toml:338"]
    D3["nip98_client<br/>crates/solid-pod-rs/Cargo.toml:342"]
    D4["notifications_consumer<br/>crates/solid-pod-rs/Cargo.toml:346"]
    D5["webhook_receiver<br/>crates/solid-pod-rs/Cargo.toml:350"]
    D6["wac_admin<br/>crates/solid-pod-rs/Cargo.toml:354"]
    D7["oidc_client, required-features = oidc<br/>crates/solid-pod-rs/Cargo.toml:358"]
    C1["cargo check -p solid-pod-rs --examples<br/>crates/solid-pod-rs/docs/examples-index.md:35"]
    C2["cargo check --examples --features oidc<br/>crates/solid-pod-rs/docs/examples-index.md:36"]

    D1 --> C1
    D2 --> C1
    D3 --> C1
    D4 --> C1
    D5 --> C1
    D6 --> C1
    D7 --> C2

    N["Without the oidc feature, oidc_client compiles to a stub that prints a hint and<br/>exits — it stays registered so the NAME remains discoverable.<br/>crates/solid-pod-rs/docs/examples-index.md:39"]
    D7 -.-> N
    N2["INVARIANT: examples-index.md states there is no standalone example, the quickest working<br/>pod being the solid-pod-rs-server binary, and the manifest declares exactly the seven above.<br/>crates/solid-pod-rs/docs/examples-index.md:10"]
    C1 -.-> N2
    N3["DIVERGENCE: neither cargo check line is a CI job — SP-09.10 shows the matrix<br/>never builds --examples, so this gate is documentation an author must run by<br/>hand."]
    C1 -.-> N3
```
