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
verified_commit: 1d9da5270
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
```

## SP-01.2 Workspace-inherited package metadata

```mermaid
classDiagram
    class WorkspacePackage {
        +version = 0.5.0-alpha.9  ../solid-pod-rs/Cargo.toml:15
        +edition = 2021  ../solid-pod-rs/Cargo.toml:16
        +license = AGPL-3.0-only  ../solid-pod-rs/Cargo.toml:17
        +repository  ../solid-pod-rs/Cargo.toml:18
        +documentation = docs.rs  ../solid-pod-rs/Cargo.toml:20
        +rust-version = 1.88  ../solid-pod-rs/Cargo.toml:22
    }
    class WorkspaceLints {
        +unsafe_code = deny  ../solid-pod-rs/Cargo.toml:28
        +rust_2018_idioms = warn  ../solid-pod-rs/Cargo.toml:29
    }
    class ReleaseProfile {
        +lto = thin  ../solid-pod-rs/Cargo.toml:32
        +codegen-units = 1  ../solid-pod-rs/Cargo.toml:33
        +strip = symbols  ../solid-pod-rs/Cargo.toml:34
    }
    class CoreCrate {
        +version.workspace = true  crates/solid-pod-rs/Cargo.toml:3
        +lib name = solid_pod_rs  crates/solid-pod-rs/Cargo.toml:17
    }
    WorkspacePackage <|-- CoreCrate
    WorkspacePackage ..> WorkspaceLints
    WorkspacePackage ..> ReleaseProfile
```

## SP-01.3 Core library module surface

```mermaid
classDiagram
    class AlwaysCompiled {
        auth  solid-pod-rs/src/lib.rs:134
        bitcoin_tx  solid-pod-rs/src/lib.rs:139
        config  solid-pod-rs/src/lib.rs:140
        error  solid-pod-rs/src/lib.rs:141
        interop  solid-pod-rs/src/lib.rs:142
        ldp  solid-pod-rs/src/lib.rs:143
        mashlib  solid-pod-rs/src/lib.rs:144
        metrics  solid-pod-rs/src/lib.rs:145
        mrc20  solid-pod-rs/src/lib.rs:146
        multitenant  solid-pod-rs/src/lib.rs:147
        payments  solid-pod-rs/src/lib.rs:148
        provenance  solid-pod-rs/src/lib.rs:149
        security  solid-pod-rs/src/lib.rs:150
        trading  solid-pod-rs/src/lib.rs:151
        wac  solid-pod-rs/src/lib.rs:152
        webid  solid-pod-rs/src/lib.rs:153
    }
    class FeatureGated {
        did_nostr_types  solid-pod-rs/src/lib.rs:163
        notifications  solid-pod-rs/src/lib.rs:174
        provision  solid-pod-rs/src/lib.rs:176
        quota  solid-pod-rs/src/lib.rs:178
        storage  solid-pod-rs/src/lib.rs:180
        oidc  solid-pod-rs/src/lib.rs:183
        export  solid-pod-rs/src/lib.rs:199
        handlers  solid-pod-rs/src/lib.rs:206
    }
    AlwaysCompiled <.. FeatureGated
    note for AlwaysCompiled "The always-compiled half is pure logic plus the crypto and protocol primitives,\nwhich is what makes the core wasm/edge surface in SP-01.4 possible: storage,\nnotifications, provision, quota and oidc — everything that needs I/O or a\nruntime — sit behind a feature."
```

## SP-01.4 `core` — the no-IO wasm/edge subset

```mermaid
flowchart TD
    CORE_F["feature core = std + js-sys + did-nostr-types<br/>crates/solid-pod-rs/Cargo.toml:107"]
    STD["feature std<br/>crates/solid-pod-rs/Cargo.toml:108"]
    TOKIO["feature tokio-runtime = tokio + tungstenite + futures-util<br/>crates/solid-pod-rs/Cargo.toml:118"]
    FSB["fs-backend = tokio-runtime + notify + cap-std<br/>crates/solid-pod-rs/Cargo.toml:120"]
    MEMB["memory-backend = tokio-runtime<br/>crates/solid-pod-rs/Cargo.toml:121"]

    CORE_F --> STD
    CORE_F -. "excludes" .-> TOKIO
    TOKIO --> FSB
    TOKIO --> MEMB

    E1["EXTERNAL: nostr-rust-forum's nostr-bbs-pod-worker compiles wasm32 against<br/>default-features = false, features = core — the WAC evaluator, WebID parser,<br/>NIP-98 structural verify and LDP parsers with no tokio and no reqwest.<br/>See the nostr-rust-forum area."]
    CORE_F -.-> E1

    E2["INVARIANT: classify_policy_read is runtime-free so the edge tier can adopt it<br/>solid-pod-rs/src/wac/resolver.rs:265"]
    CORE_F -.-> E2
```

## SP-01.5 Core feature graph — auth and identity flags

```mermaid
flowchart LR
    NIP98S["nip98-schnorr = k256<br/>crates/solid-pod-rs/Cargo.toml:123"]
    NIP98R["nip98-replay = lru + tokio-runtime<br/>crates/solid-pod-rs/Cargo.toml:130"]
    LWS["lws-cid = k256<br/>crates/solid-pod-rs/Cargo.toml:135"]
    LWSP["lws-cid-p256<br/>crates/solid-pod-rs/Cargo.toml:136"]
    LWSE["lws-cid-eddsa<br/>crates/solid-pod-rs/Cargo.toml:137"]
    LWSF["lws-cid-full<br/>crates/solid-pod-rs/Cargo.toml:138"]
    OIDCF["oidc = openidconnect + jsonwebtoken + reqwest<br/>crates/solid-pod-rs/Cargo.toml:122"]
    DPOPR["dpop-replay-cache = oidc + lru<br/>crates/solid-pod-rs/Cargo.toml:163"]
    DPOPT["dpop-symmetric-test (TEST ONLY)<br/>crates/solid-pod-rs/Cargo.toml:169"]

    LWS --> LWSP
    LWS --> LWSE
    LWSP --> LWSF
    LWSE --> LWSF
    OIDCF --> DPOPR

    G1["INVARIANT: without nip98-schnorr the verifier is a fail-CLOSED stub,<br/>never a structural-only accept<br/>solid-pod-rs/src/auth/nip98.rs:376"]
    NIP98S -.-> G1
    G2["DIVERGENCE: dpop-symmetric-test compiles the HS256 oct arm; a symmetric DPoP alg<br/>is an RFC 9449 alg-confusion vector production must reject<br/>crates/solid-pod-rs/Cargo.toml:169"]
    DPOPT -.-> G2
```

## SP-01.6 Core feature graph — JSS-parity umbrella

```mermaid
flowchart TD
    JSS["jss-v04 (parent flag, no-op alone)<br/>crates/solid-pod-rs/Cargo.toml:146"]
    SP["security-primitives<br/>crates/solid-pod-rs/Cargo.toml:150"]
    LEG["legacy-notifications (solid-0.1 WS adapter)<br/>crates/solid-pod-rs/Cargo.toml:155"]
    ORG["acl-origin (WAC acl:origin gate)<br/>crates/solid-pod-rs/Cargo.toml:159"]
    CFGL["config-loader = serde_yaml + toml<br/>crates/solid-pod-rs/Cargo.toml:178"]
    WHS["webhook-signing = ed25519-dalek + httpdate<br/>crates/solid-pod-rs/Cargo.toml:182"]
    RL["rate-limit = lru + parking_lot<br/>crates/solid-pod-rs/Cargo.toml:201"]
    QU["quota = jss-v04 + config-loader<br/>crates/solid-pod-rs/Cargo.toml:209"]
    DNT["did-nostr-types (no new deps)<br/>crates/solid-pod-rs/Cargo.toml:189"]
    MR["mrc20 = k256<br/>crates/solid-pod-rs/Cargo.toml:194"]
    DN["did-nostr<br/>crates/solid-pod-rs/Cargo.toml:195"]

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
    LIBD["core lib default = std, fs-backend,<br/>memory-backend, tokio-runtime, notifications<br/>crates/solid-pod-rs/Cargo.toml:98"]
    SRVD["server default = [] (EMPTY)<br/>crates/solid-pod-rs-server/Cargo.toml:123"]

    LIBD --> R1["a default library build has FS + memory storage,<br/>the tokio runtime and the notifications stack"]
    SRVD --> R2["a default server build has NO tls, NO git, NO forge,<br/>NO quota, NO did-nostr, NO export, NO nip05"]

    D1["INVARIANT (ADR-2004): the empty server default means git_mark_write compiles to<br/>the no-op shim, so a default build records ZERO provenance marks<br/>solid-pod-rs-server/src/lib.rs:3668"]
    SRVD -.-> D1
```

## SP-01.8 Server feature pass-throughs and the git/forge chain

```mermaid
flowchart TD
    TLS["tls = rustls + actix-web/rustls-0_23<br/>crates/solid-pod-rs-server/Cargo.toml:126"]
    RLP["rate-limit -> solid-pod-rs/rate-limit<br/>crates/solid-pod-rs-server/Cargo.toml:129"]
    QUP["quota -> solid-pod-rs/quota<br/>crates/solid-pod-rs-server/Cargo.toml:130"]
    DNP["did-nostr -> solid-pod-rs/did-nostr<br/>crates/solid-pod-rs-server/Cargo.toml:131"]
    SPP["security-primitives<br/>crates/solid-pod-rs-server/Cargo.toml:132"]
    PK["provision-keys (library surface only)<br/>crates/solid-pod-rs-server/Cargo.toml:140"]
    N05["nip05-endpoint<br/>crates/solid-pod-rs-server/Cargo.toml:144"]
    EXJ["export-jsonld<br/>crates/solid-pod-rs-server/Cargo.toml:145"]
    GITF["git = solid-pod-rs-git + solid-pod-rs/git-auto-init<br/>crates/solid-pod-rs-server/Cargo.toml:150"]
    FRG["forge = solid-pod-rs-forge + git<br/>crates/solid-pod-rs-server/Cargo.toml:155"]
    FA["forge-anchoring = forge + forge/anchoring + mrc20<br/>crates/solid-pod-rs-server/Cargo.toml:156"]
    FAN["forge-announce = forge + forge/announce<br/>crates/solid-pod-rs-server/Cargo.toml:157"]
    INS["install = solid-pod-rs/nip98-schnorr<br/>crates/solid-pod-rs-server/Cargo.toml:163"]

    GITF --> FRG
    FRG --> FA
    FRG --> FAN

    K1["INVARIANT: provision-keys mints an nsec, so no HTTP route exposes it —<br/>it stays a library surface by design<br/>crates/solid-pod-rs-server/Cargo.toml:140"]
    PK -.-> K1
    K2["DOC-DRIFT: README calls the forge 'Phases 0-3 shipped';<br/>Phases 4-7 exist only as the anchoring/announce scaffolds above"]
    FA -.-> K2
```

## SP-01.9 Sibling crate feature surfaces

```mermaid
classDiagram
    class forge {
        +default = []  crates/solid-pod-rs-forge/Cargo.toml:48
        +anchoring = solid-pod-rs/mrc20 + k256  crates/solid-pod-rs-forge/Cargo.toml:50
        +announce = solid-pod-rs-nostr + k256  crates/solid-pod-rs-forge/Cargo.toml:53
    }
    class git {
        +default = []  crates/solid-pod-rs-git/Cargo.toml:43
        +with-git-binary (e2e CGI tests)  crates/solid-pod-rs-git/Cargo.toml:44
        +git-auto-init  crates/solid-pod-rs-git/Cargo.toml:49
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
    LIC["[licenses] allow-list<br/>../solid-pod-rs/deny.toml:63"]
    BANS["[bans] wildcards = deny<br/>../solid-pod-rs/deny.toml:104"]
    MULTI["multiple-versions = warn<br/>../solid-pod-rs/deny.toml:103"]
    SRC["[sources] unknown-registry = deny<br/>../solid-pod-rs/deny.toml:128"]
    GITS["unknown-git = deny, allow-git = []<br/>../solid-pod-rs/deny.toml:129"]
    REG["allow-registry = crates.io only<br/>../solid-pod-rs/deny.toml:130"]

    ADV --> OUT["cargo-deny CI job — see SP-09.4"]
    LIC --> OUT
    BANS --> OUT
    MULTI --> OUT
    SRC --> OUT
    GITS --> OUT
    REG --> OUT

    N["INVARIANT: allow-git is empty — every dependency resolves from the crates.io<br/>index, so a git-patched dependency cannot enter a release build."]
    GITS -.-> N
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
    D1["[[example]] embed_in_actix<br/>crates/solid-pod-rs/Cargo.toml:310"]
    D2["custom_storage<br/>crates/solid-pod-rs/Cargo.toml:314"]
    D3["nip98_client<br/>crates/solid-pod-rs/Cargo.toml:318"]
    D4["notifications_consumer<br/>crates/solid-pod-rs/Cargo.toml:322"]
    D5["webhook_receiver<br/>crates/solid-pod-rs/Cargo.toml:326"]
    D6["wac_admin<br/>crates/solid-pod-rs/Cargo.toml:330"]
    D7["oidc_client, required-features = oidc<br/>crates/solid-pod-rs/Cargo.toml:336"]
    C1["cargo check -p solid-pod-rs --examples<br/>crates/solid-pod-rs/docs/examples-index.md:34"]
    C2["cargo check --examples --features oidc<br/>crates/solid-pod-rs/docs/examples-index.md:35"]

    D1 --> C1
    D2 --> C1
    D3 --> C1
    D4 --> C1
    D5 --> C1
    D6 --> C1
    D7 --> C2

    N["Without the oidc feature, oidc_client compiles to a stub that prints a hint and<br/>exits — it stays registered so the NAME remains discoverable.<br/>crates/solid-pod-rs/docs/examples-index.md:38"]
    D7 -.-> N
    N2["DOC-DRIFT: examples-index.md documents an eighth example, 'standalone', as<br/>'the quickest way to see a working pod'<br/>(crates/solid-pod-rs/docs/examples-index.md:12). No standalone.rs exists and no<br/>[[example]] declares it — the Cargo manifest has exactly the seven above. A<br/>reader following the index hits cargo error: no example target named standalone."]
    C1 -.-> N2
    N3["DIVERGENCE: neither cargo check line is a CI job — SP-09.10 shows the matrix<br/>never builds --examples, so this gate is documentation an author must run by<br/>hand. That is how the missing standalone.rs survived."]
    C1 -.-> N3
```
