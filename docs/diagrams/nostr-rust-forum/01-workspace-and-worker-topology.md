---
id: NF-01
title: Cargo workspace, the five Workers, entry points and the upstream-absorption canary
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2002, ADR-2007]
sources:
  - ../nostr-rust-forum/Cargo.toml
  - ../nostr-rust-forum/rust-toolchain.toml
  - ../nostr-rust-forum/README.md
  - ../nostr-rust-forum/docs/consumer-surface-map.md
  - ../nostr-rust-forum/crates/nostr-bbs-core/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-upstream-canary/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-upstream-canary/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/wrangler.toml
  - ../nostr-rust-forum/.github/workflows/ci.yml
verified_commit: d48a7a546
---

## NF-01.1 Fourteen crates in four layers

```mermaid
flowchart TB
    subgraph found["Foundation"]
        CORE["nostr-bbs-core<br/>nostr-rust-forum/Cargo.toml:5 - protocol, keys, signer, NIP-04/19/44/59/98, governance"]
    end
    subgraph libs["Config, federation and shared utilities"]
        CFG["nostr-bbs-config nostr-rust-forum/Cargo.toml:8"]
        MESH["nostr-bbs-mesh nostr-rust-forum/Cargo.toml:9"]
        SKILL["nostr-bbs-setup-skill nostr-rust-forum/Cargo.toml:12"]
        RATE["nostr-bbs-rate-limit nostr-rust-forum/Cargo.toml:15"]
        ASCII["nostr-bbs-ascii nostr-rust-forum/Cargo.toml:19"]
    end
    subgraph workers["Cloudflare Worker reference implementations"]
        AUTH["nostr-bbs-auth-worker nostr-rust-forum/Cargo.toml:22"]
        POD["nostr-bbs-pod-worker nostr-rust-forum/Cargo.toml:23"]
        PREV["nostr-bbs-preview-worker nostr-rust-forum/Cargo.toml:24"]
        RELAY["nostr-bbs-relay-worker nostr-rust-forum/Cargo.toml:25"]
        SEARCH["nostr-bbs-search-worker nostr-rust-forum/Cargo.toml:26"]
    end
    subgraph clients["Leptos CSR browser clients"]
        FC["nostr-bbs-forum-client nostr-rust-forum/Cargo.toml:29"]
        BBS["nostr-bbs-bbs-client nostr-rust-forum/Cargo.toml:32 - served at /community/bbs/"]
    end
    CANARY["nostr-bbs-upstream-canary nostr-rust-forum/Cargo.toml:37 - validation only, linked into no binary"]

    AUTH --> CORE
    POD --> CORE
    RELAY --> CORE
    SEARCH --> CORE
    PREV --> CORE
    FC --> CORE
    BBS --> CORE
    CFG --> RELAY
    CFG --> FC
    RATE --> AUTH
    RATE --> RELAY
    ASCII --> PREV
    ASCII --> BBS
    MESH -.->|"designed, not wired in README.md:189"| RELAY

    N1["resolver 2, edition 2021, rust-version 1.85 nostr-rust-forum/Cargo.toml:2 nostr-rust-forum/Cargo.toml:44"]
    N2["Every kit crate is pinned to the same in-tree version 1.0.0-beta.10 nostr-rust-forum/Cargo.toml:158-162,<br/>crates/nostr-bbs-core/Cargo.toml:3"]
    N3["DOC-DRIFT ADR-2007 / BASELINE known-divergence: the Cargo.toml comment above the path deps still<br/>says published to crates.io as 1.0.0-beta.3 while the tree is on beta.10 nostr-rust-forum/Cargo.toml:157"]
```

## NF-01.2 Build matrix — one target, one release profile

```mermaid
flowchart LR
    TC["rust-toolchain.toml:2 channel stable<br/>rust-toolchain.toml:3 targets wasm32-unknown-unknown"]
    PROF["release profile nostr-rust-forum/Cargo.toml:176<br/>opt-level z nostr-rust-forum/Cargo.toml:177<br/>panic abort nostr-rust-forum/Cargo.toml:181"]
    WB["worker-build --release<br/>nostr-bbs-auth-worker/wrangler.toml:6<br/>nostr-bbs-relay-worker/wrangler.toml:6<br/>nostr-bbs-pod-worker/wrangler.toml:6<br/>nostr-bbs-preview-worker/wrangler.toml:6<br/>nostr-bbs-search-worker/wrangler.toml:6"]
    SHIM["build/worker/shim.mjs entrypoint<br/>nostr-bbs-relay-worker/wrangler.toml:2"]
    CIW["declared wasm-check set is two crates<br/>nostr-rust-forum/Cargo.toml:49"]
    CIJOB["actual CI wasm job checks the WHOLE workspace<br/>.github/workflows/ci.yml:174"]

    TC --> PROF --> WB --> SHIM
    TC --> CIW
    TC --> CIJOB

    N1["panic = abort is not a tuning choice - workers and WASM cannot unwind, so the unwinding tables are<br/>dead weight nostr-rust-forum/Cargo.toml:170-175"]
    N2["DOC-DRIFT: workspace.metadata.ci.wasm-check-packages names two crates nostr-rust-forum/Cargo.toml:49<br/>with a comment saying to expand it as the secp256k1-sys cross-compile resolves nostr-rust-forum/Cargo.toml:46-48,<br/>but no workflow or script reads that key - the wasm job runs cargo check --workspace<br/>.github/workflows/ci.yml:174, made possible by installing libc6-dev-i386 .github/workflows/ci.yml:171.<br/>The metadata is inert."]
    N3["compatibility_date 2025-09-01 is identical across all five templates, e.g. nostr-bbs-search-worker/wrangler.toml:3"]
```

## NF-01.3 Worker entry points and their triggers

```mermaid
flowchart TB
    subgraph fetchentry["#[event(fetch)] handlers"]
        A["auth-worker<br/>nostr-bbs-auth-worker/src/lib.rs:143"]
        R["relay-worker<br/>nostr-bbs-relay-worker/src/lib.rs:158"]
    end
    subgraph cronentry["#[event(scheduled)] handlers"]
        AC["auth-worker scheduled<br/>nostr-bbs-auth-worker/src/lib.rs:834"]
        RC["relay-worker scheduled - every 5 min<br/>nostr-bbs-relay-worker/src/lib.rs:853<br/>trigger nostr-bbs-relay-worker/wrangler.toml:82"]
        SC["search-worker cron - every 5 min<br/>nostr-bbs-search-worker/wrangler.toml:36"]
    end
    WSUP["WebSocket upgrade to the Durable Object<br/>nostr-bbs-relay-worker/src/lib.rs:171"]
    NIP11["NIP-11 relay info on Accept application/nostr+json<br/>nostr-bbs-relay-worker/src/lib.rs:180"]
    HTTPR["HTTP route table<br/>nostr-bbs-relay-worker/src/lib.rs:226"]

    R --> WSUP
    R --> NIP11
    R --> HTTPR

    N1["Both fetch entries bootstrap their schema and the shared replay table on EVERY cold start -<br/>auth nostr-bbs-auth-worker/src/lib.rs:162, relay nostr-bbs-relay-worker/src/lib.rs:160"]
    N2["The relay binds its replay schema to REPLAY_DB, the auth worker to its own DB - the same physical<br/>database either way nostr-bbs-relay-worker/src/lib.rs:161"]
    N3["Both entries carry a cfg_attr allow(dead_code) for native builds because nothing native calls the<br/>wasm-bindgen glue nostr-bbs-relay-worker/src/lib.rs:156"]
```

## NF-01.4 Pinned dependency spine

```mermaid
classDiagram
    class Workspace {
        nostr 0.44.7 nip04 nip44 nip59 nip98 : nostr-rust-forum/Cargo.toml:58
        worker 0.8 : nostr-rust-forum/Cargo.toml:71
        leptos 0.7 csr : nostr-rust-forum/Cargo.toml:61
        k256 0.13.4 schnorr ecdh : nostr-rust-forum/Cargo.toml:77
        passkey-types 0.3 : nostr-rust-forum/Cargo.toml:74
        comrak 0.38 : nostr-rust-forum/Cargo.toml:109
        image 0.24 pure-Rust decoders only : nostr-rust-forum/Cargo.toml:168
        solid-pod-rs EXACT =0.5.0-alpha.7 core : nostr-rust-forum/Cargo.toml:155
    }
    class SolidPodRs {
        wac : pod-worker acl
        webid : pod-worker webid
        payments : pod-worker payments
        did_nostr_types : core did
    }
    Workspace --> SolidPodRs

    note for Workspace "INVARIANT ADR-2007: solid-pod-rs stays an EXACT (=) pin - a caret range would let a resolve pull a newer published alpha silently nostr-rust-forum/Cargo.toml:151-154"
    note for SolidPodRs "EXTERNAL: the consumer surface is enumerated in docs/consumer-surface-map.md:14 - see the solid-pod-rs area (SP-*) and NF-04"
```

## NF-01.5 Upstream `nostr` absorption — canary-first, nothing deleted

```mermaid
stateDiagram-v2
    [*] --> DependencyEnabled: nostr 0.44.7 with four NIP flags on<br/>nostr-rust-forum/Cargo.toml:58
    DependencyEnabled --> CanaryBuilt: nostr-bbs-upstream-canary compiles for wasm32<br/>crates/nostr-bbs-upstream-canary/Cargo.toml:2
    CanaryBuilt --> ShapeA: PASS - proceed to per-module absorption<br/>nostr-bbs-upstream-canary/src/lib.rs:17
    CanaryBuilt --> ShapeC: FAIL - patch-in-place fallback, file an upstream PR<br/>nostr-bbs-upstream-canary/src/lib.rs:19
    ShapeA --> [*]
    ShapeC --> [*]

    note right of CanaryBuilt
        Three smokes only:
        keypair round-trip nostr-bbs-upstream-canary/src/lib.rs:35
        NIP-44 v2 conversation key vs the paulmillr vector nostr-bbs-upstream-canary/src/lib.rs:56
        NIP-19 npub round-trip nostr-bbs-upstream-canary/src/lib.rs:89
        aggregated by run_all_smokes nostr-bbs-upstream-canary/src/lib.rs:106
    end note
    note right of DependencyEnabled
        INVARIANT ADR-2002: nostr-bbs-core owns on-wasm32 Schnorr until the canary
        records Shape A and a module is DELIBERATELY deleted. Enabling the upstream
        NIP flags retires nothing by itself.
    end note
    note right of ShapeC
        DOC-DRIFT: the canary's own module doc names a five-feature matrix
        nip04 nip19 nip44 nip59 nip98 (nostr-bbs-upstream-canary/src/lib.rs:10) but no smoke exercises
        nip04, nip59 or nip98 - a PASS therefore evidences three of the five.
        The same doc calls the crate nostr-upstream-canary (nostr-bbs-upstream-canary/src/lib.rs:14) while the
        package is nostr-bbs-upstream-canary (nostr-rust-forum/Cargo.toml:2).
    end note
```

## NF-01.6 Where each Worker sits in a request

```mermaid
flowchart LR
    BROWSER["Browser - forum-client or bbs-client"]
    AGENT["Agent (agentbox or any signer)"]
    RELAYW["relay-worker<br/>WebSocket NIP-01 + REST admin<br/>nostr-bbs-relay-worker/src/lib.rs:158"]
    AUTHW["auth-worker<br/>passkey + NIP-98 REST<br/>nostr-bbs-auth-worker/src/lib.rs:143"]
    PODW["pod-worker<br/>Solid LDP + WAC"]
    SEARCHW["search-worker<br/>Workers AI embeddings"]
    PREVW["preview-worker<br/>link unfurl, SSRF-guarded"]

    BROWSER -->|"WebSocket + NIP-42 AUTH"| RELAYW
    BROWSER -->|"passkey register/login, NIP-98 REST"| AUTHW
    BROWSER -->|"pod reads/writes"| PODW
    BROWSER -->|"query"| SEARCHW
    BROWSER -->|"unfurl"| PREVW
    AGENT -->|"kinds 31400-31402, 31404, 31405"| RELAYW
    AUTHW -->|"agent_registry + whitelist writes into the relay D1"| RELAYW

    N1["The relay is the ONLY access boundary that matters - the client renders what the config describes,<br/>the relay enforces deny-by-default README.md:229-231. See NF-03 and NF-08.2"]
    N2["EXTERNAL: agentbox publishes control panels into this relay and reads back the signed decision -<br/>see AB-13 and AB-14; the estate view is ES-05"]
    N3["EXTERNAL: VisionClaw is the single live consumer of the governance surface today<br/>(ontology-concept elevation, capped at five concurrent) README.md:255-258 - see VC-24"]
```
