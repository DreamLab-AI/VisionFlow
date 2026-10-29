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
  - ../nostr-rust-forum/crates/nostr-bbs-jev/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/adr/ADR-2007-kit-exact-pin-beta-channel.md
  - ../nostr-rust-forum/crates/nostr-bbs-ascii/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-zone-migrate/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-upstream-canary/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-upstream-canary/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/admin_events.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/profile_stats.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-preview-worker/wrangler.toml
  - ../nostr-rust-forum/crates/nostr-bbs-search-worker/wrangler.toml
  - ../nostr-rust-forum/.github/workflows/ci.yml
verified_commit: 8a5cf817e81b20e2b761899eab753768c0cb4988
---

## NF-01.1 Twenty crates in four layers

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
        JEV["nostr-bbs-jev nostr-rust-forum/Cargo.toml:18 - opt-in Jev typed-decision adapter, publish = false nostr-bbs-jev/Cargo.toml:4"]
        ASCII["nostr-bbs-ascii nostr-rust-forum/Cargo.toml:22"]
    end
    subgraph workers["Cloudflare Worker reference implementations"]
        AUTH["nostr-bbs-auth-worker nostr-rust-forum/Cargo.toml:25"]
        POD["nostr-bbs-pod-worker nostr-rust-forum/Cargo.toml:26"]
        PREV["nostr-bbs-preview-worker nostr-rust-forum/Cargo.toml:27"]
        RELAY["nostr-bbs-relay-worker nostr-rust-forum/Cargo.toml:28"]
        SEARCH["nostr-bbs-search-worker nostr-rust-forum/Cargo.toml:29"]
    end
    subgraph clients["Leptos CSR browser clients"]
        FC["nostr-bbs-forum-client nostr-rust-forum/Cargo.toml:32"]
        BBS["nostr-bbs-bbs-client nostr-rust-forum/Cargo.toml:35 - served at /community/bbs/"]
    end
    ZM["nostr-bbs-zone-migrate nostr-rust-forum/Cargo.toml:51 - operator CLI, native-only binary, ADR-2017 sealed-original migration"]
    PK["heads-up poker - pure engine + native house seat<br/>nostr-bbs-poker nostr-rust-forum/Cargo.toml:33<br/>nostr-bbs-poker-citizen nostr-rust-forum/Cargo.toml:36"]
    SA["nostr-bbs-sidestr-admin nostr-rust-forum/Cargo.toml:40 - operator CLI for the pinned sidestr chains"]
    GP["nostr-bbs-governance-probe nostr-rust-forum/Cargo.toml:55 - ADR-2011 M4 probe suite, one receipt per probe"]
    CANARY["nostr-bbs-upstream-canary nostr-rust-forum/Cargo.toml:60 - validation only, linked into no binary"]

    AUTH --> CORE
    POD --> CORE
    RELAY --> CORE
    SEARCH --> CORE
    PREV --> CORE
    FC --> CORE
    BBS --> CORE
    ZM --> CORE
    PK --> CORE
    SA --> CORE
    GP --> CORE
    CFG --> RELAY
    CFG --> FC
    JEV --> RELAY
    RATE --> AUTH
    RATE --> RELAY
    ASCII --> PREV
    ASCII --> BBS
    MESH -.->|"README.md:60 dotted, configured peer routing<br/>relay-worker depends on it nostr-bbs-relay-worker/Cargo.toml:27"| RELAY

    N1["resolver 2, edition 2021, rust-version 1.85 nostr-rust-forum/Cargo.toml:2 nostr-rust-forum/Cargo.toml:67"]
    N2["DIVERGENCE: the kit publishes on one beta channel but its path-dep versions are not one set - core beta.18<br/>nostr-rust-forum/Cargo.toml:188, config, mesh, rate-limit and jev beta.16 nostr-rust-forum/Cargo.toml:189-192,<br/>poker beta.15 nostr-rust-forum/Cargo.toml:194, ascii still beta.10 nostr-rust-forum/Cargo.toml:193;<br/>core declares beta.18 crates/nostr-bbs-core/Cargo.toml:3 and ascii beta.10 crates/nostr-bbs-ascii/Cargo.toml:3"]
    N3["DOC-DRIFT ADR-2007 / BASELINE known-divergence: the Cargo.toml comment above the path deps still<br/>says published to crates.io as 1.0.0-beta.3 while the tree is on beta.18 nostr-rust-forum/Cargo.toml:187"]
    N4["nostr-bbs-zone-migrate is library-plus-native-binary only, so it still passes the wasm32 workspace<br/>check nostr-rust-forum/Cargo.toml:49-51; seals plaintext channel history into sealed-original<br/>envelopes, verifies, then purges the plaintext crates/nostr-bbs-zone-migrate/Cargo.toml:5"]
```

**Drift:** the README draws the mesh as a dotted "configured peer routing" edge into the relay (`../nostr-rust-forum/README.md:60`), yet the relay-worker declares a normal path dependency on `nostr-bbs-mesh` with a `src/mesh.rs` wiring note (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml:27`).

**Open:** does the "room" deployment profile (ADR-2029, proposed, no implementation) or the NIP-17/NIP-171 interop work change the crate set? The README places that implementation on the unmerged `feat/upgrade-phase-0-1` branch (`../nostr-rust-forum/README.md:124`) and `docs/BASELINE-architecture.md` records the profile as "proposed, not implemented" (`../nostr-rust-forum/docs/BASELINE-architecture.md:271`).

## NF-01.2 Build matrix — one target, one release profile

```mermaid
flowchart LR
    TC["rust-toolchain.toml:2 channel stable<br/>rust-toolchain.toml:3 targets wasm32-unknown-unknown"]
    PROF["release profile nostr-rust-forum/Cargo.toml:208<br/>opt-level z nostr-rust-forum/Cargo.toml:209<br/>panic abort nostr-rust-forum/Cargo.toml:213"]
    WB["worker-build --release<br/>nostr-bbs-auth-worker/wrangler.toml:6<br/>nostr-bbs-relay-worker/wrangler.toml:6<br/>nostr-bbs-pod-worker/wrangler.toml:6<br/>nostr-bbs-preview-worker/wrangler.toml:6<br/>nostr-bbs-search-worker/wrangler.toml:6"]
    SHIM["build/worker/shim.mjs entrypoint<br/>nostr-bbs-relay-worker/wrangler.toml:2"]
    CIW["declared wasm-check set is three crates, jev joined it<br/>nostr-rust-forum/Cargo.toml:72"]
    CIJOB["actual CI wasm job checks the WHOLE workspace<br/>.github/workflows/ci.yml:188"]

    TC --> PROF --> WB --> SHIM
    TC --> CIW
    TC --> CIJOB

    N1["panic = abort is not a tuning choice - workers and WASM cannot unwind, so the unwinding tables are<br/>dead weight nostr-rust-forum/Cargo.toml:203-206"]
    N2["DOC-DRIFT: workspace.metadata.ci.wasm-check-packages names three crates (config, preview-worker, jev) nostr-rust-forum/Cargo.toml:72<br/>with a comment saying to expand it as the secp256k1-sys cross-compile resolves nostr-rust-forum/Cargo.toml:70-71,<br/>but no workflow or script reads that key - the wasm job runs cargo check --workspace<br/>.github/workflows/ci.yml:188, made possible by installing libc6-dev-i386 .github/workflows/ci.yml:185.<br/>The metadata is inert."]
    N3["compatibility_date 2025-09-01 is identical across all five templates, e.g. nostr-bbs-search-worker/wrangler.toml:3"]
```

## NF-01.3 Worker entry points and their triggers

```mermaid
flowchart TB
    subgraph fetchentry["#[event(fetch)] handlers"]
        A["auth-worker<br/>nostr-bbs-auth-worker/src/lib.rs:144"]
        R["relay-worker<br/>nostr-bbs-relay-worker/src/lib.rs:209"]
    end
    subgraph cronentry["#[event(scheduled)] handlers"]
        AC["auth-worker scheduled<br/>nostr-bbs-auth-worker/src/lib.rs:873"]
        RC["relay-worker scheduled - every 5 min<br/>nostr-bbs-relay-worker/src/lib.rs:1146<br/>trigger nostr-bbs-relay-worker/wrangler.toml:115"]
        SC["search-worker cron - every 5 min<br/>nostr-bbs-search-worker/wrangler.toml:36"]
    end
    WSUP["WebSocket upgrade to the Durable Object<br/>nostr-bbs-relay-worker/src/lib.rs:228"]
    NIP11["NIP-11 relay info on Accept application/nostr+json<br/>nostr-bbs-relay-worker/src/lib.rs:237"]
    HTTPR["HTTP route table<br/>nostr-bbs-relay-worker/src/lib.rs:286"]
    ADMDEL["ADR-2017 POST /api/admin/events/delete - silent kind-42 delete, no kind-5<br/>nostr-bbs-relay-worker/src/admin_events.rs:5"]
    SUCC["GET /api/profiles/successors - admin-set pubkey aliases for DM routing<br/>nostr-bbs-relay-worker/src/profile_stats.rs:6"]
    PSTATS["GET /api/profile-stats - viewer-scoped user-card activity stats<br/>nostr-bbs-relay-worker/src/profile_stats.rs:5"]

    R --> WSUP
    R --> NIP11
    R --> HTTPR
    HTTPR --> ADMDEL
    HTTPR --> SUCC
    HTTPR --> PSTATS

    N1["Both fetch entries bootstrap their schema and the shared replay table, but differently: the relay does it once per isolate<br/>behind SCHEMA_READY, after the CORS preflight and before the WebSocket branch, and latches only on success<br/>nostr-bbs-relay-worker/src/lib.rs:222, nostr-bbs-relay-worker/src/lib.rs:200; the auth worker runs it at the top of every request<br/>nostr-bbs-auth-worker/src/lib.rs:163"]
    N2["The relay binds its replay schema to REPLAY_DB, the auth worker to its own DB - the same physical<br/>database either way nostr-bbs-relay-worker/src/lib.rs:224, nostr-bbs-auth-worker/src/lib.rs:164"]
    N3["ADR-2017: deleting the kind-42 row outright, with no kind-5 tombstone, is deliberate - a kind-5<br/>would tombstone the id that a sealed-original envelope restores on every forum client<br/>nostr-bbs-relay-worker/src/lib.rs:371-373"]
```

## NF-01.4 Pinned dependency spine

```mermaid
classDiagram
    class Workspace {
        nostr 0.44.7 nip04 nip44 nip59 nip98 : nostr-rust-forum/Cargo.toml:85
        worker 0.8 : nostr-rust-forum/Cargo.toml:98
        leptos 0.7 csr : nostr-rust-forum/Cargo.toml:88
        k256 0.13.4 schnorr ecdh : nostr-rust-forum/Cargo.toml:104
        passkey-types 0.3 : nostr-rust-forum/Cargo.toml:101
        comrak 0.38 : nostr-rust-forum/Cargo.toml:136
        image 0.24 pure-Rust decoders only : nostr-rust-forum/Cargo.toml:200
        rusqlite 0.37 bundled, test-only for the shared whitelist SQL : nostr-rust-forum/Cargo.toml:78
        solid-pod-rs caret 0.5.0-alpha.14 core : nostr-rust-forum/Cargo.toml:185
    }
    class SolidPodRs {
        wac : pod-worker acl
        webid : pod-worker webid
        payments : pod-worker payments
        did_nostr_types : core did
    }
    Workspace --> SolidPodRs

    note for Workspace "TENSION ADR-2007 vs Cargo.toml: ADR-2007 and BASELINE invariant 6 say solid-pod-rs stays an EXACT (=) pin docs/adr/ADR-2007-kit-exact-pin-beta-channel.md:30, docs/BASELINE-architecture.md:192, but the manifest now declares a caret range under estate register ER-35 nostr-rust-forum/Cargo.toml:181-185; Cargo.lock still fixes what a build resolves"
    note for SolidPodRs "EXTERNAL: the consumer surface is enumerated in docs/consumer-surface-map.md:14 - see the solid-pod-rs area (SP-*) and NF-04"
```

## NF-01.5 Upstream `nostr` absorption — canary-first, nothing deleted

```mermaid
stateDiagram-v2
    [*] --> DependencyEnabled: nostr 0.44.7 with four NIP flags on<br/>nostr-rust-forum/Cargo.toml:85
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
        package is nostr-bbs-upstream-canary (crates/nostr-bbs-upstream-canary/Cargo.toml:2).
    end note
```

## NF-01.6 Where each Worker sits in a request

```mermaid
flowchart LR
    BROWSER["Browser - forum-client or bbs-client"]
    AGENT["Agent (agentbox or any signer)"]
    RELAYW["relay-worker<br/>WebSocket NIP-01 + REST admin<br/>nostr-bbs-relay-worker/src/lib.rs:209"]
    AUTHW["auth-worker<br/>passkey + NIP-98 REST<br/>nostr-bbs-auth-worker/src/lib.rs:144"]
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

    N1["The relay is the ONLY access boundary that matters - the client renders what the config describes,<br/>the relay enforces access README.md:70, and TOML, bindings and injected config must agree - a contract to check, not a guarantee README.md:72. See NF-03 (deny-by-default) and NF-08.2"]
    N2["EXTERNAL: agentbox publishes control panels into this relay and reads back the signed decision -<br/>see AB-13 and AB-14; the estate view is ES-05"]
    N3["EXTERNAL: VisionClaw is the single live consumer of the governance surface today<br/>(the documented estate consumer is the ontology-review workflow; universal use is not established) README.md:120 - see VC-24"]
```
