---
id: NF-09
title: CI gates, wrangler deploy, fixtures, benchmarks, e2e and the setup-skill scaffold
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2002, ADR-2003, ADR-2007, ADR-2016]
sources:
  - ../nostr-rust-forum/.github/workflows/ci.yml
  - ../nostr-rust-forum/.github/workflows/audit.yml
  - ../nostr-rust-forum/deny.toml
  - ../nostr-rust-forum/Cargo.toml
  - ../nostr-rust-forum/rust-toolchain.toml
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/SETUP.md
  - ../nostr-rust-forum/scripts/adr-index-gen.js
  - ../nostr-rust-forum/scripts/anti-drift-lint.sh
  - ../nostr-rust-forum/scripts/ci/dep-watchlist.sh
  - ../nostr-rust-forum/scripts/ci/dep-watchlist.txt
  - ../nostr-rust-forum/scripts/identity-vector-parity.mjs
  - ../nostr-rust-forum/scripts/package-repo.sh
  - ../nostr-rust-forum/scripts/security-audit.sh
  - ../nostr-rust-forum/scripts/sync-fixtures.sh
  - ../nostr-rust-forum/scripts/validate-forum-config.sh
  - ../nostr-rust-forum/e2e-forum-test.mjs
  - ../nostr-rust-forum/e2e-smoke-test.mjs
  - ../nostr-rust-forum/benchmarks/js-vs-wasm/bench.mjs
  - ../nostr-rust-forum/crates/nostr-bbs-setup-skill/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/transport.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/mock.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/mesh.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-upstream-canary/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-config/src/validate.rs
  - ../nostr-rust-forum/crates/nostr-bbs-config/src/ingress.rs
  - ../nostr-rust-forum/README.md
verified_commit: eb5f2323e6d72240ecb7c456181b7a674f1842f4
---

## NF-09.1 The CI gate graph

```mermaid
flowchart TB
    subgraph gates["Eight required jobs"]
        FMT["fmt - cargo fmt --all --check<br/>.github/workflows/ci.yml:70 .github/workflows/ci.yml:80"]
        CLIP["clippy - workspace, all targets, all features, -D warnings<br/>.github/workflows/ci.yml:86 .github/workflows/ci.yml:98"]
        TEST["test - cargo test --workspace --all-targets --all-features<br/>.github/workflows/ci.yml:104 .github/workflows/ci.yml:114"]
        SEC["test-security - six security-critical crates re-run alone<br/>.github/workflows/ci.yml:136 .github/workflows/ci.yml:148"]
        WASM["wasm32 check - cargo check --workspace --target wasm32-unknown-unknown<br/>.github/workflows/ci.yml:171 .github/workflows/ci.yml:188"]
        DOC["doc - cargo doc --workspace --no-deps<br/>.github/workflows/ci.yml:194 .github/workflows/ci.yml:206"]
        COV["coverage - cargo llvm-cov lcov artefact<br/>.github/workflows/ci.yml:211 .github/workflows/ci.yml:227"]
        DENY["deny - cargo deny check<br/>.github/workflows/ci.yml:239 .github/workflows/ci.yml:258"]
    end
    DG["drift-gates - ninth job, deliberately NOT a required gate<br/>.github/workflows/ci.yml:276, runs identity-vector parity .github/workflows/ci.yml:293<br/>and the anti-drift lint .github/workflows/ci.yml:296"]
    PASS["ci-pass - the single branch-protection target named CI required<br/>.github/workflows/ci.yml:303 .github/workflows/ci.yml:306"]

    FMT & CLIP & TEST & SEC & WASM & DOC & COV & DENY --> PASS
    DG -.->|"absent from the ci-pass needs list .github/workflows/ci.yml:306-314"| PASS

    N1["The test job also validates the shipped configuration contract by running the kit's own validator<br/>over forum.example.toml .github/workflows/ci.yml:121"]
    N2["The security crates re-run is nostr-bbs-core, relay-worker, pod-worker, auth-worker, preview-worker<br/>and config .github/workflows/ci.yml:150 through .github/workflows/ci.yml:155"]
    N3["The wasm32 job installs libc6-dev-i386 so the secp256k1-sys cross-compile succeeds<br/>.github/workflows/ci.yml:185 - which is why it can check the WHOLE workspace, not the three crates<br/>workspace.metadata.ci.wasm-check-packages names nostr-rust-forum/Cargo.toml:72. See NF-01.2."]
    N4["ci-pass iterates every job result and fails the aggregate unless all succeeded<br/>.github/workflows/ci.yml:329"]
    N6["The drift job needs no Rust toolchain, checks out agentbox read-only into _agentbox so the JS derivation is pinned to the real call sites<br/>.github/workflows/ci.yml:281-284, and runs only when its scripts or fixtures change - the push and PR path filters name them<br/>.github/workflows/ci.yml:33-35 .github/workflows/ci.yml:272-273"]
    N5["INVARIANT: no toolchain pin, by owner decision 2026-10-02 R8 - the forum floats on stable rust-toolchain.toml:2<br/>under -D warnings so every release is taken the day it ships BASELINE-architecture.md:244-246. Recovery when a new<br/>lint fires on code we do not own: a scoped per-item allow(clippy::lint) naming the cause, as rustc 1.99 needed on<br/>every async_trait trait - nostr-bbs-mesh/src/transport.rs:236 transport.rs:249 nostr-bbs-setup-skill/src/lib.rs:71<br/>nostr-bbs-relay-worker/src/mesh.rs:141 - never a crate-wide allow, never a pin"]
    CLIP -.- N5
    DG -.- N6
```

## NF-09.2 Supply-chain policy

```mermaid
flowchart LR
    DENYJ["deny job .github/workflows/ci.yml:239"]
    AUD["Security Audit workflow<br/>.github/workflows/audit.yml:9<br/>weekly Monday 06:17 UTC audit.yml:14<br/>plus Cargo.lock changes audit.yml:17"]
    AUDD["weekly cargo deny check advisories, same check as ci.yml<br/>.github/workflows/audit.yml:62 installed at .github/workflows/audit.yml:59"]
    DEPW["informational dependency watch list, continue-on-error<br/>.github/workflows/audit.yml:66 scripts/ci/dep-watchlist.sh:1 reading scripts/ci/dep-watchlist.txt:3"]
    SH["scripts/security-audit.sh:6 - cargo audit --deny warnings with a curated ignore list"]
    POL["deny.toml policy<br/>advisory ignores deny.toml:12<br/>licence allowlist deny.toml:25 incl. AGPL-3.0-only deny.toml:29 and MITNFA deny.toml:43<br/>ring clarification deny.toml:48"]
    HARD["wildcards = deny deny.toml:66<br/>unknown-registry = deny deny.toml:75<br/>allow-registry crates.io only deny.toml:77<br/>allow-git = [] deny.toml:78"]
    SOFT["multiple-versions = warn deny.toml:64"]

    DENYJ --> POL --> HARD
    POL --> SOFT
    AUD --> SH
    AUD --> AUDD --> POL
    AUD --> DEPW
    SH -.->|"same exceptions, no shared config format"| POL

    N1["INVARIANT: no git dependencies. allow-git is empty deny.toml:78, which is what makes the exact<br/>crates.io pin on solid-pod-rs enforceable rather than advisory - see NF-01.4 and ADR-2007"]
    N2["The advisory ignore list is duplicated by hand between deny.toml and security-audit.sh because<br/>cargo-audit has no shared config format scripts/security-audit.sh:5 - a drift risk by construction"]
    N3["MITNFA (MIT plus a no-false-attribution clause) was added for hex_lit, a rust-bitcoin 0.32<br/>dependency reached through the member wallet - ADR-2015 deny.toml:41"]
```

## NF-09.3 Repository scripts

```mermaid
flowchart TB
    ADR["adr-index-gen.js:5 - walk the ADR tree, validate frontmatter, regenerate README index"]
    DRIFT["anti-drift-lint.sh:2 - ADR-077 P3: reject SUPERSEDED Schnorr verification-key identifiers<br/>after the ADR-125 did:nostr Multikey convergence anti-drift-lint.sh:5"]
    PARITY["identity-vector-parity.mjs:3 - the JavaScript half of the ADR-2003 parity proof,<br/>loading the SAME versioned fixture as the Rust suite identity-vector-parity.mjs:6"]
    PKG["package-repo.sh:2 - package code + metadata into one review blob, ~800K tokens package-repo.sh:4"]
    SEC["security-audit.sh:6 - cargo audit with the curated ignore list"]
    SYNC["sync-fixtures.sh:4 - ADR-082 D5: clone VisionClaw, copy tests/fixtures, write a CHECKSUM"]
    VAL["validate-forum-config.sh:7 - run the kit's own nostr-bbs-config validator binary"]

    N1["INVARIANT ADR-2003: derive_subkey is a BYTE-FOR-BYTE cross-stack contract with agentbox's JS<br/>mirror derivation. The Rust half is the known-answer vector nostr-bbs-core/src/keys.rs:478 with the<br/>expected digest at nostr-bbs-core/src/keys.rs:483; the JS half is identity-vector-parity.mjs. Both read<br/>ONE fixture, so they cannot drift. EXTERNAL: see AB-11 and ES-04."]
    N2["Both anti-drift-lint.sh and identity-vector-parity.mjs are now wired into CI by the drift-gates job<br/>.github/workflows/ci.yml:293 .github/workflows/ci.yml:296, but outside the required ci-pass set .github/workflows/ci.yml:306-314.<br/>validate-forum-config.sh runs inside the required test job .github/workflows/ci.yml:121"]
    N3["Fixtures come FROM VisionClaw, so a fixture-suite change there is an unversioned input here -<br/>EXTERNAL: see the VisionClaw area (VC-*)"]
```

## NF-09.4 Config validation rules the kit enforces

```mermaid
flowchart LR
    V["validate_config<br/>nostr-bbs-config/src/validate.rs:6"]
    H["hostname must be https:// or localhost validate.rs:8"]
    R["relay.url must be wss:// or ws://localhost validate.rs:39"]
    I["ingress_policy in {allowlist, open} validate.rs:56"]
    A["admin.mode in {static, d1} validate.rs:64"]
    ING["[[ingress]] recipients well-formed and unique, ADR-2022 validate.rs:226 to nostr-bbs-config/src/ingress.rs:94"]
    Z1["zone ids unique validate.rs:241"]
    ENC["ADR-2016: encrypted && visibility = public is rejected - anonymous readers<br/>can never hold a zone key validate.rs:250"]
    Z2["slug must be lowercase a-z 0-9 hyphen validate.rs:271 validate.rs:298"]
    Z3["slug unique validate.rs:278"]
    Z4["a slug must not collide with a DIFFERENT zone's id validate.rs:283"]

    V --> H & R & I & A & ING & Z1 & ENC & Z2 & Z3 & Z4

    N1["This is what makes forum.example.toml's zeroed placeholder pubkeys fail by design - see NF-08.1"]
    N2["The slug-versus-id collision rule is what keeps the client's zone-slug route aliases unambiguous -<br/>see NF-05.2"]
    N3["ADR-2016 gate; the sealed-history migration CLI it authorises (nostr-bbs-zone-migrate, ADR-2017)<br/>is a config/zones concern - see NF-08"]
```

**Open:** `forum.example.toml` already sets `[deployment] profile = "forum"` and documents `room` as proposed under ADR-2029 (`../nostr-rust-forum/forum.example.toml:31`), but the config validator checks no profile value and no crate source mentions a channel dialect (`../nostr-rust-forum/crates/nostr-bbs-config/src/validate.rs:6`), so the key is read by nothing here.

## NF-09.5 Deploy — the wrangler sequence

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator
    participant CF as Cloudflare

    OP->>OP: rustup target add wasm32-unknown-unknown SETUP.md:9
    OP->>OP: cargo install trunk SETUP.md:14
    OP->>OP: cargo install worker-build --version 0.8.4 --locked SETUP.md:29
    OP->>CF: wrangler d1 create nostr-bbs-auth SETUP.md:67
    OP->>CF: wrangler d1 create nostr-bbs-relay SETUP.md:68
    OP->>CF: hand-run the auth schema SQL SETUP.md:75
    OP->>CF: apply the events migration, optional because the relay creates it itself SETUP.md:100 - whitelist is a checked-in migration SETUP.md:108 SETUP.md:114
    OP->>CF: apply the governance migration SETUP.md:119
    OP->>CF: create four KV namespaces SETUP.md:129
    OP->>CF: create two R2 buckets SETUP.md:138
    OP->>OP: paste resource ids into each wrangler.toml SETUP.md:145
    OP->>CF: wrangler secret put for the two secrets SETUP.md:155
    OP->>CF: wrangler deploy, one crate at a time SETUP.md:178
    OP->>CF: DNS - CNAME or Workers Routes per subdomain SETUP.md:190
    OP->>OP: FORUM_BASE=/community trunk build --release --public-url /community/ SETUP.md:275

    Note over OP: worker-build is pinned to 0.8.4 because 0.8.5 breaks the abort-handler codegen SETUP.md:29
    Note over OP: SETUP.md step 0 asks the operator to choose a deployment profile before creating any resource, but marks it proposed (ADR-2029), only forum deploys SETUP.md:32
    Note over OP: RELAY_URL must be set in nip42 mode, blank skips the relay tag check and logs an error on every AUTH SETUP.md:158 - CALIBRATION_SELECTION_KEY is a relay secret SETUP.md:165
    Note over OP: optional step 10, content-free Web Push (ADR-2023): VAPID keys, PUSH_ENABLED, VAPID_PRIVATE_KEY secret SETUP.md:373
    Note over OP: The client build is where FORUM_BASE is baked in - it is a COMPILE-TIME constant, so a base change is a rebuild, not a config flip. See NF-05.1.
    Note over CF: DIVERGENCE (narrowed at 66ca8d8): the auth schema at step 6 is still hand-run SQL pasted from the guide, but the relay tables are migration files the relay also creates at startup - see NF-08.6
```

## NF-09.6 End-to-end journeys — against production, not a fixture

```mermaid
flowchart TB
    subgraph forum["e2e-forum-test.mjs - BASE https://dreamlab-ai.com/community e2e-forum-test.mjs:3"]
        F1["1 forum homepage e2e-forum-test.mjs:52"]
        F2["2 WASM module loads e2e-forum-test.mjs:66"]
        F3["3 relay WebSocket from forum context e2e-forum-test.mjs:84"]
        F4["4 auth API e2e-forum-test.mjs:141"]
        F5["5 pod API e2e-forum-test.mjs:171"]
        F6["6 search API e2e-forum-test.mjs:201"]
        F7["7 forum navigation e2e-forum-test.mjs:235"]
        F8["8 UI elements e2e-forum-test.mjs:254"]
        F9["9 console and network audit e2e-forum-test.mjs:277"]
    end
    subgraph smoke["e2e-smoke-test.mjs - BASE https://dreamlab-ai.com e2e-smoke-test.mjs:3"]
        S1["1 homepage :43 | 2 UI :56 | 3 nav :65"]
        S2["4 WebSocket relay :72 | 5 auth CORS :101 | 6 pod health :139"]
        S3["7 search :157 | 8 route nav :196 | 9 WASM :220 | 10 console errors :235"]
    end

    N1["DIVERGENCE: both suites drive a REAL deployment over the public internet with a Nix-pinned Chromium<br/>path baked in e2e-forum-test.mjs:19. They are operator smoke tests, not CI gates - ci.yml runs neither."]
    N2["The journeys exercise all five workers from a browser context, which is the only place the CORS and<br/>ALLOWED_ORIGIN envelopes are actually proved - see NF-07.5 and NF-08.8"]
    N3["EXTERNAL: dreamlab-ai.com is a thin CONSUMER of this kit, not part of it - see the dreamlab-ai-website<br/>area (DW-*)"]
```

## NF-09.7 Benchmarks — JS versus WASM on the seven core operations

```mermaid
flowchart LR
    B["benchmarks/js-vs-wasm/bench.mjs:2"]
    SCOPE["noble / nostr-tools versus Rust-compiled WASM on 7 core ops bench.mjs:4"]
    RUN["node --experimental-wasm-modules bench.mjs bench.mjs:8"]
    OPS["1 HKDF-PRF derivation bench.mjs:183<br/>2 Schnorr sign bench.mjs:205<br/>3 NIP-44 encrypt 1KB bench.mjs:230<br/>4 NIP-44 encrypt 10KB bench.mjs:253<br/>5 NIP-44 decrypt 1KB bench.mjs:278<br/>6 event id computation bench.mjs:304<br/>7 NIP-98 token creation bench.mjs:366"]
    WARM["warmup is 10 percent of iterations, minimum 10 bench.mjs:77 bench.mjs:92"]

    B --> SCOPE --> RUN --> OPS
    B --> WARM

    N1["The benchmarked set is exactly the surface the upstream-nostr absorption would replace - so the<br/>bench is the performance half of the ADR-2002 decision the canary makes the correctness half.<br/>See NF-01.5."]
```

## NF-09.8 Canary and mesh — the two crates that prove rather than serve

```mermaid
flowchart TB
    CAN["nostr-bbs-upstream-canary<br/>not linked into any binary nostr-bbs-upstream-canary/src/lib.rs:13"]
    SHAPE["PASS to Shape A full absorption nostr-bbs-upstream-canary/src/lib.rs:17<br/>FAIL to Shape C patch-in-place nostr-bbs-upstream-canary/src/lib.rs:19"]
    MESH["nostr-bbs-mesh"]
    TRAITS["MeshSocket nostr-bbs-mesh/src/transport.rs:238<br/>MeshTransport nostr-bbs-mesh/src/transport.rs:251<br/>RelayTransport generic over any socket nostr-bbs-mesh/src/transport.rs:325"]
    ONLYIMPL["The ONLY MeshSocket impl in the tree is the test-only MockSocket<br/>nostr-bbs-mesh/src/mock.rs:249"]
    DEP["relay-worker DOES depend on the crate<br/>nostr-bbs-relay-worker/Cargo.toml:27"]
    DEAD["but its own mesh wiring is allow(dead_code)<br/>nostr-bbs-relay-worker/src/mesh.rs:48"]

    CAN --> SHAPE
    MESH --> TRAITS --> ONLYIMPL
    MESH --> DEP --> DEAD

    N1["ANOMALY O9 re-verified and REFINED: the register calls nostr-bbs-mesh a trait scaffold with no impl<br/>AND no relay import. The no-impl half holds - only MockSocket implements the socket trait. The<br/>no-import half does NOT: the relay declares the dependency at nostr-bbs-relay-worker/Cargo.toml:27 and<br/>carries a documented-but-dead PeerConnector seam. The README no longer repeats the stale claim and now<br/>draws mesh as a dotted configured-routing edge README.md:60 - see NF-03.13."]
    N2["The canary is the gate on ADR-2002: nostr-bbs-core keeps on-wasm32 Schnorr until a Shape A verdict<br/>lands AND a module is deliberately deleted - see NF-01.5"]
    N3["Both crates are DELIBERATELY inert. Neither is dead code to delete; each is a decision held open<br/>in compilable form."]
```

## NF-09.9 The setup-skill provider scaffold

```mermaid
classDiagram
    class Provider {
        tier : nostr-bbs-setup-skill/src/lib.rs:75
        provision : nostr-bbs-setup-skill/src/lib.rs:78
        render_wrangler : nostr-bbs-setup-skill/src/lib.rs:82
    }
    class SelfHostProvider {
        nostr-bbs-setup-skill/src/lib.rs:90 - only provision with real logic
    }
    class CloudflareWorkersProvider {
        nostr-bbs-setup-skill/src/lib.rs:119
    }
    class FlyDotIoProvider {
        nostr-bbs-setup-skill/src/lib.rs:149
    }
    class TurnkeyProvider {
        nostr-bbs-setup-skill/src/lib.rs:177
    }
    class KubernetesProvider {
        nostr-bbs-setup-skill/src/lib.rs:204
    }
    Provider <|.. SelfHostProvider
    Provider <|.. CloudflareWorkersProvider
    Provider <|.. FlyDotIoProvider
    Provider <|.. TurnkeyProvider
    Provider <|.. KubernetesProvider

    note for Provider "DIVERGENCE, self-declared: scaffold only - each impl returns SetupError::NotYetImplemented for unfinished methods nostr-bbs-setup-skill/src/lib.rs:11, e.g. nostr-bbs-setup-skill/src/lib.rs:111 and nostr-bbs-setup-skill/src/lib.rs:130"
    note for TurnkeyProvider "TurnkeyProvider::render_wrangler returns Unsupported by DESIGN, not by omission - a custody-tier provider never writes a wrangler.toml (variant at nostr-bbs-setup-skill/src/lib.rs:48)"
    note for SelfHostProvider "The provider abstraction is a CUSTODY ladder, not a hosting menu - each tier is a different answer to who holds the keys. Tiers are named at nostr-bbs-setup-skill/src/lib.rs:94"
```
