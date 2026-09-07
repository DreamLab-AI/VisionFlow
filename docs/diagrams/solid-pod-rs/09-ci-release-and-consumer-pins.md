---
id: SP-09
title: CI gates, the release pipeline, versioning and the consumer pin matrix
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2001]
sources:
  - ../solid-pod-rs/.github/workflows/ci.yml
  - ../solid-pod-rs/.github/workflows/release.yml
  - ../solid-pod-rs/.github/dependabot.yml
  - ../solid-pod-rs/.github/CODEOWNERS
  - ../solid-pod-rs/deny.toml
  - ../solid-pod-rs/Cargo.toml
  - ../solid-pod-rs/CHANGELOG.md
  - ../solid-pod-rs/scripts/check-diagram-staleness.sh
  - ../solid-pod-rs/scripts/parity-check.sh
  - ../solid-pod-rs/scripts/test-all.sh
  - ../solid-pod-rs/scripts/sync-fixtures.sh
  - ../solid-pod-rs/crates/solid-pod-rs/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-server/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs-git/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs/fuzz/fuzz_targets/sparql_update.rs
  - ../solid-pod-rs/crates/solid-pod-rs/benches/storage_backend_bench.rs
  - ../solid-pod-rs/crates/solid-pod-rs/benches/wac_eval_bench.rs
  - ../solid-pod-rs/crates/solid-pod-rs/benches/ldp_content_negotiation_bench.rs
  - ../solid-pod-rs/crates/solid-pod-rs/benches/nip98_verify_bench.rs
  - ../solid-pod-rs/crates/solid-pod-rs/benches/dpop_replay_bench.rs
  - ../solid-pod-rs/crates/solid-pod-rs/docs/benchmarks.md
  - ../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs
verified_commit: 1d9da5270
---

## SP-09.1 CI triggers and the concurrency guard

```mermaid
flowchart LR
    PUSH["push to main, path-filtered to crates, Cargo files and ci.yml<br/>../solid-pod-rs/.github/workflows/ci.yml:4"]
    PR["pull_request to main, same path filter<br/>../solid-pod-rs/.github/workflows/ci.yml:11"]
    CRON["schedule — Mondays 06:17 UTC<br/>../solid-pod-rs/.github/workflows/ci.yml:19"]
    WD["workflow_dispatch<br/>../solid-pod-rs/.github/workflows/ci.yml:21"]
    CG["concurrency group per workflow+ref, cancel-in-progress<br/>../solid-pod-rs/.github/workflows/ci.yml:23"]
    ENVF["RUSTFLAGS and RUSTDOCFLAGS = -D warnings<br/>../solid-pod-rs/.github/workflows/ci.yml:31"]
    JOBS["the eight jobs — SP-09.2"]

    PUSH --> CG
    PR --> CG
    CRON --> CG
    WD --> CG
    CG --> ENVF --> JOBS

    N["The weekly cron exists so cargo-audit re-runs against a moving advisory<br/>database even when nothing in the repo changed — a dependency becomes<br/>vulnerable without a commit."]
    CRON -.-> N
    N2["-D warnings is set for BOTH rustc and rustdoc at the workflow level, so a<br/>broken intra-doc link fails CI the same way a compiler warning does.<br/>../solid-pod-rs/.github/workflows/ci.yml:32"]
    ENVF -.-> N2
```

## SP-09.2 The eight CI jobs and the required-check aggregator

```mermaid
flowchart TD
    BT["build-test — the OS x toolchain x feature matrix<br/>../solid-pod-rs/.github/workflows/ci.yml:44"]
    MSRV["msrv — cargo check on 1.88<br/>../solid-pod-rs/.github/workflows/ci.yml:101"]
    WASM["wasm-check — wasm32 core surface<br/>../solid-pod-rs/.github/workflows/ci.yml:126"]
    DENY["cargo-deny<br/>../solid-pod-rs/.github/workflows/ci.yml:153"]
    AUDIT["cargo-audit<br/>../solid-pod-rs/.github/workflows/ci.yml:172"]
    COV["coverage — tarpaulin<br/>../solid-pod-rs/.github/workflows/ci.yml:190"]
    WS["workspace — every member, default features<br/>../solid-pod-rs/.github/workflows/ci.yml:236"]
    DIA["diagrams — the RES-c staleness guard<br/>../solid-pod-rs/.github/workflows/ci.yml:260"]
    REQ["ci-required — the single branch-protection check<br/>../solid-pod-rs/.github/workflows/ci.yml:273"]

    BT --> REQ
    MSRV --> REQ
    WASM --> REQ
    DENY --> REQ
    AUDIT --> REQ
    COV --> REQ
    WS --> REQ
    DIA --> REQ

    N["ci-required runs with if: always() and then ASSERTS each dependency's result<br/>explicitly (../solid-pod-rs/.github/workflows/ci.yml:287) — a skipped or<br/>cancelled job is not silently treated as success, which a plain needs: would do."]
    REQ -.-> N
    N2["INVARIANT: branch protection points at ONE check. Adding a job without adding<br/>it to the needs list and the assertion block makes it advisory, not required."]
    REQ -.-> N2
```

## SP-09.3 The build matrix

```mermaid
flowchart LR
    OS["os: ubuntu-latest, macos-latest<br/>../solid-pod-rs/.github/workflows/ci.yml:51"]
    TC["toolchain: stable, beta<br/>../solid-pod-rs/.github/workflows/ci.yml:52"]
    F1["label default, no flags<br/>../solid-pod-rs/.github/workflows/ci.yml:54"]
    F2["label oidc — memory-backend, fs-backend, oidc only<br/>../solid-pod-rs/.github/workflows/ci.yml:57"]
    F3["label all-features<br/>../solid-pod-rs/.github/workflows/ci.yml:59"]
    STEPS["fmt, check, clippy -D warnings, test, doc<br/>../solid-pod-rs/.github/workflows/ci.yml:81"]
    BETA["continue-on-error when toolchain == beta<br/>../solid-pod-rs/.github/workflows/ci.yml:47"]

    OS --> STEPS
    TC --> STEPS
    F1 --> STEPS
    F2 --> STEPS
    F3 --> STEPS
    TC --> BETA

    N["The matrix runs in crates/solid-pod-rs<br/>(../solid-pod-rs/.github/workflows/ci.yml:62), so it exercises CORE-CRATE<br/>feature combinations. The separate workspace job covers the other seven members<br/>at default features — the two jobs together are the coverage, neither alone."]
    STEPS -.-> N
    N2["beta is informational: a nightly-track regression reports without blocking the<br/>pipeline, so an upstream compiler change cannot wedge every PR."]
    BETA -.-> N2
    N3["The oidc row exists because --no-default-features plus a narrow feature set is<br/>the combination most likely to break — it catches a missing cfg guard that both<br/>default and all-features would hide."]
    F2 -.-> N3
```

## SP-09.4 Supply-chain and quality gates

```mermaid
flowchart TD
    MSRV["cargo check --all-targets on toolchain 1.88<br/>../solid-pod-rs/.github/workflows/ci.yml:117"]
    WASM["cargo check --target wasm32-unknown-unknown<br/>--no-default-features --features core<br/>../solid-pod-rs/.github/workflows/ci.yml:144"]
    DENY["cargo-deny-action against the root deny.toml<br/>../solid-pod-rs/.github/workflows/ci.yml:161"]
    AUD["cargo audit --deny warnings, at the REPO ROOT<br/>../solid-pod-rs/.github/workflows/ci.yml:185"]
    COV["cargo tarpaulin --all-features --workspace --fail-under 60<br/>../solid-pod-rs/.github/workflows/ci.yml:213"]
    CC["Codecov upload, fail_ci_if_error false<br/>../solid-pod-rs/.github/workflows/ci.yml:220"]
    WSJ["cargo clippy --workspace and cargo test --workspace<br/>../solid-pod-rs/.github/workflows/ci.yml:236"]
    DIA["bash scripts/check-diagram-staleness.sh<br/>../solid-pod-rs/.github/workflows/ci.yml:268"]

    MSRV --> GATE["ci-required"]
    WASM --> GATE
    DENY --> GATE
    AUD --> GATE
    COV --> GATE
    WSJ --> GATE
    DIA --> GATE

    N["INVARIANT: the wasm job is the ONLY thing keeping the core feature genuinely<br/>no-IO. A stray tokio import behind core compiles fine on the host and fails only<br/>here — which is what protects the edge consumer in SP-01.4."]
    WASM -.-> N
    N2["cargo audit runs at the REPO ROOT deliberately, so the whole-workspace<br/>Cargo.lock is audited rather than the core crate's slice.<br/>../solid-pod-rs/.github/workflows/ci.yml:183"]
    AUD -.-> N2
    N3["The coverage GATE is tarpaulin's --fail-under; the Codecov upload is reporting<br/>only, so a missing CODECOV_TOKEN cannot mask a passing gate.<br/>../solid-pod-rs/.github/workflows/ci.yml:225"]
    CC -.-> N3
    N4["DIVERGENCE (README status, 2026-08-19): cargo audit --deny warnings FAILS on<br/>RUSTSEC-2026-0258 in both shipped HTTP/2 stacks. The gate is real and it is<br/>currently red."]
    AUD -.-> N4
```

## SP-09.5 The diagram staleness guard

```mermaid
flowchart TD
    SRC["docs/diagrams/src/*.mmd<br/>../solid-pod-rs/scripts/check-diagram-staleness.sh:4"]
    ET["effective_time = mtime when untracked or dirty,<br/>else the last commit epoch<br/>../solid-pod-rs/scripts/check-diagram-staleness.sh:11"]
    BOTH["every source needs BOTH a .svg and a .png<br/>../solid-pod-rs/scripts/check-diagram-staleness.sh:19"]
    PASS["source and render share a commit -> equal times -> pass<br/>../solid-pod-rs/scripts/check-diagram-staleness.sh:14"]
    FAIL["source edited alone, or a render missing -> fail"]

    SRC --> ET --> PASS
    ET --> FAIL
    BOTH --> FAIL

    N["A missing render is treated as MAXIMALLY stale, so an unrendered diagram fails<br/>rather than passing by absence.<br/>../solid-pod-rs/scripts/check-diagram-staleness.sh:21"]
    BOTH -.-> N
    N2["Pure git plus coreutils — no browser and no npm, so the job needs neither the<br/>Chrome sidecar nor a renderer at CI time.<br/>../solid-pod-rs/scripts/check-diagram-staleness.sh:23"]
    ET -.-> N2
    N3["This guards the repo's OWN docs/diagrams tree (the nine .mmd sources under<br/>crates/solid-pod-rs/docs/diagrams/src). It is unrelated to the VisionFlow<br/>diagrams-as-code tree this topic lives in."]
    SRC -.-> N3
```

## SP-09.6 The release pipeline

```mermaid
sequenceDiagram
    autonumber
    participant T as a v* tag, or workflow_dispatch
    participant V as verify-tag<br/>../solid-pod-rs/.github/workflows/release.yml:26
    participant D as publish-dry-run<br/>../solid-pod-rs/.github/workflows/release.yml:70
    participant CL as changelog<br/>../solid-pod-rs/.github/workflows/release.yml:92
    participant P as cargo-publish<br/>../solid-pod-rs/.github/workflows/release.yml:186
    participant G as github-release<br/>../solid-pod-rs/.github/workflows/release.yml:143

    T->>V: on push tags v*<br/>../solid-pod-rs/.github/workflows/release.yml:5
    V->>V: strip the leading v, read the crate version<br/>../solid-pod-rs/.github/workflows/release.yml:48
    V->>V: fall back to the WORKSPACE version when the crate inherits it<br/>../solid-pod-rs/.github/workflows/release.yml:55
    alt tag does not match
        V-->>T: hard error, release refused<br/>../solid-pod-rs/.github/workflows/release.yml:63
    end
    V->>D: cargo publish --dry-run --all-features<br/>../solid-pod-rs/.github/workflows/release.yml:87
    V->>CL: git log since the previous tag, scoped to the crate
    D->>P: MANUAL APPROVAL — the crates-io-publish environment<br/>../solid-pod-rs/.github/workflows/release.yml:190
    P->>P: preflight — CARGO_REGISTRY_TOKEN must be non-empty<br/>../solid-pod-rs/.github/workflows/release.yml:205
    P->>P: skip when the version is already on the sparse index<br/>../solid-pod-rs/.github/workflows/release.yml:228
    P->>P: cargo publish --all-features<br/>../solid-pod-rs/.github/workflows/release.yml:232
    P->>G: only AFTER the registry upload succeeds<br/>../solid-pod-rs/.github/workflows/release.yml:150
    G->>G: prerelease auto-detected from an -alpha, -beta or -rc suffix<br/>../solid-pod-rs/.github/workflows/release.yml:166

    Note over G: The GitHub release runs LAST on purpose: it is the cheap, updatable artifact<br/>and the registry upload is the irreversible one. The old ordering left a GitHub<br/>release with no crate behind it when publish failed.
    Note over P: The index check makes a re-dispatched tag idempotent — a re-run after a<br/>partially failed release stays green instead of erroring on "already published".
```

## SP-09.7 Version alignment and the consumer pin matrix

```mermaid
flowchart TD
    WSV["workspace.package.version 0.5.0-alpha.9<br/>../solid-pod-rs/Cargo.toml:15"]
    INH["every crate inherits it — version.workspace = true<br/>crates/solid-pod-rs/Cargo.toml:3"]
    REL["0.5.0-alpha.9 re-pins ALL EIGHT crates and yanks the superseded versions<br/>../solid-pod-rs/CHANGELOG.md:12"]
    DRIFT["the drift it fixed: siblings sat at alpha.7 on crates.io while the root<br/>crate moved to alpha.8<br/>../solid-pod-rs/CHANGELOG.md:11"]

    WSV --> INH --> REL
    DRIFT --> REL

    AB["EXTERNAL: agentbox builds a PINNED solid-pod-rs-server binary through Nix for<br/>the native pod tier — the estate pin is v0.5.0-alpha.9. See AB-08, AB-06 and ES-08."]
    VC["EXTERNAL: VisionClaw consumes solid-pod-rs at the current alpha line with the<br/>feature set its embedded pod needs (LDP, WAC, NIP-98, WebID, did:nostr). See VC-26."]
    NF["EXTERNAL: nostr-rust-forum pins solid-pod-rs =0.5.0-alpha.7,<br/>default-features = false, features = core — an EXACT pin, behind the alpha.9 source line.<br/>See the nostr-rust-forum area."]
    DW["EXTERNAL: dreamlab-ai-website has no direct dependency; it inherits whatever<br/>the forum kit pins. See the dreamlab-ai-website area."]

    REL --> AB
    REL --> VC
    REL --> NF
    NF --> DW

    N["DOC-DRIFT: the 2026-09-05 no-version-bump note predates alpha.9.<br/>CHANGELOG.md:7-19 records the closeout release. The forum still resolves<br/>alpha.7, so local upstream fixes do not reach that consumer. Every type intended for the edge tier<br/>compiles under core. Adoption still requires publishing and resolving the<br/>new version, wiring the edge ACL/audience/replay seams and testing them;<br/>a dependency bump alone does not change caller behaviour."]
    NF -.-> N
    N2["The one source-compatibility note across alpha.8 to alpha.9 is ReplayError,<br/>which gained CapacityExhausted and is now non_exhaustive. nip98-replay is not in<br/>core, so no in-estate consumer is affected. See SP-05.5."]
    REL -.-> N2
```

## SP-09.8 Release history and what each bump changed

```mermaid
flowchart TD
    A0["0.5.0-alpha.0 — provenance and economy release<br/>../solid-pod-rs/CHANGELOG.md:340"]
    A1["0.5.0-alpha.1 — documentation only, no code change<br/>../solid-pod-rs/CHANGELOG.md:328"]
    A2["0.5.0-alpha.2 — core crate only, the siblings lagged<br/>../solid-pod-rs/CHANGELOG.md:316"]
    A3["0.5.0-alpha.3 — first whole-workspace publish<br/>../solid-pod-rs/CHANGELOG.md:292"]
    A4["0.5.0-alpha.4 — interop convergence, public API unchanged<br/>../solid-pod-rs/CHANGELOG.md:225"]
    A5["0.5.0-alpha.5 — forge Phases 0 to 3, default-off<br/>../solid-pod-rs/CHANGELOG.md:190"]
    A7["0.5.0-alpha.7 — the version the forum edge tier pins<br/>../solid-pod-rs/CHANGELOG.md:113"]
    A8["0.5.0-alpha.8 — S3 dropped, cap-std storage, atomic quota<br/>../solid-pod-rs/CHANGELOG.md:46"]
    A9["0.5.0-alpha.9 — closeout, registry re-alignment, all eight re-pinned<br/>../solid-pod-rs/CHANGELOG.md:7"]

    A0 --> A1 --> A2 --> A3 --> A4 --> A5 --> A7 --> A8 --> A9

    N["A registry-alignment bump appears twice in this line (alpha.3 and alpha.9),<br/>both times because a per-crate publish left the siblings behind the root crate.<br/>Publishing the workspace as a set is what the release job's version check<br/>(SP-09.6) now enforces."]
    A9 -.-> N
    N2["alpha.6 is absent from the changelog — the line skips from alpha.5 to alpha.7."]
    A5 -.-> N2
```

## SP-09.9 Repository governance and maintenance automation

```mermaid
flowchart LR
    CO["CODEOWNERS<br/>../solid-pod-rs/.github/CODEOWNERS:1"]
    DEP["dependabot<br/>../solid-pod-rs/.github/dependabot.yml:1"]
    DENYF["deny.toml — licences, advisories, bans, sources<br/>../solid-pod-rs/deny.toml:28"]
    TA["scripts/test-all.sh<br/>../solid-pod-rs/scripts/test-all.sh:1"]
    PC["scripts/parity-check.sh — strict parity must be >= 95%<br/>../solid-pod-rs/scripts/parity-check.sh:7"]
    SF["scripts/sync-fixtures.sh<br/>../solid-pod-rs/scripts/sync-fixtures.sh:1"]

    CO --> REVIEW["review routing"]
    DEP --> BUMP["dependency PRs"]
    DENYF --> CI["the cargo-deny job — SP-09.4"]
    TA --> LOCAL["local pre-PR loop"]
    PC --> LOCAL
    SF --> LOCAL

    N["parity-check counts rows in PARITY-CHECKLIST.md and computes strict parity:<br/>every table row starting with a number is a feature row, status is the 6th<br/>pipe-delimited field, and Shipped covers present, net-new,<br/>semantic-difference and present-by-absence.<br/>../solid-pod-rs/scripts/parity-check.sh:10"]
    PC -.-> N
    N2["DOC-DRIFT: the README claims 97.6% strict JSS parity over 230 rows tracked<br/>through JSS 0.0.220. That number is produced by this script from a checklist<br/>file, not by any executable conformance suite — it is a curated claim, and the<br/>script's own threshold is 95%."]
    PC -.-> N2
```

## SP-09.10 What CI does NOT gate

```mermaid
flowchart TD
    GATED["gated by ci-required"]
    NG1["the forge feature — the matrix runs default, oidc and all-features on the<br/>CORE crate; the workspace job runs siblings at DEFAULT features, and every<br/>sibling default is empty (SP-01.9)"]
    NG2["the git feature end-to-end — the CGI integration tests sit behind<br/>with-git-binary<br/>crates/solid-pod-rs-git/Cargo.toml:44"]
    NG3["the tls feature — no matrix row builds it<br/>crates/solid-pod-rs-server/Cargo.toml:126"]
    NG4["deployed behaviour: no job starts the binary, exercises a real cache, or<br/>crosses a restart"]
    NG5["any out-of-repo consumer's conformance — the edge ACL resolver lives in<br/>nostr-rust-forum"]

    GATED -.-> NG1
    GATED -.-> NG2
    GATED -.-> NG3
    GATED -.-> NG4
    GATED -.-> NG5

    N["DIVERGENCE (baseline, 2026-09-05): the repo's evidence is IN-PROCESS test<br/>evidence at the working tree, produced with no network access. It does not<br/>establish deployed activation, behaviour observed through a real cache or<br/>across a restart, or conformance of any out-of-repo consumer."]
    NG4 -.-> N
    N2["all-features DOES compile forge, git and tls into the core-crate matrix rows,<br/>so a compile break is caught — what is not covered is their runtime behaviour."]
    NG1 -.-> N2
```

## SP-09.11 Benches and the fuzz target — the surfaces CI never runs

```mermaid
flowchart TD
    subgraph BENCH["5 criterion benches, harness = false"]
        B1["storage_backend_bench — sequential PUT, random GET, list 10k<br/>crates/solid-pod-rs/benches/storage_backend_bench.rs:48"]
        B2["wac_eval_bench — simple, inherited, group membership<br/>crates/solid-pod-rs/benches/wac_eval_bench.rs:77"]
        B3["ldp_content_negotiation_bench — negotiate, parse, transcode<br/>crates/solid-pod-rs/benches/ldp_content_negotiation_bench.rs:33"]
        B4["nip98_verify_bench — valid and tampered tokens<br/>crates/solid-pod-rs/benches/nip98_verify_bench.rs:100"]
        B5["dpop_replay_bench — single-threaded and concurrent<br/>crates/solid-pod-rs/benches/dpop_replay_bench.rs:49"]
    end
    DECL["[[bench]] declarations<br/>crates/solid-pod-rs/Cargo.toml:342"]
    FUZZ["fuzz_target over apply_sparql_patch<br/>crates/solid-pod-rs/fuzz/fuzz_targets/sparql_update.rs:6"]
    CI["ci-required — see SP-09.2"]

    DECL --> BENCH
    BENCH -. "no CI job runs cargo bench" .-> CI
    FUZZ -. "no CI job runs cargo fuzz" .-> CI

    N["Each bench guards a hot path a diagram elsewhere describes: B1 the Storage seam<br/>(SP-06.1), B2 the WAC evaluator (SP-04.7), B3 conneg (SP-03.10), B4 the NIP-98<br/>verifier (SP-05.2), B5 the DPoP replay cache (SP-05.7). dpop_replay_bench states<br/>the contract it defends — under 1 microsecond at 10k steady-state entries<br/>(crates/solid-pod-rs/benches/dpop_replay_bench.rs:4)."]
    BENCH -.-> N
    N2["DIVERGENCE: SP-09.10 lists what CI does not gate; these are the concrete<br/>artifacts. A performance regression in any of the five, or a panic the fuzzer<br/>would find, reaches main unchallenged — the suites exist and nothing runs them."]
    CI -.-> N2
```

## SP-09.12 The SPARQL-Update fuzz contract

```mermaid
flowchart TD
    IN["fuzz_target!(|data: &[u8]|)<br/>crates/solid-pod-rs/fuzz/fuzz_targets/sparql_update.rs:6"]
    UTF{"valid UTF-8?<br/>crates/solid-pod-rs/fuzz/fuzz_targets/sparql_update.rs:11"}
    SKIP["return — not an interesting input"]
    CAP{"len > SPARQL_UPDATE_MAX_BYTES?<br/>crates/solid-pod-rs/fuzz/fuzz_targets/sparql_update.rs:16"}
    ASSERT["ASSERT the library returns Err, never panics<br/>crates/solid-pod-rs/fuzz/fuzz_targets/sparql_update.rs:20"]
    PARSE["exercise the spargebra parser below the cap<br/>crates/solid-pod-rs/fuzz/fuzz_targets/sparql_update.rs:26"]
    LIM["SPARQL_UPDATE_MAX_BYTES = 1 MiB<br/>solid-pod-rs/src/ldp.rs:1246"]

    IN --> UTF
    UTF -- no --> SKIP
    UTF -- yes --> CAP
    CAP -- yes --> ASSERT
    CAP -- no --> PARSE
    LIM --> CAP

    N["The target encodes TWO properties, not one: oversized input must be REJECTED<br/>(a positive assertion), and in-cap input must either succeed or error — never<br/>panic. A panic in a PATCH parser is reachable from an authenticated write<br/>(SP-03.5), so this is a security surface rather than a robustness nicety."]
    ASSERT -.-> N
    N2["The size cap is the first line of defence and the fuzzer deliberately spends<br/>most of its budget BELOW it<br/>(crates/solid-pod-rs/fuzz/fuzz_targets/sparql_update.rs:9) — fuzzing above the<br/>cap would only re-test the guard."]
    PARSE -.-> N2
    N3["DOC-DRIFT: docs/benchmarks.md opens 'Four criterion-based benches'<br/>(crates/solid-pod-rs/docs/benchmarks.md:3) and its Running section lists four<br/>(:11 to :14). There are FIVE bench files and five [[bench]] declarations —<br/>dpop_replay_bench (crates/solid-pod-rs/Cargo.toml:358) is in neither."]
    IN -.-> N3
```
