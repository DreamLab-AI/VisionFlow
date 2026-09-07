---
id: VW-05
title: Build, test, bench and the npm publish pipeline
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
adrs: []
sources:
  - ../vowl-wasm/Cargo.toml
  - ../vowl-wasm/.github/workflows/ci.yml
  - ../vowl-wasm/.github/workflows/release.yml
  - ../vowl-wasm/benches/layout_bench.rs
  - ../vowl-wasm/benches/parser_bench.rs
  - ../vowl-wasm/benches/interaction_bench.rs
  - ../vowl-wasm/benches/phase3_benchmarks.rs
  - ../vowl-wasm/deny.toml
verified_commit: 65e2d1e78
---

## VW-05.1 CI — job graph on push/PR/weekly cron
```mermaid
flowchart TB
    TRIG["push main / PR / weekly cron 06:17 Mon /<br/>workflow_dispatch — ci.yml:3-13"] --> FMT["fmt: cargo fmt --all -- --check"]
    TRIG --> CLIPPY["clippy: --all-targets --all-features -D warnings"]
    TRIG --> TEST["test (ubuntu, macos):<br/>cargo test --all-features --all-targets<br/>+ cargo test --all-features --doc"]
    TRIG --> DOC["doc: cargo doc --no-deps --all-features<br/>RUSTDOCFLAGS=-D warnings -D broken-intra-doc-links"]
    TRIG --> DENY["cargo-deny: check licenses advisories bans sources"]
    TRIG --> MSRV["msrv (1.85): cargo check --all-features --all-targets"]
    TRIG --> FEAT["features matrix (10 configs) — VW-05.2"]
    TRIG --> W32["wasm32 build: --target wasm32-unknown-unknown<br/>--release --features ngg1,interaction"]
    TRIG --> WPACK["wasm-pack bundle — VW-05.3 build half"]
```
- `concurrency: cancel-in-progress` scoped to `${{ github.workflow }}-${{ github.ref }}` (ci.yml:15-17) — a new push on the same ref cancels an in-flight CI run.
- `cargo-deny` is scheduled weekly independent of code changes (ci.yml:11) because it checks a live advisory database — "a green build can go red without a code change" (ci.yml:10 comment).

## VW-05.2 Feature matrix — every optional feature compiles standalone
```mermaid
flowchart LR
    NONE["none<br/>--no-default-features"]
    NGG1["ngg1"]
    MD["markdown-ontology"]
    INT["interaction"]
    DBG["debug-serde"]
    PAR["parallel"]
    SIMD["simd"]
    BUNDLE["bundle<br/>ngg1,interaction<br/>= published npm shape"]
    ESTATE["estate-source<br/>ngg1,markdown-ontology,interaction"]
    ALL["all<br/>--all-features"]
    MATRIX["ci.yml job features:<br/>cargo check --all-targets $flags<br/>RUSTFLAGS=-D warnings"]
    NONE --> MATRIX
    NGG1 --> MATRIX
    MD --> MATRIX
    INT --> MATRIX
    DBG --> MATRIX
    PAR --> MATRIX
    SIMD --> MATRIX
    BUNDLE --> MATRIX
    ESTATE --> MATRIX
    ALL --> MATRIX
```
- `RUSTFLAGS: -D warnings` is set specifically in this job, separate from the `clippy` job — "an import or item that is unused in one feature configuration is invisible to an --all-features run" (ci.yml features job comment).
- `Cargo.toml` ties tests to their features with `required-features`: `markdown_parser_test`/`owl2_validation_test` need `markdown-ontology`, `ngg1_explorer_test` needs `ngg1` (Cargo.toml `[[test]]` blocks) — a plain `cargo test` without those flags silently skips them.

## VW-05.3 Release — tag push to npm, via GitHub OIDC trusted publishing
```mermaid
sequenceDiagram
    autonumber
    participant GH as GitHub: push tag v*<br/>release.yml:3-5
    participant CI as release job<br/>release.yml:28
    participant WP as wasm-pack build<br/>--features ngg1,interaction<br/>release.yml:75-80
    participant PIN as pin package metadata<br/>repository.url, publishConfig.access=public<br/>release.yml:83-93
    participant PACK as npm pack → vowl-wasm-$VERSION-pkg.tgz<br/>release.yml:96-106
    participant REL as attach tarball + sha256<br/>to GitHub release<br/>release.yml:108-115
    participant NPM as npm publish (OIDC, no token)<br/>release.yml:121-129
    GH->>CI: checkout ref (tag or workflow_dispatch input)
    CI->>CI: assert tag == Cargo.toml version<br/>release.yml:58-69
    CI->>WP: build the bundle
    WP->>PIN: fix repository.url + sideEffects=false
    PIN->>PACK: npm pack
    PACK->>REL: softprops/action-gh-release
    PACK->>NPM: npm publish "$tarball" --access public
    NPM->>NPM: skip if @dreamlab-ai/vowl-wasm@$VERSION already published<br/>release.yml:124-127
```
- INVARIANT: `id-token: write` mints a short-lived OIDC credential; no long-lived npm token is stored anywhere — the npmjs.com trusted-publisher entry names this repo + workflow file (release.yml:14-21).
- The exact tarball attached to the GitHub release is what `npm publish` ships — "the two artefacts are byte-identical" (release.yml:117-119 comment before the publish step) — so a release asset download and the npm-installed package are provably the same bytes.
- A re-run against an existing tag (`workflow_dispatch` with `tag` input, release.yml:6-10) is safe: publish is skipped, not failed, when that version is already on the registry.

## VW-05.4 Benches — what each measures
```mermaid
flowchart TB
    LB["layout_bench.rs<br/>ForceSimulation over synthetic<br/>class/property ontologies, varying n"]
    PB["parser_bench.rs<br/>StandardParser::parse over<br/>generated JSON, varying class/property count"]
    IB["interaction_bench.rs<br/>ray_sphere_intersection +<br/>find_closest_node_hit (feature interaction)"]
    P3["phase3_benchmarks.rs<br/>tick() <10ms target, checkNodeClick <1ms/1000 nodes<br/>bundle size <1.5MB — targets stated in the file header"]
    LB & PB & IB & P3 --> HARNESS["criterion, harness=false<br/>Cargo.toml [[bench]] entries"]
    IB -.-> REQ["required-features = interaction<br/>Cargo.toml"]
    P3 -.-> REQ
```
- `phase3_benchmarks.rs` is the only bench file that states explicit numeric targets in its own header comment (benches/phase3_benchmarks.rs:1-6): tick <10ms (8ms target), `getGraphData()` serialisation <5ms, `checkNodeClick` <1ms for 1,000 nodes, bundle <1.5MB — these are aspirational annotations in the source, not enforced by any CI assertion (contrast with the bundle-symbol check in VW-04.5, which IS enforced).

## VW-05.5 Release profile — why `wasm-opt` stays off
```mermaid
flowchart LR
    PROF["[profile.release]<br/>opt-level=z, lto=true,<br/>codegen-units=1, panic=abort, strip=true<br/>Cargo.toml"] --> MEASURE["measured on 0.1.1 estate bundle:<br/>wasm-opt -Oz --strip-debug"]
    MEASURE --> RAW["raw: 1,114,091 → 1,032,999 bytes (smaller)"]
    MEASURE --> GZIP["gzip: 379,414 → 395,517 bytes (LARGER)"]
    MEASURE --> BROTLI["brotli: 279,066 → 288,007 bytes (LARGER)"]
    RAW & GZIP & BROTLI --> DECISION["package.metadata.wasm-pack.profile.release<br/>wasm-opt = false — every host serves compressed,<br/>so the smaller raw module is the worse download<br/>Cargo.toml"]
```
- `strip = true` already removes the ~190 KB `name` section (Cargo.toml comment above `[package.metadata.wasm-pack.profile.release]`); running binaryen on top of that opt-level-z + LTO build is what produced the gzip/brotli regression above.
- The decision also notes a portability cost: validating rustc's bulk-memory output needs binaryen's `-all` flag, which "ties the build to a binaryen version" — a second reason to leave it off (Cargo.toml comment).

## VW-05.6 Dependency posture — `cargo-deny` gate
```mermaid
classDiagram
    class RuntimeDeps {
        wasm-bindgen 0.2
        serde 1.0 + derive
        serde-wasm-bindgen 0.6
        js-sys / web-sys 0.3
        thiserror 1.0
        petgraph 0.6
        nalgebra 0.32
        regex 1.10 optional dep:regex
        rayon 1 optional dep:rayon
    }
    class DevDeps {
        wasm-bindgen-test 0.3
        mockall 0.12
        pretty_assertions 1.4
        criterion 0.5
    }
    class DenyGate {
        licenses
        advisories
        bans
        sources
        deny.toml + ci.yml job deny
    }
    RuntimeDeps --> DenyGate
    DevDeps --> DenyGate
```
- `[lib] crate-type = ["cdylib", "rlib"]` (Cargo.toml) — the crate ships both as a WASM-loadable `cdylib` for `wasm-pack` and as an `rlib` for native Rust consumers (tests, benches, the `docs.rs` build with `all-features`).
