---
id: VW-05
title: Build, test, bench and the npm publish pipeline
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
adrs: [vowl-wasm:ADR-001]
sources:
  - ../vowl-wasm/Cargo.toml
  - ../vowl-wasm/.github/workflows/ci.yml
  - ../vowl-wasm/.github/workflows/release.yml
  - ../vowl-wasm/benches/layout_bench.rs
  - ../vowl-wasm/benches/parser_bench.rs
  - ../vowl-wasm/benches/interaction_bench.rs
  - ../vowl-wasm/benches/phase3_benchmarks.rs
  - ../vowl-wasm/tools/release-inventory/src/main.rs
  - ../vowl-wasm/tools/release-inventory/README.md
  - ../vowl-wasm/docs/release-inventory.json
  - ../vowl-wasm/docs/UPSTREAMS.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
verified_commit: 89d1a0a32eec16e79a7fa507eaa6f8fcb92e4f0b
---

## VW-05.1 CI — job graph on push/PR/weekly cron
```mermaid
flowchart TB
    TRIG["push main / PR / weekly cron 06:17 Mon /<br/>workflow_dispatch — ci.yml:3-12"] --> RN["release-notices: fmt, test, clippy and doc of tools/release-inventory,<br/>no inventory verify here — ci.yml:25-52"]
    TRIG --> FMT["fmt: cargo fmt --all -- --check"]
    TRIG --> CLIPPY["clippy: --all-targets --all-features -D warnings"]
    TRIG --> TEST["test (ubuntu, macos):<br/>cargo test --all-features --all-targets<br/>+ cargo test --all-features --doc"]
    TRIG --> DOC["doc: cargo doc --no-deps --all-features<br/>RUSTDOCFLAGS=-D warnings -D broken-intra-doc-links"]
    TRIG --> DENY["cargo-deny: check licenses advisories bans sources"]
    TRIG --> MSRV["msrv (1.85): cargo check --all-features --all-targets"]
    TRIG --> FEAT["features matrix (8 configs) — VW-05.2"]
    TRIG --> W32["wasm32 build: --target wasm32-unknown-unknown<br/>--release --features ngg1,interaction"]
    TRIG --> WPACK["wasm-pack bundle — VW-05.3 build half"]
```
- `concurrency: cancel-in-progress` scoped to `${{ github.workflow }}-${{ github.ref }}` (ci.yml:14-16) — a new push on the same ref cancels an in-flight CI run.
- The `release-notices` job checks the release guard crate itself (fmt, tests, clippy, rustdoc with warnings denied) but deliberately does not run its `verify` against the reviewed sources: that runs in `release.yml`, because on `main` it could only fail between a source change and the next licence review (ci.yml:44-52).
- `cargo-deny` is scheduled weekly independent of code changes (ci.yml:11) because it checks a live advisory database — "a green build can go red without a code change" (ci.yml:10 comment).

## VW-05.2 Feature matrix — every optional feature compiles standalone
```mermaid
flowchart LR
    NONE["none<br/>--no-default-features"]
    NGG1["ngg1"]
    INT["interaction"]
    DBG["debug-serde"]
    PAR["parallel"]
    SIMD["simd"]
    BUNDLE["bundle<br/>ngg1,interaction<br/>= published npm shape"]
    ALL["all<br/>--all-features"]
    MATRIX["ci.yml job features:<br/>cargo check --all-targets $flags<br/>RUSTFLAGS=-D warnings<br/>ci.yml:131-160"]
    NONE --> MATRIX
    NGG1 --> MATRIX
    INT --> MATRIX
    DBG --> MATRIX
    PAR --> MATRIX
    SIMD --> MATRIX
    BUNDLE --> MATRIX
    ALL --> MATRIX
```
- `RUSTFLAGS: -D warnings` is set specifically in this job, separate from the `clippy` job — "an import or item that is unused in one feature configuration is invisible to an --all-features run" (ci.yml:156-157 comment). The earlier `markdown-ontology` and `estate-source` rows are gone with the feature (ci.yml:139-147).
- `Cargo.toml` ties one test to its feature with `required-features`: `ngg1_explorer_test` needs `ngg1` (Cargo.toml:109-111), so a plain `cargo test` without that flag silently skips it; the markdown and OWL 2 validator test targets were deleted with their code, and `ontomark_ngg1_migration` runs as the bundle gate instead (.github/workflows/ci.yml:233-239).
- `[lib] crate-type = ["cdylib", "rlib"]` (Cargo.toml:24) — the crate ships both as a WASM-loadable `cdylib` for `wasm-pack` and as an `rlib` for native Rust consumers (tests, benches, the `docs.rs` build with `all-features`).

## VW-05.3 Release — tag push to npm, via GitHub OIDC trusted publishing
```mermaid
sequenceDiagram
    autonumber
    participant GH as GitHub: push tag v*<br/>release.yml:3-5
    participant CI as release job<br/>release.yml:28
    participant INV as release-inventory verify<br/>release.yml:79-90
    participant WP as wasm-pack build<br/>--features ngg1,interaction --locked<br/>release.yml:94-97
    participant GATE as Rust and bundle contracts<br/>release.yml:99-105
    participant PIN as pin package metadata<br/>repository.url, publishConfig.access=public<br/>release.yml:110-121
    participant NTC as include reviewed notices<br/>release.yml:123-126
    participant PACK as npm pack → vowl-wasm-$VERSION-pkg.tgz<br/>release.yml:128-143
    participant REL as gh release create and upload<br/>tarball, sha256 and notices<br/>release.yml:145-159
    participant NPM as npm publish (OIDC, no token)<br/>release.yml:165-173
    GH->>CI: checkout ref (tag or workflow_dispatch input)
    CI->>CI: assert tag == Cargo.toml version<br/>release.yml:65-76
    CI->>INV: verify against docs/release-inventory.json,<br/>refuse an existing browser asset for this version
    INV->>WP: build the bundle
    WP->>GATE: cargo test all features, node --test ngg1_wasm.test.mjs
    GATE->>PIN: fix repository.url + sideEffects=false
    PIN->>NTC: package THIRD-PARTY-NOTICES.md and docs/UPSTREAMS.md into pkg
    NTC->>PACK: npm pack, dry run must list both notices
    PACK->>REL: attach to the GitHub release without clobber
    PACK->>NPM: npm publish "$tarball" --access public
    NPM->>NPM: skip if @dreamlab-ai/vowl-wasm@$VERSION already published<br/>release.yml:168-171
```
- INVARIANT: `id-token: write` mints a short-lived OIDC credential; no long-lived npm token is stored anywhere — the npmjs.com trusted-publisher entry names this repo + workflow file (release.yml:14-21).
- The exact tarball attached to the GitHub release is what `npm publish` ships — "the two artefacts are byte-identical" (release.yml:161-163 comment before the publish step) — so a release asset download and the npm-installed package are provably the same bytes.
- A re-run against an existing tag (`workflow_dispatch` with `tag` input, release.yml:7-12) is safe: publish is skipped, not failed, when that version is already on the registry.
- Since the earlier stamp the release also installs wasm-pack 0.15.0 from a checksum-verified release tarball and builds on rustc 1.99.0 (release.yml:38, release.yml:45-54).

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
    PROF["[profile.release]<br/>opt-level=z, lto=true,<br/>codegen-units=1, panic=abort, strip=true<br/>Cargo.toml:90-95"] --> MEASURE["measured on 0.1.1 estate bundle:<br/>wasm-opt -Oz --strip-debug"]
    MEASURE --> RAW["raw: 1,114,091 → 1,032,999 bytes (smaller)"]
    MEASURE --> GZIP["gzip: 379,414 → 395,517 bytes (LARGER)"]
    MEASURE --> BROTLI["brotli: 279,066 → 288,007 bytes (LARGER)"]
    RAW & GZIP & BROTLI --> DECISION["package.metadata.wasm-pack.profile.release<br/>wasm-opt = false — every host serves compressed,<br/>so the smaller raw module is the worse download<br/>Cargo.toml:143-144"]
```
- `strip = true` already removes the ~190 KB `name` section (Cargo.toml:133-142, the comment above `[package.metadata.wasm-pack.profile.release]`); running binaryen on top of that opt-level-z + LTO build is what produced the gzip/brotli regression above.
- The decision also notes a portability cost: validating rustc's bulk-memory output needs binaryen's `-all` flag, which "ties the build to a binaryen version" — a second reason to leave it off (Cargo.toml:139-141).

## VW-05.6 Reviewed release inventory — the guard between a source change and a browser tarball
```mermaid
flowchart TB
    INVJ["docs/release-inventory.json: version, source commit, target, features ngg1 and interaction,<br/>rustc and wasm-pack versions, a SHA-256 per source file<br/>docs/release-inventory.json:2-12"]
    VER["verify ROOT TAG RUSTC WASM_PACK refuses drift in manifest version, tag, toolchain or any hashed source<br/>tools/release-inventory/src/main.rs:72"]
    PKG["package copies full notices and source URLs into pkg and its npm files allowlist<br/>tools/release-inventory/src/main.rs:115"]
    REL["release.yml runs verify before the build and package before npm pack<br/>.github/workflows/release.yml:79-90 and .github/workflows/release.yml:123-126"]
    DRY["npm pack dry run must list THIRD-PARTY-NOTICES.md and docs/UPSTREAMS.md<br/>.github/workflows/release.yml:133-134"]
    FROZEN["an existing browser tarball for the version is never replaced, a corrected notice ships as a separate companion asset<br/>.github/workflows/release.yml:86-90"]
    INVJ --> VER
    VER --> REL
    PKG --> REL
    REL --> DRY
    REL --> FROZEN
    N1["INVARIANT: hashes are updated only after the new dependency closure and notices are reviewed, never to bypass a failed check<br/>tools/release-inventory/README.md:18-22"]
    N2["TENSION: the guard runs on tags and manual dispatch only, so on main a source edit can leave docs/release-inventory.json stale until the next release attempt fails<br/>.github/workflows/ci.yml:50-52 and .github/workflows/release.yml:79-85"]
    VER --- N1
    REL --- N2
```

**What it shows.** A private Rust tool binds the crate version, tag, features, compilers and every source file to a reviewed inventory, and the release workflow runs it before building and again when packaging, so a browser tarball cannot ship stale licence notices.
**Why it is this way.** The frozen 0.2.0 tarball carried only the project and WebVOWL licences; full third-party notices were added as a separate public asset rather than by overwriting an immutable release (docs/UPSTREAMS.md:3, tools/release-inventory/README.md:18-22).
