# Migration and provenance reproduction — 1 November 2026

Use clean, disposable exports of dream-engine `815f963b1cbd6999e78144ce5475005ae1dd1d8e` at .audit-source and agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf` at .audit-adapter. The latter export includes services/dream-engine, lib/dream-engine.nix and docs/developer/dream-engine-extraction-ledger.md. No production source is patched.

```bash
mkdir .audit-source .audit-adapter .audit-tmp .audit-legacy
git -C /home/devuser/workspace/dream-engine archive 815f963b1cbd6999e78144ce5475005ae1dd1d8e | tar -x -C .audit-source
git -C /home/devuser/workspace/project/agentbox archive 9dfc27b16e82e9786535a55e790710493dd51fdf services/dream-engine lib/dream-engine.nix docs/developer/dream-engine-extraction-ledger.md | tar -x -C .audit-adapter
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-source/Cargo.toml --locked --offline -p dream-core --test ts_parity -- --nocapture
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-adapter-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --test night_e2e -- --nocapture --test-threads=1
```

Both suites pass: nine library compatibility tests and eleven isolated host-night tests. Execution used Rust/Cargo 1.99.1, Node 22.23.3 and the pre-existing Cargo cache. It does not establish the declared minimum compiler version or live host acceptance. Host fixtures keep fixture directories containing its transport shims and logs; they are retained before cleanup. Its empty memory endpoints fail URL construction before networking or database use. The model and forge are doubles, and transport shims accept only .invalid hostnames. Several night-level services are explicitly disabled in setup_env.

For the direct old/new comparison, export the original TypeScript validator and add the Rust example only to the disposable copy:

```bash
git -C /home/devuser/workspace/dream-machine show b77093e564ef689923fb523b0e0b18d2bb4a21a4:packages/ledger/src/rowContract.ts > .audit-legacy/rowContract.ts
mkdir -p .audit-source/crates/dream-core/examples
cp docs/estate-review/evidence/2026-11-01/dream-migration-ledger-probe.rs .audit-source/crates/dream-core/examples/migration_ledger_probe.rs
CARGO_TARGET_DIR="$PWD/.audit-target" cargo run --manifest-path .audit-source/Cargo.toml --locked --offline -p dream-core --example migration_ledger_probe < docs/estate-review/evidence/2026-11-01/dream-migration-ledger-cases.json
node --experimental-strip-types docs/estate-review/evidence/2026-11-01/dream-migration-ledger-oracle.mjs .audit-legacy/rowContract.ts < docs/estate-review/evidence/2026-11-01/dream-migration-ledger-cases.json
```

[Input cases](dream-migration-ledger-cases.json) encode all invisible characters with JSON escapes. [Rust output](dream-migration-ledger-rust.json), [TypeScript output](dream-migration-ledger-typescript.json) and [comparison](dream-migration-ledger-differential.json) retain the exact results. Two agree; the four U+FEFF/U+0085 cases differ; VETOED-prefix tightening is the expected documented divergence. These observations do not exercise executable candidate acceptance.

Package commands use the actual Cargo resolver and file selector. Run them before adding the audit example, or on another clean export, to reproduce the retained lists:

```bash
cargo metadata --manifest-path .audit-source/Cargo.toml --locked --offline --format-version 1
cargo metadata --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --format-version 1
for package in dream-cli dream-core dream-gate failure-fingerprint pairwise-gate; do
  cargo package --manifest-path .audit-source/Cargo.toml --locked --offline --list --allow-dirty -p "$package"
done
cargo package --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --list --allow-dirty -p dream-engine
```

An initial package-list attempt refused the untracked export; [initial evidence](dream-migration-package-lists-initial.json) is retained. The final invocation permits that disposable export and lists files only. It neither packages nor publishes. [Dependency receipt](dream-migration-dependency-receipt.json) records metadata and compares all 36 Rust source files in four engine crates with the corresponding tar archive members and cache files. Each archive SHA-256 matches the adapter's Cargo.lock checksum. Full metadata is retained outside Git with a hash receipt; no current registry or daemon identity is inferred from it. [Cargo-deny availability](dream-migration-deny-availability.log) records that the command is missing; no substitute licence scan is presented as cargo-deny.

Diagram gates run from the isolated VisionFlow worktree with sibling checkout links:

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only dream-engine/05 --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only dream-engine/01 --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only agentbox/23- --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
```

DE-05's two panels and AB-23.3 are converted from SVG to PNG with ImageMagick and visually reviewed. DE-01 is rerendered after its prose qualification. Rendering other AB-23 panels does not close their semantic review. An initial DE-05 gate retains the corrected prose-line limit error. [Retention receipt](dream-migration-retention.json) identifies source exports, original TypeScript oracle, metadata, fixture logs and rendered artefacts. Cargo targets are disposable caches; source and evidence are preserved before removing the worktree.
