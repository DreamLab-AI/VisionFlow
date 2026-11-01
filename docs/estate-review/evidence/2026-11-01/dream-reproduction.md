# Dream-engine audit reproduction — 1 November 2026

Source revision: `815f963b1cbd6999e78144ce5475005ae1dd1d8e`. Tests ran on Linux with `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, Cargo 1.99.1 and rustfmt 1.10.0-stable. This does not certify the declared minimum supported Rust version.

From a disposable audit directory with the engine checkout accessible, export the pinned source and run the unchanged suite first:

```bash
mkdir .audit-source .audit-tmp
git -C /home/devuser/workspace/dream-engine archive 815f963b1cbd6999e78144ce5475005ae1dd1d8e | tar -x -C .audit-source
CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-source/Cargo.toml --workspace --locked --offline
```

[Baseline output](dream-baseline.log) records 555 passing tests, zero failures and zero ignored tests. Offline execution requires the dependencies already in Cargo's cache. It does not download or authenticate packages.

Copy [the audit test source](dream-contract-probes.rs) to `.audit-source/crates/dream-core/tests/estate_contract_probe.rs`, then run:

```bash
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-source/Cargo.toml --locked --offline -p dream-core --test estate_contract_probe -- --nocapture
```

[Final probe output](dream-contract-probes.log) records 11 passing assertions about observed defects and caller obligations. [Earlier output](dream-contract-probes-first.log) records the first eight probes. Neither result means the defects have been repaired. Real Git and rustfmt binaries must be on PATH. Temporary fixtures and their candidate worktrees are removed by the tests, which also check that only the fixture's primary worktree remains.

Formatter traversal targets a harmless Rust canary in a disposable directory outside the fixture candidate worktree. No production file is targeted. A separate evaluator intentionally modifies its fixture's tracked file. Direct gate probes construct evidence explicitly; they do not run full host admission or demonstrate a forged production receipt. The JUnit probe does exercise the actual parser before the gate. Its top-level receipt is supplied as passing, so the finding does not bypass a non-zero evaluator exit.

[Source comparison](dream-consumed-source-comparison.json) records eight byte-identical files in the checkout and local registry cache. Cache checksum manifests are absent. [Source manifest](dream-source-manifest.json) records declared diagram inputs, revision comparisons and ADR headings. It is not a proof of the daemon's loaded binary.

Diagram checks used sibling checkout links in the isolated VisionFlow worktree:

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only dream-engine/ --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
```

[Final focused check](dream-diagram-gate.log) verifies 41 citations and renders six panels. All six PNG previews were visually inspected. [Initial gate output](dream-diagram-gate-initial.log) includes an overly broad topic selector, paragraph-count and anchor errors; [initial visual-edit output](dream-diagram-visual-initial.log) records accidentally shortened front-matter source paths. Those audit-document mistakes were corrected before the final successful check. Rendering AB-23 during that initial selection does not constitute semantic assessment of its 22 panels.

[Corpus regeneration](dream-corpus-regeneration.log) and [final corpus check](dream-final-corpus-check.log) record structural coverage. They do not substitute for semantic review of every panel. [Retention receipt](dream-retention.json) lists hashes and sizes of the source export, previews, rendered diagrams and copied ignored artefacts preserved outside the task worktree. The Cargo target cache is disposable and is not retained. Production repositories were not modified by these experiments.
