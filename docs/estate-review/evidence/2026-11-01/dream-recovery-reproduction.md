# Recovery and source-supply reproduction — 1 November 2026

Use a disposable directory. Export dream-engine commit `815f963b1cbd6999e78144ce5475005ae1dd1d8e`, then copy [the probe file](dream-recovery-source-probes.rs) to `.audit-source/crates/dream-core/tests/estate_recovery_source.rs`:

```bash
mkdir .audit-source .audit-tmp
git -C /home/devuser/workspace/dream-engine archive 815f963b1cbd6999e78144ce5475005ae1dd1d8e | tar -x -C .audit-source
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-source/Cargo.toml --locked --offline -p dream-core --test estate_recovery_source -- --nocapture
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-source/Cargo.toml --locked --offline -p dream-core runstate::tests -- --nocapture
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-source/Cargo.toml --locked --offline -p dream-core source::tests -- --nocapture
```

Tests used Rust/Cargo 1.99.1 with rustfmt 1.10.0-stable on Linux. Offline execution depends on the pre-existing Cargo cache; it does not certify downloaded package integrity or minimum-version support. [Probe output](dream-recovery-source-probes.log) records ten passing tests; [journal output](dream-recovery-existing-tests.log) records 24 existing tests and [source output](dream-source-existing-tests.log) records 29. Other test binaries report filtered tests, not additional executed coverage. The unchanged-source 555-test baseline is linked in the preceding report.

Source fixtures use a local Git repository, fixed in-process planner responses and identity redaction. The Unicode test explicitly enables Git's default quoting behaviour. No network or live provider is called. Process-death tests re-execute only their own test function in a child, wait for its real journal write, verify the lock is held, kill and reap it, then test restart classification. The two child cases cover ModelCalled and Persisted. They do not exercise a real forge, machine failure, storage durability or an interruption inside reset. Fixture files are under the disposable temporary directory and removed by their owners.

Diagram commands, from the isolated VisionFlow worktree with sibling checkout links:

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only dream-engine/03 --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only agentbox/23- --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
```

[DE-03 gate](dream-recovery-diagram-gate.log) and [adapter gate](dream-adapter-recovery-gate.log) pass. DE-03's three panels and the two changed AB-23 panels were converted from SVG to PNG with ImageMagick and visually inspected. Rendering all 22 AB-23 panels is a syntax/size check; this pass semantically assesses only panels 7 and 21. Initial logs retain the corrected paragraph-count and citation-anchor errors: [DE-03](dream-recovery-diagram-gate-initial.log), [AB-23](dream-adapter-recovery-gate-initial.log). An unavailable optional CairoSVG module was replaced by the installed ImageMagick renderer; no dependency was installed.

[Corpus regeneration](dream-recovery-corpus-regeneration.log) and [final check](dream-recovery-final-corpus-check.log) record structural coverage. [Input manifest](dream-recovery-source-manifest.json) binds reviewed source to commit and retained diagram pins. [Retention receipt](dream-recovery-retention.json) lists the exported source with its added test, five previews, all rendered files and the two copied ignored citation artefacts. Cargo build output is disposable; retained files are hash-verified before worktree removal. No production implementation is changed.
