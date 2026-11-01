# Forge and seam reproduction — 1 November 2026

Use a disposable directory. Export dream-engine commit `815f963b1cbd6999e78144ce5475005ae1dd1d8e`, then copy [the probe file](dream-forge-seam-probes.rs) and [offline gh source](dream-forge-fake-gh.rs) as shown:

```bash
mkdir .audit-source .audit-tmp .audit-bin
git -C /home/devuser/workspace/dream-engine archive 815f963b1cbd6999e78144ce5475005ae1dd1d8e | tar -x -C .audit-source
cp docs/estate-review/evidence/2026-11-01/dream-forge-seam-probes.rs .audit-source/crates/dream-core/tests/estate_forge_seams.rs
rustc --edition 2021 docs/estate-review/evidence/2026-11-01/dream-forge-fake-gh.rs -o .audit-bin/gh
DREAM_AUDIT_GH_BIN="$PWD/.audit-bin" TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-source/Cargo.toml --locked --offline -p dream-core --test estate_forge_seams -- --nocapture --test-threads=1
for module in seams automerge persist runner; do
  TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-source/Cargo.toml --locked --offline -p dream-core --lib "${module}::tests" -- --nocapture
done
```

Execution used Rust/Cargo 1.99.1 on Linux and the existing Cargo cache. Offline success is not a dependency integrity or minimum-version certification. [Final probe output](dream-forge-seam-probes.log) records 13 passes; existing module logs record 48, for 61 total. [Initial output](dream-forge-seam-probes-initial.log) retains a compilation error from spelling TimedOut as Timeout; the committed probe source corrects it. No production source was patched.

Each fixture makes a Git repository and bare origin below TMPDIR, disables its hooks, and sets a state directory with the marker offline-dream-audit. The replacement gh refuses calls without this marker and handles only enumerated fixture commands. Fixtures put it first on PATH and restore environment variables on drop; serial test execution is required. Local Git pushes are real. Remote GitHub effects are simulated responses and recorded argv only. No credentials, model endpoint, real forge or notification service is used.

A relocated-crate fixture tracks a valid package named dream-gate in components/dream-gate. The empty-URL fixture deliberately returns success with no stdout. The timeout fixture runs a real sleep command under LocalRunner's timeout. A rejected verdict tests the helper's publication precondition, not a bypass of the adapter's acceptance guard. Temporary fixture directories are removed by their owners.

Diagram commands run from the isolated VisionFlow worktree with sibling checkout links:

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only dream-engine/04 --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only agentbox/23- --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
```

DE-04's three panels and AB-23.6 were converted from SVG with ImageMagick and visually inspected. All 22 AB-23 panels render, but only panel 6 receives a new semantic assessment here. An [initial adapter log](dream-forge-adapter-gate-initial.log) retains a punctuation-only source anchor, subsequently changed to the operative line. Structural regeneration covers the full root corpus; semantic coverage remains bounded by the report.

[Input manifest](dream-forge-source-manifest.json) binds source and decisions to revisions. [Retention receipt](dream-forge-retention.json) lists the clean export with its added test, offline executable, four previews, rendered files and copied ignored citation artefacts. Retained files are hash-verified before removing the worktree. Cargo build output and empty fixture directories are disposable.
