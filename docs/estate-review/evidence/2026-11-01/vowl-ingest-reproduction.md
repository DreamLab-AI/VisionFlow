# Reproducing the vowl ingest and release probes

Source: vowl-wasm `28af4365a68d9aacfd7c04f63299cc96ad717ce3`.
Run in a disposable source archive on an executable filesystem, with separate Cargo target directories. Do not reuse the production checkout's target or ignored `pkg/`. No production source changes are required.

1. Export the revision with `git archive` and extract it into a new directory. Copy `vowl-ingest-probes.rs` beside the existing tests as `tests/estate_ingest_probe.rs`.
2. Run `cargo test --manifest-path <archive>/Cargo.toml --locked --offline --test estate_ingest_probe` with an absolute `CARGO_TARGET_DIR` reserved for this archive. The recorded run passes 11 tests. Assertions establish present defects, so a corrected implementation may intentionally fail them until expectations change.
3. Before adding guard probes, run `cargo test --manifest-path <archive>/tools/release-inventory/Cargo.toml --locked --offline` with a second isolated target directory. Six existing tests pass.
4. Insert the contents of `vowl-release-guard-probes.rs` immediately before the final closing brace of the existing test module in `tools/release-inventory/src/main.rs`. Rerun the same command: nine tests pass, including three new audit cases.
5. Run `python3 vowl-release-flow-probes.py <archive>`. It extracts exact workflow shell blocks and uses local shell functions in place of cargo, gh and npm. Node executes the workflow's JSON handling. Assertions cover attachment followed by publish failure/retry, and an existing registry version with different fixture bytes. No service mutation occurs.

Executed source, rendered diagrams and previews are retained under the destination in `vowl-ingest-retention.json`, with per-file SHA-256 comparisons. Cargo target directories are disposable caches and are excluded. The source archive contains the added audit tests; the production source was read-only.

`vowl-inventory-current.json` records two invocations of the built inventory tool against the production source: recorded rustc/wasm-pack strings pass; actual local rustc 1.99.1 fails the reviewed 1.99.0 comparison. Passing supplied strings verifies source/notice agreement, not the local compiler identity.

External observations are time-dependent read-only results, not fixtures that promise future registry state. The registry was `https://registry.npmjs.org/`. Queries were `npm view @dreamlab-ai/vowl-wasm@0.2.2 version dist.integrity dist.tarball --json` and `npm view @dreamlab-ai/vowl-wasm versions dist-tags --json`. GitHub release, action-run and job-log metadata were read through the GitHub connector; only selected metadata and the publication error excerpt are retained. Asset digests are GitHub-reported, not freshly recomputed from a downloaded package.

An initial diagram run failed on a blank source anchor. A visual revision then exposed another blank anchor and a Mermaid semicolon parse error. Both failures are retained in `vowl-ingest-diagram-gate-initial.log` and `vowl-ingest-diagram-visual-initial.log`; the final gate passes after correction. Visual checks also found overflowing class members and overlapping transition labels that the structural gate alone did not detect.
