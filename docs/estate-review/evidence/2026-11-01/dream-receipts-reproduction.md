# Annexe and receipt reproduction — 1 November 2026

Export agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf` and append the audit tests to its existing night fixture. Do not replace the upstream fixture or run these commands from a live service directory.

```bash
mkdir .audit-adapter .audit-tmp
git -C /home/devuser/workspace/project/agentbox archive 9dfc27b16e82e9786535a55e790710493dd51fdf services/dream-engine | tar -x -C .audit-adapter
cat docs/estate-review/evidence/2026-11-01/dream-receipts-probes.rs >> .audit-adapter/services/dream-engine/tests/night_e2e.rs
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --test night_e2e audit_receipts -- --nocapture --test-threads=1
```

Five tests pass with eleven existing tests filtered out, using Rust/Cargo 1.99.1 and the pre-existing registry cache. Production source is unmodified. Serial execution is required because the existing fixture sets process-wide environment once. It puts fake SSH/SCP executables first on PATH; those refuse hosts not ending in .invalid, execute the supplied shell command locally, and record calls. No real SSH or SCP process is reached by the probes.

The runner test invokes both LocalRunner and SshRunner, compares a failing producer with a successful tail, a two-failure pipeline, handled/hidden failure, missing cwd, exit 255 and immediate exit 124. It does not wait out a real timeout or test a broken network connection. It observes the same production outcome classifier used by receipts.

The overlay test builds two local Git repositories, validates the absolute sibling declaration through DreamConfig::from_json, and uses the actual archive/copy/extraction functions. Both repositories map to the same annexe leaf. Assertions check replacement of source plus survival of a target-only evaluator script. The HEAD test captures baseline A, commits B, dispatches and reads B's files from the local annexe substitute. It does not manufacture a full accepted night with mismatched inputs or claim to observe a live race.

The no-patch test runs one full Engine::run_night with a scripted ACCEPT report, recording forge and local evaluators. It asserts INCONCLUSIVE, exactly one Unproven veto, an operator inbox alert, and no forge request. Night-level journal delivery, sweep, digest, governance, forum suggestions and source reading are disabled; empty memory endpoints fail before networking/database use. No supervisor, real night directory or forum is touched.

The selector test calls the public finding selector on a same-date quoted table example and on a long emoji hypothesis. The latter result breaks its own finding contract before the adapter's repair step. This is not evidence that the adapter appends that invalid value unchanged.

An initial five-test run passed. Additional positive-tail and exact-veto assertions were added before the final passing run; both logs are retained. The tests assert observed behaviour, including defects, rather than desired acceptance behaviour.

Diagram gates run with declared sibling checkout links and read-only copies of the primary checkout's ignored website/build-receipt.json and texput.log:

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only agentbox/23- --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
inspect-text docs/estate-review/2026-11-01-dream-annexe-and-receipts.md
slop-scan docs/estate-review/2026-11-01-dream-annexe-and-receipts.md --severity high
```

An initial render exposed five punctuation/blank-line citations; they were corrected and the topic rerendered with zero warnings. Four changed panels were converted from SVG to PNG and visually inspected. Structural rendering of the other panels does not close their semantic review. The retention receipt identifies source exports, kept fixture files and rendered previews; build targets are disposable caches.
