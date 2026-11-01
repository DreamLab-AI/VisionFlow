# Governance audit reproduction — 1 November 2026

Export agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf` into a disposable directory. Never run these fixtures against a live inbox or relay.

```bash
mkdir .audit-adapter .audit-tmp
git -C /home/devuser/workspace/project/agentbox archive 9dfc27b16e82e9786535a55e790710493dd51fdf services/dream-engine config/hooks/dream-inbox-surface.cjs tests/config/dream-inbox-surface.test.mjs | tar -x -C .audit-adapter
cp docs/estate-review/evidence/2026-11-01/dream-governance-probes.rs .audit-adapter/services/dream-engine/tests/audit_governance.rs
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --test audit_governance -- --nocapture --test-threads=1
for scope in governance relay inbox digest zone_crypto; do
  TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --lib "${scope}::" -- --test-threads=1 || exit 1
done
TMPDIR="$PWD/.audit-tmp" node .audit-adapter/tests/config/dream-inbox-surface.test.mjs
```

Eleven new Rust observation tests pass, as do 55 existing Rust tests across the five selected modules and 12 hook assertions. Rust/Cargo 1.99.1 and the existing offline registry cache were used. The first eleven-test run passed; the custom-ledger fixture was strengthened to validate its config through the actual loader and the final eleven-test run passed. Production source is unchanged.

The decision probes use two disposable signing keys created by the existing crypto library. They test optional request binding, absent local role validation, newest-before-binding selection, delegation, terminal local resolution and panel acknowledgement. These are consumer-boundary tests; no forged signature passes the production planner and no deployed relay admission gate is bypassed.

The reopen test starts with a closed and an open record sharing an ID and observes which record mark_published updates and what withdrawal the planner emits. The stale-snapshot test explicitly interleaves public load/resolve/save calls; it does not claim to observe a concurrent production race. Digest and zone probes are pure. No digest is published, and selecting Plain is not evidence that a live relay would admit plaintext. Case context uses a validated custom ledgerPath with an existing fixture file.

Two loopback transport tests exercise actual RelaySession and governance withdrawal. The server sends a prompt NIP-42 challenge, answers with scripted OK frames and responds to queries with CLOSED. It can return an invalidly signed event outside the requested filter. The withdrawal test forces inbox receipt persistence to fail by placing a directory at its fixed temporary-file path; it observes two acknowledged sends and an unmarked inbox. DREAM_RELAY points to loopback and the key FILE variable points to a fixture file. Prior environment values are restored after the test. Run serially because that one fixture changes process environment. No real secret is read.

Temporary fixtures self-clean after assertions; the source export, rendered SVGs, PNG previews and copied ignored corpus inputs are retained under the retention receipt's directory. Logs contain observation results. The build cache is disposable.

Diagram commands, with declared sibling checkout links and copies of ignored website/build-receipt.json and texput.log:

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only agentbox/23- --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
inspect-text docs/estate-review/2026-11-01-dream-governance-and-delivery.md
slop-scan docs/estate-review/2026-11-01-dream-governance-and-delivery.md --severity high
```

Initial rendering found five blank/punctuation citations; the final citations were corrected against the relevant statements, not merely moved to arbitrary nonblank lines. Four changed panels were converted using ImageMagick and visually inspected. Structural rendering does not close semantic review of the remaining panels. The manifest preserves the older topic pin and verifies selected unchanged blocks where surrounding files changed.
