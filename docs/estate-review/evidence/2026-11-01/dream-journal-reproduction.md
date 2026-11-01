# Journal audit reproduction — 1 November 2026

Use a clean export of agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf`; do not execute activation scripts against the live workspace. The commands below assume the audit evidence is in the current VisionFlow checkout.

```bash
mkdir .audit-adapter .audit-tmp
git -C /home/devuser/workspace/project/agentbox archive 9dfc27b16e82e9786535a55e790710493dd51fdf services/dream-engine management-api scripts/activation/adr-2071-api-down-night.sh scripts/activation/adr-2087-check.sh tests/config/adr-2071-api-down-night.test.sh | tar -x -C .audit-adapter
cp docs/estate-review/evidence/2026-11-01/dream-journal-probes.rs .audit-adapter/services/dream-engine/tests/audit_journal.rs
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --test audit_journal -- --nocapture --test-threads=1
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --lib journal:: -- --nocapture
TMPDIR="$PWD/.audit-tmp" bash .audit-adapter/tests/config/adr-2071-api-down-night.test.sh
ln -s /home/devuser/workspace/project/agentbox/management-api/node_modules .audit-adapter/management-api/node_modules
TMPDIR="$PWD/.audit-tmp" node docs/estate-review/evidence/2026-11-01/dream-journal-route-probes.cjs .audit-adapter
```

The pre-existing Rust registry cache and management-api node_modules were used; no dependencies were installed or updated. Rust/Cargo is 1.99.1. The Node probe uses actual exported route/journal/events modules, Fastify injection and a temporary directory; it neither starts the full API nor invokes its authentication stack. Its concurrency assertion calls the journal's public append directly with the actual disk adapter. Four assertions pass: changed-payload deduplication, unhydrated reset, concurrent sequence duplication and hash-chain verification. Event files are retained, not deleted by the probe.

The ten Rust tests use five loopback/session cases, four temporary tick scenarios and the exact C3 block extracted from the checker. The C3 block is run with a fixture record function, temporary files and no live supervisor. All tick calls receive env_clear, an explicit temporary WORKSPACE, a local supervisor substitute and a pinned clock. The full checker is never run. The original nine-test run passed; a tenth C3 case was then added and the final ten-test suite passed. Existing six journal tests and 26 tick assertions pass. These are observation tests, including defects, not remediation acceptance tests.

The temporary Rust fixtures are removed by tempfile after their assertions. Their returned state and status are captured in the test log. Node JSONL evidence, source export and rendered images are retained under the path in the retention receipt. The dependency symlink and compiled target cache are not part of the retained export.

Diagram validation uses the declared sibling checkout links and read-only copies of ignored website/build-receipt.json and texput.log:

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only agentbox/23- --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
inspect-text docs/estate-review/2026-11-01-dream-journal-and-api-down.md
slop-scan docs/estate-review/2026-11-01-dream-journal-and-api-down.md --severity high
```

The first render found a blank-line citation, which was corrected; the final topic render has zero warnings. Both changed panels were visually inspected after conversion to PNG. Whole-corpus validation checks structure and pins; it does not certify unreviewed diagram semantics or current deployment.
