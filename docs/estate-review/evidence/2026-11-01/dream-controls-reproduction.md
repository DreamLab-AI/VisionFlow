# Adapter controls reproduction — 1 November 2026

Use a disposable export of agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf`. Include management-api/package.json: without its CommonJS context, Node inherits VisionFlow's module type and the discovery comparison cannot load its exports correctly.

```bash
mkdir .audit-adapter .audit-tmp
git -C /home/devuser/workspace/project/agentbox archive 9dfc27b16e82e9786535a55e790710493dd51fdf services/dream-engine management-api/lib/dream-ledger.js management-api/package.json | tar -x -C .audit-adapter
cp docs/estate-review/evidence/2026-11-01/dream-controls-probe.rs .audit-adapter/services/dream-engine/tests/audit_controls.rs
cat docs/estate-review/evidence/2026-11-01/dream-controls-night-probe.rs >> .audit-adapter/services/dream-engine/tests/night_e2e.rs
AUDIT_LEDGER_JS="$PWD/.audit-adapter/management-api/lib/dream-ledger.js" TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --test audit_controls --test night_e2e audit_controls -- --nocapture --test-threads=1
```

Three tests pass, with eleven existing host tests filtered out. Rust/Cargo 1.99.1 and Node 22.23.3 use the pre-existing dependency cache. Production code is unmodified. Tests are deliberately serial because the upstream fixture sets process-wide environment. Run them in a disposable export, not a live service process.

The configuration probe replaces relevant provider/key variables with synthetic fixture settings and never calls a provider. A fake scp records arguments and exits without transfer. The discovery probe invokes the unchanged JavaScript module on a temporary link. Night fixtures substitute model, forge and annexe transport, disable night-level integrations, and configure empty memory endpoints. No live credentials are printed, no supervisor is stopped and no forum/forge event is published.

The completed-night case asserts unchanged run-state bytes, an extra HANDOFF ledger row, replacement of report.md, and exactly one model/forge call across both invocations. The pause case runs only a targeted dry-run after confirming the paused roster returns None. The discovery test compares Rust and JavaScript on the same directory tree.

Initial logs retain an incorrect probe API assumption and the missing CommonJS package context; these were fixture errors, corrected before the final result. An expanded report-preservation assertion was then added and the focused tests rerun. No failing observation is presented as a passing acceptance test.

From the isolated VisionFlow worktree, with declared sibling checkout links and read-only copies of the primary checkout's ignored website/build-receipt.json and texput.log:

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only agentbox/23- --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
inspect-text docs/estate-review/2026-11-01-dream-adapter-controls-and-egress.md
slop-scan docs/estate-review/2026-11-01-dream-adapter-controls-and-egress.md --severity high
```

Four changed panels were converted from SVG to PNG with ImageMagick and visually reviewed. The first render exposed citation and width errors, retained in the initial log. An initial corpus pass also lacked those two ignored artefacts; its failure is retained and the complete pass was rerun after copying them. Citation labels were shortened after visual review to prevent long paths overflowing boxes; the final topic gate reports zero warnings and renders all 22 panels. Rendering the other eighteen panels does not mean all were semantically reviewed in this pass.

The retention receipt identifies the isolated source, fixture files and final previews/rendered diagrams. Cargo targets are disposable build caches. The source manifest binds current inputs and confirms unchanged selected diagram inputs at their retained pins. The preceding migration receipt established that the consumed core Rust sources match the pinned extracted repository.
