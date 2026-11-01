# Execution audit reproduction — 1 November 2026

Use disposable exports of agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf` and dream-engine `815f963b1cbd6999e78144ce5475005ae1dd1d8e`. Production source is unchanged. Never direct these fixtures at an actual annexe or credential store.

```bash
mkdir .audit-adapter .audit-core .audit-tmp
git -C /home/devuser/workspace/project/agentbox archive 9dfc27b16e82e9786535a55e790710493dd51fdf services/dream-engine | tar -x -C .audit-adapter
git -C /home/devuser/workspace/dream-engine archive 815f963b1cbd6999e78144ce5475005ae1dd1d8e | tar -x -C .audit-core
cp docs/estate-review/evidence/2026-11-01/dream-execution-probes.rs .audit-adapter/services/dream-engine/tests/audit_execution.rs
TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --test audit_execution -- --nocapture --test-threads=1
for scope in jev_triage dispatch candidate_flow prompt min_free_tests; do
  TMPDIR="$PWD/.audit-tmp" CARGO_TARGET_DIR="$PWD/.audit-target" cargo test --manifest-path .audit-adapter/services/dream-engine/Cargo.toml --locked --offline --lib "${scope}::" -- --test-threads=1 || exit 1
done
```

Run serially: the SSH fixtures temporarily prepend a private executable directory to PATH and restore it on drop. The substitute SSH runs only the production probe's stdin script locally. The health fixture supplies a df executable; the failed-cleanup fixture supplies a failing find. Both stay under tempfile directories. The age-based cleanup fixture extracts the exact template from the archived engine.rs, sets only its fixture directory's mtime, executes it locally and asserts deletion. It does not run the nightly engine, SSH to a host, or delete a production path.

Triage uses an injected scripted transport, never HTTP. It verifies a failed annotation write still returns defer, initial fallback preserves an older deferred file, sixteen disciplines sent versus twenty recorded, missing delete/rename path hints, an unbounded catalogue field, and no requests when disabled. Existing tests include one loopback redirect test.

The two repair tests initialise temporary Git repositories with obvious non-secret canaries. A protected-path check succeeds for each fixture patch; git then rejects its deliberately wrong context. The actual adapter candidate flow invokes a capturing Llm implementation and returns DidNotApply after the stub declines repair. Assertions check the prompt contains the withheld marker, the fixture checkout is clean and only its primary worktree remains. No valid key material is generated or read, and no model endpoint is contacted.

The first ten-test run passed. After inspection found the repair-withholding gap, two candidate-flow probes were added; the final twelve-test run passed. Fifty existing checks passed. Rust/Cargo 1.99.1 and the offline cache were used; this is not MSRV, power-loss, live-host or model-accuracy evidence. The consumed-source receipt compares eight core/gate files with the local cache; missing cache checksum manifests prevent a package-authenticity claim.

With declared sibling checkout links and copies of the two existing ignored corpus inputs (website/build-receipt.json and texput.log):

```bash
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --only agentbox/23- --check --render --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --strict-citations
VISIONFLOW_JEV=off node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --check-verification
inspect-text docs/estate-review/2026-11-01-dream-execution-and-triage.md
slop-scan docs/estate-review/2026-11-01-dream-execution-and-triage.md --severity high
```

Initial rendering identified four punctuation citations and a sequence-label syntax error. Those were corrected against the statements, followed by manual citation inspection. The class view's overflowing fields were replaced with a legible type map; other long labels were shortened. Six changed panels were rendered to SVG, converted to PNG with ImageMagick and visually inspected. Initial prose scanning flagged four repeated article-led openings, which were edited. These checks do not imply whole-estate semantic completion.

Source exports, rendered SVGs, previews and ignored corpus inputs are retained under the directory named in dream-execution-retention.json and hashed before worktree removal. The Cargo target is disposable. Temporary fixture worktrees self-clean; no production source checkout was changed.
