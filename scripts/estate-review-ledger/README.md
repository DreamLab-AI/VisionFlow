# Estate review ledger

This private Rust CLI inventories an explicit repository scope and reconciles manually reviewed evidence. It reads source repositories and writes one JSON output. It does not run product tests, modify an ADR, infer coverage from a mention, or declare the estate audit complete.

Run from the VisionFlow checkout, replacing the first two paths for another workspace:

```sh
cargo run --locked --offline --manifest-path scripts/estate-review-ledger/Cargo.toml -- \
  /home/devuser/workspace /home/devuser/workspace/VisionFlow \
  docs/estate-review/completion/scope.json \
  docs/estate-review/completion/claims.json \
  /tmp/estate-review-snapshot.json
```

The first argument is the workspace root. The second selects the VisionFlow checkout, including an isolated worktree. Other checkouts come only from `scope.json`. The final argument is the output path. Its parent directory must already exist; output is replaced through a temporary file after a successful scan.

## What the ledger records

- Git-tracked and nonignored untracked Markdown candidates in `adr`/`adrs` directories, plus filenames containing an ADR token or compact identifier such as `ADR071`. “Quadratic” and “headroom” do not match. Files with other naming conventions still require discovery through references.
- The declared checkout HEAD and dirty flag, document hash, parsed heading lines and titles. Fenced examples and YAML front matter are not sections. Setext headings and duplicate headings are retained separately by ordinal.
- Existing VisionFlow diagram topics and their Mermaid panels. The parser uses the preceding heading as panel identity. Ancillary README diagrams and excluded archive/render directories are not catalogue topics.
- Explicit review claims whose document and report hashes still match. A whole-document claim records the report against its heading units. A panel claim covers only named panels; other panels retain their own state.

Missing checkouts stay in the scope output. Nested Git owners are excluded from parent inventories, and duplicate canonical checkout roots are rejected. Invalid relative paths, unreadable candidates, changed HEAD/status during a scan, absent reports, unknown panels and unmatched claims fail the run instead of silently dropping work. A changed document or report produces a stale-evidence record and receives no current coverage credit.

The scan is an observation, not an atomic estate snapshot. The before/after HEAD and status check does not detect every content race in an already-dirty file. Per-file hashes identify the bytes actually read. Report/source-document identity also does not prove that the implementation files used by an older report are unchanged; those source manifests and deployment receipts still need their own checks.

## Adding a review claim

Edit the [claim manifest](../../docs/estate-review/completion/claims.json) only after reading the report's scope and checking its evidence. Each claim names `repo`, `path`, `source_sha256`, `report`, `report_sha256`, `coverage` and a human-readable `basis`. `kind` is `adr` or `topic`. ADR coverage is `whole-document`; topic coverage is `selected-panels` with a nonempty `panels` array.

A report that merely cites an ADR is insufficient. A renderer result is insufficient. A report that reviews one consumer of a compound ADR must not receive a whole-document claim. For partial ADR work, leave whole-document status unreconciled until a separate section-level claim format and explicit mapping are provided; the report itself remains the evidence.

Hashes bind the human assertion to artefacts. They do not independently prove that assertion. “Assessment recorded” means the named report contains the scoped assessment, including its open findings and limitations. It does not mean the feature is implemented, safe, deployed or fixed. “Unreconciled” means no qualifying claim has been entered; earlier review may exist.

## Validation

```sh
cargo fmt --manifest-path scripts/estate-review-ledger/Cargo.toml --check
cargo test --locked --offline --manifest-path scripts/estate-review-ledger/Cargo.toml
cargo clippy --locked --offline --manifest-path scripts/estate-review-ledger/Cargo.toml --all-targets -- -D warnings
```

Nine tests cover Markdown examples and heading identity, path boundaries, discovery/classification, a SHA256 published vector, and an isolated Git fixture. That fixture checks omitted unrelated/nested repositories, retained missing scope, partial panel credit, unknown panel refusal, changed-report/source invalidation and unmatched-claim refusal. No production checkout is modified by tests.

This tool is private audit machinery (`publish = false`), not a reusable published library.
