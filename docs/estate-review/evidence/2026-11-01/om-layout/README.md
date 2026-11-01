# Layout and shared-check evidence

Evidence for the [layout assessment](../../../2026-11-01-ontomark-layout-and-shared-checks.md).

`verification.json` pins 21 baseline source exports at Ontomark
`0caae5d4b695cb720942b9b9212017c00a80ad41`. Full-file hashes are checked against
Git; ranged exports preserve line numbers. ADR-OM-008 is assessed in full;
ADR-OM-007 is retained as context only.

`audit_layout.rs` contains nine baseline probes. Eight run the actual CLI; the
furniture case calls the actual loader. Reproduction: add the file as
`crates/ontomark-cli/tests/audit_layout.rs` in a disposable archive of that revision,
then run `cargo test --locked --offline -p ontomark --test audit_layout`.
`probes-final.log` records nine passes reproducing the defects. These are
counterexample assertions, not evidence that the bad behaviour is desirable.

`cli-initial.log` retains an initial wrong expectation of exit 2 for validation
failure; the documented code is 1 and `cli-final.log` corrects that assertion.
The final baseline also passed five native plain-vault CLI tests, ten profile
tests and six bundle-contract tests. No real private pages were used.

The implementation evidence records a separate red/green suite. Generic
required-key enforcement was attempted, failed an existing build fixture and
was removed because the profile's schema list is broader than runtime policy.
That finding remains open; it must not be counted among the repaired layout cases.

`old-pin-diff.patch` includes the CLI, build module, page loader and body parser
between historical topic pin `560e50c1a35bb9486b9f43ebf074b816c1e01349` and the
baseline. Page/body CommonMark changes shift lines; selected-panel comparisons
do not justify describing the whole old implementation as identical.

The CPU local-reader harness includes the actual production module and the port
traits/types extracted from the same checkout. It is temporary test machinery,
not a new shipped crate. Its results do not stand in for server linking or runtime
activation. The server check records separately hydrated generated PTX and the
schema from its pinned Agentbox gitlink, with hashes in `server-build-inputs.json`.

Security-review receipts retain their original pre-remediation scope. The owner
medium finding was reproduced and fixed with a rollback regression. Consumer
findings concern feature-off compatibility behaviour. No high-severity finding
was reported; that is a bounded review result, not a universal security claim.

## Final implementation and receipts

`implementation-sources.json` binds owner and consumer source exports to Git
revisions and full-file SHA-256 values. `journal-*.log` records the final owner
0.6.3/0.7.2 gates and publication; `consumer-journal-*.log` records actual registry
0.6.3 consumption. Earlier `owner-*` and `consumer-registry-*` runs concern the
preceding 0.6.2 release. Test totals include unit, integration and doctest results;
ignored tests are reported separately. The compiler's 29 migration cases include
eight new boundary regressions; the first six are retained red-to-green evidence.

The original owner security scratch was lost during the cleanup incident described
in `owner-cleanup-incident.json`. `owner-security-final/` is a fresh review at
`a55033c`, before the journal patch. Its journal findings prompted the patch;
its custom-public-gate finding stays open. `consumer-security/` retains the
consumer review. Neither receipt is represented as a post-journal independent
review. `consumer-registry-tests.log` was truncated by the incident; the complete
successful rerun is `consumer-registry-tests-final.log` and the later final patch
run is `consumer-journal-tests.log`.

To repeat final acceptance, check out the pinned owner and run workspace tests,
compiler tests with `--all-features` and `--no-default-features`, strict workspace
Clippy, workspace rustdoc with `RUSTDOCFLAGS=-Dwarnings`, and release WASM builds
for `ontomark-wasm` and `ontomark-ingest`. For the consumer, run the commands in its
exported implementation receipt with `--locked`. The retained `local-harness/`
is audit machinery: restore it as `.audit-local` at the consumer root to resolve
its relative paths before running Cargo. Its lock and source are preserved;
build caches are excluded. Generated PTX/schema inputs are identified by
`server-build-inputs.json`; no deployed server or real-corpus parity is claimed.

Primary comparisons were reopened on 2026-11-01: Obsidian's official storage
page supports the folder/nesting observation, and Quartz's Private Pages page
distinguishes filter plugins from parsing exclusions. Those comparisons are
inferences about useful interfaces, not certification of either compiler.

Pinned source exports and raw patches retain their original bytes. Consequently,
`git diff --check` reports whitespace in patch context lines and the exported
consumer Cargo manifest's trailing blank line. The authored-file whitespace check
passes separately; evidence is not rewritten to make that check appear clean.
