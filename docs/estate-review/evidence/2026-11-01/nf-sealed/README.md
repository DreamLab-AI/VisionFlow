# Sealed-history audit evidence

Report: [forum sealed history](../../../2026-11-01-forum-sealed-history.md).

`source-manifest.json` binds 28 forum exports to `c39f41f104b114d6a3fa29013daaf6fa58b45080`, checks the earlier diagram revision and the website-consumed kit pin. `pin-diff.patch` records the two files with argument-conversion differences at the latter pin. `agentbox-source-manifest.json` binds three separate reader/consumer/test exports. `sha256.json` hashes this evidence directory except itself.

The fresh scoped result is **103 native Rust tests, seven Rust documentation tests, 47 existing JavaScript tests and four JavaScript probe cases**, all passing. Of the native Rust tests, six are audit additions: five migration probes and one member-reader probe. Passing reproductions assert the current deficient behaviour; they do not mean those deficiencies were fixed.

| Log | Scope |
| --- | --- |
| `logs/migrate-tests.log` | Entire migrator package: 38 existing native tests, five added probes and seven documentation tests |
| `logs/client-tests.log` | Ten existing read-cache tests plus the added key-tag mismatch case |
| `logs/core-tests.log` | 25 sealed-format tests; other package tests filtered out |
| `logs/relay-sealed-tests.log` | Ten existing pure relay admission/timestamp cases |
| `logs/relay-delete-tests.log` | Fourteen existing parser/disposition/response/audit-shape tests, not a running D1 handler |
| `logs/agentbox-tests.log` | 47 existing zone-key and agent-consumer tests |
| `logs/agentbox-boundaries.log` | Good envelope, duplicate-property acceptance, missing-zk fallback and unknown-version rejection |

The client build reported existing dependency future-incompatibility warnings. It passed. An initial command incorrectly selected a nonexistent client library target; rerunning the package's binary test target produced the recorded successful result. Citation checks initially caught a mistyped revision and punctuation-only source anchors; corrected output is retained. These were audit setup errors, not product failures.

## Reproduce

Use a disposable archive of the pinned forum source; no source checkout or production keys are required. Copy `probes/audit_boundaries.rs` into `crates/nostr-bbs-zone-migrate/tests/` and apply `probes/client-reader.patch` with `git apply --unidiff-zero` at the archive root. Then run:

```sh
cargo test --locked --offline -p nostr-bbs-zone-migrate
cargo test --locked --offline -p nostr-bbs-forum-client zone_crypto::read_cache
cargo test --locked --offline -p nostr-bbs-core sealed::
cargo test --locked --offline -p nostr-bbs-relay-worker sealed_original_tests
cargo test --locked --offline -p nostr-bbs-relay-worker admin_events::tests
```

Dependencies must already be cached for offline mode. In this environment `/tmp` is not executable; `CARGO_TARGET_DIR=/home/devuser/workspace/nostr-rust-forum/target` supplied an executable build directory. The archive itself was `/tmp/vf-nf-sealed-artifacts/forum`. The new test file was formatted after its successful run; no semantic changes followed.

At the pinned agentbox checkout with its existing dependencies:

```sh
node --test tests/sovereign/zone-keys.node-test.js
node /absolute/path/to/probes/agentbox-boundaries.cjs /absolute/path/to/agentbox
```

The added probes use fixed synthetic keys and make no network calls. The migration fake models query/publish/delete responses, not real Worker/D1 or transport behaviour. The malformed admin-envelope probe therefore demonstrates the engine's acceptance rule; the separate actual member-reader test demonstrates rejection. Production reachability depends on the stated admin/import precondition.

Native unit cases do not prove fresh-client browser usability, live permissions, deployment identity, database failure behaviour, post-removal rotation, backup deletion or historical embedding cleanup. Those limits remain open in the report.

Command logs have trailing blank lines removed. Source exports are byte-for-byte copies. Rendered SVGs and visual PNGs remain in the separately archived local audit artefacts; `visual-review.json` binds the inspected versions by hash.
