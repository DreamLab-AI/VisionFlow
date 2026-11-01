# Device-key evidence and reproduction

`source-manifest.json` pins fifteen exact source exports at forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. Fourteen match the website-consumed revision; the auth-device difference is test-side k256 type adaptation. All fifteen match the old NF-02/NF-08 diagram stamp. No deployed revision is asserted.

The selected checks total 76:

| Run | Existing | Audit additions | Boundary |
| --- | ---: | ---: | --- |
| Auth-worker `devices::` | 34 | 5 | Two extracted client request expressions; actual server row serialisation and copied unchanged client types/parser |
| Core `feature_gate::` | 8 | 0 | Exact-value rule and worker posture helpers |
| Relay `device_gate_tests` | 10 | 0 | Production gate/identity helpers; not real Worker bindings |
| Standalone cache fixture | 8 | 7 | Five unchanged extracted production functions plus full cache module; mocked clock/lookup/admin/zone boundaries |
| SQLite registry statements | 0 | 4 | Exact source queries, synthetic rows and scheduled interleaving; no NIP-98 or D1 runtime |

Extract a clean `git archive` of the pinned forum revision into a disposable directory outside the primary checkout. Run `python3 prepare_tests.py --kit <archive-directory> --out <fixture-directory>` from this evidence directory. The script refuses a Git checkout or a copy already carrying the test module, appends the five contract probes to the archive's auth-worker device module, and generates the standalone Rust cache fixture. Compile/run:

```sh
cargo test --manifest-path <archive-directory>/Cargo.toml -p nostr-bbs-auth-worker --lib devices::
cargo test --manifest-path <archive-directory>/Cargo.toml -p nostr-bbs-core --lib feature_gate::
cargo test --manifest-path <archive-directory>/Cargo.toml -p nostr-bbs-relay-worker --features test-exports --test device_gate_tests
rustc --edition=2021 --test <fixture-directory>/native/harness.rs -o <executable-path>
<executable-path>
python3 sql-boundaries.py
```

Angle-bracket paths are placeholders to replace. Put the executable on an executable filesystem; this session used the owned worktree under `~/workspace/wt/`. Delete the disposable archive and binary afterwards. The session reused the existing ignored forum Cargo target directory. No production implementation was edited.

`audit-client-contract.rs` retains the appended module; `native/extracted-methods.json` hashes the five source functions; `native/read_cache.rs` is copied unchanged. The fixture's absent row represents a revoked mapping because the production SQL filters `revoked=0`; mock failure returns the same `None` as the production error path. It does not execute actual D1, admin resolution or zone configuration. The old-context test deliberately retains a resolved context; it does not measure a real 400-second request. The single-poll runner fails on unexpected suspension.

`device-sql.json` contains the registry schema, owner-principal probe, upsert and scoped revoke with source line numbers. `sql-boundaries.py` executes them using SQLite. Transfer/reactivation cases assume the caller has satisfied authentication and device proof; neither is bypassed or simulated. The race is an explicit statement schedule, not a live concurrent request trace. The owner-scoped revoke control passes.

Counterexample tests assert current behaviour and should change when the implementation is repaired. A green result therefore records a reproduced gap; it does not certify the feature. Empty-list and integer-normalised controls show why hand-authored fixtures missed the server/client numeric mismatch.

`render-inputs/` and its hash manifest retain six NF-20 panels plus NF-02.7 and NF-08.8. All eight changed panels were visually inspected; NF-08.8 receives only a client-parser edge correction. Full-topic renders include unchanged panels that were not newly source-certified. Citation/freshness logs and prose checks establish their stated mechanical scope only. `web-comparison.json` records primary-source comparisons and the proposed user journey.

No browser registration, actual Worker HTTP response, D1 transaction, live account, reconnect or hibernation journey was exercised. Full scratch archive, generated renders and cleanup receipts remain under `/tmp/vf-nf-device-artifacts/` for this session; the source/test evidence here is durable.
