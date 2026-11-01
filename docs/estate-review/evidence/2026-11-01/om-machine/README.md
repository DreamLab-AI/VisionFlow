# Machine-bundle audit evidence

Evidence for the [section assessment and findings](../../../2026-11-01-ontomark-machine-bundle-contract.md). The [manifest](verification.json) records 28 source exports at five repository revisions, selected line ranges, whole-file hashes, test counts and diagram checks. Full source files are retained unchanged; range exports preserve original line numbers.

## Reproduction

Export Ontomark revision `0caae5d4b695cb720942b9b9212017c00a80ad41` into a disposable directory on an executable filesystem. Copy [audit_machine.rs](audit_machine.rs) into `crates/ontomark-compile/tests/`. From that archive run:

```sh
cargo test --locked --offline -p ontomark-compile --test machine_scope
cargo test --locked --offline -p ontomark --test machine_scope_cli
cargo test --locked --offline -p ontomark-compile --test audit_machine -- --nocapture
cargo tree --locked --offline -p ontomark --edges normal --format '{p} {l}'
```

Offline reproduction requires the dependencies already cached. The CLI package name is `ontomark`; `cli-tests.log` retains the initial wrong-package failure. `machine-tests.log` records 24 passing compiler cases and `cli-tests-final.log` records four actual CLI cases. `probes-initial.log` records five passing probes with two unused-import warnings. The audit harness removes those imports; `probes-final-clean.log` records five passes without those warnings. No product code was changed.

The probes use the existing `estate-small` fixture, fixed time and source labels. They check a real 32-bit path-hash collision/refusal, private-name masking asymmetry, profile omission from content identity, vocabulary comment equivalence and acceptance of a caller-supplied commit label. They assert actual output fields or bytes and publication failure, rather than just reproducing the naming formula. [slug-collision.json](slug-collision.json) records the candidate-search result.

## Diagram and source receipts

`render-om03-final.log`, `render-om06.log` and `render-om11.log` record 16 successful renders and 394 pinned citations, with no warnings or working-tree fallback. Earlier render logs retain the initial corrections. The seven assessed panels were visually inspected; the remaining nine receive render validation only.

[old-pin-diff.patch](old-pin-diff.patch) records the differences in the two nonidentical source files used by the older OM-03/OM-06 pins. Build-module differences concern a test fixture; CLI differences add ingest outside the reviewed build path. Other compared runtime files and machine tests are byte-identical at the old and current revisions, as recorded by `old_pin_identity`. Historical topic stamps remain in place because this batch reviews selected panels.

The default CLI dependency tree is retained in `cli-dependency-licences.txt`. Its licence labels are source metadata, not a legal opinion or an all-features inventory. Cross-repository source excerpts establish current code/configuration boundaries only. No production build, private distribution deployment, full consumer parity, embedding service or real-corpus performance measurement is claimed.
