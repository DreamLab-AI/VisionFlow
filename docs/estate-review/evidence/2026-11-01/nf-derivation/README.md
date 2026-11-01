# Subkey derivation evidence

`source-manifest.json` pins thirteen exact files: ten forum and three agentbox exports. All forum exports match the legacy diagram stamp; nine match the website-consumed revision. The remaining `keys.rs` changes are k256 argument conversions. Source differences are retained separately. Current source identity is not deployed identity.

Validation runs:

| Run | Count | Boundary |
| --- | ---: | --- |
| Existing native key unit tests | 22 | Actual core implementation |
| Existing native shared-vector tests | 5 | Versioned fixture, 13 vectors and invalid-root/rotation controls |
| Native audit tests | 3 | Tag bytes, old signature validity, lack of implicit service context |
| JavaScript selection probes | 12 | Verbatim extracted code with synthetic root/config/key-provider inputs |
| Unchanged parity-runner scenarios | 8 | Baseline and seven controlled variants; expected successes and failures |

To reproduce the native runs, extract `git archive c39f41f104b114d6a3fa29013daaf6fa58b45080` from `nostr-rust-forum` into a disposable directory. Copy `audit-derivation.rs` into its `crates/nostr-bbs-core/tests/audit_derivation.rs`, then run:

```sh
cargo test --manifest-path <archive>/Cargo.toml -p nostr-bbs-core --lib keys::
cargo test --manifest-path <archive>/Cargo.toml -p nostr-bbs-core --test identity_subkey_vectors
cargo test --manifest-path <archive>/Cargo.toml -p nostr-bbs-core --test audit_derivation
node audit-js.mjs <disposable-output-directory>
```

Replace the angle-bracket paths. The JavaScript harness reads the retained exports next to itself; it never imports the self-executing production entrypoints. It extracts exact bounded function/statement slices, hashes them and evaluates only those slices in a VM. `operatorKeyHex`, environment inputs and logging are injected. Public test roots come from the fixture. The forced zero HMAC case is a mock validation-boundary test, not a real invalid-child preimage. Gateway root public-key validation and downstream nostr-tools signing are outside the extracted boundary. No browser or wasm bridge is run.

`audit-results.json` records twelve probe results and the eight unchanged-runner outcomes. Baseline succeeds with 13 vectors, three invalid roots and 18 checks. Absent sources, source expressions only in comments, and a changed gateway default still pass. An unrelated HKDF comment, a missing expression, a corrupted vector and a missing fixture fail. The changed-default scenario also evaluates the altered selector and asserts its output actually differs. Mutations affect disposable copies only. Green audit results therefore include demonstrated gaps, not implementation repairs.

The 18 baseline runner checks overlap its vector and structural assertions and are not counted as 18 additional independent audit cases. `extracted-source.json` pins the six executed source slices. Logs retain compiler/test outputs. Full scratch sources and isolated variants are archived under `/tmp/vf-nf-derivation-artifacts/` for the session; this directory retains the durable reproduction kit.

`render-inputs/` retains four NF-21 panels plus the four changed legacy panels, with hashes and visual-inspection flags. Citation and freshness checks establish only resolvable locations and declared revisions. `remote-branch-rules.json` retains read-only GitHub responses: main reports unprotected and no active rules at the recorded time and remote commit. No remote setting changed. No deployment, message, key registry, private root or live process was changed or inspected. Existing user files and foreign worktrees are preserved.
