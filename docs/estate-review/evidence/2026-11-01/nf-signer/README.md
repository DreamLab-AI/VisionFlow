# BBS signer audit evidence

Source revision: `nostr-rust-forum` `c39f41f104b114d6a3fa29013daaf6fa58b45080`. Twelve exact source exports and SHA-256 values are retained in `source/` and `source-manifest.json`. Nine match the website-consumed revision; the three core API adaptations are retained as JSON-encoded diffs. All twelve match the existing NF-05 source stamp.

`audit-lifecycle.rs` contains eight test additions, appended only to an archived copy of `crates/nostr-bbs-bbs-client/src/signer.rs`. No production function or dependency manifest was changed. `native-signer.log` records twenty passing tests (twelve existing plus eight audit); `native-nip07.log` records two existing tests. The latter only checks stored public-key access and the native unavailable-extension result.

To reproduce, extract `git archive c39f41f104b114d6a3fa29013daaf6fa58b45080` from the forum repository into a disposable directory, append `audit-lifecycle.rs` to that copy's BBS `src/signer.rs`, then run from the copied workspace:

```sh
cargo test -p nostr-bbs-bbs-client --lib signer::
cargo test -p nostr-bbs-bbs-client --lib nip07::
```

Remove the disposable directory afterwards. This run used `/home/devuser/workspace/wt/VisionFlow-nf-signer/.audit-signer/kit` and the existing ignored forum Cargo target directory. The single-poll helper deliberately fails if a local signing future is pending; it does not simulate browser promises. Synthetic keys use repeated `01` and `02` bytes. No operator secrets or messages are involved.

The counterexamples assert current behaviour, including old signer survival and permissive metadata selection. They will need updating when those behaviours are fixed. A local author-mismatch test is a passing control. Invalid pasted/baked input retaining the old account is recorded without assuming that every failed switch must log out.

`render-inputs/` retains Mermaid and SVG for all seven changed panels; `render-manifest.json` hashes them. All seven were visually inspected after rendering. The complete 23-panel render run includes sixteen unchanged NF-05 panels which were not newly semantically certified. `validation.json` and the gate logs distinguish citation/structure checks from actual behaviour. `prose-checks.json` retains the final invisible-character and high-severity prose checks. `web-comparison.json` records primary sources and the scope of the market inference.

Browser adoption/reload, origin separation, provider account switching, delayed prompts, clipboard state, IndexedDB deletion failures and live relay AUTH rejection remain unexecuted. No browser or deployment acceptance claim follows from native tests or source tracing. Runtime-source archive and worktree cleanup receipts are retained under `/tmp/vf-nf-signer-artifacts/` in this session; durable source/test/render evidence is committed here.
