# Poker audit reproduction

Use the consumed kit revision `9c36b6032640c89879ae52fe25da99493979a0b6` in an isolated writable copy. Keep its Cargo.lock. The audit extracted regular files/directories from `git archive`, omitting the unrelated absolute core fixture symlink listed in archive-skipped-links.json. Do not run these checks against a live house service.

Run the existing native suite from that copy:

```sh
cargo test --locked --offline -p nostr-bbs-poker -p nostr-bbs-poker-citizen
```

Copy `estate_boundaries.rs` into `crates/nostr-bbs-poker-citizen/tests/`, then run:

```sh
cargo test --locked --offline -p nostr-bbs-poker-citizen --test estate_boundaries
```

The six tests call the real public state-machine/ledger APIs. They assert counterexamples rather than desired corrected behaviour. Injected balances, debt and times are synthetic; no sockets, chain mutations or production keys are involved. The baseline native suite was run before adding these tests.

For the script scenarios, put this directory in an executable scratch location and provide its `node_modules` with the website's installed nostr-tools dependency (the audit reused that installation, without a new npm install). The copies `provision.mjs` and `probe.mjs` are byte-identical to the website entry points at `b5eb1ecf96599ebe154ebf23bbe7c703d2e653a5`. Run from this directory:

```sh
node run.mjs
node run-probe.mjs
```

The parents create random synthetic keys and delete them after execution. Children receive a restricted environment. Preloaded mocks replace WebSocket and reject fetch; no production relay or HTTP request is sent. The fake WebSocket records a scheduled send on a closed connection instead of throwing: that case proves an early ACCEPTED message and premature close, not the final exit behaviour of every real WebSocket implementation. A separate rejection fixture proves zero exit on an explicit publication rejection. The hand fixture really encrypts its synthetic replies but deliberately changes the commitment and supplies no replay record or transfer.

The native suite does not run the browser/WASM UI, inject runner crashes, exercise ambiguous producer responses or attest deployed binaries. Full audit limitations are in the report.
