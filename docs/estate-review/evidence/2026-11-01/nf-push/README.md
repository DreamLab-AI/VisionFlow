# Push audit evidence

Native tests ran in a fresh archive of forum c39f41f104b114d6a3fa29013daaf6fa58b45080, sharing the existing ignored Cargo build cache. Append `push-audit-tests.rs` to `crates/nostr-bbs-relay-worker/src/push.rs` in a temporary source copy, then run:

```sh
cargo test -p nostr-bbs-relay-worker push --lib
cargo test -p nostr-bbs-core push --lib
cargo test -p nostr-bbs-forum-client push
cargo test -p nostr-bbs-forum-client dm_verdict
```

The worker run includes 45 original and six audit tests; core/client/DM filters contain seven/seven/one existing tests. The client has no library target; the initial `--lib` attempt was corrected.

The browser fixtures are independently runnable from this directory:

```sh
node --test browser/tests/sw-push.test.mjs browser/bridge-audit.test.mjs
```

This runs fourteen original worker tests plus four worker audit cases and one bridge case. `browser/sw.js` and `browser/js/push.js` are unchanged exported product source. The test helper adds rejection/stall controls; the bridge VM strips only module export keywords. No actual browser, network request, notification or subscription is involved. SQLite tests demonstrate deliberately ordered statements, not measured Cloudflare scheduling.
