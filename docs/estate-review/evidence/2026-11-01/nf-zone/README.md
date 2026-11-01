# Zone-key audit evidence

Source baseline: forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. The source manifest binds 22 byte-exact exports. All match diagram revision `1b27f79`; 19 match website-consumed pin `9c36b60`. Differences are ADR-2016 close-out text/metadata and two `SigningKey::from_bytes` argument conversions in auth and a read-cache test. This does not identify the currently deployed revision.

## Reproduction

Create a disposable `git archive` of the pinned forum commit and apply `probes/zone-probes.patch` with `git apply --unidiff-zero`. With dependencies cached, run:

```sh
cargo test --locked --offline -p nostr-bbs-forum-client zone_crypto::
cargo test --locked --offline -p nostr-bbs-relay-worker zone_config::tests
cargo test --locked --offline -p nostr-bbs-config
```

This run used `/tmp/vf-nf-zone-artifacts/forum`; `/tmp` does not allow executable output, so `CARGO_TARGET_DIR=/home/devuser/workspace/nostr-rust-forum/target` supplied an executable build directory. Product source checkouts were unchanged. Test fixtures use generated disposable keys, ordinary core NIP-44 and an in-memory KV, without network calls or production data. Random generation is used for valid cryptographic material; none of the asserted failures depends on a rare random value.

| Receipt | Result and boundary |
| --- | --- |
| `logs/client-tests.log` | 41 passed: 38 existing tests plus three audit probes. The planner stops retrying old history after newest-key ACK; two independent loaded ledger instances lose a recipient; the real write/decrypt functions permit a stale writer and old key holder. |
| `logs/relay-tests.log` | 12 passed: 11 existing zone-config tests and one added probe showing the real shape predicate accepts an old epoch after a new epoch. This is not full Worker/ACL/transport admission. Other package targets selected zero matching tests and receive no credit. |
| `logs/config-tests.log` | 74 existing native tests and three doc tests passed. |
| `probes/zone-probes.patch` | Four test-only additions to the disposable archive; assertions capture current behaviour, not a proposed fix. |
| `logs/render-nf05.log`, `logs/render-nf27.log` | 22 panels rendered. Only the changed NF-05.16 and new NF-27.1–5 were visually inspected and receive review credit. |

Source-only findings cover conflicting key replacement, account/relay scoping, missing routing policy and queued-versus-accepted status. Native probes do not execute IndexedDB, browser account switching, negative ACK delivery, full Worker admission or a real removal/rotation. Those tests remain open. A removed member's ability to obtain ciphertext is not demonstrated by decryption tests.

Initial citation checks caught two punctuation-only anchors; they were corrected. Visual review found two overflowing sequence notes; explicit line breaks corrected them. Those were audit-artefact issues. Client compilation reported an existing dependency future-incompatibility warning; the build and tests passed. Logs retain command output with trailing blank lines removed. Source exports retain their original bytes.

## External comparison

Checked on 2026-11-01: [NIP-44](https://raw.githubusercontent.com/nostr-protocol/nips/master/44.md) for forward-secrecy/post-compromise limitations; [Marmot README](https://raw.githubusercontent.com/marmot-protocol/marmot/master/README.md) for adopted status, MLS group agreement and redundant transport. These are primary protocol descriptions, not independent implementation certifications. Anagram directory retrieval failed and search yielded no relevant result; this batch does not claim independent verification of its recovery draft or vectors.
