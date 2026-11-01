# NIP-171 key-plane audit evidence

Eighteen exact source files are retained in the manifest from local forum upgrade `0b7979c404f41414cfdbefffa28fb6d5ae1b1242` and agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf`. This does not identify a deployed revision. ADR-2028 has identical bytes on forum main `c39f41f104b114d6a3fa29013daaf6fa58b45080`. The local remote-tracking upgrade ref is older and was not refreshed.

## Reproduction

Archive the pinned forum revision into a disposable directory. Its tracked fixture symlink is absolute; replace only `crates/nostr-bbs-core/tests/fixtures` with `../../../tests/fixtures`, using the archived branch fixtures. `archive-adjustment.json` records this relocation. Apply `probes/keyplane-probes.patch` with `git apply --unidiff-zero`. No production source change is in that patch.

```sh
cargo test --locked --offline -p nostr-bbs-forum-client zone_crypto::
cargo test --locked --offline -p nostr-bbs-relay-worker zone_groups::
cargo test --locked --offline -p nostr-bbs-zone-migrate grants::
cargo test --locked --offline -p nostr-bbs-zone-migrate --test migration ticket_
```

This execution used `/tmp/vf-nf-keyplane-artifacts/forum` with `CARGO_TARGET_DIR=/home/devuser/workspace/nostr-rust-forum/target`, because `/tmp` is not executable. The shared ignored build directory does not change tracked production source. Native tests use synthetic keys and local fixtures, not real zone content or accounts.

Run agentbox's existing `node --test tests/sovereign/zone-keys.node-test.js`. For the added probe, copy `probes/keyplane-audit.node-test.js` into `tests/sovereign/` in a disposable copy of the pinned agentbox tree and run it with Node's test runner. It requires the existing `management-api/node_modules/nostr-tools` dependency and unchanged `management-api/lib/zone-keys.js`; use the pinned repository dependency lock. Our probe ran against byte-exact module copies with a dependency-directory symlink. The existing 47-test suite ran read-only against the pinned checkout, creates/removes its own temporary stores and uses synthetic fixtures. No gateway entrypoint, deployment or real message was invoked.

| Receipt | Result and boundary |
| --- | --- |
| `logs/client-tests.log` | 153 passed: 151 existing, two added. Actual read/decrypt functions demonstrate unreadable retained-key history; actual signed recovery records demonstrate acceptance of a valid older prefix. |
| `logs/relay-tests.log` | Ten passed: nine existing and one added. Actual constants/SQL/pure predicates reproduce the handler's two-loop algorithm in SQLite; no D1 or WebSocket handler execution. Other filtered targets run zero tests and receive no credit. |
| `logs/migrator-tests.log` | Five existing grant unit tests passed. This filter excludes the integration ticket tests; they have a separate receipt. |
| `logs/migrator-ticket-tests.log` | Four existing ticket integration tests passed with an in-memory relay. No production migration. |
| `logs/agentbox-tests.log` | 47 existing tests passed, including signed ticket validation and agent routing. |
| `logs/agentbox-probe.log` | One new probe passes: authenticated conflicting tickets leave the first key selectable for encrypted sending. No message is sent. |

Total: 220 tests, including four additions. Existing compilation warnings are retained in logs. Probes assert current behaviour; their passing does not mean the reported failures are fixed. Source-only findings cover missing owner workflows, member-history distribution, rollout discard, cutover and reader hint handling. Browser persistence, full Worker admission, current deployment configuration and external-client compatibility remain untested.

`source-searches.json` records targeted call-site/table searches in the disposable archive; it contains added test call sites too. No production owner caller or table reset is found in that inspected source tree. Searches do not prove absence from private deployment scripts or another revision.

## External comparison

Primary sources checked on 2026-11-01: [Anagram NIP-171 draft](https://raw.githubusercontent.com/lnbits/anagram/main/nips/nip171.md), [Anagram private-group behaviour](https://raw.githubusercontent.com/lnbits/anagram/main/docs/private-groups.md) and [NIP-59](https://raw.githubusercontent.com/nostr-protocol/nips/master/59.md). The report bounds its comparison to documented protocol/product behaviour, not a runtime certification. No external source document is copied into this evidence bundle.

## Diagram and document verification

NF-28.1–6 rendered with Pele and all six were visually inspected. The first citation pass found two empty/punctuation-only anchors; they were corrected, and all new anchors were manually checked for relevance. Visual review prompted replacing “uniqueness race” with “existing binding” to describe the sequential fixture accurately. The final full-corpus gate reports 198 topics, 1,700 panels, 25,149 revision-verified citations, no warnings/fallbacks/unresolved citations and nine existing allowlisted artefact citations. The generated verification/index check passes. NF-05's Mermaid bodies are byte-identical; its implementation-status prose was corrected without extra panel credit.
