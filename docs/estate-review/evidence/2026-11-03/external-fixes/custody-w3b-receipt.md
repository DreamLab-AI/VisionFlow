# W3b source readiness receipt — 2026-11-03

Original reviewed branch: `fix/custody-w3b`, base `6dbaaddc8`, head `ba798738f79b08787c445bdd5f68f1aa06f31c81`. Final integrated main is `b83771c21`, including root's exact producer pins/locks and URL/ledger repairs, own delivery/calendar amendment `85c6b8ffd`, integration `e923658ff`, and final keyed review matrix. Root owns exact fetched-source gate receipts.

Implemented isolation-aware callers: JunkieJarvis, nightly forum suggestions, live-mirror self-DMs, gateway operator/default and fixed derived identity, Dream governance and encrypted/plain digest. Isolation selects the port before local identity material is read. Port refusal never falls back to the flag-off key reader. Flag-off caller compatibility is retained.

The governed port adds bounded, typed NIP-17 wrapping/opening, authenticated forum grant opening, fixed kind-42 zone sealing, mirror self-wrap, and independently marked governance signing. Crypto is delegated to existing `nostr-bbs-core` NIP-44/NIP-59/signing code. Outer recipient cardinality/key, body bounds and inner kind are checked. Receipt write failure still prevents returned admission. `mirror_key` is public-only; old `secret: true` grants fail load. Human governance kinds remain forbidden for every key. The optional independently provisioned governance signer can publish panels/cases and withdraw its own events; public lookup and allowlisted relay authentication do not grant other signing authority. Same-UID capabilities are not represented as separate process isolation.

| Gate | Result / evidence |
| --- | --- |
| Bridge full native suite | 213 native tests plus 5 doctests pass, `/tmp/custody-port-full-tests.log` |
| Bridge native Clippy | All targets, warnings denied, `/tmp/custody-port-clippy.log` |
| Actual identity-port IPC | 15 tests, no skips, including real encrypted self-DM through the shared Node client; `/tmp/custody-port-ipc-tests.log` |
| Focused JavaScript callers | 105 Jest tests pass; `--forceExit` used for existing pending test handles, `/tmp/custody-js-tests.log` |
| Dream host after rebase | 193 library tests with locked registry dream-core 0.4.1; `/tmp/custody-dream-rebased-tests.log` |
| Dream host Clippy after rebase | All targets, warnings denied, `/tmp/custody-dream-rebased-clippy.log` |
| Dream host rustdoc | Warnings denied, before the unrelated dependency rebase; `/tmp/custody-dream-doc.log` |
| Source checks | Changed JavaScript syntax, Rust formatting and whitespace checks pass |
| External security gate | Original raw BLOCK: 12 findings (critical1, medium8, bug3). Revalidation of 11 findings at integrated e923: fixed5, false-positive2, true-positive3, uncertain1; twelfth reply-content policy remains open. `/tmp/custody-w3b-deepsec/receipt.json`, `/tmp/custody-w3b-revalidate.log`, `/tmp/custody-w3b-revalidated-findings.json`. No blanket PASS claimed. |

Final delivery/calendar controls pass 79 Node tests, 106 focused Jest tests, 25 supervisor tests and the 720-name inventory. The final matrix is `docs/estate-closeout/w3b-security-review-2026-11-03.md` on Agentbox main; root preserves durable receipts in `VisionFlow/docs/estate-review/evidence/2026-11-03/external-fixes/custody-w3b-review/`. Full Deepsec checkpoint and local metadata were archived to `/tmp/custody-w3b-deepsec-checkpoint.tgz` and `/tmp/custody-w3b-local-metadata.tgz` before worktree teardown. Owned build caches are reproducible and contain no unique gate receipts.

Dependency boundary: the old Nix forum source `fe36bf36` lacks `unwrap_gift_kind`. The root coordinator must move the bridge/Colloquy forum pins together to `a3812b8e79846b569c1f19054f0774e94fed1248` and validate the exact fetched source with committed locks alongside final Solid alpha.19. Workspace source tests above do not substitute for that gate.

Owner-only remaining boundaries: independently provision the governance key, admit its public identity and role at the forum, rebuild and rehearse with isolation enabled, verify receipt ownership and encrypted-zone journeys, complete host Docker permissions, browser Podkey minting and historical secret purge. This work read no live role key, minted no identity, changed no role flag, authenticated no hardware key and performed no deployment.
