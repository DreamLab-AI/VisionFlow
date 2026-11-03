# W3b source readiness receipt — 2026-11-03

Reviewed branch: `fix/custody-w3b`, base `6dbaaddc8`, head `ba798738f79b08787c445bdd5f68f1aa06f31c81`. Root has merged these two commits into its repair branch and owns the final exact fetched-source locks/pins gate. No Cargo manifest, lock or Nix pin change is part of this branch.

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
| External security gate | In progress on the exact clean diff; configured Claude SDK/local, minimal thinking, HIGH threshold, 20-minute cap. `/tmp/custody-w3b-deepsec/plan.json` |

Dependency boundary: the old Nix forum source `fe36bf36` lacks `unwrap_gift_kind`. The root coordinator must move the bridge/Colloquy forum pins together to `a3812b8e79846b569c1f19054f0774e94fed1248` and validate the exact fetched source with committed locks alongside final Solid alpha.19. Workspace source tests above do not substitute for that gate.

Owner-only remaining boundaries: independently provision the governance key, admit its public identity and role at the forum, rebuild and rehearse with isolation enabled, verify receipt ownership and encrypted-zone journeys, complete host Docker permissions, browser Podkey minting and historical secret purge. This work read no live role key, minted no identity, changed no role flag, authenticated no hardware key and performed no deployment.
