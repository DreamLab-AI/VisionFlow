# Forum implementation receipt — 2026-11-03

Phase 1 merged to local main as `7852fe8` from `944e844`. Phase 2 merged as `a3812b8e79846b569c1f19054f0774e94fed1248`. The owned worktree and branch have been removed; primary main is clean and the only remaining forum worktree.

Implemented: both passkey register-verification requests carry NIP-98; server persists the random options salt against the registration challenge and atomically consumes it with credential insert. Device clients send owner-bound signed proof; Worker integer wire fields and conditional same-owner/nonrevoked upsert prevent transfer/unrevocation. Auth, moderation, admin listing and pod/payment cleanup use relay dynamic admin authority without legacy members/cache fallback. Invite code/id compatibility and SQL set-union merge; push snapshot/token/sequence compare-and-swap, captured send-sequence bookkeeping and localStorage teardown credentials; persisted trust-sweep cursor; explicit disclosure registered boolean.

Gate receipts for the exact phase-1 diff:

- `/tmp/forum-external-tests-final.log`: auth253 + BBS165 + pod211 + relay491 =1120 native tests pass.
- `/tmp/forum-external-clippy-final.log`: affected three Workers, all native targets, warnings denied, pass.
- `/tmp/forum-external-wasm-final.log`: auth/relay/pod and both WASM clients compile, pass.
- `/tmp/forum-external-doc.log`: affected Worker rustdoc warnings denied, pass.
- Formatting and whitespace checks pass; six owner-local ADR repair links resolve.
- `/tmp/forum-external-wasm-clippy.log`: four pre-existing BBS WASM lint errors (redundant_ok in passkey, type_complexity/redundant_closure in search, question_mark in signer). Native Worker Clippy and WASM compilation are distinct successes; no clean WASM-client-Clippy claim.
- `/tmp/forum-external-deepsec/receipt.json`: TIMEOUT, gate124, deepsec137; SDK first batches produced no result. Partial scanner coverage is not a security PASS. Worktree deepsec state must be archived before teardown.

Phase 2 complete: `classify_policy_read` governs pod R2/KV reads and candidate ACL parsing. Present invalid/oversized/read-failed child policy terminates with deny; only absence inherits. Local64KiB cap retained. Existing D1ReplayStore implements Solid ReplayStore using worker-rs SendFuture, and the old kit trait delegates to it, so every Worker verification route consumes the shared nonce contract. Missing write metadata fails closed using alpha19 ReplayError::Unavailable. Registry checksum34d78d76e62681e6ae3f54ffd91f92e917ce5cf94342c889a21533bf1f2d3a27 is committed; no path patch remains.

Phase2 final results: `/tmp/forum-alpha19-tests.log` auth253+BBS165+pod212+rate2+relay491=1123 library tests plus existing integration/doc tests pass using locked registry resolution. `/tmp/forum-alpha19-restored-final.log` confirms the final restored replay module and pod214 tests plus integration/doc gates. `/tmp/forum-alpha19-native-clippy.log` four workers all native targets -Dwarnings passes; `/tmp/forum-alpha19-wasm-clippy.log` four workers WASM library -Dwarnings passes; `/tmp/forum-alpha19-clients.log` both clients WASM compile passes. Native-only SQLite fixtures are target gated. A workspace disk-full interruption was corrected by clearing only owned caches, restoring the uncommitted replay source, then rerunning these gates. Ignored scanner/checkpoint state archived at `/tmp/forum-external-ignored-evidence.tgz` before teardown. No push, deployment, account/key mutation or flag change performed.

Operational limits: pod RELAY_DB must bind the instance relay database (kit template supplies explicit placeholder). Existing credentials whose original PRF salt was lost by older code cannot be safely rewritten and may need authenticated recovery/re-enrolment. Full ADR2014 migration and static-bootstrap removal are not claimed. Real authenticator, Cloudflare D1, browser logout and role-on deployment acceptance remain separate runtime checks.
