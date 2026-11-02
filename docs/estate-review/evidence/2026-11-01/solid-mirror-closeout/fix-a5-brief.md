# Commit A fix round 5: brief

**Rules**, the same as round 4:
- Write each test first, capture it red, then make the fix and show it green.
- Use `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.
- Run all seven gates, plus `--features mirror,anchoring,announce`.
- Put the evidence in `fix-a5/`: the interdiff from round 4, the full patch, the red and green logs, and the report.
- Do not commit, push, or create or remove worktrees.
- Reply with only the report path.

**Sources:**
- `audit-r2-codex.txt`: the final answer block, findings C1 to C7;
- `review-a2-security.md`: findings R1 to R5 and I1.

Both files are in this folder. Line numbers refer to the round 4 tree (`final-a4.patch`).

## Items

1. **Verify before changing pending state (C1).**
   - An event must not clear `pending`, or take the inbox slot from a verified event, until its signature and tags are verified.
   - **Option:** verify in `enqueue`, or keep `pending` until `on_event` confirms the new state was verified and acted on.
   - **Test:** a forged newer state does not cancel a pending retry, and the retry still fires.
2. **Timer and enqueue race (C2, R3).** In the timer arm, put the pending event back unless the slot already holds a state that `supersedes` it, so `or_insert` can no longer drop it.
   - **Test:** pending S2, with older S1.5 in the slot when the timer fires; S2 is still retried.
3. **Transient fetch and materialise failures are retried (C3, R4).**
   - Schedule `retry_at` with the same floor and backoff for fetch errors (network, timeout) and materialise errors.
   - Do not retry a fetch that hit the size cap.
4. **Retry floor from failure time (C4).** Compute `retry_at` from a fresh clock reading taken after the failure, not from the `now` sampled at the start of the attempt.
5. **Tie ordering (C5).** "Seen" and "acted on" must use the same `(created_at, id)` ordering. A lower-ID state at the same timestamp as an applied higher-ID sibling must be applied, or must be refused consistently, never recorded as newest and then skipped. Pick one and test it.
6. **Retry classification (C6, R2).** These become terminal for the state:
   - growth cap hit during the push;
   - a `mark_clean_store` command error, which is terminal, or awaits migration.

   Repeated publish timeouts on the same state become terminal after N attempts. Choose N, and document it.
7. **Rescope the unreachable-object scan (R1). This overturns locked decision 13.**
   - Today every followed owner-pushed repository with dangling objects, which is nearly all of them, goes to 503 permanently, and the owner can't push to recover.
   - Run `holds_unreachable_objects` only for a store with mirror history from before this release: a mirror record that already existed while the store was unmarked, or legacy config or `refs/mirror` traces.
   - A store first followed under the new code is an owner's store, and its unreachable objects are the owner's own.
   - Persist the awaits-migration verdict in the record, so the scan runs once, not on every event.
   - Move `prepare_served` after the replay check, so a replayed event costs no scan.
   - Stream the scan output, or use `git fsck --unreachable --no-reflogs --connectivity-only`, instead of buffering both full listings in memory.
   - **Tests:**
     - an owner-pushed repository with one commit left behind by a force push is followed and stays readable;
     - a legacy-style store (a record exists, the store is unmarked, and it holds unreachable objects) still waits for migration;
     - a replayed event does not run the scan.
8. **Ceiling and quarantine (R5).** Leave quarantine directories out of the `MAX_SERVED_BYTES` baseline.
9. **Docs (C7, I1, and the two CHANGELOG lines):**
   - **Ceiling:** the rustdoc and the CHANGELOG must say it is an admission threshold, that one push can grow the store by up to the growth cap past it, and that measurement is by polling.
   - **CHANGELOG:** "Leftover quarantine directories are removed" becomes "stale ones (older than the publish timeout)".
   - **`RefUpdate::expected` and `PublishError` rustdoc:** a stale lease is retried against the fresh value, so a ref moved behind the engine is overwritten on the retry.
   - **Codex round 4 Lows:** fix the stale "sweeps any it finds" comment and the "both sweeps" test name.

## Out of scope

- I2: the lock re-arms its retry every ~1 s. It is cheap; note it only.
- Anything in Commit B (migration).
