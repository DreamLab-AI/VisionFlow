# Review brief: round 2 production code (not yet independently reviewed)

**Rules:** read-only. Do not run cargo, do not edit anything, and do not create worktrees.

**Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, uncommitted on top of `31fe898`. It is the full tree after rounds 1–4 (`final-a4.patch`, sha256 `04dac91b…`).

**Diff to review:** `docs/estate-review/evidence/2026-11-01/solid-mirror-closeout/fix-a2-interdiff.patch`. Read it together with the current source; rounds 3 and 4 changed some of the same functions.

**Context:**
- the round 2 report, `fix-a2/fix-a2-report.md`;
- your own earlier delta review, `review-a-security-delta.md` (M1, N1–N3, T1).

## Changes under review

- the retry floor (2 s minimum, doubling to 60 s, even when `minIntervalMs` is 0);
- waking a sleeping retry when a newer state arrives;
- `PublishError`: retryable versus permanent;
- the `AwaitsMigration` skip;
- `MAX_SERVED_BYTES`;
- path-boundary redaction;
- arming a retry only for the newest state (the replayed-retry fix).

## Questions

Answer each with file:line.

1. **Retry versus replay.**
   - When a retry runs, does it fetch again and check the verdict again, or does it reuse an earlier result?
   - Can anything older than `newest_seen_at` be applied through a retry or a wake?
   - While a retry is pending, states are dropped. Can that drop a state that should be applied, for example a newer state, or the pending state with new content?
2. **Locks while waiting.**
   - Is the in-process mutex or the file lock held while the worker sleeps for a retry?
   - On the wake path, is the lock order still the in-process mutex first, then the file lock?
3. **Retry classification.** For each failure below, say whether it is counted as retryable or permanent, and whether that is right:
   - stale lease
   - receiver ref lock
   - ceiling exceeded
   - timeout
   - oversize
   - fetch failure
   - `AwaitsMigration`

   A permanent error marked retryable recreates D1 (a hot loop). A transient error marked permanent strands the mirror.
4. **Ceiling overshoot.** `MAX_SERVED_BYTES` is checked only before the push, so one push can overshoot it by up to the growth cap. Do the rustdoc and the CHANGELOG say so?
5. **Anything else** material in this code.

## Output

- Write your findings to `docs/estate-review/evidence/2026-11-01/solid-mirror-closeout/review-a2-security.md`. Rank them by severity and mark each one CONFIRMED or PLAUSIBLE.
- Reply with only that path.
