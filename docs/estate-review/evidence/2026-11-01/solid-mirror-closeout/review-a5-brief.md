# Review brief: round 5 (Commit A)

**Rules:** read-only. Do not run cargo, do not edit anything, and do not create worktrees.

**Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, uncommitted on top of `31fe898`.

**Diff to review:** `docs/estate-review/evidence/2026-11-01/solid-mirror-closeout/fix-a5/fix-a5-interdiff.patch` (`CHANGELOG.md` and `mirror.rs`).
- Read it **together with the current source**: earlier rounds' code interacts with these changes.

**What round 5 was asked to do:** `fix-a5-brief.md`.
**What it says it did:** `fix-a5/fix-a5-report.md`.
**The findings it answers:** `audit-r2-codex.txt` (C1–C7) and `review-a2-security.md` (R1–R5, I1).

## Questions

Answer each with file:line.

1. **Each finding.** For every C1–C7 and R1–R5, is it FIXED, PARTLY FIXED or NOT FIXED?
2. **Rescoped scan (item 7, the big change).** The check for objects no ref reaches now runs only for stores with mirror history from before this release.
   - **Confidentiality, most important:** can a store that really holds held mirror objects from the old code escape the scan and be marked as published?
     - Check every legacy signal: a record already present while the store is unmarked, legacy config, `refs/mirror` traces.
     - Check every order of events: an announcement before a state, a restart, a record deleted by hand, and a candidate folder present or absent.
   - **Availability:** can an ordinary store an owner pushed (dangling objects from a force push) still end up stuck at 503?
   - **Saved verdict:** can it go stale in either direction, or be written by a forged or replayed event?
   - **Scan cost:** is it really skipped for replayed events now? Is its memory use bounded?
3. **Verify in `enqueue` (item 1).**
   - Is the signature checked before any slot or pending change?
   - Is the CPU cost of verifying every relay event acceptable, or can a relay flood it?
4. **Retry classification.** Are all of these retryable or terminal as they should be, with no hot loops and no stranded mirrors?
   - fetch failures, including the SSRF/DNS refusal now treated as terminal;
   - materialise failures;
   - growth cap;
   - repeated timeouts;
   - scan errors.
5. **Tie ordering (C5).** Are "seen" and "acted on" now consistent?
6. **Anything new** introduced by round 5.

## Output

- Write your findings to `docs/estate-review/evidence/2026-11-01/solid-mirror-closeout/review-a5-<you>.md`, where `<you>` is `security` or `codex`.
- Rank them by severity and mark each one CONFIRMED or PLAUSIBLE.
- Say plainly if nothing material remains.
