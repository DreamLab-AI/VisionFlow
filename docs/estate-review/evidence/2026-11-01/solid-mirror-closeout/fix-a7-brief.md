# Commit A fix round 7: micro-brief (the last round before commit)

**Rules**, the same as round 6:
- Write each test first, capture it red, then make the fix and show it green.
- Use `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.
- Run all nine gates.
- Save evidence in `.../solid-mirror-closeout/fix-a7/`: a report, the red logs, the gate logs, `fix-a7.patch` (the full diff against 31fe898) and `fix-a7-interdiff.patch` (the diff against `final-a6.patch`).
- Do not commit, push, or create or remove worktrees.
- Reply with only the path to the report.

**Source:** `audit-a6-codex.txt` (the last `codex` block).

Do these items and nothing more:

1. **A dangling record symlink counts as Unreadable, not Missing.**
   - In `MirrorStore::load`, run `symlink_metadata` on the record path before reading it.
   - If a directory entry exists but reading it gives `NotFound` (a dangling link), or the entry is a symlink at all, return `Unreadable`.
   - Return Missing only when no directory entry exists.
2. **S6 remainder: a `symbolic-ref` command failure is an error, not evidence.**
   - In `holds_unreachable_objects` (~`mirror.rs:1729`), separate a confirmed detached or non-branch HEAD (positive evidence) from a timeout, a spawn failure or an unexpected exit code.
   - Return `Err` for the failures, so they retry without persisting a verdict.
3. **`gave_up` must stay given up after the timeout limit.**
   - Once `timeouts >= MAX_TIMEOUTS` for the newest state, a later `Retry::Yes` failure of that state must not re-arm automatic retries.
   - Check the limit before scheduling, for any retryable class.
4. **Docs:**
   - Move `acted_on_at`'s rustdoc back onto `acted_on_at` (~`mirror.rs:2670`).
   - The alternates comment (~`mirror.rs:2065`) must say that errors are returned as `Err`, not "count as present".
