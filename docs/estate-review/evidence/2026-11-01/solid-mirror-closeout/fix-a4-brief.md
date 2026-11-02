# Commit A fix round 4: brief

Same rules as round 3:
- Write each test first, capture it red, then make the fix and show it green.
- Use `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.
- Run all seven gates on the final tree.
- Put the evidence in `fix-a4/`, with a round-4-only interdiff.
- Do not commit, push, or create or remove any worktree.
- Do not change existing assertions.

Source: the Codex round-3 audit, `audit-a3-codex.txt` (read its final block).

## Items

1. **`has_mirror_record` fails open** (`lib.rs:596`).
   - **Problem:** `metadata().is_ok()` treats permission or I/O errors as "no record".
   - **Fix:** return `Result<bool>`. `NotFound` means false. Any other error must refuse the write (409 or 503) in every writer guard, and in `unpublished_mirror` it must keep the gate closed.
   - **Test:** make the record path unreadable without relying on chmod (root ignores modes). For example, make the record path's parent a file, or inject an error through a seam. Assert the write is refused.

2. **The reachability roots include `HEAD`** (`mirror.rs:1504`).
   - **Problem:** `rev-list --all` counts `HEAD`, so a detached `HEAD` at an unreferenced commit passes the check.
   - **Fix:** walk only `refs/heads/*` and `refs/tags/*`, plus any other non-`refs/mirror` refs you decide to keep (list them in the rustdoc). Do not include `HEAD`.
   - **Test:** detach `HEAD` at an otherwise unreferenced commit. Assert `AwaitsMigration`.

3. **No check for alternates.**
   - **Fix:** `mark_clean_store` returns `AwaitsMigration` when served has `objects/info/alternates` or `objects/info/http-alternates`. Also have `publish` refuse a served store that has gained an alternates file (terminal).
   - **Test:** add one for each.

4. **The post-failure sweep cannot tell which quarantine is ours** (`sweep_new_quarantines`, `mirror.rs:1836`).
   - **Fix:** delete `sweep_new_quarantines` entirely. Leftovers from a failed push are cleaned only by the pre-push stale sweep on a later run. The size baseline already tolerates them, and the hard ceiling counts them, which is acceptable.
   - Also make `newest_mtime` treat an unreadable entry as "fresh", not skip it.
   - Correct the CHANGELOG line that claims a concurrent push "cannot be broken". The honest statement: only quarantines older than `PUBLISH_TIMEOUT` are removed; an owner push stalled longer than that may lose its quarantine and fail (availability only, nothing is exposed).
   - **Tests:**
     - Update the round-3 test so that the fresh quarantine is created during a failed publication (through the existing test seam or a hook-free injection) and survives.
     - Change `publication_leaves_no_receive_quarantine_behind` only if its assertion now has to describe the stale sweep. That is allowed, as long as you explain the change in the report.

5. **The writer tests don't prove nothing was written.**
   - The web-edit and merge-into-mirror tests must also assert that the served object set is unchanged: compare the `cat-file --batch-all-objects --batch-check` output before and after.

6. **Rustdoc in the wrong place:** the "Locked decision 3" doc belongs on `mark_clean_store`, not on `holds_unreachable_objects` (`mirror.rs:1493`).

7. **Record as a decision; no code change.** `pull_merge_info` runs `merge-tree --write-tree` in the base store, but for a mirror base the head can only be a ref already held. Anything written is derived from approved content only. Say this in the `has_mirror_record` or `resolve_head` rustdoc in one sentence.

## Report

Write `fix-a4/fix-a4-report.md`. For each item, give the red output and the fix location, plus the gate results. Reply with only the report path.
