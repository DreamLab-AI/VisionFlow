# Commit B1 fix round 3 (micro; tests first)

**Same rules as `b1-fix2-brief.md`.** Evidence goes in `b1/fix3/`, with an interdiff against `b1/fix2/after-fix2.patch`. Write the report to `b1/fix3/report.md` and reply with only its path.

**Input:** `audit-b1-fix2-codex.txt`, the final `codex` block, findings 1 and 2. I have confirmed finding 1 against the source.

## Must fix

1. **Persist the journal before any rejection branch** (Codex High 1).
   - In `migrate_one`, when `JournalRead::Absent` leads to `Detected::Rebuild`, write the new journal to disk straight away, using the same durable write `rebuild` uses. That must happen before `contained`, the attempt check, `dir_state` and every later branch that returns `Failed`.
   - If that write fails, return `Outcome::Ungated`, so the pass is incomplete and the service-wide flag gates the store.
   - Audit **every** `return Outcome::Failed` in `migrate_one` and its callees. On each one, either a journal is confirmed on disk, or the function returns `Ungated` instead. List them in the report.
   - **Test:** a lost-record legacy store with held objects and `refs/mirror/*`, whose `mirror-recovery/<owner>` directory is inaccessible. Include a precondition check that the probe really gets `EACCES`. After boot, a held read returns 503, either because a journal is present or because `migration_incomplete` is set.
   - **Mutation:** remove the early persist. The test turns red.
2. **Final size check after the push** (Codex Medium 2).
   - When the capped push completes, measure staging once more. If it exceeds the cap, fail closed, the same way as a watcher-detected overrun.
   - **Test:** a fast push with a cap smaller than the result. The hook can make the watcher's interval long, so only the final check can catch it.
   - **Mutation:** remove the final check. The test turns red.

## Gates

All nine, with `b1/gates.sh` and your own `OUT`, on the final tree.
