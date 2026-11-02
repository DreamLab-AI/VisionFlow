# Review brief: Commit B1, fix round 1

**Rules:** read-only. Do not run cargo, do not edit anything, and do not create worktrees. The lead runs the gates.

**Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, uncommitted, on `d6133e7`.

**Read:**
- `b1/fix1/report.md`;
- `b1/fix1/interdiff-vs-after-c.patch` (what this round changed);
- the current source: `crates/solid-pod-rs-forge/src/migrate.rs`, `lib.rs`, `forks.rs` and `migrate_tests.rs`;
- your own earlier review (`review-b1-security.md` or `review-b1-code.md`), and `audit-b1-codex.txt`.

**Task:**
1. For each finding you raised earlier, say whether it is FIXED, PARTLY FIXED or NOT FIXED, with `file:line`.
2. Look for new defects this round introduced, most of all any path where a held object is served or a store is ungated wrongly.

**Check specifically:**
- **The held definition.** Is H computed from the same R that is copied? Can any ref outside R put an object into staging: through replace refs, `GIT_NO_REPLACE_OBJECTS` missing on a migration path, alternates, a shallow store, or a ref added between listing and push? Does verify run when H is empty and `refs/mirror` is gone?
- **The detection-error journal.** Can it be removed when it shouldn't be? Can any flag-gate path be skipped? Is `migration_incomplete` cleared too early? Does the 64-hex owner condition miss stores the readers serve?
- **`served_is_ours` and the `migration-id`:**
  - Can an attacker who can write under `repos/` forge a matching id?
  - `migration-id` is inside the served store. Is it ever served or readable (raw, HTML, JSON, smart-HTTP dumb paths such as `/info`, archive)?
- **The `read_regular` same-inode check, and path containment.** Is there any TOCTOU left?
- **`smart_http_path`.** Does mapping a suffix-less path to `<name>.git` open anything ungated? Does it break the engine or the owner-push path (auto-init)?
- **Process-group killing.** Can it kill something outside the migration?
- **Notes now dropped** (a deviation). Do you accept it?
- **The residual "lock-timeout gate journals are never re-detected".** Is it fail-closed?

**Output:** a security reviewer writes `review-b1-fix1-security.md`; a code reviewer writes `review-b1-fix1-code.md`. Reply with only the path.
