# Commit B1 fix round 2 (rust-engineer; tests first; narrow)

**Same rules as `b1-fix-brief.md`:**
- Write each test first, capture it red, then make the fix and show it green.
- Use your own target directory and `--locked --offline`, one cargo command at a time.
- Do not commit, and do not create or remove a worktree.
- Restore mutated files with `cp`, then `touch` them and check their sha256.
- Evidence goes in `b1/fix2/`, with an interdiff against `b1/fix1/after-fix1.patch`.
- Write the report to `b1/fix2/report.md` and reply with only its path.

**Inputs:**
- `audit-b1-fix1-codex.txt`: the final `codex` block;
- `review-b1-fix1-security.md` (N1–N6);
- `review-b1-fix1-code.md` (N1–N3).

## Must fix (fail-open paths)

1. **Every journal gates.** `journal_gates` returns true for any journal, `published` included (Codex High 1, security N2, code N1).
   - `finish` removes the journal once the record is rewritten and the marker is in place.
   - A store is briefly 503 between publish and finish. That is accepted.
   - Check that no other path relies on a `published` journal being open.
   - **Test:** a lost-record legacy store with a regular `published` journal at `attempts: 3` and no matching migration id. Boot, then a held read returns 503. Repeat across a second boot.
2. **The incomplete-pass fallback gates marked stores too** (Codex High 2).
   - While `migration_incomplete` is set, `unpublished_mirror` returns true for every store under a 64-hex owner, marked or not, unless that store has a valid record *and* a report bound to its `migration-id`. Simpler still is all of them; pick the simpler and say which.
   - Readers and detection must agree on the marker check: both use the regular-leaf-file test.
   - **Test:** a marked, flagged store with no bound report, and an unwritable migrations directory. A held read returns 503.
3. **Enumeration errors are not silent** (security N1).
   - Any error listing owners or repositories, or reading a directory entry's type, makes the pass incomplete: set `migration_incomplete` and log at error level.
   - **Test:** make an owner directory unreadable, with a precondition check that `EACCES` really occurs. After boot, the flag is set and that owner's held store returns 503.
4. **Detection returns an explicit Clean** (Codex Medium 5).
   - `is_dir` and every other probe distinguish *absent* from *error*. A detection-error journal is removed only when detection returns `Clean`, meaning every signal was examined successfully and found negative.
   - The marked/unflagged shortcut is not `Clean` for that purpose: leave the journal.
   - **Test:** a detection-error journal, then a boot where the store-dir probe errors. The journal stays.
5. **Neutralise grafts on the migration's walks** (security N3).
   - Set `GIT_GRAFT_FILE` to an empty file under `mirror-work/`, or `/dev/null`, after checking that git accepts it, alongside `GIT_NO_REPLACE_OBJECTS=1` in `migration_command`.
   - **Test:** a store with an `info/grafts` line grafting held history onto an approved commit. The held blob is not copied.
6. **Never copy `refs/replace/*`** (code N3). Drop them as `outside-roots` and report them.

## Should fix
- Code N2: put the misplaced doc comment back on `unpublished_mirror_error`.
- Security N6: the CHANGELOG precision items.

## Record only (do not fix)

Accepted as outside the threat model, because each needs write access to the server user's filesystem, and an actor with that can already serve anything:
- Codex 3: replaying the `migration-id`;
- Codex 4 and security N4: containment and TOCTOU in `read_regular`;
- Codex 6: a recycled process-group ID;
- security N5: an operator-relocated recovery symlink, which fails closed.

## Gates

All nine, with `b1/gates.sh` and your own `OUT`, on the final tree. Include one mutation per Must-fix item, showing the new test catches it.
