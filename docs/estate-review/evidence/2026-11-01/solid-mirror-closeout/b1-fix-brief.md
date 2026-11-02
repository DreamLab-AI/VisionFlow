# Commit B1 fix round 1 (rust-engineer; tests first)

**Same rules as `b1-brief.md`:**
- Write each test first, capture it red, then make the fix and show it green.
- Use `CARGO_HOME=/home/devuser/workspace/.cargo`, `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`, `CARGO_BUILD_JOBS=4` and `--locked --offline`. Run one cargo command at a time.
- Do not commit, push, or create or remove a worktree.
- Evidence goes in a new `b1/fix1/` folder, with an interdiff against `b1/after-c.patch`.
- Restore mutated files with `cp`, then `touch` them and check their sha256. `mv` keeps the old mtime, and cargo then reuses a stale build.
- Write the report to `b1/fix1/report.md` and reply with only its path.

**Inputs** (in the evidence folder):
- `review-b1-security.md` (F1–F7);
- `review-b1-code.md` (H1, H2, M1, M2, L1–L7);
- `audit-b1-codex.txt`: the final `codex` block, items 1–6 plus "other material findings".

**Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`.

## Must fix (confidentiality: each is a path where held objects get served)

1. **One held definition, used for both the filter and verify.** This covers Codex #5, F2 and the C17 case.
   - **Root set R** = the `refs/heads/*` and `refs/tags/*` that will actually be copied: they pass `safe_ref`, have a valid SHA, and are not dropped for any reason.
   - **Copy only R**, plus non-root refs whose whole closure lies inside R's closure (`rev-list --objects <ref> --not R` is empty). Every other ref is dropped as `outside-roots`.
   - **Held set H** = `rev-list --objects --all --not R`, run on the source. That covers `refs/mirror`, dropped tags and anything else outside R.
   - **Verify** that staging contains no object in H, whether or not H is empty and whether or not `refs/mirror` exists. Keep the memory cap and fail closed past it.
   - The verify step must also fail on any object in staging that R's closure in the source does not reach.
   - **Tests:**
     - a lost-record store with `refs/mirror` removed, hideRefs only, and `refs/misc/*` reaching the held commit;
     - Codex's long or unsafe-name tag pointing at held, alongside a misc ref;
     - a C17 force-pushed-away held branch.

     All must be C15-hard, and all fail before the fix.
2. **Detection errors fail closed** (F1, M1, Codex #2; this reverses deviation 5).
   - Any error from `legacy_traces`, the alternates stat, or the record load writes a gate journal for that store: `state: failed`, `reason: detection-error: <redacted>`, `attempts: 0`.
   - On a later boot, if detection succeeds and finds the store clean, remove **only** a journal whose reason starts with `detection-error`.
   - If the runtime build fails or the migration thread panics, set a service-wide flag. While it's set, `unpublished_mirror` returns 503 for every unmarked store under a 64-hex owner that has no valid record. Log it loudly.
   - **Test:** inject a trace-check failure, for example by making the store's config unreadable, through a seam or real file modes. Then a held read returns 503.
3. **Journal reads use `symlink_metadata`** (Codex #1, L1).
   - A journal path that is a symlink or any non-regular file gates.
   - So does any error other than `NotFound`.
   - Apply this to `journal_gates`, `journal_exists`, `read_journal`, and the report and `.held` reads.
   - **Contain** the journal's `recovery` and `staging` strings: they must canonicalise to inside `mirror-recovery/` and `mirror-work/staging/`, otherwise the migration fails closed.
   - **Tests:** a dangling-symlink journal, and a symlink to a `published` journal. Both gate.
4. **`h_git_smart` accepts exactly one path shape** (Codex #4, L6, F4).
   - Allowed: `/<owner>/<name>[.git]/<one of GIT_SUFFIXES>`.
   - Anything else, including deeper nesting, any segment starting with `.`, or an encoded separator, returns 404 before the CGI runs.
   - **Test:** a nested bare repository at `repos/<owner>/<dir>/x` is not fetchable.

## Must fix (crash recovery and availability; fails closed today)

5. **Bind the report and the swap to the journal** (H1, H2, Codex `Finish`).
   - At build time, write the journal's uuid into staging as a file (for example `migration-id`) that is never served.
   - **Resume from `Moved` with served present:** if served's `migration-id` matches the journal, finish (`Published`, then clear the flag). Otherwise treat it as an intruder and fail closed.
   - **`Detected::Finish`:** require a report whose uuid matches served's `migration-id`.
   - **The resumed path** writes `Moved` before its rename.
   - **Never overwrite an existing report** for the same `recovery`, and never build with `source = p.served` once the journal has reached `Moved` or later.
   - **Tests:**
     - a crash after the second rename, on both the normal and the resumed path;
     - a stale report with a substituted store.
6. **Attempts count when an attempt starts** (F3). Apply the `MAX_ATTEMPTS` check to `Building`, `Verified` and `Failed` alike.
   - **Test:** three simulated crashes mid-build, then a fourth boot gives up with the store still gated.
7. **`take_lock` has a deadline.** Use `try_lock` polling, bounded by `STORE_BUDGET`. On timeout, write a gate journal (decision 21) and skip the store.
8. **Durability.** Fsync the journal, the report and `.held`/`.dropped`, and their parent directories, before each rename that depends on them.
9. **On a budget timeout, kill the process groups** of all children, not only the direct child (L4).
10. **Size cap on migration git calls.** Use `run_capped` with A's `MAX_REPO_BYTES`, as is done on the engine side.

## Should fix

11. **L5:** a non-UTF-8 ref name is dropped as `unsafe-name` and reported. It is never renamed.
12. **F6:** creating a fork onto a name that has a journal is refused with 503.
13. **L2:** the gate journal written when the lock can't be taken uses an exclusive-create write, never an overwrite.
14. **L7 / CHANGELOG:**
    - state exactly what migration does: roots only, and refs outside them are dropped;
    - say that detection errors gate;
    - drop A's stale "not migrated yet" sentence;
    - say that forks are not swept until B2.

## Record only (do not fix)

These go in the report as accepted:
- **F5:** an owner push in flight across instances can be lost;
- **F7 / L3:** boot time is unbounded; document it;
- the operator runbook for the fsck give-up.

## Gates

All nine, from `b1/gates.sh` with your own `OUT`, on the final tree. Include a mutation per Must-fix item, showing the new test catches it.
