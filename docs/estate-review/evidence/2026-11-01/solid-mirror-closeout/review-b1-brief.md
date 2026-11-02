# Review brief: Commit B1 (SP-GIT-02 migration, mirror class)

**Rules:** read-only. Do not run cargo, do not edit anything, and do not create worktrees. The lead is running the gates.

**Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, base `d6133e7` (Commit A merged). Uncommitted. The full diff is `b1/after-c.patch` in the evidence folder `/home/devuser/workspace/VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout/`.

**Read first:** `b-design.md` with locked decisions 21–26 at the end of `publication-design.md`, then `b1/b1-report.md`.

**The claim under review:** after `ForgeService::new` returns, no object that was held in a pre-release served store can be read through any reader (raw, HTML, JSON, archive, smart-HTTP) or reached by any writer, for record-bearing, lost-record and unreadable-record legacy stores. That must hold whether migration succeeds, fails, crashes part-way or is resumed.

## Highest priority

1. **Decision 21 edits Commit A's reader gate.** `ForgeService::unpublished_mirror` in `lib.rs` now checks `migrate::journal_gates` before the marker. Check:
   - every reader and writer path still reaches it;
   - a journal in any state other than `published` gates;
   - the gate can't be opened by a journal that is malformed, unreadable, a symlink, or written by an attacker. Who can write under `mirror-work/`?
2. **Every way migration can fail, with the store still gated afterwards:** fsck reject, verify hit, timeout, oversize, the held-set cap, a lock that can't be taken, a crash at each stage, the `MAX_ATTEMPTS` give-up. Look specifically at report deviation 5: a git error during detection leaves the store as it is, with no journal. For a lost-record store, that leaves held objects served with 200. Is that an exploitable fail-open? Can a request make the trace check fail?
3. **The swap.** Is there any moment where served is a half-built store, or a path an attacker controls, or where the old store is reachable under `repos/`? Where does the recovery copy live? Can the CGI export it?
4. **Decision 22 (`h_git_smart`):** `PathTraversal` now returns 404. Check the report's note that only the first two path segments are checked. Can any path depth or encoding reach a bare repository that skips the marker gate? Is `<owner>/<dir>/x` reachable?
5. **The ref filter and verify:**
   - `safe_ref` versus `check-ref-format`: can a dropped ref still have its objects pushed?
   - `reaches_held` and `held_set` correctness;
   - that `verify` counts objects in staging, not refs;
   - deviation 7: `Detected::Finish` trusts that a `.migration.json` exists. Can that be forged or left stale so a store is finished without verification?
6. **Concurrency:** two processes (t16), the engine against migration (`refuse_while_migrating`), and the in-process lock order.

## Also check

- **The five changed `fix_a` tests:** only the `forge_at` line moved, and no assertion was weakened.
- **The `run_at_boot` scoped thread and runtime:** does it deadlock or panic inside actix or tokio, and is there an unbounded boot time?
- **The CHANGELOG** describes the behaviour accurately, including "forks not yet swept".

## Out of scope (B2)

Forks, the transitive sweep, the `.dropped` trigger and the `.fork-*.tmp` sweep. Flag only if B1 makes them worse.

## Output

Write `review-b1-<security|code>.md` in the evidence folder. Findings ranked by severity, each with a `file:line` anchor, a concrete failure scenario, and CONFIRMED or PLAUSIBLE. Reply with only the path.
