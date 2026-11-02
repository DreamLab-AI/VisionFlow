# Code review: Commit B1 (SP-GIT-02 mirror-class migration)

- **Reviewer:** code-reviewer (correctness focus). Read-only: no cargo was run, and no file was edited outside this one.
- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, `fix/mirror-migration`, uncommitted on top of `d6133e7`. The diff is `b1/after-c.patch`.
- **Line anchors:** `crates/solid-pod-rs-forge/src/` unless stated otherwise.
- **Labels:**
  - CONFIRMED: traced end to end in the code and still holds after an attempt to falsify it.
  - PLAUSIBLE: I believe it, but could not trace it fully, or it depends on environment.

## Verdict

**The claim holds for the happy path and for most failures:** after `new`, no held object is readable for record-bearing, lost-record or unreadable-record stores.

**Three exceptions:**
- **The crash-resume logic mishandles a crash after the second rename** (H1, H2). It fails closed with no exposure. The cost: a correctly migrated store stays at 503 for good, and on one path the migration report that B2 depends on is overwritten.
- **Deviation 5 is a narrow fail-open on lost-record stores** (M1). It contradicts decision 21's "any failure".
- **Gaps in test strength** (M2) mean neither of the above is caught.

---

## High

### H1: a crash after `rename(staging → served)` and before the `Published` write leaves the store gated for good. CONFIRMED

- **Anchors:** `migrate.rs:1004-1027` (swap), `migrate.rs:853-864` (resume branch).
- **State:** in the normal path (`moved == false`), the journal is written as `Moved` at :1004-1005, then staging is renamed into `repos/` at :1009-1013. A crash or kill before :1024-1025 leaves:
  - the journal at `state: "moved"`;
  - `repos/<o>/<n>.git` holding the rebuilt, marked, verified store;
  - `mirror-recovery/<o>/<n>.git.<ts>` holding the old store.
- **Resume:** `migrate_one` matches `Stage::Moved if is_dir(&p.served) && is_dir(&recovery)` at :853. It reads this as "a store appeared in the gap", tries to set aside a staging directory that no longer exists, logs an error and returns `Failed`.
  - It does not increment `attempts`, so this repeats on every boot.
  - The flag is never cleared, and the journal stays at `moved`.
- **Effect:**
  - `journal_gates` (lib.rs:622) returns 503 on every reader for a correctly migrated store, for good.
  - `refuse_while_migrating` (mirror.rs:1243) keeps the engine from ever following it.
  - Only an operator who knows to delete the journal recovers it.
- **Why the branch cannot tell:** "served present" in the `Moved` state can mean either "the swap finished" or "an intruder appeared". Distinguishing needs one of:
  - a check that served is the staging store named in the journal (for example, write the staging uuid into the store and compare it);
  - a separate `Placing` state written before the second rename.

### H2: resume from `moved` writes no `Moved` state, so a second crash re-runs the build against the new store and overwrites the report. CONFIRMED

- **Anchors:** `migrate.rs:927-948`, `:981-989`, `:990-1008`.
- **The resumed path never re-enters `Moved`:** when the journal says `moved` and served is absent (t12's scenario), `rebuild` sets `moved = true`, writes `Building` (:947), then `Verified` (:981). It skips the `if !moved` block, so it never writes `Moved` again, and renames staging into served (:1013).
- **Second crash:** a crash between :1013 and :1025 leaves the journal at `verified`, with served (the new store) and recovery both present. The same state arises on the normal path when the `Moved` journal write at :1005 fails, which is only logged.
- **Next boot:**
  - `migrate_one` falls through to `rebuild` (`Verified` matches no special arm).
  - `moved = false`, so `source = p.served`, which is now the rebuilt store.
  - `build` runs against it. There are no `refs/mirror`, so the held set is empty and the build succeeds.
  - `write_report` (:986) then overwrites `<recovery>.held` with an empty list and `<recovery>.dropped` with roughly nothing, and rewrites `<recovery>.migration.json` to describe a rebuild of the rebuilt store. The original lists are lost. B2's fork trigger (`.dropped`) and fork filter (`.held`) depend on them (design §2.4; report "Out of scope").
  - `rename(served → recovery)` (:992) then fails with `ENOTEMPTY`, because recovery exists. That is `fail`, `attempts + 1`.
  - After three boots the journal is `failed` with `attempts = 3`, and the correct store stays at 503 for good, as in H1.
- **Fix direction:**
  - Write the report before the first rename only once. Never let a rebuild whose source is `p.served` write over an existing report for the same `recovery`.
  - Have the resumed path record its own placement state (see H1).
  - `fresh_recovery` checks only at journal creation, so it does not protect this.
- **Not covered by any test:** t12 starts at `moved` with served absent, and t13 starts at `published`. Neither crashes after the second rename.

---

## Medium

### M1: deviation 5 is a fail-open on lost-record stores and contradicts decision 21. PLAUSIBLE

- **Anchors:** `migrate.rs:652-658`, `:667-704`.
- **Why there is no gate:** `detect` returns `Detected::None` on any error from `legacy_traces`, so no journal is written. A lost-record store has no record, so `unpublished_mirror` (lib.rs:622-631) finds no journal and no record and serves it. That includes held objects, with 200.
- **The probes short-circuit (legacy_traces):**
  - An error from the alternates `stat` (:674) or from `for-each-ref` (:677-690) returns `Err` before the config probe (:694) runs.
  - `has_report` (:660) is never reached.
  - So one failing probe hides a positive hideRefs or report signal that would have been decisive.
- **Can a request cause it?**
  - Not at boot. Detection runs before the listener, and no request can make `for-each-ref` or `config` fail.
  - The realistic triggers are environmental and can recur on every boot:
    - the 15 s `quick` timeout (:668) on a slow or very large store, for example a huge `packed-refs`;
    - an `EACCES` on `objects/info/` (:674).
  - **Self-limiting in part:** an error that breaks git itself, such as a corrupt config, also breaks the readers. That does not hold for a timeout, or for an alternates `stat` error while packs are still readable.
- **Severity:** Medium rather than High. It is decision 26's pre-existing exposure surviving on an error path, not a new one.
- **Fix:**
  - Evaluate every probe. If any one is positive, return `Rebuild` despite the others' errors.
  - If none is positive and one errored, a gate journal is the decision-21 answer.
  - **Constraint:** the `Failed` resume path (:849, then `rebuild` at :867) does not re-run `detect`. A gate journal written for a detection error therefore needs its own state that forces re-detection, or a plain owner store at a 64-hex owner would be rebuilt (and lose its PR heads) on the next boot.
- **Not mentioned in the CHANGELOG** (see L7).

### M2: gaps in test strength. CONFIRMED (by reading the tests)

**Nothing exercises these paths:**
- A crash after the second rename (H1, H2): `Moved` or `Verified` with served and recovery both present.
- `Detected::Finish` (`migrate.rs:640-645`, `:814-822`): a store with no journal that is marked, flagged and has a report. t13 starts at `published`, which goes through `Stage::Published` instead.
- The `Moved` "appeared in the gap" branch (:853).
- The lock-failure gate journal (:777-803).
- A journal that is unreadable or malformed. `journal_gates` (:443-449) should gate it; no test asserts this.
- A detection error (M1).

**t16 does not prove the lock** (`migrate_tests.rs:940-974`):
- The two children are spawned back to back. If the first finishes before the second detects, the second sees a marked store and skips, so t16 passes with or without `take_lock`.
- There is no lock mutation alongside M1-M3. A mutation run with the lock removed would show whether t16 ever goes red. If it does not, add a barrier, for example a slow `DirMeasure` or a held lock that the parent releases.

**t21 stops early** (`migrate_tests.rs:1112-1117`): it asserts `attempts == 3`, but does not re-check `held_leaks` or the 503 after the boot that gives up, which is the state decision 21 cares about.

**t18 covers three journal states** (`migrate_tests.rs:1010`): `published`, `building` and `failed`. It does not cover `moved`, `verified` or an unparseable journal.

---

## Low

### L1: a dangling-symlink journal opens the reader gate. CONFIRMED (needs write access under `mirror-work/`)

- **Anchor:** `migrate.rs:438-452`.
- **How:** `journal_gates` uses `tokio::fs::read`, which follows symlinks. For a dangling symlink, `read` returns `NotFound`, so the function returns `false` and the gate opens.
- **Inconsistent with its neighbours:**
  - `journal_exists` (:456-464) uses `symlink_metadata` and treats the same entry as present.
  - `MirrorStore::load` treats any symlink as `Unreadable` (`mirror_record.rs:333-336`), and A pins that with `fix_a::a_dangling_record_symlink_keeps_the_gate_closed`.
- **Likewise:** a symlink to a file reading `{"state":"published",…}` opens the gate.
- **Who can write there:** `.forge/` is blocked by `DotfileGuard`, so only the server user or an operator can write under `mirror-work/`. Not remotely exploitable.
- **Fix:** match the record rule. If `symlink_metadata` reports a symlink, gate.

### L2: the lock-failure gate journal is written without the lock. PLAUSIBLE

- **Anchor:** `migrate.rs:783-801`.
- **Race:** `read_journal` returns `None`, then a peer instance holding the lock migrates the store and deletes its journal, then this instance writes `failed` with `attempts = 0`.
- **Effect:** a migrated, marked store sits at 503 until the next boot, which rebuilds the clean store again and creates a second recovery entry.
- **Severity:** fails closed. It needs a lock I/O error, so it is rare.

### L3: boot time is unbounded. CONFIRMED (by design, but undocumented)

- **Anchor:** `migrate.rs:734-749`.
- **The lock wait has no limit:** `take_lock` blocks in `File::lock` with no timeout. A peer instance holding the lock (a stuck engine `apply`, or a peer's own 600 s migration) stalls `ForgeService::new` indefinitely.
- **The total is not capped either:** per store, the bound is `STORE_BUDGET` (600 s, which wraps only `build`, :959) plus the lock wait. Stores run serially and failures retry three times, so a plugin directory with N legacy stores can block boot for N × 600 s per boot.
- **Lazy path:** this runs inside `web::block` (server `lib.rs:5089`) while the `REGISTRY` std mutex is held. Every forge request waits on it. That is intended, but there is no ceiling.
- **Suggestion:** use `try_lock` with a bounded retry, and on timeout write the gate journal, as for a lock error.

### L4: the outer budget timeout does not kill process groups. CONFIRMED

- **Anchors:** `migrate.rs:959`, `:243-283`.
- **What goes wrong:** when `tokio::time::timeout(STORE_BUDGET, build(..))` fires, it drops the inner future.
  - `kill_on_drop` kills only the `git push` leader.
  - The receive-pack and pack-objects processes in its process group (made by `process_group(0)`) are not killed: `kill_group` runs only on the runner's own timeout or cap branch.
  - `remove_dir(&staging)` (:966) then races with the orphans still writing into staging.
- **Effect:** availability only. Staging is in `mirror-work/` and is cleared at the next boot.

### L5: non-UTF-8 ref names are renamed or lost without a report. CONFIRMED

- **Anchors:** `migrate.rs:1147-1151`, `:1179`, `:1363`.
- **Renamed:** the ref listing goes through `from_utf8_lossy`. A ref with invalid UTF-8 becomes a name containing `U+FFFD`, which passes `safe_ref` and is pushed under the new name. `verify` compares lossy against lossy, so it passes.
- **Lost:** two such refs that map to the same lossy name collide in the `BTreeMap` (:1179). One is silently missing from both `copied` and `dropped`.
- **Effect:** this breaks I2 ("no approved ref is lost": served, or in recovery and reported). The ref is in recovery, but it is not reported.
- **Fix:** drop any line that is not valid UTF-8 as `unsafe-name`.

### L6: smart HTTP gates `<name>.git`, but the CGI can resolve another directory. PLAUSIBLE

- **Anchor:** `lib.rs:785-800`.
- **Suffix search:** `git-http-backend` resolves a path non-strictly, as `enter_repo(dir, 0)`, trying `X/.git`, then `X`, then `X.git/.git`, then `X.git`. For `/forge/<o>/<n>/info/refs`, a bare repository at `repos/<o>/<n>` (no suffix) is served first, while the forge gates `repos/<o>/<n>.git`. If that path is missing, the gate returns `Ok`.
- **Deeper paths:** the report's own observation stands. The router accepts any depth that ends in a git suffix (`router.rs:406-412`), so `/<o>/<n>.git/x/info/refs` gates `<o>/<n>` but serves `repos/<o>/<n>.git/x`.
- **Reachable today?** No forge code creates either shape, and the dot-named `.fork-*.tmp` is closed by decision 22.
- **Fix:** require the exact shape `/<owner>/<name>(.git)?/<suffix>` (the report's suggestion closes both).

### L7: CHANGELOG accuracy. CONFIRMED

- **Stale sentence:** part 1's "Repositories mirrored before this change are not migrated yet and stay refused (503) until the migration change lands" (`CHANGELOG.md:67-68`) is now false for mirror-class stores. Suggested wording: "…are migrated at boot (part 2); forks of them are not swept yet."
- **Overclaim:** "makes the migration resumable after a crash at any step" (`CHANGELOG.md:35`) is not true, because of H1 and H2.
- **Inaccurate for post-move failures:** "A migration that fails … leaves the old store in place behind a `failed` journal" (`CHANGELOG.md:40-44`). After the first rename, the old store is in recovery and served is absent (404).
- **Understated:** "stays refused until an operator acts" understates the operator side:
  - No recourse is implemented (no `receive.fsck.<id>=ignore` constant).
  - Deleting the journal on a lost-record store re-exposes held objects until the next boot. The report says so (Observations); the CHANGELOG should too.
- **Missing:** the CHANGELOG does not mention deviation 5, that a detection error leaves the store as it is (M1).
- **Accurate:** "Forks of legacy mirrors are not swept yet" (`CHANGELOG.md:33`).

---

## Checked and found sound

- **The five moved `fix_a` tests** (mirror.rs diff hunks at about 6506, 6555, 8158, 8457 and 8870):
  - Each moves only the `forge_at` line above the legacy-shape construction, plus a two-line comment. In the compare test, `forge_at(&plugin).handle(` becomes `forge.handle(`.
  - No assertion changed.
  - Each still exercises A's per-request gate, which reads the record and marker on every request, not at construction.
  - `c-full-run1.txt:386-419` shows all five failing before the move, with migration having opened the gate. So the move is needed and does not weaken the tests.
- **The decision-21 hunk reaches every path:**
  - `resolve_repo` (lib.rs:590-602) and `gated_git_dir` (:692-703) both call `unpublished_mirror`.
  - Writers resolve first:
    - the fork source (`forks.rs:271`, so a gated source cannot be cloned with `--local` into an ungated fork);
    - web edit (`web_edit.rs:598`);
    - merge (`merge.rs:294`, `:308`);
    - pulls and compare (`gated_git_dir`, throughout);
    - smart HTTP (lib.rs:785-800).
  - Every state other than `published` gates, and so does a journal that does not parse (`migrate.rs:443-449`). A `published` journal falls through to the marker check, so a stale one opens nothing that no journal would not.
- **Deviation 7 (`Detected::Finish`):** it only clears the flag and wipes the candidate. Readability is governed by the marker, which only migration (after verification) or the engine (never for a flagged store) writes. A forged or stale report cannot open a reader.
- **Swap atomicity (I3):**
  - The staging store is marked only after `verify` succeeds (:977) and is renamed into `repos/` only after that.
  - Between the two renames, served is absent (404) while the journal gates.
  - Recovery is under `plugin_dir/mirror-recovery/`, outside `GIT_PROJECT_ROOT=repos`, and `assert_migrated` checks that nothing is left under `repos/`.
- **The ref filter and verify:**
  - A dropped ref is never in a refspec (:1283). Its objects can arrive only through a copied ref's closure, and `verify` checks for that at object level (`present_count`, :1396-1416, counting objects in staging, not refs).
  - `--glob=refs/mirror` implies `/*`, and the `*` matches across `/`.
  - The held set subtracts the full closure of the roots (:1236-1251), which handles C16.
  - Test 4 and mutation M1 pin the object-level check.
- **Legacy tags:** the pre-A engine fetched `+refs/heads/*:refs/mirror/*` with `--no-tags` (`git show 30135fe^:…/mirror.rs:1438-1442`). Auto-followed upstream tags into held branches therefore do not sit in `refs/tags/`, so copying `refs/tags/*` unconditionally does not leak. I checked only that one revision.
- **`run_at_boot`** (:474-490): a scoped thread with its own current-thread runtime, joined. There is no `block_in_place` and no nested `block_on`, and a panic is caught by `join`. Blocking the caller's actix thread at boot happens before the listener exists. t17 (forge and server) covers it.
- **Residual risk, not a finding:** the mirror-class filter uses the held set `H` computed from the *current* `refs/mirror` tips. A non-root ref reaching a historical held object, one whose branch upstream force-pushed away before migration (the C17 shape), is copied. This is design Risk 4 applied to the mirror class. It needs a non-root ref in a mirror store, which only owner pushes create.
