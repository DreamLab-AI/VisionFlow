# Commit B2 report (SP-GIT-02): forks of legacy mirrors

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-fork-sweep`, `HEAD` `8ba3372`. Nothing is committed or staged; `migrate_fork.rs` and `migrate_fork_tests.rs` are intent-to-add only (`git add -N`). No worktree was created or removed.
- **Patch:** `after-b2.patch` (`git diff 8ba3372`, sha256 `54c95bf6c776d6ff…`): 7 files, +1971 / −98.

| File | Change |
|---|---|
| `migrate.rs` | +365 / −65 (class plumbing, `build` split, file-fed stdin, fork pass hook, the gated-plan guard) |
| `migrate_fork.rs` (new) | +676 |
| `migrate_fork_tests.rs` (new) | +803 |
| `forks.rs` | +61 / −30 (hoist of `build_fork`, fallible config reader) |
| `lib.rs` | +19 / −3 (incomplete-pass fork rule) |
| `migrate_tests.rs` | +3 (`mod fork;`) |
| `CHANGELOG.md` | +44 |

- **Environment:** every cargo command used `b1/cargo.sh` (`CARGO_HOME=/home/devuser/workspace/.cargo`, `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`, `CARGO_BUILD_JOBS=4`), with `--locked --offline`, one at a time. No dependency was added.

## Gates (final tree): all nine exit 0

`gates.txt`, run with `OUT=b2/gates.txt bash b1/gates.sh`. It contains no `FAILED` line.

| # | Gate | Exit | Tests |
|---|---|---|---|
| 1 | `cargo fmt --all --check` | 0 | |
| 2 | forge `--features mirror` | 0 | lib 405 passed (B1 final: 391) |
| 3 | forge (default features) | 0 | lib 284 passed (B1 final: 270) |
| 4 | forge `--features mirror,anchoring,announce` | 0 | lib 466 passed (B1 final: 452) |
| 5 | clippy forge (all features) `-D warnings` | 0 | |
| 6 | server `--features forge-full forge` | 0 | |
| 7 | clippy forge `--features mirror` `-D warnings` | 0 | |
| 8 | clippy server `--features forge-full` `-D warnings` | 0 | |
| 9 | `cargo doc` forge `--features mirror` (`RUSTDOCFLAGS=-D warnings`) | 0 | |

The fork tests are ungated, so they run in gates 2, 3 and 4. `gates-round1.txt` is the first full run, before the advisor's review round (also all nine green: 403 / 282 / 464).

## Order of work

1. **Seam, no behaviour change.** `ForgeService::build_fork`'s body moved into `forks::build_fork_at(repo_root, src, tmp, owner, name)`; the method delegates to it. Tests must make a fork with production code *before* `ForgeService::new` runs, because `new` migrates the parent first.
2. **Tests, red on the B1 tree** (`after-tests-red.patch`, `red.txt`, `red-summary.txt`). Every red is an assertion failure.
3. **Implementation.** Test 6 failed once more on the first green run (`green1.txt`); the cause is under "Defect found while implementing". After that all 12 fork tests passed (`green.txt`), and the full forge suites passed under both feature sets with **no B1 test changed or knocked on** (`full-run*.txt`). Then clippy, mutations M1–M8, the CHANGELOG and docs, and the gates (`gates-round1.txt`).
4. **Review round.** The advisor's pre-handoff review found two gaps.
   - **A fail-open path.** A fork that already had a non-gate journal skipped `fork::detect`, so its plan's gate was never consulted: with an ancestor still behind a journal, the plan held no reports, H was empty, and the rebuild published every ref, a ref at held included.
   - **An untested guard.** The stamping test never reached `rebuilt_against`.

   I added two tests (`after-tests-red-2.patch`, `red-2a.txt`, `red-2b.txt`) and the guard: `migrate_store` now refuses to rebuild a fork whose plan is gated, returning `Failed` behind its journal. All 14 fork tests then passed (`green-2.txt`). I also replaced `present_roots`' one `cat-file` per root with a single batch call. Then mutations M9–M10, and the final gates.

## Tests: red, then green

All are in `migrate_fork_tests.rs` (a child of `migrate_tests`, sharing its fixtures). How each fixture meets C15:
- **Hardlinked fork.** Every fork made with `forks::build_fork_at` (`clone --local --bare`) asserts as a precondition that it shares a pack inode with its parent. That covers every test except the two below.
- **`.keep` pack.** The untouched-fork guard pushes approved objects into a fresh fork and keeps its pack with a `.keep` file. It cannot be hardlinked, because a hardlinked fork holds dropped objects by construction.
- **No fork store.** The journalled-source test makes no fork; it checks an HTTP refusal.

Every legacy parent is B1's `legacy()` fixture, which also has a `.keep` pack and `refs/misc/*` refs reaching held.

| Test | What it checks | Red on the B1 tree |
|---|---|---|
| `b2_t6_a_fork_of_a_legacy_mirror_is_rebuilt` (design 6) | A fork marked by hand (decision 19), with its own branch `feature` and a PR head `refs/forge/heads/dave/x`, is rebuilt. Held is denied on every fork reader; `main`, `feature` and the PR head survive; `forge.parent`, both `refs/forge/` hides and the marker are kept; `receive.denyDeletes` is unset; one recovery entry; no journal left | `held commit served by the fork: …/raw/<held>/held.txt -> 200 (held bytes: true)` |
| `b2_t7_a_fork_of_a_fork_is_rebuilt` (design 7) | `dave/sample`, a fork of `carol/sample`, a fork of the legacy mirror, is rebuilt (C2) | `held commit served by the fork of a fork: …/dave/… -> 200` |
| `b2_t7b_a_fork_taken_before_a_force_push_is_rebuilt` (design 7b, C17) | The fork was taken while `side = X1`; upstream then force-pushed `side` to an unrelated X2 and the legacy store refetched. Preconditions: X1 is in the fork and X2 is not, so an H trigger has no hit. X1 is denied and gone from the fork | `X1 served by the fork: …/raw/<X1>/held.txt -> 200 (held bytes: true)`. This confirms the readers serve the unreachable X1. |
| `b2_t8_a_fork_ref_reaching_held_is_dropped_and_reported` (design 8, C9, D1) | `refs/heads/republished` at the held commit is dropped and listed in the fork report as `reaches-held` (report `class: fork`); `feature` and `main` survive and are served | `held commit served by the fork … -> 200` |
| `b2_t19_a_stale_partial_fork_is_moved_to_recovery` (design 19, D4) | A hardlinked `.fork-stale.tmp`, two hours old, is moved to `mirror-recovery/carol/.fork-stale.tmp.<ts>`; a fresh `.fork-fresh.tmp` (control) stays | `the stale partial fork is still under repos/` |
| `b2_an_unreadable_fork_config_makes_the_pass_incomplete_and_gates` | The fork's `config` is at mode 000 (precondition: `EACCES`). The pass is incomplete, held is denied, and approved content from the fork is 503 | `the pass did not report incomplete` |
| `b2_a_fork_cycle_terminates_and_the_fork_is_swept` | Legacy M names `carol/sample` as `forge.parent`, and carol names M. The boot runs on a thread under a 120 s `recv_timeout`; it terminates, the fork is swept and M is migrated | `held commit served by the fork in a cycle … -> 200` |
| `b2_a_fork_with_no_dropped_object_is_left_untouched` | A fork holding only approved objects (pushed `main`, `.keep` pack) gets no journal and no recovery entry; its config bytes and pack files (by inode) are unchanged, it has no migration id, and it is served | **Green on the B1 tree, by design:** a guard, like B1's t5. Its meaningful red is mutation M6 (`the clean fork was moved to recovery`). |
| `b2_a_fork_of_an_unmigrated_parent_is_gated` | The parent's migration fails (t21's fsck-bad commit; precondition: the parent journal exists). Its fork is gated: held denied, approved 503 | `held commit served by the fork of an unmigrated parent … -> 200` |
| `b2_a_followed_fork_gets_both_treatments` | A fork under a 64-hex owner with a pre-release mirror record and a branch `side` at the held commit. After boot, `side` is dropped and `main` kept; it has the marker, `receive.denyDeletes = true` and `forge.parent`; approved content is served | `held commit served by the followed fork … -> 200` (B1's mirror-class rebuild copied the root `side`) |
| `b2_a_rebuilt_fork_is_not_rebuilt_again_and_the_parent_is_stamped` | The C17 fixture with a fork ref `keep-x1` at X1 (risk 4: in D, not in H, so kept). Boot 2 leaves the fork's migration id and recovery entries unchanged, and the parent's report carries `forksCleanAt` | `boot 1 did not rebuild the fork` |
| `b2_a_journalled_fork_with_a_gated_ancestor_is_not_rebuilt` (review round) | carol is forked before the parent is broken, then given `republished` at held; the parent's migration then fails (t21's fsck-bad commit), and carol has a planted `failed` fork journal (`attempts: 1`, no `detection-error` prefix). Carol is not rebuilt: its journal stays, there is no recovery entry, held is denied, and approved is 503 | `held commit served by the fork: …/raw/<held>/held.txt -> 200 (held bytes: true)` (the fail-open path) |
| `b2_a_rebuilt_fork_is_not_rebuilt_again_while_stamping_is_blocked` (review round) | The C17 fixture plus `keep-x1`, plus a planted erin fork journal that has given up (`attempts: 3`), which keeps the parent's report unstamped on both boots (precondition asserted). Boot 2 leaves carol's migration id and recovery entries unchanged, through `rebuilt_against` | **Green on the pre-review tree:** the guard already existed but was untested. Its red is mutation M10 (`boot 2 rebuilt the fork`). |
| `b2_a_fork_of_a_journalled_source_is_refused` (item 6) | A fork request whose source has a journal is refused with 503 and no fork is created | **Green on the B1 tree, by design:** B1 already refuses it, through `resolve_repo` → `journal_gates`. This test confirms that behaviour. |

## Implementation

All new code is in `migrate_fork.rs`, a child module of `migrate` (`#[path]`), so it uses B1's private machinery directly.

**B1 machinery made class-aware.** A `Kind { Mirror, Fork(Plan) }` is threaded through. `migrate_one` became `migrate_store(.., kind)` (`migrate_one` is the `Mirror` wrapper). The following take the kind:
- `detect_for`, which uses B1's `detect` for mirrors and `fork::detect` for forks;
- `gate`, `new_journal` (journal `class`, plus a new optional `parent` field so a resumed fork knows its lineage while its store is in recovery), `rebuild`, `build`, `seal(.., marked)`, `served_is_ours(.., marked)` and `write_report` (`class`, plus `parent`, `followed` and `ancestors` for a fork).

A journal of the other class is left alone and gates. The mirror pass skips fork journals, which the fork pass resumes.

**Item 1: discovery.** The fork pass lists stores with B1's `served_stores` (errors make the pass incomplete) and reads every store's config with the new `forks::read_fork_parent_checked`, which, unlike `read_fork_parent`, returns read errors instead of "not a fork". Any config error makes the pass incomplete (decision 28). Lineage is walked transitively from each fork's parent:
- it is cycle-safe through a visited set, and bounded by the store count;
- `count_forks` is not used;
- fork journals are added to the targets, using their `parent`.

**Item 2: trigger.** A fork is rebuilt when it holds any id from an unstamped ancestor report's `.dropped` file. The files are streamed into `cat-file --batch-check`, which stops at the first hit; this is the new file-fed stdin, `Input::Files`, with regular-file checks. The check runs under B1's `MIGRATION_ENV` (replacements and grafts off), never consults the marker, and never uses H. A fork whose served migration id matches a fork report listing every trigger report id among its `ancestors` counts as clean: it was rebuilt against them, and what remains of D is reached by its own refs (risk 4). When the pass is complete and no fork failed, every unstamped mirror-class report gets `forksCleanAt` (rewritten with tmp, fsync, rename and a directory fsync; every other field kept).

**Item 3: rebuild.**
- **Refs:** every ref is copied except `refs/mirror/*` (`mirror`), `refs/replace/*` (`replace`) and unsafe names. Each copied ref, pull-request heads included, must not reach the inherited H beyond the ancestors' roots present in the fork: `reaches_held` with `--not <present ancestor roots>`, present-checked first (C12). A ref that does reach H is dropped as `reaches-held` and reported (D1).
- **Verification:** staging is checked against the inherited H, with every copied ref as a root.
- **Config:** for a fork, an allow-list from the old config: `forge.parent`, `http.receivepack` when it is a boolean, and `uploadpack.hideRefs`/`transfer.hideRefs` entries equal to `refs/forge/`. There is no served config. The marker is written only if the old fork had one.
- **Followed fork:** mirror config and marker; roots that reach the inherited H are dropped first, then B1's mirror selection runs on the surviving roots.
- **Report:** `.held` lists the inherited held objects actually left behind.
- **Unchanged:** grafts and replacements stay off for walks and the push.

**Item 4: locks.** A fork's lock is `lock_path`, which already accepts `valid_name_segment` owners. The fork pass runs after the whole mirror pass, and each `migrate_store` takes and releases its own lock, so two locks are never held at once.

**Item 5: partial forks.** `repos/<owner>/.fork-*.tmp` directories at least 3600 s old (by mtime) are renamed to `mirror-recovery/<owner>/<entry>.<ts>`, with directory fsyncs. Errors are logged only, because this is cleanup.

**Item 6: fork creation against a journal.**
- **Source:** refused by `resolve_repo` → `unpublished_mirror` → `journal_gates` (503). Shown by `b2_a_fork_of_a_journalled_source_is_refused`.
- **Target:** refused by B1's `journal_exists` check (503). Shown by B1's `f12_a_fork_onto_a_journalled_name_is_refused`.
- **A journal appearing mid-fork, within one instance:** migration runs to completion inside `ForgeService::new` before any request is handled, and the engine never writes migration journals, so no journal can appear while `h_api_fork` runs. I did not add a re-check between `build_fork` and the rename. Several instances are out of scope (decision 32).

## Mutations (each caught; file restored and checked by sha256; test green again)

`mutate.py`, `mutation-M*.txt`, `mutation-M*-restored.txt`, and `mutation-summary.txt` (which joins `mutation-summary-M1-M8.txt` and `mutation-summary-M9-M10.txt`; M9 and M10 were run on their own with `mutate.py M9-… M10-…`). For each mutation the driver copies the file, mutates it, runs the test, restores it with `cp` and `touch`, checks the sha256, and re-runs the test. The final gates ran after the mutations.

| Mutation | Security fix | Test | Red message |
|---|---|---|---|
| M1: the trigger streams `.held` instead of `.dropped` | trigger (D, not H) | 7b | `X1 served by the fork … -> 200` |
| M2: the fork ref filter never drops | inherited-held filter | 8 | `republished kept` |
| M3: lineage stops at the direct parent | transitivity | 7 | `held commit served by the fork of a fork` |
| M4: the partial-fork sweep is not called | tmp sweep | 19 | `the stale partial fork is still under repos/` |
| M5: the incomplete-pass fork rule is off | incomplete pass gates forks | unreadable config | `the fork is not gated` (200, not 503) |
| M6: a fork with no D object is rebuilt anyway | untouched guard | untouched | `the clean fork was moved to recovery` |
| M7: a followed fork's roots are not filtered | followed fork | followed | `the held branch was kept` |
| M8: an ancestor's journal does not gate its forks | unmigrated ancestor | unmigrated parent | `held commit served by the fork of an unmigrated parent` |
| M9: a gated plan with a journal still rebuilds | gated-plan guard (review round) | journalled fork, gated ancestor | `held commit served by the fork` |
| M10: `rebuilt_against` always answers no | no re-rebuild of a risk-4 fork | stamping blocked | `boot 2 rebuilt the fork` |

## Defect found while implementing

The first green run failed test 6 with a fork journal reason of `push into staging failed`. The fork's allow-listed `transfer.hideRefs = refs/forge/` was in staging's config before the push, and receive-pack refuses updates to a hidden ref, so the PR head `refs/forge/heads/dave/x` was rejected. Control C13 showed only that hides in the *source* do not block a push; a hide in the *target* does. **Fix:** a fork's config is now written after the push. Test 6 also asserts that both hides survive.

## Deviations from the brief and the design

1. **Decision 28, extended:** an incomplete pass now also refuses every **fork**: any repository whose `config` names a `forge.parent` or cannot be read. This is in `unpublished_mirror`, and runs only while `migration_incomplete` is set. Decision 28 gated only 64-hex owners, but forks live under any owner, and the brief's unreadable-config test ("incomplete, and it must gate") cannot pass without it.
2. **B1 behaviour change:** a mirror-class rebuild now keeps a valid `forge.parent` from the old config. Without this a followed fork loses its lineage once the mirror pass rebuilds it, and its descendants are never swept. A config that cannot be read now fails that mirror-class migration, which stays gated behind its journal.
3. **More fail-closed gates than the brief lists.** A fork is gated (journal reason `detection-error: …`, re-examined every boot through `fork::detect`) when:
   - its lineage passes a 64-hex store still behind a journal;
   - an ancestor's report, `.held` or `.dropped` is missing or unreadable;
   - an ancestor's lineage could not be read;
   - the inherited H passes `MAX_HELD_IDS`.
4. **Orphaned lineage:** a fork whose chain reaches a store that is gone is checked against **every** mirror-class report (and filtered against all their held sets), rather than left as a residual. This is the advisor's option (a).
5. **Stamping is conservative.** If any fork is gated before its ancestors are known, no report is stamped on that boot.
6. **Three tests are guards, not reds.** The untouched-fork guard's red is mutation M6, and the blocked-stamping test's is M10. The journalled-source test confirms B1's existing refusal.
7. **Report shape:** fork reports carry `class: "fork"`, `parent`, `followed` and `ancestors` (the ancestor report ids). Only `class: "mirror"` reports feed the trigger and the filter.
8. **`migrate::verify`** is kept as a `#[cfg(test)]` wrapper over the new `verify_with_roots`, so B1's tests are unchanged.
9. **`forksCleanAt`, not `forks_clean_at`.** The brief and design name the stamp `forks_clean_at`; I used camelCase to match every other report key (`droppedObjects`, `bytesBase64`).
10. **Decision 30, extended to forks.** A fork whose rebuilt copy exceeds 512 MiB fails, like any other migration failure, and stays gated (503) behind its journal. This is in the CHANGELOG.
11. **A gated plan never rebuilds** (review round). This is in `migrate_store`, just before `rebuild`, and applies to any fork whose plan carries a gate, whatever its journal says. Resuming a fork whose rebuilt store is already in place (`served_is_ours` → `publish`) is unaffected.

## Removed or changed assertions

None. `after-b2.patch` removes no `assert`, `expect` or `panic` line, and no B1 test changed.

## Out of scope (B3, recorded only)

- operator recourse for stores that stay gated;
- the engine's race between process exit and the size check;
- ancestor-directory fsync;
- descendant-process cleanup.

Also carried: several instances sharing a plugin directory (decision 32), so there is no mid-fork journal re-check, and risk 4 (accepted, decision 25), which the two rebuilt-fork tests pin as behaviour.

New residuals, recorded and not fixed:
- **Pruned parent.** If a legacy parent had already garbage-collected an object such as X1 before migration, that object is in neither D nor H. A fork still holding it in a hardlinked pack is then rebuilt only if it also holds some other D object. This is in the CHANGELOG.
- **Stale roots from a followed fork.** A followed fork's mirror report lists the roots of its mirror-class rebuild, and the fork pass may later drop one of them for reaching the inherited H. In a descendant's `--not` list that stale root can shield held objects from the descendant's filter. Verification still checks staging against H, so the descendant's migration fails closed and stays gated, not leaked.

## Evidence index (`b2/`)

| File | What it holds |
|---|---|
| `after-tests-red.patch`, `after-b2.patch` | Tree snapshots against `8ba3372` |
| `red.txt`, `red-summary.txt` | Reds |
| `green1.txt` | The hideRefs defect |
| `after-tests-red-2.patch`, `red-2a.txt`, `red-2b.txt`, `green-2.txt` | Review round |
| `gates-round1.txt`, `gates-round1.txt.patch` | First full gate run |
| `green.txt`, `full-run.txt`, `full-runfeaturesmirror.txt` | Green runs |
| `mutate.py`, `mutation-*.txt`, `mutation-summary.txt` | Mutations |
| `gates.txt`, `gates.txt.patch` | Gates |
| `SHA256SUMS` | Checksums |
