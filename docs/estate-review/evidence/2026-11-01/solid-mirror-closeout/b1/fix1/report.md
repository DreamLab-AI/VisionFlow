# Commit B1 fix round 1 report (SP-GIT-02)

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, `HEAD` `d6133e7`. Nothing is committed, and no worktree was created or removed.
- **Final patch:** `after-fix1.patch` (= `gates.txt.patch`, sha256 `433a55b65b8e5c4f…`): 10 files, +5425 / −576 against `d6133e7`.
- **Interdiff against `b1/after-c.patch`:** `interdiff-vs-after-c.patch`. It is a real source interdiff: `HEAD` was exported with `git archive`, `after-c.patch` applied to the export in scratch, and each file compared with the working tree using `git diff --no-index`.

| File | Change |
|---|---|
| `migrate.rs` | +1185 / −382 |
| `migrate_tests.rs` | +846 / −6 |
| `lib.rs` | +63 / −37 |
| `forks.rs` | +7 |
| `mirror.rs` | +4 / −4 (`MAX_REPO_BYTES` hoisted; t13 fixture line) |
| `router.rs` | `GIT_SUFFIXES` made `pub(crate)` |
| `CHANGELOG.md` | +92 / −35 |

- **New files:** none since B1. The four B1 files are still intent-to-add (`git add -N`), and no content is staged.
- **Environment:** every cargo command used `b1/cargo.sh` (`CARGO_HOME=/home/devuser/workspace/.cargo`, `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`, `CARGO_BUILD_JOBS=4`), with `--locked --offline`, one command at a time.

## Gates (final tree): all nine exit 0

`gates.txt`, run with `OUT=b1/fix1/gates.txt bash b1/gates.sh`:

| # | Gate | Exit | Tests |
|---|---|---|---|
| 1 | `cargo fmt --all --check` | 0 | |
| 2 | forge `--features mirror` | 0 | lib 383 passed, plus the integration and doc suites |
| 3 | forge (default features) | 0 | lib 262 passed |
| 4 | forge `--features mirror,anchoring,announce` | 0 | lib 444 passed |
| 5 | clippy forge (all features) `-D warnings` | 0 | |
| 6 | server `--features forge-full forge` | 0 | includes both `forge_migration_runtime` tests |
| 7 | clippy forge `--features mirror` `-D warnings` | 0 | |
| 8 | clippy server `--features forge-full` `-D warnings` | 0 | |
| 9 | `cargo doc` forge `--features mirror` (`RUSTDOCFLAGS=-D warnings`) | 0 | |

## Order of work (tests first)

1. **Seams first, with no behaviour change** (`after-seams.patch`). I added a `#[cfg(test)]` `test_hooks` registry, keyed by the absolute plugin path, because tests run in parallel and environment variables are process-wide. Through it a test can shorten the lock wait, store budget or size cap, abort mid-build, panic at boot, or run a slow child. The only environment variable signals a child process (f6), as t16 already does. In the same step I hoisted `MAX_REPO_BYTES` into `migrate`; `mirror` re-exports it with `pub use`. The migrate tests stayed green on both feature sets.
2. **Tests, captured red on the pre-fix tree** (`after-tests-red.patch`, `red-mirror.txt`, `red-mirror-summary.txt`, `red-f5-resumed.txt`):
   - All 16 new tests failed, each on an assertion about behaviour.
   - Three tests (f1 C17, f4, f11) first failed on an incidental assertion. I reordered them so the behavioural check runs first, and re-captured.
   - f5 loops over three crash states and stops at the first failure. So I also captured the two resumed-path cases separately, with a temporary reorder that was restored with `cp`, `touch` and a sha check. They show H2 exactly: the store was rebuilt from served, the report was overwritten, then `the old store was not moved: Directory not empty (os error 39)`.
   - The B1 migration tests were red at the same time, but only on the new `migration-id` assertion in `assert_migrated`.
3. **Fixes.** All 32 migrate tests passed on the first run (`run1.txt`), as did the full forge suite under both feature sets (`full-run1*.txt`). Then the mutations, then a first gate run (`gates-round1.txt`, all nine exit 0).
4. **Second round, after the advisor's review of the finished round.** Three more confidentiality gaps came up, each confirmed before fixing. Tests first again: `after-tests-red-r2.patch` and `red-r2.txt` (all three red on behaviour), then `green-r2.txt`. Mutations: `mutation-summary-r2.txt`. Then the final gates (`gates.txt`).
   - **Replacement refs widen the roots' closure** (item 1). Probe: with an owner-planted `refs/replace/<m2>` whose tree adds the held blob, `rev-list --objects <m2>` lists the held blob; with `GIT_NO_REPLACE_OBJECTS=1` it does not, and `push` sends the originals. `refs/misc/blob` then passed the filter and both verify checks. Fix: the migration-only runners (`for_each_line`, `run_with_input`, through the new `migration_command`) set `GIT_NO_REPLACE_OBJECTS=1`; the engine is untouched. Test `f1_a_replace_ref_does_not_widen_the_roots` (red: `a replacement smuggled the held blob into the rebuilt store`); mutation M1c is caught.
   - **A regular-file `published` journal was finished without checking the served store** (Codex #1, second half). On a lost-record store this removed the journal and served the held objects for the rest of the boot. Fix: only `served_is_ours` (marked, with the journal's migration id) leads to publish and finish, whatever the state. A `published` or `moved` journal counts as moved even when the `moved` field is missing, so a store without a match fails closed. Test `f3_a_forged_published_journal_does_not_open_a_store` (red: held served); mutation M3c is caught. The t13 fixture gains a `migration-id` file matching its journal id; see changed tests.
   - **A served store behind a symlink was skipped** (a regression from this round). I had changed `is_dir` to `symlink_metadata`, but `repo::list_owner` and the readers follow symlinks, so a record-less legacy store symlinked to another disk was skipped by detection and served. Fix: `is_dir` follows symlinks again, as in `after-c`. Test `f2_a_symlinked_served_store_is_detected` (red: held served); mutation M11 is caught.

## Must fix

| # | Finding | Fix | Test (red, then green) | Mutation (caught) |
|---|---|---|---|---|
| 1 | Codex #5, F2, C17 | Roots R are the copied `refs/heads/*` and `refs/tags/*`. H is closure(all refs) minus closure(R). A non-root ref is copied only if it reaches nothing in H, otherwise dropped as `outside-roots`. Verify checks H unconditionally and adds a check that every staging object lies in closure(R) in the source | `f1_lost_record_store_without_refs_mirror_copies_roots_only` (red: held served), `f1_a_dropped_tag_does_not_shield_held_objects` (red: held served), `f1_force_pushed_away_held_history_is_not_copied` (red: X1's blob served, 200). All C15-hard. t4 gains a verify call with an empty H | M1a: H built from `--glob=refs/mirror` turns all three f1 tests red. M1b: the outside-roots check disabled turns t4 red |
| 2 | F1, M1, Codex #2 | Signals are evaluated independently: any positive means rebuild. With no positive and any error, a `failed` journal is written: `reason: detection-error: <redacted>`, `attempts: 0`. A later boot that finds the store clean removes only that kind of journal. If the runtime fails, the thread panics, or a store cannot be gated by a journal, `run_at_boot` returns `false`, the new `ForgeService.migration_incomplete` is set, and `unpublished_mirror` refuses every unmarked store under a 64-hex owner (logged at error) | `f2_a_detection_error_gates_and_a_clean_recheck_ungates`: `objects/info` at mode 000, with a precondition assert that `EACCES` really occurs. Red: served 200. It also checks that a journal with another reason is kept. `f2_a_panicked_migration_gates_unmarked_mirror_class_stores` (red: held served) | M2a: detection error mapped to `None` is caught. M2b: flag ignored is caught |
| 3 | Codex #1, L1 | `read_journal` uses `symlink_metadata`; any non-regular file or any error other than `NotFound` is `Bad` and gates. The file is read with `read_regular` (lstat, open, fstat, same dev/inode), which also covers record, report and migration-id reads. Containment: `recovery` must be exactly `mirror-recovery/<owner>/<name>.git.<digits>` and `staging` exactly `mirror-work/staging/<owner>/<name>.git.<alnum>`, both as plain components, and the existing parent must canonicalise to the same place under the canonical plugin dir. Otherwise the migration fails closed (journal left, store gated) | `f3_a_symlinked_journal_gates`: a dangling symlink and a symlink to a `published` journal both give 503, before and after a boot that leaves the symlink in place. Red: 200. `f3_a_journal_recovery_outside_mirror_recovery_fails_closed` (red: wrote to `plugin/escape`) | M3a: a bad journal opening the gate is caught. M3b: containment bypassed is caught |
| 4 | Codex #4, L6, F4 | `smart_http_path` accepts only `/<owner>/<name>[.git]<suffix>` with both segments passing `valid_name_segment` (no `/`, `%` or leading `.`); anything else is 404 before the CGI. The CGI is always given `/<owner>/<name>.git<suffix>`, so it opens the directory the gates check | `f4_smart_http_accepts_one_path_shape`: nested `repos/bob/dir/x`, nested `dir/y.git`, suffix-less bare `repos/bob/plain`, a dot-led segment and `%2F` are refused. Positive controls: `real.git` and suffix-less `real`. Red: `/forge/bob/dir/x/info/refs -> 200` advertising the nested repo | M4: nested path passed to the CGI is caught |
| 5 | H1, H2, Codex `Finish` | Staging gets `migration-id` (the journal id) before the marker. On every boot the journal is read; if it is `published`, or served is marked and carries this journal's id, the migration is published and finished, never rebuilt (this check runs before the attempt limit). `moved` is stored in the journal, so once the old store has moved it is read only from recovery. Served present while recovery is also present is an intruder (fail closed). `moved` set with recovery gone fails. Both paths write `Moved` before the second rename. A report is never overwritten: an existing report with the same id stands (resume), one with another id fails closed, and a new one is created via `hard_link`. `Detected::Finish` requires a report whose `id` equals served's `migration-id`; a marked, flagged store without that binding is rebuilt | `f5_a_crash_after_the_second_rename_is_finished_not_rebuilt` over (`moved`, true), (`verified`, true) and (`verified`, false), each built from a real migration and then rewound. Asserts: journal gone, report bytes unchanged, one recovery entry, flag cleared. `f5_a_stale_report_does_not_finish_a_substituted_store` (red: held served) | M5a: the served-is-ours check removed is caught. M5b: an unbound finish is caught |
| 6 | F3 | `rebuild` increments and persists `attempts` before building; `fail()` no longer increments. `MAX_ATTEMPTS` applies to every state except a store already in place | `f6_three_crashes_mid_build_then_the_boot_gives_up`: three child processes abort after the `building` write, then the fourth boot gives up with attempts 3, the store gated and unmarked. Red: the fourth boot migrated. t21 still passes (attempts 1, then 3) | M6 is caught |
| 7 | F7, L3, Codex #6 | `take_lock` polls `File::try_lock` up to `STORE_BUDGET` (600 s). On timeout a gate journal is created exclusively and the store is skipped | `f7_a_held_lock_times_out_and_gates`: `new` runs on a thread and is checked with `recv_timeout`. Expects a journal mentioning the lock and the store gated; once the lock is released, the next boot migrates. Red: `waited on the lock past its deadline: Timeout` | M7 is caught |
| 8 | Codex durability | `write_journal`, report, `.held`, `.dropped`, record rewrite, `migration-id` and marker: write, `sync_all`, rename or `hard_link`, then `fsync` of the directory. The recovery and `repos/<owner>` directories are `fsync`ed after each rename | None. No process-level test can observe `fsync` without a power-cut harness, so no mutation is claimed | n/a |
| 9 | L4, Codex | A `tokio::task_local!` registry of process-group leaders is noted on spawn by every migration runner (`run_output`, `for_each_line`, `run_with_input`) and forgotten on normal completion. On a budget timeout, every group still registered is killed before staging is removed. The engine runs outside the scope, so it is unaffected | `f9_a_budget_timeout_kills_the_whole_process_group`: a 2 s budget and a git alias that backgrounds `sleep 41.4142135`. Red: `orphaned helpers survived` | M9 is caught |
| 10 | Codex #2 (size) | The push into staging runs `run_output` with cap `(limits.size_cap = MAX_REPO_BYTES, sizecap::dir_size)`, watching staging | `f10_a_staging_store_past_the_size_cap_fails_closed` (cap hooked to 1 byte). Red: migrated past the cap | M10 is caught |

**Mutation evidence.** `mutate.py` runs each mutation as copy, mutate, test, restore with `cp`, `touch`, and a sha256 check. Output is in `mutation-*.txt`, summarised in `mutation-summary.txt` (13 mutations) and `mutation-summary-r2.txt` (3: M1c, M3c, M11). All 16 mutations were caught, and every file was restored with a matching sha256. The final gates ran after the mutations.

## Should fix

| # | Fix | Evidence |
|---|---|---|
| 11 (L5) | Ref lines are parsed as bytes. A name that is not UTF-8 is dropped as `unsafe-name` (reported under its lossy name) and never copied | `f11_a_non_utf8_ref_is_dropped_not_renamed` (red: the store was migrated with the ref under a lossy name) |
| 12 (F6) | `h_api_fork` returns 503 (`Busy`) when `journal_exists` for the destination | `f12_a_fork_onto_a_journalled_name_is_refused` (red: 201) |
| 13 (L2) | Both lock-less and detection-error gate journals go through `create_exclusive`: a temp file, `fsync`, then `hard_link`, which fails with `EEXIST` and never replaces | Covered by f2 and f7 (gate journals written). No separate race test |
| 14 (L7) | CHANGELOG states roots-only copying, `outside-roots`, that notes are not served, that detection errors gate, the incomplete-pass gate, the 512 MiB cap, crash recovery, and that forks are not swept until B2. A's stale "not migrated yet" sentence is replaced with "migrated at boot (part 2)". The `migrate.rs` module docs are rewritten to match | Gate 9 clean |

## Changed test assertions (for the removed-assert grep)

1. **t1 (`notes`):** the assertion that the note survives in served was replaced. Item 1 forces this: a notes commit is never reachable from a branch or tag, so a notes ref is always outside the roots. The test now asserts that `refs/notes/` is absent from served, that the report lists `refs/notes/commits` as `outside-roots`, and that the note is still readable in the recovery copy. Widening R to `refs/notes/*` was rejected, because a note can point at any blob, which recreates F2.
2. **t4:** `refs/misc/blob` and `refs/misc/stash-held` are expected as `outside-roots`, not `reaches-held`; the reason was renamed by item 1. t4 also gains the empty-H verify assertion.
3. **`assert_migrated`:** gains an assertion that served's `migration-id` equals the report's `id`. This is an addition.
4. **t13 (`mirror.rs`, `migration_b1`):** the hand-built fixture gains one line, a `migration-id` file reading `t13`, its journal's id. A `published` journal is now bound to the store it names (Codex #1). No assertion changed.
5. **Removed against `d6133e7`:** still only the four doctest asserts that B1 moved verbatim into `mirror_record.rs`. No other assert was removed.

## Deviations and decisions for the lead

- **Notes are no longer served after migration.** This follows from item 1 and is stated in the CHANGELOG. Design test 1's "note readable" is now "note in recovery".
- **Suffix-less smart HTTP now opens `<name>.git`.** The CGI is handed `/<owner>/<name>.git/<suffix>`. Before, `/forge/<o>/<n>/info/refs` opened a bare `repos/<o>/<n>` directory, never the forge's `<n>.git`. A suffix-less push now auto-inits `<n>.git`, not `<n>`. No test in either crate used the old behaviour.
- **Legacy stores over 512 MiB now fail closed.** Item 10 caps staging at `MAX_REPO_BYTES`, so such a store stays gated (503) and needs an operator.
- **A marked store whose record is still flagged, with no report bound to its migration id, is rebuilt.** It is not left alone. That is the conservative reading of item 5.
- **The `Ungated` outcome.** If even the initial or gate journal cannot be written (Codex #2: unwritable migrations directory), the pass reports incomplete and the service-wide flag gates every unmarked 64-hex store. The "runtime could not be built" branch shares this flag path, but I could not force it in a test; the panic test and M2b cover the flag.
- **`first_line` and `write_marker`** are now used only by the engine, so they are gated `#[cfg(feature = "mirror")]`. Migration writes its marker through its own durable `seal()`. The engine's marker write is unchanged.
- **Not addressed (outside the brief):**
  - The code review's M2 note that t16 does not prove the lock. There is still no barrier or lock mutation.
  - t21 still asserts attempts after give-up but not the 503. f6 asserts the 503 after give-up.

## Residuals (known, fail closed)

- **Lock-timeout gate journals are never re-detected.** They are ordinary `failed` journals with 0 attempts, so the next boot rebuilds the store. Two consequences:
  - If a peer instance finished the store first (what is left of L2), its clean store is rebuilt once more and a second recovery entry made.
  - A plain owner store at a 64-hex owner that hit both a detection error and a lock timeout loses its re-detection, and is rebuilt (dropping refs outside the roots, kept in recovery) instead of being ungated.

  I did not widen the "remove only `detection-error`" rule, because the brief is explicit.

## Accepted, recorded only

- **F5:** an owner push in flight on another instance during migration can land in the recovery copy and be lost from what is served. Integrity of the owner's own push only; nothing held is exposed. Runbook step 0 (every instance stopped or running B) applies.
- **F7/L3:** boot time is bounded per store (lock wait 600 s, then build 600 s) but not in total. Stores run serially, and the lazy path holds `REGISTRY` inside `web::block` meanwhile. Documented in the CHANGELOG.
- **Operator runbook for an fsck give-up:** no recourse is built in (no `receive.fsck.<id>=ignore` constant). Deleting the journal of a lost-record legacy store serves its old objects, held ones included, until the next boot re-detects and gates it, so the server must be stopped first. Documented in the CHANGELOG.

## Evidence index (`b1/fix1/`)

| File | What it holds |
|---|---|
| `after-seams.patch`, `after-tests-red.patch`, `after-tests-red-r2.patch`, `after-fix1.patch` | Tree snapshots |
| `interdiff-vs-after-c.patch` | Source interdiff against `b1/after-c.patch` |
| `red-mirror.txt`, `red-mirror-summary.txt`, `red-f5-resumed.txt`, `red-r2.txt` | Pre-fix reds |
| `run1.txt`, `full-run1featuresmirror.txt`, `full-run1.txt`, `green-r2.txt` | First green runs |
| `mutate.py`, `mutation-*.txt`, `mutation-summary.txt`, `mutation-summary-r2.txt` | Mutations |
| `gates.txt` | Final nine gates (`gates-round1.txt` is the first, pre-second-round run) |
| `SHA256SUMS` | Checksums |
