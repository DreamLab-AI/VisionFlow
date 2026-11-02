# Commit B1 report (SP-GIT-02): hoist, decision 22, mirror-class migration

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, base `d6133e7`.
- **Nothing committed, staged, pushed, or worktree-changed.** I added the four new files with `git add -N` (intent to add, no content staged), so a plain `git diff` (the form `lead-gates-a8.sh` uses) includes them. `git status` lists them as `A`.
- **Final patch:** `after-c.patch` (8 files, +3670 / −554).
- **Interdiffs:**
  - `interdiff-a.patch`: (a) against `d6133e7`;
  - `interdiff-b.patch`: (b) against (a);
  - `interdiff-c.patch`: (c) against (b).
  - Each one is a diff between two consecutive `git diff` snapshots.
- **Build environment:** every cargo run went through `cargo.sh` with the brief's `CARGO_HOME`, `CARGO_TARGET_DIR` and `CARGO_BUILD_JOBS=4`, plus `--locked --offline`. No two cargo runs overlapped.
- **Gate script:** `gates.sh` is `lead-gates-a8.sh` with its output redirected into this folder. The commands are unchanged.

## Gate exits (all nine, each step)

| Gate | (a) `gates-a.txt` | (b) `gates-b.txt` | (c) `gates-c.txt` |
|---|---|---|---|
| 1 fmt --check | 0 | 0 | 0 |
| 2 forge test `mirror` | 0 | 0 | 0 (lib 364 passed) |
| 3 forge test default | 0 | 0 | 0 (lib 243 passed) |
| 4 forge test `mirror,anchoring,announce` | 0 | 0 | 0 (lib 425 passed) |
| 5 server test `forge-full` filter `forge` | 0 | 0 | 0 (includes the two new `forge_migration_*` tests) |
| 6 clippy forge `mirror` | 0 | 0 | 0 |
| 7 clippy server `forge-full` | 0 | 0 | 0 |
| 8 clippy forge `mirror,anchoring,announce` | 0 | 0 | 0 |
| 9 rustdoc `-D warnings` forge `mirror` | 0 | 0 | 0 |

- The first (a) run failed gate 1 only, on formatting (`gates-a-attempt1-fmt-red.txt`). I ran `cargo fmt` and re-ran all nine gates.
- **(c) ran twice.** `gates-c-round1.txt` was all green. I then hardened the t15 and server fixtures and fixed one doc comment, so `gates-c.txt` is the re-run on the final tree. That run is all green, and `after-c.patch` is its snapshot.
- **Extra check, not one of the nine:** clippy on forge with default features, `--all-targets -D warnings`, exits 0 (`extra-clippy-default.txt`).
- **Not run:** rustdoc without `mirror`. It is not a gate.

## Step (a): pure hoist

**New ungated `pub mod migrate`** (`src/migrate.rs`). Moved here unchanged:
- `RECEIVE_PACK` and `SERVED_CONFIG`, now `pub` (they were private consts in `mirror`);
- `hides_mirror_refs`;
- `init_bare` and `write_served_config`, the init and config half of `build_served`;
- the process-group runner: `run_in`, `run_output`, `first_line`, `kill_group`, `GIT_TIMED_OUT`, `GREW_PAST_CAP`, `DirMeasure`.

New in the module: `pub fn lock_path(plugin_dir, owner, name)`, which accepts any `valid_name_segment` owner and name, and `AWAITS_MIGRATION_KEY`.

**Engine side.** `MirrorStore::lock_file` keeps its own `is_hex64`/`is_repo_name` check and then delegates, so the engine accepts exactly what it did before. `GitCli`'s methods are now thin wrappers, and its test seams stay on `GitCli`.

**Record loading.**
- `MirrorRecord`, its nested types, `MirrorStore`, `StoredRecord`, `is_hex64` and `is_repo_name` moved into a new private ungated module, `src/mirror_record.rs`.
- `mirror` re-exports them, so their only public path is still `mirror::*`.
- Migration therefore uses the engine's own `MirrorStore::load`. One parser gives identical missing/unreadable/valid verdicts. A JSON record that is valid but lacks `events` is `Unreadable` to both.

**No behaviour change.** Every `mirror.rs` hunk in (a) lies above `mod tests` (line 3262 then). No existing `#[test]` changed in (a).

**Test 20** (`migrate_tests::t20_hoisted_constants_and_record_key_are_pinned`):
- **Red** (`red-a-t20.txt`): a compile error, as expected for a module that does not exist yet:
  - 9 × `cannot find 'migrate'`;
  - `cannot find 'mirror_record'`;
  - `lock_file is private`.
- **Pins:**
  - both constants, byte for byte;
  - the lock path for a 64-hex owner, plus the fork-owner and refusal cases;
  - with `mirror`, equality with `MirrorStore::lock_file`;
  - the `awaitsMigration` serde key, both ways.
- **Fix:** `migrate.rs`, `mirror_record.rs`, and the `lib.rs` module lines.

## Step (b): decision 22 (a Commit A defect)

**Test** `migrate_tests::t22_smart_http_refuses_a_dot_named_bare_repo`, ungated. It uses the real `git-http-backend`.

**Red** (`red-b-t22.txt`): `/forge/bob/.fork-x.tmp/info/refs -> 200`, and the body advertises the tmp repo's `refs/heads/main` SHA, which confirms the defect.
- Positive control: `bob/real.git` returns 200 and advertises its SHA.
- The same request with POST `git-upload-pack` and a want is also asserted.

**Fix** (`lib.rs` `h_git_smart`, marker-gate block):
- `Ok` or `NotFound` lets the request go on; a missing repository is still the CGI's to answer.
- `PathTraversal` returns `NotFound` (404).
- Any other error is returned.
- `ForgeError::status_code()` is unchanged: `PathTraversal` stays 400 on every other route.

**Green:** `green-b-t22.txt`.

**Recorded, not changed:**
- Raw `/forge/bob/.fork-x.tmp/raw/...` was already refused with 400 through `resolve_repo`.
- `GIT_SUFFIXES` holds only smart paths, so dumb HTTP never reaches `h_git_smart`.

## Step (c): mirror-class migration with decision 21

### Red per test, and where the fix is

**Where the red output is:**
- `red-c-mirror.txt` and `red-c-default.txt`, with one-line summaries in `*-summary.txt`;
- `red-c-server-t17.txt`;
- the red tree is `after-c-tests-red.patch`.

**Red scaffold.** For the red run, `migrate::verify` existed only as a stub returning `Ok(())`, so test 4 compiled and every red is an assertion. The stub was replaced by the real function.

**Fixtures.** Every fixture is C15-hard:
- the pack is kept by a `.keep` file;
- `refs/misc/stash-held` reaches the held commit;
- `refs/misc/blob` points straight at the held blob;
- the held commit re-adds an approved blob (C16).

The fixture also carries an annotated tag `v1`, a note, and a foreign PR head.

| Test | Red message (before the fix) | Where the fix is |
|---|---|---|
| 1 `t1_restart_migrates_a_record_bearing_legacy_store` | `approved raw: {"error":"busy: '…/sample' is a mirror awaiting migration…"}` (503) | `migrate::run_at_boot` and `rebuild`, `build`, `finish`; `ForgeService::new` calls it |
| 2/3 `t2_t3_lost_record_legacy_store_is_migrated` (ungated) | `held commit served: …/raw/<held>/held.txt -> 200 (held bytes: true)`, plus blob HTML 200. Red under `mirror` **and** under default features (test 3); this is decision 26's live exposure | `migrate::detect` (traces), `rebuild` |
| 4 `t4_direct_held_blob_ref_is_dropped_and_verify_catches_it` | `held commit served: … -> 200` | `build` (filter: `reaches_held`, `held_set`) and `migrate::verify` (object-level `present_count`) |
| 5 `t5_fixture_survives_delete_and_gc` | It pins fixture hardness, not migration, so it passes on the unfixed tree. Its red is a weak fixture (`red-c-t5-weak-fixture.txt`): with the `.keep` write and both `refs/misc/*` lines removed from `legacy()`, it fails with `the fixture is weak: delete-plus-gc removed the held blob`. The file was then restored, `touch`ed and its sha256 checked | — (fixture guard) |
| 9 `t9_owner_pushed_mirror_keeps_its_tag` | `approved raw: busy … awaiting migration` (503) | `build`: tags always copied |
| 10 `t10_mirror_pull_request_head_is_not_copied` | `approved raw: busy …` (503) | `build`: `refs/forge/heads/*` dropped, reason `pull-request-head`. PR pages, commits, files and API return < 500 afterwards |
| 11 `t11_unreadable_record_store_is_rebuilt_and_the_record_kept` | `approved raw: busy …` (503) | `detect` (`Unreadable` is a target); `write_report` (record bytes in base64); `finish` never touches an unreadable record |
| 12 `t12_crash_after_the_first_rename_resumes_and_materialise_refuses` (`mirror`) | `materialise ran while a migration journal exists` | `GitCli::refuse_while_migrating` in `materialise_inner` and `prepare`; `migrate_one` and `rebuild` resume `moved` from the recovery copy |
| 13 `t13_crash_after_publish_clears_the_flag_and_the_engine_follows` (`mirror`) | `flag not cleared: {… "awaitsMigration":true …}` | `migrate_one` on `Stage::Published` calls `finish` |
| 14 `t14_restart_is_idempotent` | `approved raw: busy …` (503) | `detect` returns `None` for a marked, unflagged store. The second `new` leaves the recovery entries and the served refs and objects unchanged |
| 15 `t15_redelivered_state_is_applied_after_migration` (`mirror`) | `the served repository predates publication isolation and awaits migration` | `finish` clears the flag; A's replay behaviour does the rest |
| 16 `t16_two_processes_migrate_once` | `approved raw: busy …` (503) | Two real child processes (the test binary re-spawned) run `new` on one plugin directory. Exactly one recovery entry, through `take_lock` plus a re-read and re-detect under the lock |
| 17 `t17_new_inside_a_current_thread_runtime` and `t17_…_multi_thread_runtime` (forge); `forge_migration_completes_inside_an_actix_system` and `…_actix_web_test` (server) | Forge: `held commit served … -> 200`. Server: `assertion left == right failed: b1-server-held-bytes` (held raw 200) | `run_at_boot`: a scoped `std::thread` that builds its own current-thread runtime and is joined |
| 18 `t18_smart_http_push_is_refused_while_a_journal_exists` | `push with a published journal  left: 401` (the push reached the CGI's auth) | `h_git_smart` write branch: `migrate::journal_exists`, 503. Reads with a `published` journal return 200; with `building` or `failed`, 503 |
| 21 (decision 21) `t21_failed_migration_keeps_the_store_gated` | `held commit served: …/raw/<held>/held.txt -> 200` | `unpublished_mirror` checks `migrate::journal_gates` first. `fail` writes `state: failed`, the reason (staging's `badTimezone` fsck error is in it) and `attempts`. `MAX_ATTEMPTS` = 3: a fourth boot stops retrying, and the test asserts `attempts == 3` after three more `new` calls |

**Fixtures hardened after the first red.**
- As first written, t15 and the server fixture were not C15-hard: t15 had no `.keep` pack and no non-mirror ref reaching held, and the server fixture had no `.keep`.
- Both now have the full shape. t15 also asserts that `refs/misc/*` is absent from served, and the server test that the held commit is absent from the served store.
- Their red was re-captured on the hardened fixtures with the boot migration switched off (`run_at_boot` commented out in `lib.rs`; restored, `touch`ed, sha256 checked): `red-c-hardened-boot-off.txt`. t15 fails with `…awaits migration`, and both server tests with held raw 200. Green: `green-c-hardened.txt`.
- **C15 does not apply to t13, t18 and t22**, which hold no held content: a clean published store, a plain owner store, and a dot-named bare repository.

**Decision 21 edits A's reader gate.** Name this hunk in every review brief: the `lib.rs` `ForgeService::unpublished_mirror` gains a journal check ahead of the marker check. It covers `resolve_repo`, `gated_git_dir` and `h_git_smart`.

### Mutation results

Method: `migrate.rs` copied to scratch, mutated, the named tests run, the file restored with `cp`, then `touch`ed and its sha256 checked. All mutated tests are green again afterwards (`mutation-restored-green.txt`).

| Mutation | File | Result |
|---|---|---|
| M1: verification off (`verify` returns `Ok` at once) | `mutation-M1-verify-off.txt` | **t4 red:** `verification passed a staging store holding a held blob: Ok(())`. t1 stays green, as expected: the ref filter keeps held objects out on that path |
| M2: journal gate off (`journal_gates` returns false) | `mutation-M2-journal-gate-off.txt` | **t21 red:** `held commit served: …/raw/<held>/held.txt -> 200` |
| M3: tombstone off (`journal_exists` returns false) | `mutation-M3-tombstone-off.txt` | **t18 red:** `push with a published journal left: 401`. **t12 red:** `materialise ran while a migration journal exists` |

### Existing tests changed

These five tests built a legacy shape and *then* called `forge_at`. Migration now rebuilds such a store, so they would test B rather than A's gate.
- In each one I moved only the construction line: `let forge = forge_at(&plugin);` now comes before the legacy shape is made. A two-line comment explains why.
- **No assertion was removed or changed.**
- In the compare test, the inline `forge_at(&plugin).handle(` became `forge.handle(`, with the `let` moved above.

The tests:
1. `fix_a::a_legacy_record_store_with_an_unreachable_object_awaits_migration`
2. `fix_a::an_unreadable_record_awaits_migration_and_is_left_in_place`
3. `fix_a::an_unmarked_mirror_is_refused_on_its_summary_and_left_out_of_the_listing`
4. `fix_a::a_compare_never_reads_a_head_from_an_unmarked_mirror`
5. `fix_a::a_dangling_record_symlink_keeps_the_gate_closed`

All five still assert A's 503 path. Their failures before the move are in `c-full-run1.txt`.

**Removed asserts.** Against `d6133e7`, the patch removes four assert lines, all in doctests: the `MirrorStore::read` and `MirrorStore::load` examples. They moved verbatim into `mirror_record.rs`. Step (c) removes no assert. (`interdiff-c.patch` is a diff of two patches, so lines whose hunk context shifted appear there as `-+`.)

**Doctests.** The examples on `MirrorStore::read` and `MirrorStore::load` are now wrapped in a hidden `# #[cfg(feature = "mirror")] { … # }` block, with their content unchanged. Doctests of a private module run under every feature set, and these examples name the feature-gated public path.

## Deviations from the design, with reasons

1. **More was hoisted than the list.** The record *types* moved, not only `load`, because migration's verdict must use the engine's parser (see step (a)). `write_marker` moved in (c), and `mirror` delegates to it. A shared `git_command` builder now backs the runner, with the same flags.
2. **Ref names are checked with a pure `safe_ref` function, not by spawning `git check-ref-format` per ref.**
   - The names come from `for-each-ref`, so git has already accepted them.
   - `safe_ref` refuses `:`, `+`-style modifiers, glob characters, `..`, `@{`, control characters and the rest.
   - A ref that fails it is dropped and reported as `unsafe-name`, never pushed.
3. **Resume from `moved` with staging present rebuilds from the recovery copy** instead of re-verifying the old staging. This is simpler and never trusts a partial staging store; the design allows either.
4. **The lock-failure gate journal** is written as `failed` with `attempts: 0`, so an environmental lock failure does not use up retries.
5. **Detection errors leave the store as it is.** If git fails on the trace check, the store is not gated by a new journal, and a warning is logged. Gating on a transient error would put a plain owner store at 503 for good after 3 boots.
6. **Memory.** The held set is capped at `MAX_HELD_IDS` = 16 M, and a store past the cap fails closed. The `.dropped` computation holds staging's object ids, and then the dropped ids, in memory with no cap (risk 2: measure later).
7. **`Detected::Finish`** (marked, flagged, report present) requires only that a `.migration.json` exist. The report is written only after verification passes, and the marker on served is written only by migration or the engine.
8. **Test 1's smart-HTTP check is fetch-by-want only.** It does not clone and then run `cat-file`. The object-store assertion (`cat-file -e <held>` fails in served) is the stronger proof (decision 15).
9. **Test 16 lives in the forge crate**: a self-spawning lib test with two real processes, not a server integration test. It therefore runs in gates 2 to 4, and gate 5's filter does not apply to it.
10. **CHANGELOG.** Part 2 is added under Security, and the new module under Changed. Part 1's sentence "Repositories mirrored before this change are not migrated yet…" was left as A wrote it, because it is A's entry. The lead may want to reword it, since forks are still not swept.

## Observations for the lead

- **Operator recourse after 3 failed attempts is not implemented.** Design §6's fsck runbook (a second fixed receive-pack constant with `receive.fsck.<id>=ignore`) is not in B1. The store stays at 503 behind its `failed` journal. Deleting that journal on a lost-record store ungates it, and the old store with its held objects is served, until the next boot re-detects and retries. The operator must stop the server before deleting a journal.

- **`h_git_smart` checks only the first two path segments.**
  - A bare repository nested deeper, such as `repos/<owner>/<dir>/x`, would still reach the CGI through `/forge/<owner>/<dir>/x/info/refs`.
  - No forge code creates such a directory: `.fork-*.tmp` sits at depth 2.
  - Requiring the exact shape `/<owner>/<name>[.git]/<suffix>` would close it. That is outside decision 22.
- **Boot cost.**
  - Every `ForgeService::new` now spawns one thread and one runtime.
  - It stats each repository's marker and record.
  - For each unmarked 64-hex store with no record signal, it spawns two `git` processes (`for-each-ref`, `config --list`).
  - The test suite's wall-clock barely moved: lib tests take about 5 s with `mirror`.
- **Out of scope (B2):** forks (tests 6–8, 7b, 19), the transitive sweep, the `.dropped` trigger, the `.fork-*.tmp` sweep, and `forks_clean_at`. The `.held` and `.dropped` files B2 consumes are written: one id per line, sorted.

## Evidence index

| What | Files |
|---|---|
| Red and green | `red-a-t20.txt`, `red-b-t22.txt`, `green-b-t22.txt`, `red-c-mirror.txt`, `red-c-default.txt`, `red-c-*-summary.txt`, `red-c-server-t17.txt` |
| Snapshots and interdiffs | `after-a.patch`, `after-b.patch`, `after-c-tests-red.patch`, `after-c.patch`, `interdiff-{a,b,c}.patch` |
| Gates | `gates-{a,b,c}.txt` (each with its `.patch` snapshot), `gates-a-attempt1-fmt-red.txt`, `extra-clippy-default.txt` |
| Mutations | `mutation-M{1,2,3}-*.txt`, `mutation-restored-green.txt` |
| Intermediate runs | `c-run1.txt`, `c-full-run1.txt` (the five A tests failing before their construction line moved), `c-full-run2.txt` |
| Checksums | `SHA256SUMS` |
