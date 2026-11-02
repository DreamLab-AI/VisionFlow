# Commit B1 fix round 2 report (SP-GIT-02)

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, `HEAD` `d6133e7`. Nothing is committed and nothing is staged (the four B1 new files are still intent-to-add only). No worktree was created or removed.
- **Final patch:** `after-fix2.patch` (= `gates.txt.patch`, sha256 `e566725522904983…`): 10 files, +5836 / −576 against `d6133e7`.
- **Interdiff against `b1/fix1/after-fix1.patch`:** `interdiff-vs-after-fix1.patch`. It is a source interdiff: `HEAD` exported with `git archive`, `after-fix1.patch` applied in scratch, and each file compared with `git diff --no-index`.

| File | Change since fix 1 |
|---|---|
| `migrate.rs` | +243 / −64 |
| `migrate_tests.rs` | +219 / −7 |
| `lib.rs` | +25 / −19 |
| `CHANGELOG.md` | +36 / −22 |

- **Environment:** every cargo command used `b1/cargo.sh` (`CARGO_HOME=/home/devuser/workspace/.cargo`, `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`, `CARGO_BUILD_JOBS=4`), with `--locked --offline`, one command at a time.

## Gates (final tree): all nine exit 0

`gates.txt`, run with `OUT=b1/fix2/gates.txt bash b1/gates.sh`:

| # | Gate | Exit | Tests |
|---|---|---|---|
| 1 | `cargo fmt --all --check` | 0 | |
| 2 | forge `--features mirror` | 0 | lib 389 passed |
| 3 | forge (default features) | 0 | lib 268 passed |
| 4 | forge `--features mirror,anchoring,announce` | 0 | lib 450 passed |
| 5 | clippy forge (all features) `-D warnings` | 0 | |
| 6 | server `--features forge-full forge` | 0 | includes both `forge_migration_runtime` tests |
| 7 | clippy forge `--features mirror` `-D warnings` | 0 | |
| 8 | clippy server `--features forge-full` `-D warnings` | 0 | |
| 9 | `cargo doc` forge `--features mirror` (`RUSTDOCFLAGS=-D warnings`) | 0 | |

No `FAILED` line appears in `gates.txt`.

## Order of work

1. **Probe first.** In a scratch store, an `info/grafts` line naming a held commit as an extra parent of an approved one put the held blob into `rev-list --objects <approved>`. A plain `git push` of the approved commit then sent the held blob. With `GIT_GRAFT_FILE=/dev/null`, both stopped, and git accepted the setting. So the env var has to reach the push as well as the walks.
2. **Tests, captured red on the fix-1 tree** (`after-tests-red.patch`, `red.txt`, `red-summary.txt`). All seven were red on behaviour (six new tests and t18's changed row):

| Test | Pre-fix red |
|---|---|
| `g1_a_published_journal_gates_across_boots` | `boot 1: held commit served` |
| `g2_an_incomplete_pass_gates_a_marked_store` | `held commit served` (the flag was set, but the marked store was open) |
| `g3_an_unlistable_owner_makes_the_pass_incomplete` | `held commit served` |
| `g4_a_failing_store_probe_leaves_the_detection_error_journal` | `a detection-error journal was removed without a clean examination` |
| `g5_grafts_do_not_widen_the_roots` | `a graft smuggled held history into the rebuilt store` |
| `g6_replace_refs_are_never_copied` | `a replace ref was copied` |
| t18, `published` row (now expects 503) | `read with a published journal` (200) |

3. **Fixes.** All seven tests then passed (`green.txt`). The full forge suites showed two knock-ons. f2's marked-store control is explained under changed tests; its failing run is kept in `knockon-full-run*.txt`. t21 lost `badTimezone` from its failure reason because git's graft-deprecation hints filled the four kept stderr lines; it is fixed under item 5. I deleted the output of that first run by mistake; the journal reason it showed was `push into staging failed: hint: Support for <GIT_DIR>/info/grafts is deprecated | …`. After those were fixed, the full suites passed under both feature sets (`full-run*.txt`). Then the mutations, then the gates.

## Must fix

| # | Fix | Test | Mutation (caught) |
|---|---|---|---|
| 1 | `journal_gates` returns `true` for anything at the journal path, `published` included. `finish` already removes the journal after rewriting the record, so a migrated store is 503 only between the `published` write and `finish` (accepted). I checked which paths could rely on an open `published` journal: the readers reach the journal only through `unpublished_mirror`, the engine's tombstone check (`journal_exists`) already treated every journal as present, and t18's `published` row was the only test that relied on it | g1: a lost-record legacy store with a regular `published` journal at `attempts: 3` and no matching id. It is gated after boot 1 and after boot 2 | N1: `published` treated as open turns g1 red |
| 2 | **The simpler option:** while `migration_incomplete` is set, `unpublished_mirror` refuses every store under a 64-hex owner, marked or not, with no record or report exception. The reader's marker test is now the same as detection's, a regular file at the store root (`symlink_metadata(..).is_file()`; it used to follow symlinks with `metadata`) | g2: a marked, flagged store with no bound report and a read-only `mirror-work/migrations` (precondition: `EACCES` on create). The flag is set, held is denied and approved is 503 | N2: the flag skipping marked stores turns g2 red |
| 3 | Migration has its own enumeration, `served_stores` and `journaled`, built on `names_in` and `dir_state`. Any error listing a directory, reading an entry, or checking an entry's type (other than `NotFound`) is logged at error level and marks the listing incomplete; `run` then returns `false`, so `migration_incomplete` is set. `repo::list_all` is no longer used by migration. Symlinked stores are still followed, as the readers follow them | g3: the owner directory at mode 0311, searchable but not listable (precondition: `read_dir` gets `EACCES` while `stat` of the store works). The flag is set, held is denied and approved is 503 | N3: listing errors ignored turns g3 red |
| 4 | `Detected::None` is split into `Skip` (out of scope, no store, or a marked store whose record is not flagged; nothing proven) and `Clean` (an unmarked store whose every signal was examined successfully and found negative). Only `Clean` removes a `detection-error` journal; `Skip` leaves it. `dir_state` tells absent (`NotFound`) from error, so a store-directory error is `Detected::Error`. In `migrate_one`, an error probing the served or recovery path returns `Failed` and leaves the journal untouched (it gates, by item 1). The remaining boolean `is_dir` counts an error as present, so it fails closed | g4: a `detection-error` journal on a plain 64-hex store, and a boot with the owner directory at mode 000 (precondition: the store probe gets `EACCES`). The journal stays | N4: a probe error read as `Clean` turns g4 red |
| 5 | `MIGRATION_ENV` = `GIT_NO_REPLACE_OBJECTS=1`, `GIT_GRAFT_FILE=/dev/null` and `GIT_ADVICE=0`. It is applied by `migration_command` (every walk and check) and by the push, through the new `run_output_env`; the engine's `run_output` is unchanged. `GIT_ADVICE=0`, and dropping `hint:` lines from the push error, stop git's graft-deprecation hints from crowding the real reason out of a failure (t21 caught this) | g5: a commit planted under `refs/misc/g5` (kept in a `.keep` pack, C15-hard) and grafted onto `m2` through `info/grafts`. Its blob is not copied, and the ref is dropped as `outside-roots` | N5: the graft file not neutralised turns g5 red |
| 6 | `refs/replace/*` is dropped as `outside-roots` without any reachability check, and reported | g6: `refs/replace/<held>` pointing at the approved `m2`. It is not copied and the report lists it | N6: the drop removed turns g6 red |

**Mutation evidence.** `mutate.py` is the fix-1 driver with this round's six mutations: copy, mutate, run the test, restore with `cp`, `touch`, sha256 check. Outputs are in `mutation-N*.txt`, summarised in `mutation-summary.txt`. All six were caught and every file was restored with a matching sha256. The final gates ran after the mutations.

## Should fix

- **Code N2:** the doc comment `/// The 503 every reader returns for an unmigrated mirror.` is back on `unpublished_mirror_error`; `smart_http_path` keeps only its own doc.
- **Security N6 (CHANGELOG precision):**
  - The held set is described as "every object any ref reaches that the roots do not". History that no ref reaches any more is kept out by the verification's outside-the-roots check, not by H.
  - The size cap is described as applying to the rebuilt, roots-only copy, polled every 500 ms while the push runs.
- **Other CHANGELOG changes:**
  - **Journals:** every journal gates, and a migrated store is briefly refused between `published` and `finish`.
  - **Incomplete pass:** it gates marked stores too, and it now covers enumeration errors.
  - **`detection-error` journals:** removed only after a full, clean examination.
  - **Walks and push:** grafts and replacements are ignored in both.
  - **`refs/replace/*`:** never copied.
  - **The migration id:** described as continuity bookkeeping, not an integrity check.
- **Module docs:** the `migrate.rs` docs, `journal_path`, `journal_gates`, `run`, `unpublished_mirror` and the `migration_incomplete` field doc are updated to match.

## Changed test assertions (for the removed-assert grep)

1. **t18, `published` row:** `read.status` was `200` for `published` and `503` otherwise; it is now `503` for every state. Item 1 requires this: a `published` journal now gates.
2. **f2_a_panicked control:** `marked store -> 200` is now `-> 503`, with the message `a marked mirror-class store is not refused`. Item 2 requires this: an incomplete pass refuses marked stores under a 64-hex owner too. The non-mirror-owner control (`bob/plain -> 200`) is unchanged.
3. **No other assertion was removed** against `after-fix1.patch`. The other `-` lines the grep finds in the interdiff are log or doc text containing "panicked" (`interdiff-vs-after-fix1.patch`).

## Deviations and notes for the lead

- **Item 2: I picked "all of them".** While the flag is set, every store under a 64-hex owner is 503, including clean, marked, migrated stores, until a restart completes the pass. This is more availability loss than the "valid record and bound report" exception would cause, and it needs no extra reads on the reader path.
- **Item 4: `Skip` keeps a `detection-error` journal for good** if the store is now marked but its record is not flagged, or the store directory is gone. Such a store stays 503 until an operator removes the journal. That fails closed, and it is what the brief asks ("leave the journal").
- **Item 5:** `GIT_GRAFT_FILE=/dev/null` was checked empirically in scratch on git 2.56.0. `/dev/null` was used because git accepted it; no file under `mirror-work/` was needed. A legacy `shallow` file is not neutralised. `receive-pack` refuses a shallow update into staging by default, so it fails closed (security review, "Shallow").
- **Not changed:** the engine's own marker checks (`mirror.rs` 1464 and 1911) still use `metadata`. They decide engine behaviour, not reader access, and the brief limits the marker alignment to readers and detection.

## Accepted, recorded only (not fixed, per the brief)

Each needs write access to the server user's filesystem, and an actor with that access can already serve anything:

- **Codex 3:** replaying the `migration-id`. The CHANGELOG now calls it continuity bookkeeping, not integrity.
- **Codex 4 and security N4:** containment and TOCTOU gaps in `read_regular`: a FIFO swap between `lstat` and `open`, and parent-directory symlinks.
- **Codex 6:** a recycled process-group ID in the budget-timeout kill.
- **Security N5:** an operator-relocated `mirror-recovery/` symlink. This fails closed.

Still recorded from fix 1: F5 (an owner push in flight across instances), F7/L3 (boot time bounded per store but not in total), the operator runbook for an fsck give-up, and lock-timeout gate journals not being re-detected.

## Evidence index (`b1/fix2/`)

| File | What it holds |
|---|---|
| `after-tests-red.patch`, `after-fix2.patch` | Tree snapshots |
| `interdiff-vs-after-fix1.patch` | Source interdiff |
| `red.txt`, `red-summary.txt` | Pre-fix reds |
| `green.txt`, `full-run.txt`, `full-runfeaturesmirror.txt` | Green runs |
| `knockon-full-run.txt`, `knockon-full-run-featuresmirror.txt` | The f2 knock-on before it was fixed (t21 had already been fixed by then) |
| `mutate.py`, `mutation-N*.txt`, `mutation-summary.txt` | Mutations |
| `gates.txt` | Final nine gates |
| `SHA256SUMS` | Checksums |
