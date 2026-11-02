# Commit B1 fix round 3 report (SP-GIT-02)

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, `HEAD` `d6133e7`. Nothing is committed or staged; the four B1 new files are still intent-to-add only. No worktree was created or removed.
- **Final patch:** `after-fix3.patch` (= `gates.txt.patch`, sha256 `97d0eb3cf2241039…`): 10 files, +5961 / −576 against `d6133e7`.
- **Interdiff against `b1/fix2/after-fix2.patch`:** `interdiff-vs-after-fix2.patch`, a source interdiff made the same way as in fix 2 (`git archive HEAD`, `after-fix2.patch` applied in scratch, then `git diff --no-index` per file).

| File | Change since fix 2 |
|---|---|
| `migrate.rs` | +60 / −13 |
| `migrate_tests.rs` | +68 / −0 |
| `CHANGELOG.md` | +12 / −2 |

  `lib.rs` and `mirror.rs` are unchanged.
- **Environment:** every cargo command used `b1/cargo.sh` (`CARGO_HOME=/home/devuser/workspace/.cargo`, `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`, `CARGO_BUILD_JOBS=4`), with `--locked --offline`, one at a time.

## Gates (final tree): all nine exit 0

`gates.txt`, run with `OUT=b1/fix3/gates.txt bash b1/gates.sh`. It contains no `FAILED` line.

| # | Gate | Exit | Tests |
|---|---|---|---|
| 1 | `cargo fmt --all --check` | 0 | |
| 2 | forge `--features mirror` | 0 | lib 391 passed |
| 3 | forge (default features) | 0 | lib 270 passed |
| 4 | forge `--features mirror,anchoring,announce` | 0 | lib 452 passed |
| 5 | clippy forge (all features) `-D warnings` | 0 | |
| 6 | server `--features forge-full forge` | 0 | |
| 7 | clippy forge `--features mirror` `-D warnings` | 0 | |
| 8 | clippy server `--features forge-full` `-D warnings` | 0 | |
| 9 | `cargo doc` forge `--features mirror` (`RUSTDOCFLAGS=-D warnings`) | 0 | |

## Order of work

1. **Test seam first, no behaviour change.** I added `Hooks::size_poll` and threaded it through `Limits` → `build` → `push_refs` → `run_output_env` as a `SizePoll` pair (delay before the first measurement, then the interval). In production the pair is the constant `SIZE_POLL = (0, 500 ms)`, the same timing as before. The engine's `run_output` passes `SIZE_POLL`, so the engine is unchanged.
2. **Tests, red on the fix-2 tree** (`after-tests-red.patch`, `red.txt`, `red-summary.txt`). Both are assertion failures:

| Test | Pre-fix red |
|---|---|
| `h1_a_recovery_probe_error_still_gates_the_store` | `no journal and a complete pass: nothing gates the store` |
| `h2_a_fast_push_past_the_cap_is_caught_by_the_final_check` | `the store was migrated past the size cap` |

3. **Fixes.** Both tests then passed (`green.txt`). The full forge suites passed under both feature sets (`full-run*.txt`), with no knock-on, and clippy was clean. Then the mutations, then the gates.

## Must fix

| # | Fix | Test | Mutation (caught) |
|---|---|---|---|
| 1 | In `migrate_one`, `JournalRead::Absent` → `Detected::Rebuild` now writes the new journal at once with `write_journal` (the same durable write `rebuild` uses: temporary file, `fsync`, rename, directory `fsync`). This happens before `contained`, the detection-error branch, the attempt check and `dir_state`. If the write fails, it returns `Outcome::Ungated`, which makes the pass incomplete and sets `migration_incomplete`. The new journal is `building` with `attempts: 0`; `rebuild` still counts the attempt when it starts. | h1: a lost-record legacy store (held objects, `refs/mirror/*`, hideRefs) with `mirror-recovery/<owner>` at mode 000. Precondition: `symlink_metadata` of a child gets `EACCES`. After boot a journal exists or the pass is incomplete, no held object is served, and the approved read is 503. Post-fix, the journal exists. | M1: the early persist removed turns h1 red with the same pre-fix message |
| 2 | After each push batch in `push_refs` succeeds, staging is measured once more with `sizecap::dir_size` (off the runtime). Above `size_cap` it returns `GREW_PAST_CAP`, the same error as a watcher-detected overrun, so `rebuild` removes staging and `fail`s with a reason containing "size cap". This is in the migration's push only; the engine's capped runs are untouched. | h2: `size_cap: 1` with `size_poll: 3600 s`, so the watcher never measures during the push and only the final check can catch it. The journal is `failed` with a "size cap" reason, and the store is gated. | M2: the final check removed turns h2 red (`migrated past the size cap`) |

**Mutation evidence:** `mutate.py`, `mutation-M1-*.txt`, `mutation-M2-*.txt` and `mutation-summary.txt`. For each mutation the driver copies the file, mutates it, runs the test, restores it with `cp` and `touch`, checks the sha256, and re-runs the test green. Both were caught, both restored files matched their sha256, and both re-ran green. The final gates ran after the mutations.

## Audit: every `Failed` in `migrate_one` and its callees

Line numbers are in the final `migrate.rs`. For each place, a journal is confirmed on disk, or the code now returns `Ungated`.

| Line | Where | Why a journal is on disk |
|---|---|---|
| 1391 → `gate` (1309) | lock not taken | `gate` returns `Failed` only when `create_exclusive` succeeded (it wrote the journal, or an entry already existed); otherwise `Ungated` (1313) |
| 1405 | `JournalRead::Bad` | `read_journal` returns `Bad` only when an entry exists at the path, or the path cannot be checked (not `NotFound`). `journal_gates` treats both as gating, because it opens only on `Absent`. |
| 1411 → `gate` | `Detected::Error` | as `gate` above |
| 1420 | new | the early persist failed → `Ungated` |
| 1430 | `contained` refused | the journal was read `Present`, or was just persisted (item 1) |
| 1450, 1456, 1460 | detection-error re-examination: journal not removed, still an error, `Skip` | the journal was read `Present`; at 1450 `remove_file` failed, so it is still there |
| 1478 | `attempts >= MAX_ATTEMPTS` | `Present` (a new journal has 0 attempts and cannot reach this) |
| 1485 | `dir_state` error on served or recovery | `Present`, or persisted by item 1. **This is the Codex High 1 path, closed by h1.** |
| 1499 | a store appeared during migration | `Present`, or persisted |
| `fail` (1592) | every later failure | `fail` now returns `gated_or_ungated`: `Failed` only if `journal_exists`, else `Ungated`. Every caller is after a successful journal write, so this is defensive. |
| `finished` (1533) | `finish` failed (from `Detected::Finish` with an absent journal, or from `publish`) | now `gated_or_ungated`. On the `Detected::Finish` path no journal exists, so a failed `finish` now returns `Ungated` and the pass is incomplete. Before, it returned `Failed` with no journal. That store is already the rebuilt one, so the change only over-gates. |
| `rebuild` (1676) | the attempt journal write failed | `gated_or_ungated` (the same logic as before, now through the helper) |

`journal_exists` counts a path that cannot be checked as present. In that case `journal_gates` (which opens only on `Absent`) also gates, so "confirmed" there means the readers refuse the store.

## Documentation

- **Size cap (CHANGELOG, Codex Medium 2):** it now says staging is measured once more after each push completes. It also says git may briefly write past the cap between polls before the push is killed or the final check fails it. The statement that an oversized store "is not migrated" is now backed by the final check, not by sampling alone.
- **Journal first (new CHANGELOG bullet):** a journal is written as soon as detection decides to rebuild. If it cannot be written, or a failure leaves no journal, the pass is incomplete and every 64-hex owner's store is refused until a boot completes it.

## Changed or removed assertions

None. The interdiff against `after-fix2.patch` removes no `assert`, `expect` or `panic` line.

## Deviations and notes for the lead

- **`finished` and `fail` now use `gated_or_ungated`.** This goes slightly beyond item 1's minimum, so that every `Failed` in the audit is backed by a journal check. The one behaviour change is the `Detected::Finish` path when `finish` fails: it now returns `Ungated`, the pass is incomplete, and there is a temporary service-wide refusal.
- **The final size check is per batch in `push_refs`, not in the shared `run_output_env`.** That keeps the engine's capped runs and their `push_measure` test seams unchanged. The engine has the same race between process completion and the measurement poll. It is outside this brief, and I recorded it here rather than changing it.
- **A persisted `building` journal whose `contained` check or `dir_state` keeps failing gates the store on every boot, without counting attempts.** That fails closed, and an operator clears it (as with fix 2's `Skip` case).

## Still recorded, not fixed

All carried from earlier rounds: Codex 3, 4 and 6; security N4 and N5; F5; F7/L3; the fsck give-up runbook; lock-timeout gate journals not re-detected. The newly noted engine-side poll race is listed above.

## Evidence index (`b1/fix3/`)

| File | What it holds |
|---|---|
| `after-tests-red.patch`, `after-fix3.patch` | Tree snapshots |
| `interdiff-vs-after-fix2.patch` | Source interdiff |
| `red.txt`, `red-summary.txt` | Pre-fix reds |
| `green.txt`, `full-run.txt`, `full-runfeaturesmirror.txt` | Green runs |
| `mutate.py`, `mutation-*.txt`, `mutation-summary.txt` | Mutations |
| `gates.txt`, `gates.txt.patch` | Final nine gates |
| `SHA256SUMS` | Checksums |
