# Commit A fix round 7: report

All four items are done, and all nine gates exit 0. Nothing is committed: the worktree /home/devuser/workspace/wt/solid-pod-rs-mirror-publication is at HEAD 31fe898, with rounds 1–7 uncommitted.

## Evidence

- **Starting tree:** `git diff HEAD` was byte-identical to `../final-a6.patch` (SHA-256 `6dfafcf5…6e53`).
- **This round only:** `fix-a7-interdiff.patch` (342 lines; `lib.rs` and `mirror.rs`), built from a scratch index of HEAD plus `final-a6.patch`. It removes no `assert`.
- **Full diff against 31fe898:** `fix-a7.patch` (SHA-256 begins `822693cea35067cb`).
- **Logs:**
  - `red-raw.log`: 4 tests red before any production change;
  - `green-forge.log`: gate 2;
  - `fix-a7-gates.txt`, with logs in `gate-logs/` and the script in `run-gates.sh`.

## Items

### 1. A record symlink is Unreadable, not Missing

`MirrorStore::load` (`mirror.rs:~1180`) now runs `symlink_metadata` first:
- **no directory entry:** `Missing`;
- **a symlink, dangling or not:** `Unreadable("is a symbolic link")`;
- **any other `symlink_metadata` error:** `Unreadable`;
- **a regular entry:** read as before, except that a read error, including `NotFound` from a race after the stat, is now `Unreadable`.

**One change outside the item's wording.** The reader gate's `has_mirror_record` (`lib.rs:622`) used `metadata`, which follows the link. A dangling record symlink therefore read as "no record", and the unmarked store beside it was served. I changed it to `symlink_metadata`. This is one call, and without it item 1 has no effect on readers.

**Tests:**

| Test | What it checks | Red output |
|---|---|---|
| `a_record_symlink_is_unreadable_not_missing` (`:9333`) | `load` on a dangling link and on a link to a valid record, and the record API giving 503 | "a dangling record symlink loaded as Missing" |
| `a_dangling_record_symlink_keeps_the_gate_closed` (`:9367`) | an unmarked owner store beside a dangling record symlink, read through raw | 200 with the file's content (the gate was open) |

### 2. A failed `symbolic-ref` is an error, not evidence

**New helper.** `run_in` was split, with no change to its behaviour. The new `run_output` (`:1991`) returns git's whole `Output`, or `Err` only when git could not start, timed out, or was killed by the cap. `run_in` is `run_output` plus the existing exit-status check.

**`holds_unreachable_objects` (`:~1730`) now reads the exit status of `symbolic-ref -q HEAD`:**

| Result | Treated as |
|---|---|
| exit 0 and `refs/heads/*` | carry on with the scan |
| exit 0 and anything else | evidence |
| exit 1 (git's silent "detached") | evidence |
| any other exit code, a timeout or a spawn failure | `Err` |

An `Err` takes the round-6 retry path, so no verdict is persisted.

**Tests:**
- `a_failed_head_check_is_an_error_not_evidence` (`:9390`) runs the check on a directory that is not a repository (fatal, exit 128). Red: `Ok(true)`.
- `a_detached_head_is_still_evidence` (`:9404`) is a control and was green before. The round-4 detached-HEAD test still passes too.

### 3. A state that gave up stays given up

`Mirror::failed` (`:3275`) now computes `gave_up = retry != Retry::No && timeouts >= MAX_TIMEOUTS`. The limit is checked before scheduling for every retryable class, not only `Counted`.

- **Test:** `a_state_that_gave_up_stays_given_up` (`:9424`). Three publish timeouts reach the limit, then a stale-lease failure (`Retry::Yes`) of the same state follows.
- **Red:** "a state past the limit was re-armed".
- **After the fix:** the reason carries "(3 timeouts of this state; not retried)".

### 4. Docs

- `acted_on_at`'s rustdoc is back on `acted_on_at` (`:2747`). It had been sitting above `Retry`.
- The `has_alternates` rustdoc now says that a path which cannot be checked is returned as `Err`: neither presence nor absence, so the store is neither marked nor given a verdict.
- The `MirrorStore::load` and `failed` rustdoc are updated for items 1 and 3.

## Gates (`fix-a7-gates.txt`)

| # | Command | Result |
|---|---|---|
| 1 | `cargo fmt --all --check` | exit 0 |
| 2 | forge tests, `--features mirror` | exit 0: 345 lib + 13 + 18 + 66 doc |
| 3 | forge tests, default features | exit 0: 228 + 13 + 18 + 51 |
| 4 | server tests, `--features forge-full forge` | exit 0: 3 |
| 5 | clippy `-D warnings`, forge `mirror` | exit 0 |
| 6 | clippy `-D warnings`, server `forge-full` | exit 0 |
| 7 | `cargo doc --no-deps` with `RUSTDOCFLAGS=-D warnings`, forge `mirror` | exit 0 |
| 8 | forge tests, `--features mirror,anchoring,announce` | exit 0: 406 lib + 9 + 13 + 23 + 85 doc |
| 9 | clippy `-D warnings`, forge `mirror,anchoring,announce` | exit 0 |

Residuals from round 6 are unchanged: the unbounded first-follow prepare retry, Codex Medium 4, permanent scan errors, I4, and MSRV 1.89 unverified.
