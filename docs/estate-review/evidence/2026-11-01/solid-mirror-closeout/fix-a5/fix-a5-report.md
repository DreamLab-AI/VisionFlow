# Commit A fix round 5: report

All nine items are fixed, and all nine gates exit 0 on the final tree: the seven standard gates, plus `cargo test` and `cargo clippy -D warnings` with `--features mirror,anchoring,announce`. Nothing is committed. The worktree /home/devuser/workspace/wt/solid-pod-rs-mirror-publication is at HEAD 31fe898, with rounds 1 to 5 uncommitted.

- **Starting tree:** `git diff HEAD` before this round was byte-identical to `../final-a4.patch` (same SHA-256).
- **Round 5 only:** `fix-a5-interdiff.patch` (1,811 lines; `CHANGELOG.md` and `mirror.rs`). I built it from a scratch index of HEAD plus `final-a4.patch`.
- **Full diff against HEAD:** `fix-a5.patch`.
- **Red output:** `red-raw.log` (12 tests) and `red-raw-forcepush.log`.
- **Green output:** `green-forge.log`, a copy of gate 2.
- **Mutations:** `mutations.txt`.
- **Gates:** `fix-a5-gates.txt`, with logs in `gate-logs/` and the script in `run-gates.sh`.
- **Toolchain:** rustc 1.99.1 and git 2.56.0, with `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.

## How the red output was captured

1. **Behaviour-neutral refactor first,** with the suite green (308 tests). It changed these things:
   - `MirrorGit::prepare` gained a `scan: bool` argument; the engine passed `true`, as before.
   - `MirrorRecord` gained `isolated`, `awaits_migration` and `publish_timeouts`.
   - `PublishError` gained `timed_out`.
   - I added `push_failure`, initially just `PublishError::from`.
   - I added the shared constants `GIT_TIMED_OUT` and `GREW_PAST_CAP`.
   - I added a `#[cfg(test)] push_measure` seam on `GitCli`.
2. **Red run:** the 12 new tests, against that tree.
3. **Fixes,** then green.

Three tests were not red in step 2:
- **The force-push test** first failed in its own setup: `HEAD~1` does not exist on a root commit. I changed the setup only, to an orphan branch, and re-ran it red (`red-raw-forcepush.log`: "awaits migration").
- **`a_legacy_record_store_with_an_unreachable_object_awaits_migration`** is a guard and passed before the fix. Mutation M3 (the engine never asks for the scan) kills it.
- **Five tests added after the final review** are guards that pass on the fixed code: `an_oversize_fetch_is_not_retried`, `fetch_head_is_not_a_root_of_the_scan`, `a_notes_ref_is_not_a_root_of_the_scan`, `the_scan_marks_a_store_with_nothing_unreachable` and `push_failures_are_classified`. Mutations M1, M2 and M4 show that they discriminate.

No existing assertion changed. I renamed one test (item 9). The round 3–4 direct `git.prepare(…)` calls now pass `true`; that changes the call, not the assertion.

## Items

### 1. C1: verify before changing pending state

- **Fix:** `enqueue` checks the signature (`verify_event`) after the far-future check and before the slot comparison (`mirror.rs:2778`). Only a verified event can take a slot, so only a verified event can replace a pending retry.
- **Test:** `a_forged_newer_state_does_not_cancel_a_pending_retry` (`:7996`). S2 is pending behind a ref lock, then a newer state arrives with its tags tampered after signing, then the lock is removed.
- **Red:** `timed out waiting for the pending retry of S2 to publish`.

### 2. C2 and R3: the timer and enqueue race

- **Fix:** in the timer arm, the pending state goes back into its slot unless the slot already holds a state that supersedes it or is it (`:2865`).
- **Test:** `a_pending_retry_is_not_lost_to_an_older_state_in_the_slot` (`:8024`). It puts S1.5 straight into `inbox` with no wake, so the timer arm is the one that runs.
- **Red:** `timed out waiting for the pending retry of S2 to publish`.

### 3. C3 and R4: transient fetch and materialise failures

- **Fetch:** `fetch_checked` (`:2970`) now returns `(reason, retryable)`.
  - A git failure is retryable, except `GREW_PAST_CAP`. Both cap paths (the watcher kill in `run_in`, and the check `fetch_inner` makes after the fetch) return that exact constant, which redaction leaves unchanged.
  - **Decision:** a refused clone URL or host (`bad URL`, `URL has no host/port`, `clone URL refused`, `clone host refused: …`) is terminal. That includes DNS failures inside the SSRF check, because `SsrfCheck` returns only a string and they can't be told apart.
- **Materialise:** a failure now counts towards `failures` and schedules a retry (`:3283`).
- **Tests and red output:**

| Test | Red output |
|---|---|
| `a_transiently_failed_fetch_is_retried` (`:8050`): the upstream is renamed away and then back | `timed out waiting for the state to fail with a retry scheduled` |
| `a_failed_materialise_schedules_a_retry` (`:8071`) | `a failed materialise was not retried` |

- **Guard:** `an_oversize_fetch_is_not_retried` (`:8509`). Mutation M1, which makes every fetch error retryable, kills it.

### 4. C4: retry floor from the failure time

- **Fix:** `schedule_retry` (`:3058`) is used for materialise, fetch and publish failures. It sets `retry_at = clock.now_ms()` read at failure time, plus `max(backoff_interval, backoff_delay)`.
- **Test:** `the_retry_floor_counts_from_the_failure` (`:8109`) uses a clock that steps 5 s at every reading.
- **Red:** `the retry is 2000 ms after the attempt started, not 2000 ms after the failure`.

### 5. C5: tie ordering

- **Decision:** the lower-id sibling is applied. That matches NIP-01, `supersedes` and the replay check.
- **Fix:** the planner's "acted on" comparison goes through `acted_on_at` (`:2564`). A state at the same second as the last one acted on, with a lower id, counts as newer than it. `PlanInput` is unchanged, so there is no public API change.
- **Test:** `a_lower_id_sibling_of_an_applied_state_is_applied` (`:8138`).
- **Red:** `the lower-id sibling was not applied: … Skip, reason: "not newer than the last state acted on"`.

### 6. C6 and R2: retry classification

- **`push_failure` (`:1414`):**
  - `GREW_PAST_CAP` is terminal;
  - `GIT_TIMED_OUT` is retryable, with `timed_out` set;
  - anything else (a stale lease, a receiver ref lock) is retryable.
- **`publish` no longer scans or marks** (`:2262`). It has no record to tell an owner's store from one with mirror history, and `prepare` has already marked every store the engine publishes into. For an unmarked store:
  - legacy traces give `AWAITS_MIGRATION`, terminal;
  - otherwise it is refused as "not marked for publication", terminal;
  - any error during these checks is terminal.
- **The engine side of "command error is terminal":** `prepare_served` (`:3024`) returns `Err` for a new-style store. `apply_locked` then records "the served repository could not be prepared: …" as a Skip, with no retry and not acted on. For a first record it saves nothing.
- **Timeouts:** `MAX_PUBLISH_TIMEOUTS = 3` (`:138`, documented). `MirrorRecord::publish_timeouts` counts consecutive timeouts of the newest state. It resets when a newer state is seen, on a non-timeout failure, and on success. The third timeout clears `retry_at` and adds "(3 timeouts in a row; not retried)" to the reason (`:3394`).
- **Tests and red output:**

| Test | Red output |
|---|---|
| `a_push_killed_by_the_growth_cap_is_terminal` (`:8182`); the measure seam returns `u64::MAX` at the first poll | `a growth-cap kill was classed retryable` |
| `a_store_check_error_in_publish_is_terminal` (`:8208`) | `…classed retryable: fatal: not a git repository: '<served>'` |
| `repeated_publish_timeouts_of_one_state_become_terminal` (`:8317`), with a fake `Spy` adapter | `timeout 3 of the same state was retried  left: Some(…)` |

- **Guard:** `push_failures_are_classified` (`:8606`) covers the real classifier, which the `Spy` fake bypasses.

### 7. R1: rescoping the scan (overturns locked decision 13)

**Who is scanned.** `MirrorRecord::isolated` (`:1030`) is written `true` by every new record. A record file from before this release lacks it, so serde reads `false`. `prepare_served` asks `prepare` for the scan only when such a pre-release record exists. An owner's store first followed now is never scanned. Legacy layout and alternates are still checked for every store.

**The verdict is persisted.** `MirrorRecord::awaits_migration` (`:1035`) is set on the first `AwaitsMigration`, and later events return it without calling `prepare`. A scan that fails on a pre-release store also counts as awaiting migration, and that verdict is persisted: this fails closed, and nothing clears it before Commit B.

**Order in `apply_locked`.** The record is read, then the far-future and replay checks run, and `prepare` runs after them. When no record exists yet, `prepare` runs first (`early`, `:3133`), so no first record is ever saved beside an unprepared store. A record beside an unmarked store would gate it at 503.

**Order in `on_announce` (`:2931`).** The replay check runs, then `prepare`, then the save. A `prepare` failure with no record saves nothing.

**The scan is streamed** (`holds_unreachable_objects`, `:1605`, which uses the new `first_line` reader at `:1651`):
1. `HEAD` must be symbolic to `refs/heads/*`; otherwise the store counts as holding an unreachable object.
2. `for-each-ref` is read line by line and stops at the first ref outside `refs/heads/` and `refs/tags/`, which also counts.
3. `git fsck --unreachable --no-reflogs --connectivity-only --no-progress` is read line by line and stops at the first `unreachable` or `dangling` line.

Nothing is buffered whole, and every step has the 120 s budget, including the final wait. The branches-and-tags-only roots from round 4 still hold. `FETCH_HEAD` is not an fsck root: `fetch_head_is_not_a_root_of_the_scan` passes.

The legacy check was split out into `legacy_traces` (`:1718`), and the "Locked decision 3" rustdoc on `mark_clean_store` (`:1761`) now states the revised rule.

**Tests and red output:**

| Test | Red output |
|---|---|
| `an_owner_repo_with_a_force_pushed_away_commit_is_followed_and_served` (`:8353`) | "awaits migration" |
| `a_legacy_record_store_with_an_unreachable_object_awaits_migration` (`:8426`), a guard; the record JSON has no `isolated` field | green before; M3 kills it |
| `the_store_is_examined_once_and_never_for_a_replay` (`:8449`), counting `prepare` calls | `a replay examined the store  left: 2 right: 1` |

`fetch_head_is_not_a_root_of_the_scan`, `a_notes_ref_is_not_a_root_of_the_scan` and `the_scan_marks_a_store_with_nothing_unreachable` (`:8571`–`8596`) are guards; M2 kills the notes test.

Mutation M4 (no detached-`HEAD` pre-check) kills the round-4 detached-`HEAD` test.

### 8. R5: ceiling and quarantine

- **Ceiling:** the check uses the store's size minus its quarantine directories (`:2333`).
- **Growth watcher:** its limit is still the full size plus `MAX_REPO_BYTES` (`:2339`), so a large young quarantine does not kill the push. That kill would now be terminal.
- **Test:** `young_quarantines_do_not_count_towards_the_ceiling` (`:8487`), with a fresh quarantine holding a sparse file of `MAX_SERVED_BYTES + 1`.
- **Red:** `PublishError { reason: "the served repository is past its size ceiling (2147483648 bytes)" … }`.

### 9. Docs

- **`MAX_SERVED_BYTES` rustdoc (`:130`) and CHANGELOG:** it is an admission threshold that excludes quarantines. One push may still add up to `MAX_REPO_BYTES`, measured by polling every 500 ms, so a store can reach about 2.5 GiB.
- **CHANGELOG:** "Leftover … are removed" now reads "Stale … (older than the publish timeout, 120 s) are removed".
- **`RefUpdate::expected` and `PublishError` rustdoc:** a stale lease is retried against served's current value, so a branch moved behind the engine is overwritten on the retry.
- **Codex round 4 Lows:**
  - the "sweeps any it finds" doc on `publication_leaves_no_receive_quarantine_behind` is corrected;
  - `a_fresh_quarantine_survives_both_sweeps_…` is renamed `a_fresh_quarantine_survives_a_failed_and_a_later_publish_and_an_old_one_does_not`.
- **Also updated:**
  - `Prepared::AwaitsMigration`;
  - the `MirrorGit::prepare` contract, including when `Mirror` calls it;
  - the `on_event`, `apply`, `enqueue` and `drain` rustdoc, for the new order and verify-first;
  - `MirrorRecord::failures` and `retry_at`;
  - every new public item;
  - the CHANGELOG bullets for marking, retries, and the breaking changes (`prepare(owner, name, scan)`, the record fields, `PublishError::timed_out`, `MAX_PUBLISH_TIMEOUTS`).

## Decisions and behaviour changes for review

- **Decision 13 is overturned** as the brief directs. I did not edit `publication-design.md`.
- **The public record JSON** gains `isolated: true` on every new record. `awaitsMigration` and `publishTimeouts` appear only when they are set.
- **A direct caller of `GitCli::publish`** on an unmarked clean store now gets a terminal refusal; `publish` no longer marks.
- **A web edit, merge or publish** is unaffected beyond earlier rounds.

## Residuals (not fixed)

- Fetch failures are retried without limit, at the capped backoff (60 s with `minIntervalMs: 0`). Fetch timeouts have no counter like publish timeouts do.
- **I2:** a persistent lock failure re-arms the retry about every 1 s. Noted only, as the brief says.
- MSRV 1.89 is still unverified here.

## Gates (`fix-a5-gates.txt`)

| # | Command | Result |
|---|---|---|
| 1 | `cargo fmt --all --check` | exit 0 |
| 2 | `cargo test --locked --offline -p solid-pod-rs-forge --features mirror` | exit 0: 326 lib + 13 + 18 + 65 doc tests |
| 3 | `cargo test --locked --offline -p solid-pod-rs-forge` | exit 0: 228 + 13 + 18 + 51 |
| 4 | `cargo test --locked --offline -p solid-pod-rs-server --features forge-full forge` | exit 0 |
| 5 | `cargo clippy … -p solid-pod-rs-forge --features mirror --all-targets -- -D warnings` | exit 0 |
| 6 | `cargo clippy … -p solid-pod-rs-server --features forge-full --all-targets -- -D warnings` | exit 0 |
| 7 | `RUSTDOCFLAGS=-D warnings cargo doc … --no-deps -p solid-pod-rs-forge --features mirror` | exit 0 |
| 8 | `cargo test --locked --offline -p solid-pod-rs-forge --features mirror,anchoring,announce` | exit 0: 387 lib + 9 + 13 + 23 + 84 doc tests |
| 9 | `cargo clippy … -p solid-pod-rs-forge --features mirror,anchoring,announce --all-targets -- -D warnings` | exit 0 |
