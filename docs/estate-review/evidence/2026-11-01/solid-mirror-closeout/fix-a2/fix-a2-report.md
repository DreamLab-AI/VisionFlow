# Commit A fix round 2: report (uncommitted; the worktree is otherwise untouched)

- **Worktree:** /home/devuser/workspace/wt/solid-pod-rs-mirror-publication, at HEAD 31fe898, with round 1 and round 2 both uncommitted.
- **Full diff against HEAD:** `../fix-a2.patch`.
- **Round 2 only:** `../fix-a2-interdiff.patch` (1,031 lines; `ci.yml`, `CHANGELOG.md` and `mirror.rs`). I built it from a scratch index of HEAD plus `fix-a.patch`, which applied cleanly, so `fix-a.patch` is exactly the round 1 state.
- **Gates:** all seven exit 0, in both run 1 and run 2 (`fix-a2-gates.txt`, logs in `gate-logs/` from run 2, script in `run-gates.sh`). The toolchain was rustc 1.99.1 and git 2.56.0. Run 2 followed the drain re-arm fix below.

## Findings addressed

| Finding | Fix | Test (red before the fix?) |
|---|---|---|
| D1 hot retry loop | A transient publish failure sets `retry_at = now + max(backoff_interval(base, failures), backoff_delay(failures))`, so the retry waits at least 2 s, doubling up to 60 s, even when `minIntervalMs` is 0. | `a_failed_publish_waits_before_its_retry_even_with_no_rate_limit`: RED ("retry scheduled 0 ms after the failure") |
| D1 optional / N2 terminal failures | `MirrorGit::prepare` now returns `Result<Prepared, String>`, where `Prepared` is `Ready` or `AwaitsMigration`. For a legacy store, `apply_locked` records `Skip` with the reason "awaits migration". It does not fetch, does not build a candidate, sets no `retry_at` and leaves `last_event` unset. `newest_seen_*` still advances, so the same state redelivered after migration passes the replay check. `publish` now returns `PublishError { reason, retryable }`. Terminal (not retryable) errors are: awaits migration, past the size ceiling, ref/sha/lease refused, a duplicated branch, an unsafe key, and a non-UTF-8 path. Every other error is retryable. | `a_legacy_store_is_neither_fetched_nor_retried`: RED (FetchFailed, not Skip) |
| N2 the worker sleeps through newer states | New `Mirror::wake: Notify`. `enqueue` calls `notify_waiters()`. `drain` keeps the retry as `pending` and waits with `select!` on `sleep_until` against the notification, enabled before it reads the inbox so no wakeup is lost. A newer state replaces the pending retry and is applied at once. An older state, or the same state again, is dropped while a retry is pending (`supersedes` uses the inbox and replay order). An announcement is applied and the retry is kept. A `retry_at` on an announcement's record no longer requeues the announcement. | `a_newer_state_wakes_a_worker_waiting_to_retry`: RED before the fix, but only through its control (the retry was 0 ms away). The wake itself is shown by mutation MW (`mutations.txt`): with the notification arm disabled, the test times out. |
| N1 no bound on the served store's total size | New `pub const MAX_SERVED_BYTES = 4 × MAX_REPO_BYTES` (2 GiB). `publish` refuses, as terminal, when the swept baseline is past it, before any push starts. The per-push growth cap is unchanged. I did not use the `gc --prune=now` option, because it races served readers. | `a_served_store_past_the_ceiling_refuses_publication_without_retry`: RED ("1 of 1 moved") |
| T1 test gap | Test only. S2 fails to publish, then S1.5 (older than S2) is refused as a replay and served is unchanged. S2 is then re-applied directly and lands. | `an_older_state_during_a_publish_retry_is_refused_as_a_replay`: green before the fix, as both reviewers expected (the code was already correct) |
| Found in final review: a replay inherits a stale `retry_at` (the same hole was in round 1's `drain`) | Every early-return path in `apply_locked` hands back the persisted `retry_at`, which belongs to the newest state. An older state refused as a replay (for example after a restart, when S2's `retry_at` is in the past) would re-arm itself every 50 ms for ever. The fix has two parts. `drain` now arms a retry only when `rec.newest_seen_id == ev.id`. Every re-arm wait is also clamped to at least `MIN_REQUEUE_MS` (1 s), so a past `retry_at`, or a lock that keeps failing, cannot spin. | `a_replay_does_not_inherit_the_newest_state_retry`: RED (the worker never went idle; `red-replay-retry.log`) |
| D2 redaction mangles unrelated text | The bare plugin root is no longer replaced. `redact` now replaces served, then candidate, then `work_root` (as `<plugin>/mirror-work`), then `repo_root` (as `<plugin>/repos`). Each replacement happens only at a path boundary (new `replace_path`): the character before the match must not belong to a path segment, and neither may the one after. | `redaction_leaves_unrelated_text_alone_under_a_short_plugin_root`: RED (`https://host<plugin>/x.git`) |
| D3 the record test assumes a non-root user | The test now fails publication with `plant_ref_lock` instead of chmod 0555, so it works as root, and the `ReadOnly` helper is deleted. A new control checks that the reason contains `<served>`: receive-pack's first stderr line names `<served>/./refs/heads/main.lock`, which I confirmed by probing git. Without that control the test could pass vacuously. | Green before the fix, because the redaction already existed. Mutation MR (redact as the identity) kills it. |
| D4 MSRV step has no `--locked` | Added `--locked` to `cargo check -p solid-pod-rs-forge --features mirror --all-targets`. | Unverified: rustup is not installed and only rustc 1.99.1 is available. Nothing new needs more than 1.82 (`is_none_or`, `sleep_until`, `Notify`). |

## Other changes

- **Docs:**
  - The stale `MirrorRecord::failures` and `retry_at` rustdoc is corrected.
  - The `prepare`, `publish` and `drain` rustdoc is updated.
  - `PublishError` has a doctest. That brings the doctests to 65; round 1 had 64.
- **CHANGELOG:** the Breaking entry now names `Prepared` and `PublishError`. The retry bullet describes the floor, the wake, the terminal refusals and the ceiling. A new bullet covers the legacy no-fetch behaviour.
- **Call sites:** `an_adapter_error_names_no_store_path` now reads `e.reason`. `ParkAfterFetch` is updated for the new signatures. No other `MirrorGit` impl exists in the workspace.

## Not done

- **The brief arrived truncated.** It ended after "do NOT commit, push, or create/remove worktree…". I took the scope to be every finding in `review-a-code-delta.md` and `review-a-security-delta.md`. Please confirm that the two items below are out of scope.
- **States dropped during a retry wait:** an older state, or a redelivery of the same one, is dropped while a retry is pending rather than recorded as a replay. Dropping the identical redelivery is deliberate: applying it would retry early and skip the backoff. Disclosed, not changed.

- **N3** (sweeping young quarantines): optional, Low. A plain mtime filter would break `publication_leaves_no_receive_quarantine_behind`, and it would weaken M2 after our own killed push, because that quarantine would stay counted in the baseline for up to 120 s. The pre-record window stays as documented at `sweep_quarantine`.
- **ADR-2009 `hideRefs` wording:** not requested.

## Evidence

- `fix-a2-red.txt`: red and green status of the new tests before the fix, with the panic messages. The raw log is `red-raw.log`, taken before any production change.
- `mutations.txt`: MW (no wake) and MR (identity redaction).
- `green-mirror.log`: the mirror test run after the fix.
