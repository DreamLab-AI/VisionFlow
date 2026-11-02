# Security delta review — Commit A fix round

- Scope: the worktree diff against `31fe89856d0c` at review time, compared with `impl-a.patch` (the version reviewed in `review-a-security.md`), and `fix-a/fix-a-report.md`.
- Method: read-only. No cargo, no edits to the worktree. All line numbers are in `crates/solid-pod-rs-forge/src/` unless stated otherwise.

## Verdict

- (a) M1, L1, L2, L3 and I1 are fixed. M2's stall is fixed, but the fix removes the only bound on the served store's total size (N1 below).
- (b) The fix round introduces one new Medium (N1) and two Low findings (N2, N3). It also leaves one test gap (T1). Nothing new lets a held or unapproved object reach served.

## New findings, ranked

### N1 — Medium: the served store's total size is now unbounded

`mirror.rs:1857-1875`

- **Problem:** the publish cap is now `baseline + MAX_REPO_BYTES`, where the baseline is measured after the quarantine sweep. Each push may therefore add up to 512 MiB, and nothing limits the total. Served never shrinks:
  - commits orphaned by approved force moves stay (locked decision 7);
  - only `gc --auto` prunes, and only after its two-week expiry.

  The old total cap was a bound, even if an unintended one.
- **Attack path:** the followed key's owner, or anyone who controls the upstream once a state is signed, repeats the following:
  1. Push a fresh history of about 500 MiB upstream, under the candidate cap.
  2. Sign a state that passes. With no mark, `branch_verdict` moves any present commit. With a mark, the owner anchors a trail.
  3. Force-move the branch to an unrelated history.

  Each round lands up to 512 MiB in served. The candidate is wiped and rebuilt when it passes its own cap, so it does not stop this. The rate limit (`MIN_FETCH_INTERVAL_MS`, 30 s by default) allows roughly 1 GiB a minute. The disk under `data_root` fills, which takes the whole pod down, not only this mirror.
- **Who can reach it:** only the holder of the followed key. The impact still falls on the operator, which is why this is Medium.
- **Fix:** keep the growth cap, and add a hard ceiling on the total. Two options:
  - Refuse to publish when `baseline` is already past a ceiling (for example `4 × MAX_REPO_BYTES` or an operator setting), with a clear record reason and no kill loop.
  - Or prune unreachable objects in served after an approved non-fast-forward move: `git gc --prune=now` under the lock. That is safe, because served holds only approved history, and decision 7 allows orphaned commits to go.
- **Test:** add one where served is past the ceiling, publication is refused with a reason, and nothing is killed mid-push.

### N2 — Low: publish failures now retry indefinitely, and each retry stalls the repository's worker

`mirror.rs:2672-2676`, with `2335-2347`

- **Problem:** every publish failure now sets `retry_at`. The `drain` worker then sleeps in place for up to `MAX_BACKOFF_MS` (1 h) before requeuing the state, and holds the repository's `busy` slot while it sleeps. Newer states for that repository sit in the inbox until the sleep ends.
- **Effect:** some failures are permanent, for example a legacy store refused by `mark_clean_store` ("predates publication isolation"). For those, the worker sleeps, refetches up to 512 MiB, fails again, and doubles the backoff, indefinitely. The cost is bounded and nothing leaks, but:
  - a newer state that would fix things waits up to an hour;
  - a stuck legacy mirror costs a full fetch every hour until Commit B.
- **Fix:**
  - Classify the publish error. Retry only transient failures (a lock, the timeout, a stale lease); treat "predates publication isolation" and "ref, sha or lease refused" as terminal.
  - Make the sleep wake when the inbox gains a newer event for the key, for example with `tokio::select!` on a per-key `Notify`.

### N3 — Low: the sweep can break a concurrent owner push in the window before the first record

`mirror.rs:1648-1661`, `1857`, `1884`

- **Scope:** `sweep_quarantine` deletes only directories, by `file_type()`, which does not follow symlinks. They must sit directly in `served/objects/` and start with `tmp_objdir-incoming-`, so it cannot delete anything outside receive-pack quarantines.
- **The window:** a followed repository the owner pushed themselves, at the first apply. `prepare_served` runs before any record exists, so the smart-HTTP write guard (`lib.rs:~663`) still accepts the owner's pushes. A concurrent owner push can lose its quarantine part-way through. receive-pack then fails, and in the worst case leaves an unreferenced `.pack` with no `.idx`. The rustdoc acknowledges this.
- **Impact:** availability or hygiene only. A quarantine holds only what that pusher sent, and the refs do not move.
- **Fix (optional):** skip quarantines younger than `PUBLISH_TIMEOUT` (by mtime); only one that old can be left over from a killed push.

### T1 — test gap: an older state arriving during the retry window

- **Code:** correct. `newest_seen_at`/`newest_seen_id` advance to S2 at `mirror.rs:2528-2529`, before planning, and are saved at `2696` even when the publish fails (`2672`). An owner-signed state S1.5 with `S1 < S1.5 < S2` then hits the replay check at `2509` and is skipped. S2's own retry passes, because the tie check requires `ev.id != newest_seen_id` (`2507`). Item 3 therefore does not reopen M1's rollback.
- **Gap:** nothing pins this. `a_transiently_failed_publish_is_retried_for_the_same_state` (`mirror.rs:5640`) covers only the retry of S2 itself.
- **Fix:** add a test. Fail S2's publish, then apply S1.5. Assert `last_seen.reason` says "older than a state already seen" and that served is unchanged. Then release the lock and assert that S2 lands.

## The five specific checks

1. **Item 3, retry.** Correct, as described under T1. `last_event` stays at S1 (`2672-2685`), `newest_seen_*` advances to S2, and an older state in between is refused. There is no test for that last part.

2. **Item 4, lock order.** Sound.
   - `on_event` (`2218-2220`) takes the in-process mutex, then `lock_repo`, and passes the file into `apply_locked` (`2233`). The announcement path holds the same file through `prepare_served` and `on_announce` (`2223-2231`).
   - `lock_repo` is taken once per event; `apply_locked` never takes it.
   - The public `apply` (`2443-2446`) takes only `lock_repo`, never the mutex. Mixing it with `on_event` cannot create a cycle: there is only one edge, mutex → flock. Two `flock`s on separate opens in one process serialise rather than deadlock, because Linux `flock` locks per open file description.
   - `drain` and the retry path take no lock themselves; they call `on_event`. `BusyGuard::drop` takes only the std `busy`/`inbox` mutexes, never across an await.
   - No production code outside tests calls `apply`.

3. **Item 6, explicit `--git-dir`.** Sound.
   - `run_in` (`1506-1531`) passes `--git-dir=<git_dir>` on every spawn, and every spawn goes through it: `run` → `run_capped` → `run_capped_with` → `run_in`.
   - For the push, `git_dir` is the candidate and the working directory is served, which is only the size watcher's target (`1867-1875`).
   - `--git-dir` sets `GIT_DIR` for git's children. `GIT_DIR` is in git's `local_repo_env`, which the local transport clears before it runs the `--receive-pack` command, so receive-pack does not inherit the candidate. It operates on the served path git appends: `served_s`, after `--` (`1849-1851`), shell-quoted.
   - `RECEIVE_PACK` is still the fixed literal (`1250-1252`) and is never interpolated.
   - The real-push tests that assert served moves (`a_small_publish_succeeds_into_a_served_store_already_over_the_cap`, `an_atomic_two_branch_publish_with_one_stale_lease_moves_neither`, the hooks test) would fail if receive-pack had written into the candidate.
   - The `ensure_candidate` rebuild (`1362-1375`) removes and re-initialises only `self.candidate(owner, name)`: `work_root/candidates/<hex64>/<repo>.git`, with both segments validated. That path can never resolve to served, which sits under `repo_root`, a sibling of `work_root`. `remove_dir_all` on a symlink removes only the link.

4. **Item 5, sweep and baseline.**
   - The sweep is confined to receive-pack quarantine directories (N3).
   - The baseline does not open a way for held objects in. The push still carries only the approved history, and the cap only kills.
   - It does remove the total bound (N1).
   - The watcher now measures before its first sleep (`1544-1553`), which closes the 500 ms window at start.

5. **Item 8, reader gate.** Sound for content.
   - `gated_git_dir` (`lib.rs:604-616`) is now used by:
     - smart and dumb HTTP (`lib.rs:~688-698`, which stops only on the gate's `Busy` refusal);
     - `repo_summary` (`api.rs:321`), now limited to `refs/heads refs/tags` (`api.rs:333-334`);
     - `h_api_repo`, which propagates the 503 (`api.rs:421`);
     - `resolve_head`, for both the base (`compare.rs:499`) and the cross-repository head (`compare.rs:527-531`, with `Busy` propagated, so a head in an unmarked mirror is never fetched into the base);
     - `release_head_ref` (`compare.rs:619`), `compare_data` (`compare.rs:652`);
     - `pull_merge_info` and `merge_pull` (`merge.rs:224`, `306`);
     - pulls (`pulls.rs:541`, `601`, `1161`).
   - The remaining direct `repo_git_dir` calls read no content from an existing store:
     - `forks.rs:109` reads fork-parent metadata from config;
     - `forks.rs:283` builds the fork destination path, and the source is gated by `resolve_repo` at `forks.rs:271`.
   - Leaving the repository out of the listing (`api.rs:364`) leaks nothing beyond its existence, which is already public:
     - the owner page lists names via `list_owner` (`lib.rs:620`);
     - `api/repos/<o>/<n>/mirror` serves the record.
   - The repo summary route returns 503, consistent with the rest. The report's "one-line flip" to 503 the whole listing is not needed for security.

## Status of the original findings

| ID | Status | Evidence |
|---|---|---|
| M1 | Fixed | The announcement path takes `lock_repo` under the mutex (`mirror.rs:2220-2231`); a lock failure drops the announcement unsaved. Test: `an_announcement_waits_for_the_repository_lock_and_marks_a_clean_store` (`5694`). |
| M2 | Stall fixed; total bound lost (N1) | The quarantine is swept before and after a failed publish (`1857`, `1884`); the cap is on growth (`1858-1864`). Tests: `a_small_publish_succeeds_into_a_served_store_already_over_the_cap` (`5759`), `publication_leaves_no_receive_quarantine_behind` (`5807`). |
| L1 | Fixed | Explicit `--git-dir` on every spawn (`1515-1518`); `is_bare_store` plus a full rebuild (`1362-1375`, `1625-1635`). Test: `a_candidate_missing_its_objects_is_rebuilt_not_resolved_to_a_parent_repo` (`5851`). |
| L2 | Fixed for paths | `redact` replaces the served, candidate and plugin paths (`1691-1706`) on every adapter method that can return a path-bearing error (`1936-1963`, `2031-2041`). `GitCli::new` makes the root absolute, so the prefix matches. Remaining, not new: the upstream's `remote:` stderr is still stored verbatim in the public record, and a path git prints in canonical form behind a symlinked `data_root` would not match. Tests: `5904`, `5950`. |
| L3 | Fixed | See check 5. Tests: `6050`, `6075`, `6093`. |
| I1 | Fixed | `prepare` (mark a clean store) runs under the lock before any record is saved, for both kinds (`2230`, `2478`, `2407-2414`). The trait method has no default (`1164-1177`). Tests: `5496`, `5539`. |
| I2 | Documented | `MirrorGit::publish` rustdoc (`1202-1209`) and `apply` rustdoc (`2428-2431`). |
| I3 | Unchanged | Not in scope of this round. The cancellation tests `6208` and `6232` show served is untouched when an apply is aborted. |

## Legacy window

The list in `review-a-security.md` still stands, with these changes:
- the ungated readers are closed;
- legacy mirrors now also retry hourly (N2);
- forks made from legacy mirrors before Commit A remain ungated and still hold held objects. That remains a Commit B item.
