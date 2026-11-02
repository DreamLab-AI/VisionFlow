# Commit A code review: mirror publication isolation

Worktree: /home/devuser/workspace/wt/solid-pod-rs-mirror-publication (branch fix/mirror-publication-isolation, base 31fe898). The review was read-only: no cargo was run, nothing was built and nothing in the worktree was edited.

**Verdict:** no confidentiality hole. On every path traced, held objects stay out of served. Defects 1 and 2 are new in commit A. Defect 3 existed before but now affects more cases. 4–7 are smaller.

## Production defects (most severe first)

### 1. A clean owner-pushed repo that gets followed returns 503 permanently (CONFIRMED, medium)

- **Where:** `mirror.rs:1791` (empty-update early return in `publish`, before `mark_clean_store` at 1807), `mirror.rs:2410` (`apply` does not call `publish` when nothing passed), `mirror.rs:2324-2333` (skip and rate-limit paths save the record), `lib.rs:581-593` (`unpublished_mirror`).
- **What fails:** many paths save a `MirrorRecord` without ever publishing: `on_announce`, the "already at this state" skip, the rate-limit hold, fetch failure, and an all-held state. Once a record exists, `unpublished_mirror` returns true, so every reader answers 503.
- The marker for an existing clean store is written only inside `publish`, and only when at least one branch is approved.
- **Concrete case:** a 64-hex owner pushes `repo`, and the operator follows it.
  - The first 30618 equals the pushed heads, so the plan skips with "already at this state".
  - The record is saved and no marker is ever written.
  - Raw, HTML, JSON, archive and smart-HTTP all return 503 until upstream moves *and* that move passes its verdict.
  - Locked decision 2 names exactly this scenario.
- **Fix:** add a trait method, e.g. `prepare(owner, name)`, implemented in `GitCli` as `mark_clean_store(dir)`. Call it in `apply` straight after `lock_repo`, when `exists()` is true, before planning. Also make sure no record is written for an existing store before that step: run `on_announce`'s save under the same lock and after the step (see 4).
- **Test gap:** no test covers this. The marker is only asserted on stores created by `materialise`.

### 2. Backoff never escalates when publishing keeps failing (CONFIRMED, low-medium)

- **Where:** `mirror.rs:2359-2361` resets `failures = 0` on fetch success; `mirror.rs:2449-2451` then adds 1 on publish failure.
- **What fails:** the counter goes 0 → 1 on every state. A persistent publish failure refetches from upstream at `backoff_interval(base, 1)` = 2× `min_interval` forever. The expected example is any legacy store refused by `mark_clean_store` until commit B lands.
- **Fix:** move `rec.failures = 0` to after publication, and only when `publish_error.is_none()`.

### 3. A failed publish is never retried for the same state (CONFIRMED, existed before but now affects more cases)

- **Where:** `mirror.rs:2438-2441` sets `rec.last_event` even when publish failed. `plan_mirror` at `mirror.rs:638` then skips any state with `created_at <= last_seen_at`. No `retry_at` is set, so `drain` (`mirror.rs:2144-2156`) never requeues the state.
- **What fails:** after a transient failure, served stays behind until upstream publishes a strictly newer 30618. Examples of transient failures:
  - a timeout
  - a stale `refs/heads/main.lock`
  - the size-cap kill in 5
- The old code did the same on `update_ref` errors, but with `--atomic` one bad ref now strands every approved branch.
- **Fix:** on `publish_error`, leave `last_event` unchanged and set `rec.retry_at = Some(now + backoff_interval(min_interval_ms, failures))` so `drain` requeues the state.

### 4. The rustdoc overstates the cross-process lock (CONFIRMED, low)

- **Where:** `mirror.rs:2225-2228` says two instances cannot "race the record's read-modify-write". But `on_event` (`mirror.rs:2037-2040`) runs `on_announce` under the in-process `lock_for` only, not `lock_repo`.
- **What fails:**
  - Instance A's `apply` reads the record at 2252 while holding the file lock.
  - Instance B's `on_announce` saves a new `clone` and `announce_at`.
  - A's save at 2463 overwrites them, and the new clone URL is lost until the next 30617.
- **Fix:** take `lock_repo` in `on_event` around both the announce and apply branches (and drop the inner one from `apply`), or narrow the doc.

### 5. The publish size cap measures the whole served store, not growth (PLAUSIBLE, low-medium)

- **Where:** `mirror.rs:1833-1842`. `run_capped(&served, …, Some(MAX_REPO_BYTES))` kills the push once `dir_size(served) > 512 MiB`, but the comment says publication "may grow it by at most the cap".
- **Why served can exceed the cap while the candidate stays under it:**
  - served never shrinks: commits orphaned by approved force moves stay (locked decision 7);
  - receive-pack leaves loose objects for small pushes;
  - an owner-pushed repo can already be large.
- **What fails:** from then on, every publish that runs longer than the 500 ms poll is killed. Publish fails, the next state is skipped (3), `failures` sticks at 1 (2), and receive-pack can be killed mid-update (leaving `*.lock` or `tmp_objdir-incoming-*`).
- **Fix:** take a baseline `dir_size(served)` (via `spawn_blocking`) before the push and pass `Some(baseline.saturating_add(MAX_REPO_BYTES))`. Otherwise, correct the comment and accept the absolute limit as policy.

### 6. The plugin path can leak into the public record (CONFIRMED by reading, low)

- **Where:** `mirror.rs:1845` redacts only `served_s`, and only for the push.
- **Not redacted:**
  - a `-C <candidate>` failure ("fatal: cannot change to '<plugin_dir>/mirror-work/candidates/…'");
  - errors from `mark_clean_store`, `write_marker` and `write_served_config` (`io::Error` text containing paths).
- All of these reach `seen.reason` and the branch verdict (`mirror.rs:2426`, `2453`), which the mirror API serves.
- **Fix:** apply the replacement (served path, candidate path and `work_root`) to the whole `publish` result, e.g. an inner function plus `map_err` in the trait method.

### 7. A doc comment now sits on the wrong function (CONFIRMED, trivial)

- **Where:** `lib.rs:139-145`. `unpublished_mirror_error` was inserted between `/// Current wall-clock time in Unix seconds.` and `fn now_secs`.
- **What fails:** the wall-clock comment now documents the 503 helper, and `now_secs` has no doc.
- **Fix:** move the new function and its own comment above that line.

## Test integrity

- **Real git:** these all use real `GitCli` against real stores:
  - rows 2, 3, 4, 6, 7, 8, 9 and 10 (engine `RaceClock` variant);
  - row 15 (failed fetch, oversize, cancellation);
  - the three adapter lease tests (`publish_with_a_current_lease…`, `…stale_lease…`, `…empty_lease…`).
  - The cancellation test wraps `GitCli` in a delegating `ParkAfterFetch`, which is fine.
- **Production argv:**
  - The stale-lease tests (engine and adapter) and the row 8 non-fast-forward move all drive the production `publish` argv, with no `+` and a per-ref lease.
  - `RaceClock` relies on `apply` reading the clock exactly once, at `mirror.rs:2261`, straight after `local_heads`. The test documents this coupling.
- **Fail on revert:** rows 2, 4, 6, 7, 8, 10 and 15 (and the adapter lease tests) would fail on the reverted code. Each asserts that held or refused objects are absent from served, which the old shared store violates. Row 15 also asserts the candidate path exists.
- **Pass on revert:**
  - **Row 5 (smart-HTTP), `mirror.rs:3751`:**
    - The old served store already had `hideRefs`, and the earlier Git HTTP fix (CHANGELOG Security, second bullet) already refuses unadvertised wants, so it cannot tell the fix apart from the old code.
    - It also skips silently when `git-http-backend` is missing.
    - `if let Ok(r) = held` passes when `handle` returns `Err`.
    - Suggested: assert on `Err` too, or require the status to be 4xx/5xx.
  - **Row 9 (hooks), `mirror.rs:4050`:**
    - The old path used `update-ref`, never receive-pack, so no receive hooks could run.
    - It is still a valid guard for the new path: it overrides served `core.hooksPath` with planted hooks, so it catches a `RECEIVE_PACK` that drops `-c core.hooksPath=/dev/null`.
- **Missing:** a test for defect 1 (followed owner-pushed store, in-sync first state, readers expected 200).

## MSRV, CI and docs

- The `ci.yml`, `Cargo.toml`, `README.md` and `clippy.toml` edits change only the MSRV, 1.88 → 1.89. Nothing else changed.
- `File::lock` (stabilised in 1.89) is the only newer std API. `Option::is_none_or` (1.82) and `std::path::absolute` (1.79) are older.
- **CI caveat:** the msrv job runs `cargo check --all-targets` in `crates/solid-pod-rs` with default features, so the forge `mirror` code that calls `File::lock` is never compiled at 1.89 in CI.
- **Stale 1.88 mentions:**
  - `crates/solid-pod-rs/docs/explanation/comparison-vs-jss.md:18`
  - `crates/solid-pod-rs/docs/tutorials/01-your-first-pod.md:15`
- `crates/solid-pod-rs/docs/adr/ADR-2009-…md:44,74` still describes the replaced `hideRefs`-on-served layout. This is not blocking.

## Reuse and dead code

- `update_ref`, `set_head` and `drop_mirror_refs_and_gc` are fully removed from the code.
- `publish`, `mark_clean_store`, `init_bare` and `write_served_config` spawn through `self.run` / `run_capped` → `hardened_git_async`, so there is no duplicate spawn logic.
- `kill_group` existed before this change.
- `repo/browse.rs:101` `is_full_object_id` partly duplicates `guard::is_sha` (`guard.rs:168`). It could be `matches!(rev.len(), 40 | 64) && crate::guard::is_sha(rev)`.

## Changelog

The CHANGELOG records:
- the trait break (`update_ref` and `set_head` removed, `publish` added);
- the new `RefUpdate` and `PUBLICATION_MARKER`;
- that verdict reads now evaluate the candidate;
- the move to atomic publication;
- the cross-process lock;
- the `GitCli::new(plugin_dir/repos)` precondition;
- the MSRV bump;
- the `/commits/<full sha>` 404 change.

## Checked and sound

- **Publication invariant:**
  - Refspecs are exactly the approved `(sha, ref)` pairs, pushed by SHA, never `refs/mirror/*`.
  - Held branches are never turned into a `RefUpdate` (`mirror.rs:2396-2404`).
  - Duplicate refs are rejected.
  - An empty update list returns early, avoiding a `push.default` fallback.
  - Refs, SHAs and leases are validated with `is_branch_ref` and `is_sha40`.
- **Push argv:**
  - one `--force-with-lease=<ref>:<expect|empty>` per ref, no `+`;
  - the served path after `--`;
  - `--atomic --porcelain --no-verify`;
  - `GIT_ALLOW_PROTOCOL=file`, a process group and a 120 s timeout.
- **`RECEIVE_PACK`:** a compile-time constant with nothing interpolated, pinned by the row 17 test against `HARDENED_CONFIG` plus the receive keys.
- **`porcelain_rejection`:** parses `<flag>\t<src>:<dst>\t<summary>` correctly for SHA sources, treats `!` and missing lines as failures, and runs after a zero exit as a second check.
- **HEAD:** filtered to pushed refs in both `apply` (2405) and `publish` (1851), set only after a successful push, and a failure is only logged.
- **`BranchOutcome.moved` on publish failure:** false for every branch, with "; publish failed: <reason>" appended to approved verdicts, `moved = 0` and `seen.action = FetchFailed`. One unavoidable caveat: if git is killed after receive-pack committed the refs, the record says not moved. That is harmless, because the next `local_heads` snapshot reflects the truth.
- **Lease snapshot:** `local_heads` reads served. If it fails it returns an empty map, so leases become "must not exist" and the push is refused (fail closed).
- **Lock (`lock_repo`):**
  - It is taken before any record read in `apply` and runs on `spawn_blocking`.
  - The `File` drops on every return path.
  - A cancelled `apply` releases the lock when the `spawn_blocking` result is dropped.
  - `flock` is per open file description, so it also serialises within one process.
  - If the lock fails, the record is not saved.
- **Lock path:** derived from `MirrorStore.dir.parent()` = `plugin_dir`, which matches `GitCli`'s `work_root`.
- **Cleanup and layout:**
  - The candidate is removed whole on a failed, killed or oversize fetch.
  - Staging is removed when `build_served` fails.
  - The marker (written to a temp file, then renamed) and the fixed config are written in staging before the rename into `repos/`, so `repos/` never shows a half-built or unmarked new store.
  - Nothing private lives under `repos/`.
  - `--prune` is on the candidate fetch.
  - Size accounting is on the candidate.
- **`mark_clean_store`:** refuses any store with `refs/mirror` or `uploadpack/transfer.hideRefs` containing `refs/mirror`. Every pre-fix mirror store had `hideRefs`, so all of them stay refused and gated.
- **Marker gate:**
  - It covers `resolve_repo` and `h_git_smart`.
  - Direct `repo_git_dir` users (compare, pulls, merge, forks) reach it through `resolve_repo` at their entry point, or only fetch the `refs/heads` closure of a head repo.
  - It is defined outside `cfg(feature = "mirror")`.
  - `ForgeError::Busy` maps to 503 (`error.rs:71`).
- **`commit_log`:** a full object id that does not resolve now returns 404, while unborn branches still list as empty.
