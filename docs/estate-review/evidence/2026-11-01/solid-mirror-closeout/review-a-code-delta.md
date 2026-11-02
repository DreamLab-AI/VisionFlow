# Commit A fix round: code delta review

- **Worktree:** /home/devuser/workspace/wt/solid-pod-rs-mirror-publication (diff against 31fe89856d0c).
- **Method:** I rebuilt the reviewed state as base 31fe898 plus `impl-a.patch` in a scratch tree, then diffed it file by file against the current worktree. The review was read-only: no cargo was run and nothing in the worktree was edited.
- **Inputs read:**
  - `fix-a/fix-a-report.md`, `fix-a-red.txt`, `fix-a-mutations.txt` and `fix-a-test-baseline.diff`;
  - the full production interdiff for `mirror.rs`, `lib.rs`, `api.rs`, `compare.rs`, `pulls.rs` and `merge.rs`;
  - CI, the CHANGELOG and the docs;
  - every new `fix_a` test.

**Verdict:**
- (a) Findings 1–7 and the test notes are fixed, with one caveat on 3 (D1).
- (b) The fix round adds one new correctness defect, D1. It is a hot retry loop when `minIntervalMs` is 0, which the code explicitly allows. Everything else below is low or a nit.

## (b) New problems, most severe first

### D1. A failed publish is retried in a hot loop when `minIntervalMs` is 0 (CONFIRMED, medium)

- **Where:**
  - `mirror.rs:2676`: `rec.retry_at = Some(now + backoff_interval(self.cfg.min_interval_ms, rec.failures))`;
  - `backoff_interval` (`mirror.rs:202`): `base_ms * 2^failures`, which is 0 for any `failures` when `base_ms = 0`;
  - `drain` (`mirror.rs` `drain`, the `retry_at` branch): `sleep(wait.min(MAX_BACKOFF_MS) + 50)`, then it requeues the same event.
- **Zero is a supported config:** `parse_mirror_config` takes `minIntervalMs` unclamped, and its own doctest (`mirror.rs:~290-298`) sets `"minIntervalMs": 0` and asserts it.
- **What fails:** with `minIntervalMs: 0` and a publish that keeps failing, the worker re-runs the full apply about every 50 ms, for ever. Each cycle takes one of the four fetch permits and runs `git fetch` from the clone URL (remote HTTP for a 30617-supplied clone), then `git push`. The escalating `failures` count has no effect because the base is 0.
- **Persistent failures that trigger it:**
  - every legacy store, which `mark_clean_store` refuses by design until commit B;
  - a read-only served store;
  - a planted `refs/heads/*.lock`.
- **Before the fix round:** the same state was never retried (finding 3), so this loop did not exist.
- **Test coupling:** `a_transiently_failed_publish_is_retried_for_the_same_state` (`mirror.rs:5640`) uses `quiet_cfg` (`min_interval_ms: 0`), so it passes partly because of this 50 ms loop.
- **Fix:** floor the retry delay independently of the rate limit. For example:
  - `let floor = backoff_interval(self.cfg.min_interval_ms.max(MIN_FETCH_INTERVAL_MS), rec.failures); rec.retry_at = Some(now + floor);`
  - or use `backoff_delay(rec.failures)`.
  - For the test, inject a clock or shorten the wait some other way.
- **Optional:** cap retries for refusals that are known to be permanent (the `mark_clean_store` "awaits migration" error) and do not set `retry_at` for them. A legacy store would then stop refetching until a newer state arrives or migration runs. With the default 30 s base it otherwise refetches every hour at most, for ever.

### D2. The redaction can mangle unrelated text when the plugin path is short (CONFIRMED, nit)

- **Where:** `GitCli::redact` (`mirror.rs`, `impl GitCli { fn redact … }`).
- **What fails:** every occurrence of the plugin directory string is replaced, with only a `len() > 1` guard. With `plugin_dir = /data`, an error such as `unable to access 'https://host/data/x.git'` becomes `https://host<plugin>/x.git`. Nothing leaks; the record text is just garbled.
- **Fix:** replace only the served and candidate paths and the `work_root` (`<plugin>/mirror-work`) prefix, and stop replacing the bare plugin root. Alternatively, replace it only at a path boundary (followed by `/` or a quote).

### D3. The item 7 record test depends on not running as root (PLAUSIBLE, low)

- **Where:** `a_failed_publication_keeps_the_plugin_path_out_of_the_saved_record` (`mirror.rs:5904`) uses `chmod 0555` (`ReadOnly`) to make publication fail.
- **What fails:** root ignores the mode, so in a root CI container the control assertion "publish failed" fails. The test fails loudly rather than passing silently.
- **Fix:** use the planted `refs/heads/main.lock` (`plant_ref_lock`) as the failure trigger, since it doesn't depend on permissions, and check that the resulting error carries a path. Alternatively, skip with a log line when the effective uid is 0.

### D4. The MSRV job compiles the mirror code at 1.89 but has not been run (NOTE)

- **Where:** `ci.yml` msrv job, new step `cargo check -p solid-pod-rs-forge --features mirror --all-targets`.
- It does run in `crates/solid-pod-rs`, which is a workspace member (root `Cargo.toml` members include `crates/solid-pod-rs-forge`). `-p` therefore resolves the forge package and `--features mirror` applies to it. That compiles `File::lock` and the test-only `File::lock`/`unlock` at 1.89, which answers my earlier CI note.
- It is still unverified: the report says only rustc 1.99.1 was available offline. The first CI run is the proof.
- The step has no `--locked`, unlike the local gates. Consider `cargo check --locked …`.

## (a) Status of the earlier findings

### 1. A followed clean store returns 503: FIXED

- **The fix:** `MirrorGit::prepare` is a required method, with no default. `GitCli::prepare` calls `mark_clean_store` when served exists. `Mirror::prepare_served` calls `prepare` when `exists()`.
- **Call order checked:**
  - In `apply_locked`, `prepare_served` runs right after the lock match and before `read_state`, the store read, and every `self.save` (future skew, replay, plan skip and rate limit, no clone, materialise failure, fetch failure, and the final save).
  - The lock-failure branch returns without saving.
  - In `on_event` for 30617, `lock_repo` is taken, then `prepare_served`, then `on_announce` (its only save).
  - So no record is written for an existing store before it is marked.
- **Tests:** `a_followed_in_sync_owner_pushed_repo_is_marked_and_served` and `…first_state_is_all_held_is_served` were red before the fix (`fix-a-red.txt`) and assert the marker plus a 200. The announce test also checks the marker.
- A store `prepare` refuses (legacy layout) is only logged and stays gated, as intended.

### 2. `failures` oscillated between 0 and 1: FIXED

- The reset on fetch success is gone (`mirror.rs:2583-2586`).
- `failures` is now:
  - incremented on fetch failure (unchanged) and on publish failure (`mirror.rs:2675`);
  - reset to 0 only in the success branch (published, or nothing passed: `mirror.rs:2680`).
- Skip paths leave it unchanged, as before.
- `repeated_publish_failures_escalate_the_failure_count` was red before the fix and asserts 1, 2, 3, then 0. It uses strictly newer states, so the retry path is not involved.

### 3. A failed publish was never retried: FIXED, but see D1

On publish failure:
- `last_event` is left unchanged (now set only in the success branch);
- `retry_at` is set;
- `last_fetch` and `last_seen` are recorded.

Traced state transitions for the retry of the same event:
- **Replay gate:** `newest_seen_at`/`newest_seen_id` still advance on a publish failure (set before planning, unchanged). The retried event has `created_at == newest_seen_at` and `id == newest_seen_id`, so `loses_tie` is false and it passes. An older event is still refused as a replay, which is correct.
- **Plan:** `last_seen_at` comes from the unchanged older `last_event`, so the state is "newer" and plans as a fetch.
- **Rate limit:** `last_fetch_at = now` was set before the failed fetch. `retry_at = now + backoff_interval(base, failures)` uses the same floor as the plan's `min_interval_ms`. `drain` waits `retry_at - now + 50 ms`, so the rate limit has passed and the plan fetches.
- **Newer events:** `drain` reinserts with `or_insert`, so a newer state that arrived during the wait replaces the retry.
- **Restart:** the relay redelivers the same event, and the same reasoning applies.
- **Lock:** the file lock is released before `drain` sleeps (it drops when `apply_locked` and `on_event` return).
- `a_transiently_failed_publish_is_retried_for_the_same_state` was red before the fix and goes through `enqueue`/`drain`. See D1 on its 0 ms base.

### 4. Record read-modify-write under the cross-process lock: FIXED

- `on_event` takes the in-process mutex, then `lock_repo`, for both kinds. State events go to `apply_locked` with that lock, so it is not taken twice.
- The public `apply` takes the lock itself, and nothing in the workspace calls `apply` or `on_event` from outside `mirror.rs`, so there is no nested `flock`.
- A 30617 that cannot take the lock is dropped unsaved. This is documented in the rustdoc and is acceptable.
- `an_announcement_waits_for_the_repository_lock_and_marks_a_clean_store` holds the `flock` from another file descriptor, so it does discriminate. It was red before the fix.
- The rustdoc at `apply` and `on_event` is now accurate.

### 5. The publish size cap measured the whole store: FIXED

- `limit = dir_size(served)` is measured as a baseline on `spawn_blocking` after `sweep_quarantine`, plus `MAX_REPO_BYTES`.
- The watcher now measures before its first sleep, so the cap holds even for pushes shorter than 500 ms.
- `MAX_REPO_BYTES` and the module docs were updated.
- Quarantine directories are swept before the push and after a failed push, under the apply lock. The one caveat (an owner push racing in before the first record exists) is documented at `sweep_quarantine`.
- `a_small_publish_succeeds_into_a_served_store_already_over_the_cap` was only red once the watcher measured first. Against an absolute cap it is now red deterministically, which makes it a valid regression test.
- `publication_leaves_no_receive_quarantine_behind` was red before the fix.

### 6. Paths leaked into the public record: FIXED, with nit D2

- `GitCli::redact` (served → candidate → plugin) is applied at the trait boundary of `materialise`, `prepare`, `fetch` and `publish`, which are the only methods that return error strings. Error strings from `mark_clean_store` and the IO steps now pass through it.
- `--git-dir` is absolute, so git names paths in the form `redact` matches.
- `an_adapter_error_names_no_store_path` (candidate removed) and the record test (D3) were red after the `--git-dir` change.

### 7. A doc comment sat on the wrong function: FIXED

- In `lib.rs`, `unpublished_mirror_error` now sits above the wall-clock doc, which is back on `now_secs`.

### Test note: row 5 (`if let Ok`): FIXED

- The test now `.expect`s `Ok` and asserts there is no `PACK`.
- Its rustdoc says it passes before the fix (upload-pack answers `ERR … not our ref`) and that it is defence in depth only.

### Test note: MSRV CI job: FIXED, unverified (D4)

### Test note: docs still saying 1.88: FIXED

- `comparison-vs-jss.md:18` and `01-your-first-pod.md:15` now say 1.89.
- `clippy.toml` keeps "lock floor 1.88" on purpose, which is accurate.

### Not fixed (not requested)

- ADR-2009 still describes `hideRefs` on served. This is not blocking.

## New tests: would each fail on its bug?

- **Items 1, 2, 3, 4, 5 (quarantine), 6, 7 (adapter) and 8 (×3):** red before the fix, per `fix-a-red.txt`. Each one's own assertion names the defect. I read each assertion and agree.
- **Item 5 (over cap) and item 7 (record):** went red only after the intermediate changes (measure-first watcher, absolute `--git-dir`). Against a revert of the cap or redaction fix on today's code, both fail.
- **Items 11 and 12 (coverage, green before the fix):** the mutation evidence is adequate, with one qualification.
  - M3 (served alternates onto the candidate) kills all four through their *own* isolation assertions:
    - "held commit in served";
    - the held raw read returning held bytes;
    - "held side commit in served";
    - "refused main commit in served".
  - M1 (no `--atomic`) kills the atomic test through its own assertion.
  - The qualification: M2's kills of the two in-flight tests and the all-held owner-pushed test come from fixture controls ("nothing in state 2 passes", "control: nothing passes"), not from the isolation assertions. M2 therefore does not show those assertions discriminate; M3 does.
  - Also, the in-flight tests check isolation of the candidate *in general*: the held commit was fetched by `side_fixture`, not by the hanging fetch. They show that served is untouched while a fetch is running, which is what row 6 asks for.
- **Item 12 receiver ref-lock test:** it reports, but does not assert, whether `main2` landed unreferenced, which is acceptable by design §2. Its isolation assertions do discriminate (M2, M3).
- **Item 13:** test-only change. The baseline diff confirms the rest of the original module is unchanged apart from the `prepare` delegation in `ParkAfterFetch` and the `fix_a` module opener.

## CHANGELOG (semver)

- The `### Changed` Breaking entry now lists the required `MirrorGit::prepare(owner, name)` alongside `publish`, `RefUpdate` and `PUBLICATION_MARKER`, and states the lock on both `apply` and `on_event`.
- New bullets cover:
  - the retry and backoff behaviour;
  - the cap now bounding growth;
  - the quarantine sweep;
  - `lastPush` read from branches and tags only.
- Security covers:
  - the wider gate (pulls, merge, compare including the head repository, `GET /api/repos`, and listings that leave the repository out);
  - `--git-dir`;
  - redaction.
- This is complete for the public API.
- The listing deviation (leave out rather than 503) matches the CHANGELOG and the `PUBLICATION_MARKER` rustdoc.

## Other changes checked and sound

- **`run_in` / `--git-dir`:**
  - Every git spawn in `GitCli` now passes an absolute `--git-dir`, so git never searches upward.
  - `ensure_candidate` rebuilds a store that fails `is_bare_store`.
  - The push runs with `--git-dir=<candidate>` and `cwd=served`; the served path still follows `--`.
  - `RECEIVE_PACK`, the leases, the no-`+` rule and the argv order are unchanged from `impl-a.patch`.
- **Marker gate (`gated_git_dir`, `lib.rs`):**
  - It is the same check as `resolve_repo`'s and is applied at:
    - `resolve_head` (base, and the head repository with `Busy` propagated);
    - `release_head_ref`, `compare_data`;
    - pulls (3 sites), merge (2 sites);
    - `repo_summary`, `h_git_smart`.
  - `h_api_repo` propagates `Busy` as 503.
  - A missing repository still falls through to the caller's existing handling, so 404 behaviour is unchanged.
- **`repo_summary` `lastPush`:** restricted to `refs/heads` and `refs/tags`.
- **`merge.rs:441`:** `crate::repo::browse` under `cfg(anchoring)` compiles without the removed `use`.
