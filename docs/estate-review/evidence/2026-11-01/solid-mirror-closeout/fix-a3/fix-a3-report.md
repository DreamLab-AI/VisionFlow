# Commit A fix round 3: report

All four items are fixed, and all seven gates exit 0 on the final tree. Nothing is committed: the worktree /home/devuser/workspace/wt/solid-pod-rs-mirror-publication is at HEAD 31fe898, with rounds 1 to 3 uncommitted.

- **Round 3 only:** `fix-a3-interdiff.patch` (708 lines, 7 files). I built it from a scratch index of HEAD plus `../fix-a2.patch`, which applied cleanly.
- **Full diff against HEAD:** `fix-a3.patch`.
- **Toolchain:** rustc 1.99.1 and git 2.56.0, with `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.
- **Existing assertions:** none changed. The only edits to existing tests are the two tightenings item 4 asks for, and a backdate inside the `plant_quarantine` fixture that item 3 asks for.

## Item 1: Codex High 1, a foreign head injected into a mirror's served store

### Fix

- **New helpers in `lib.rs`:**
  - `has_mirror_record` (`lib.rs:596`) checks for the record path. It works without the `mirror` feature, and `unpublished_mirror` now uses it.
  - `refuse_mirror_write` (`lib.rs:610`) returns 409 with the message "'o/r' is a mirror: only its publication writes to it".
- **`compare.rs:528`, `resolve_head`:** once a foreign fetch is allowed, a base that has a mirror record (marked or not) gets this treatment:
  - `HeadAccess::Caller` (compare, and opening a pull request) gets a 409, before the head lock, the config heal and the fetch.
  - `ExistingPr` and `Cached` read only the `refs/forge/heads/*` already held, which for a mirror is none. Viewing an existing pull request therefore does not turn into a 409 or a write.
- **Other writers:**
  - `merge.rs:296`: `merge_pull` refuses before the pull lock (merge-tree, commit-tree and update-ref write into served).
  - `web_edit.rs:602`: the web edit refuses (hash-object, commit-tree, update-ref).
  - `forks.rs:285`: a fork whose destination is a followed mirror's name refuses, because the fork would become the served store.
  - `compare.rs:631`: `release_head_ref` writes nothing for a mirror.
- **Smart-HTTP push:** it was already refused for mirrors, so it is unchanged.
- **Releases:** there is no release write path (`releases.rs` writes git only in its tests).

### Red output (`red-raw.log`, `red-raw-compare.log`, `red-raw-pr.log`)

| Test | Red output |
|---|---|
| `a_compare_into_a_mirror_never_fetches_a_foreign_head` | `a foreign blob reached served`. Before I reordered the assertions it showed 200, with `aheadBy:1` and dana's commit in the diff. |
| `a_pull_request_into_a_mirror_never_fetches_a_foreign_head` | `a foreign blob reached served`. The fetch happens before the later 403 check on `resourceUrl`. |
| `a_web_edit_into_a_mirror_is_refused` | `left: 200 right: 409`, with `{"changed":true,"commit":"a52b0e5…"}` |
| `a_pull_request_cannot_be_merged_into_a_mirror` | `left: 200 right: 409`, with `{"fastForward":true,…,"state":"merged"}` |

Each test asserts that the blob is absent from served (`cat-file -e`), that no `refs/forge/heads/*` was written, and the 409.

**Green:** `green-forge.log`, and gate 2.

## Item 2: Codex High 2, a store with unreachable objects gets marked

### Fix

- **New `GitCli::holds_unreachable_objects` (`mirror.rs:1504`):** every id that `cat-file --batch-all-objects --batch-check=%(objectname)` lists must appear in `rev-list --objects --exclude=refs/mirror/* --all`. Each git call has a 120 s budget.
- **`mark_clean_store` (`mirror.rs:1573`):** returns `Prepared::AwaitsMigration` when the store is legacy or holds such an object. This is the same refusal path as the legacy layout:
  - no marker is written and the store stays gated;
  - `apply` neither fetches for it nor retries it;
  - `publish` refuses it as terminal.
- **Short-circuit:** a store already known to be legacy skips the walk.
- **Error handling:** if the check itself fails (for example a timeout), the error is passed on and the store is not marked.

### Red output

`a_store_holding_an_unreachable_object_is_not_marked`: `left: Ok(Ready) right: Ok(AwaitsMigration)`.

The test checks a store with a mirror record, no `refs/mirror`, no `hideRefs`, and a `hash-object -w` held blob. It asserts that no marker is written and that the raw route returns 503. As a control, the same layout without the blob returns `Ready` and is marked.

**Green:** gate 2.

## Item 3: Security N3, sweep age

### Fix

- **`sweep_stale_quarantines` (`mirror.rs:1820`):** runs before the push. It removes a quarantine only when its newest mtime is older than `PUBLISH_TIMEOUT`. The newest mtime is taken over the directory, its entries and their entries (`newest_mtime`, `mirror.rs:1787`; it does not follow symlinks).
- **Snapshot:** publish records the quarantine names present before the push (`quarantines_in`, `mirror.rs:1767`).
- **`sweep_new_quarantines` (`mirror.rs:1836`):** runs after a failed push and removes only quarantines that are not in that snapshot.
- **Test fixture:** `plant_quarantine` now backdates the planted leftover, its `pack/` and its directory with `File::set_modified`. Without that, `publication_leaves_no_receive_quarantine_behind` (unchanged, still green) would plant a "fresh" quarantine.

### Red output

`a_fresh_quarantine_survives_both_sweeps_and_an_old_one_does_not`: `after a failed publish  left: []  right: ["tmp_objdir-incoming-fresh"]`.

The test asserts that the fresh quarantine survives the failed and then the successful publish, and that the old one is gone.

**Mutation MQ** (the post-failure sweep removes every quarantine) kills it with the same message (`mutations.txt`).

## Item 4: two tests tightened

- **`a_receiver_ref_lock_failure_is_recorded_and_lands_only_approved_history`:** the report-only `eprintln!` is replaced by two assertions (`mirror.rs:6717`):
  - the reason contains `cannot lock ref`;
  - the approved commit `main2` is present in served, so the quarantine was migrated before the ref transaction failed.

  Both pass on the real code. To show they discriminate, mutation ML3 makes the locked publication fail before its push. The test then fails with `publish failed: mutant: failed before the push`, whereas before the tightening its negative assertions passed.

  `mutations.txt` also lists two void attempts. ML did not fire for this single-branch publish. ML2 broke the fixture's first publish rather than the new assertions. Both are marked void there.
- **`held_side_commit_is_not_fetchable_over_smart_http`:** a missing `git-http-backend` now panics through `expect` (`mirror.rs:4317`) instead of printing SKIP and returning, and the rustdoc is updated. It passes here because the backend is present. I could not exercise the panic path without removing git's exec path, so this is confirmed by reading the code only.
- **Red:** none. These are tightenings of tests that already pass, not bug fixes.

## Other changes

- **CHANGELOG, Security:** two bullets. One says forge writes into mirrors are refused (409). The other covers the reachability rule for marking and the sweep age.
- **Rustdoc:** `Prepared::AwaitsMigration`, `mark_clean_store` and the sweep functions are documented.

## Gates (`fix-a3-gates.txt`, logs in `gate-logs/`, script in `run-gates.sh`)

| # | Command | Result |
|---|---|---|
| 1 | `cargo fmt --all --check` | exit 0 |
| 2 | `cargo test --locked --offline -p solid-pod-rs-forge --features mirror` | exit 0: 303 lib + 13 + 18 + 65 doc tests |
| 3 | `cargo test --locked --offline -p solid-pod-rs-forge` | exit 0: 228 + 13 + 18 + 51 |
| 4 | `cargo test --locked --offline -p solid-pod-rs-server --features forge-full forge` | exit 0 |
| 5 | `cargo clippy … -p solid-pod-rs-forge --features mirror --all-targets -- -D warnings` | exit 0 |
| 6 | `cargo clippy … -p solid-pod-rs-server --features forge-full --all-targets -- -D warnings` | exit 0 |
| 7 | `RUSTDOCFLAGS=-D warnings cargo doc … --no-deps -p solid-pod-rs-forge --features mirror` | exit 0 |

## Notes for review

- **Unreachable objects block marking:** an owner-pushed repository with ordinary dangling objects, for example from a force push, is now refused marking until migration. That is the brief's rule, but it will catch clean-but-unpruned repositories too.
- **Legacy reason text:** the recorded reason is still the legacy "predates publication isolation and awaits migration" for both kinds of refusal.
- **Pull requests opened before a repository became a mirror:** they can no longer be merged into it, and their views show only a head that is already held, which for a mirror is none.
