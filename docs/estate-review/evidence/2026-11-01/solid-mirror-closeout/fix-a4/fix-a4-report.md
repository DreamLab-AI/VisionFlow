# Commit A fix round 4: report

All seven items are done, and all seven gates exit 0 on the final tree. Nothing is committed: the worktree /home/devuser/workspace/wt/solid-pod-rs-mirror-publication is at HEAD 31fe898, with rounds 1 to 4 uncommitted.

- **Round 4 only:** `fix-a4-interdiff.patch` (683 lines, 5 files). I built it from a scratch index of HEAD plus `../fix-a3/fix-a3.patch`, which applied cleanly.
- **Full diff against HEAD:** `fix-a4.patch`.
- **Toolchain:** rustc 1.99.1 and git 2.56.0, with `CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target`.
- **Red output:** `red-raw.log` (items 1 to 4) and `red-raw-scan.log` (`newest_mtime`). Green: `green-forge.log` and gate 2. Mutation: `mutations.txt`.
- **Existing assertions:** none changed or removed.
  - Item 4 changes the setup of the round-3 quarantine test only: the fresh quarantine is now created during the failed publication.
  - Item 5 adds assertions to two tests.
  - `publication_leaves_no_receive_quarantine_behind` needed no change. Its planted quarantine is backdated, so the stale sweep before the push removes it in both the failed and the successful case. Its doc comment still says "sweeps any it finds", which is slightly loose now; I left the text alone.

## Item 1: `has_mirror_record` fails open

### Fix

- **`lib.rs:606`:** `has_mirror_record` now returns `Result<bool, ForgeError>`. `NotFound` means `false`. Any other error is logged and becomes `ForgeError::Busy` (503).
- **Callers:**
  - `refuse_mirror_write` passes the error on with `?`, so merge, web edit and fork all refuse.
  - `resolve_head` (`compare.rs:532`) passes it on as well.
  - `release_head_ref` (`compare.rs:637`) treats an error as a mirror and writes nothing.
  - `unpublished_mirror` (`lib.rs:585`) uses `unwrap_or(true)`, so the reader gate stays closed.
- **`web_edit.rs:574`:** `edit_failure` now maps `Busy` to 503.
  - **Side effect:** a web edit into an unmarked mirror used to return a generic 500 "edit failed". It now returns the gate's 503.

### Test and red output

`an_unreadable_record_path_refuses_writes_and_keeps_the_gate_closed` (`mirror.rs:7405`) replaces `plugin/mirror/<owner>` with a file, so `stat` fails with ENOTDIR. This does not depend on file modes.

- **Write:** the owner's web edit must return 409 or 503, with `main` and the object set unchanged. Red: `a write went ahead with an unreadable record: 200 {"changed":true,"commit":"8d94f1b…","ok":true,…}`.
- **Gate:** after the marker is removed, a raw read must return 503.

## Item 2: reachability roots

### Fix

- **`holds_unreachable_objects` (`mirror.rs:1511`):** now walks `rev-list --objects --branches --tags`.
- **Rustdoc:** the roots are `refs/heads/*` and `refs/tags/*` only. Not `HEAD`, `refs/mirror/*`, `refs/forge/heads/*`, `refs/notes/*`, `refs/stash` or anything else.
  - A store whose objects only those refs reach stays unmarked.
  - That includes an owner repository that holds pull request heads.

### Test and red output

`a_detached_head_at_an_unreferenced_commit_is_not_marked` (`mirror.rs:7453`) uses `commit-tree`, then `update-ref --no-deref HEAD`. Red: `left: Ok(Ready) right: Ok(AwaitsMigration)`.

## Item 3: alternates

### Fix

- **New `has_alternates` (`mirror.rs:1785`):** returns true when `objects/info/alternates` or `objects/info/http-alternates` exists. It uses `symlink_metadata` and fails closed: any error other than "not found" counts as present.
- **`mark_clean_store` (`mirror.rs:1552`):** returns `AwaitsMigration` when an unmarked store has alternates.
- **`publish` (`mirror.rs:2087`):** refuses a marked store that has gained alternates, as a terminal failure: "the served repository borrows objects through alternates". The check runs after `mark_clean_store` and before the sweep and the push.

### Tests and red output

| Test | Red output |
|---|---|
| `a_store_with_alternates_is_not_marked` (`mirror.rs:7492`), local and HTTP variants | `left: Ok(Ready) right: Ok(AwaitsMigration)` |
| `a_marked_store_that_gains_alternates_refuses_publication` (`mirror.rs:7528`), which also asserts `!retryable` and `main` unchanged | `published into a store with alternates: ()` |

## Item 4: the post-failure sweep

### Fix

- **Removed:** `sweep_new_quarantines` and the snapshot taken before the push are deleted.
- **Comment at `mirror.rs:2151`:** a failed push's leftover is not swept. The stale sweep removes it on a later publication; until then the size baseline tolerates it and the ceiling counts it.
- **`sweep_stale_quarantines` rustdoc (`mirror.rs:1862`):** says age is a heuristic, and that an owner push stalled longer than `PUBLISH_TIMEOUT` may lose its quarantine and fail. That affects availability only.
- **`newest_mtime` (`mirror.rs:1823`):** returns `None` (no age, so the quarantine counts as fresh) if any `read_dir`, entry or `modified()` in the scan fails, instead of skipping that entry.
- **CHANGELOG:** I replaced the "concurrent push is not broken" line with the wording from the brief.

### Test seam

I added a `#[cfg(test)] during_push: Option<fn(&Path)>` field on `GitCli` (`mirror.rs:1381`). It is called just before the publication push (`mirror.rs:2130`), and compiled out of release builds.

I added it before capturing the red output, so the red run is against round-3 behaviour. There was no seam inside `GitCli::publish` before this, and a real concurrent owner push cannot be timed deterministically.

### Tests and red output

- **`a_fresh_quarantine_survives_both_sweeps_and_an_old_one_does_not` (`mirror.rs:7314`):** the fresh quarantine is now created through the seam during the failed publication. Red: `after a failed publish  left: []  right: ["tmp_objdir-incoming-fresh"]`.
- **`a_quarantine_that_cannot_be_scanned_has_no_age` (`mirror.rs:7567`):** sets `pack/` to mode 000 and calls `newest_mtime` directly. Red: `left: Some(SystemTime { tv_sec: 1793582953, … }) right: None`.
  - **Why it asks `newest_mtime` directly:** my first version checked the sweep's outcome and passed vacuously on the old code, because `remove_dir_all` also fails on the unreadable directory.
  - **Under root:** root reads through mode 000, so the test prints `NOT RUN …` and returns. It ran here, as devuser.

## Item 5: the writer tests prove nothing was written

- **New helper:** `object_set` (`mirror.rs:7392`) returns the output of `cat-file --batch-all-objects --batch-check`.
- **Assertions added:** `a_web_edit_into_a_mirror_is_refused` and `a_pull_request_cannot_be_merged_into_a_mirror` now assert the object set is the same before and after the request. The unreadable-record test does too.
- **Red:** none, because the code already refused before writing.
- **Mutation MO** makes `refuse_mirror_write` `hash-object -w` an unreachable blob into served and then refuse. It kills both tests on the new assertions: "a refused web edit wrote objects into served" and "a refused merge wrote objects into served".

## Item 6: rustdoc placement

The "Locked decision 3" doc is now on `mark_clean_store` (`mirror.rs:1552`), extended to cover reachability and alternates. `holds_unreachable_objects` keeps its own doc.

## Item 7: decision recorded

The comment in `resolve_head` (`compare.rs:526-532`) now says: for a mirror base, `pull_merge_info` still runs `merge-tree --write-tree` in served, but only against a head ref it already holds, so whatever that writes is derived from approved content alone.

There is no code change for this item.

## Gates (`fix-a4-gates.txt`, logs in `gate-logs/`, script in `run-gates.sh`)

| # | Command | Result |
|---|---|---|
| 1 | `cargo fmt --all --check` | exit 0 |
| 2 | `cargo test --locked --offline -p solid-pod-rs-forge --features mirror` | exit 0: 308 lib + 13 + 18 + 65 doc tests |
| 3 | `cargo test --locked --offline -p solid-pod-rs-forge` | exit 0: 228 + 13 + 18 + 51 |
| 4 | `cargo test --locked --offline -p solid-pod-rs-server --features forge-full forge` | exit 0 |
| 5 | `cargo clippy … -p solid-pod-rs-forge --features mirror --all-targets -- -D warnings` | exit 0 |
| 6 | `cargo clippy … -p solid-pod-rs-server --features forge-full --all-targets -- -D warnings` | exit 0 |
| 7 | `RUSTDOCFLAGS=-D warnings cargo doc … --no-deps -p solid-pod-rs-forge --features mirror` | exit 0 |
