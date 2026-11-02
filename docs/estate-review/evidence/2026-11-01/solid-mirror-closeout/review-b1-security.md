# Security review: Commit B1 (SP-GIT-02 mirror-class migration)

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, base `d6133e7`, uncommitted (`b1/after-c.patch`).
- **Method:** read-only code review. No cargo was run, nothing was edited, and no worktree was created.
- **Anchors:** `migrate.rs` and `lib.rs` are under `crates/solid-pod-rs-forge/src/`. Line numbers are from the working tree.
- **Reviewer:** security-reviewer agent (Claude). Advisor consulted once before writing.

## Verdict

The claim holds for **record-bearing and unreadable-record stores** under every failure path I traced: a migration that fails, crashes part-way or is resumed. It does **not fully hold for lost-record stores**:

- **F1:** a detection error leaves the store ungated and served with 200.
- **F2:** when `refs/mirror` is gone, the held filter and the object-level verify are both empty.

The swap, the recovery location, the decision-22 hunk and the reader gate (decision 21) are sound. No finding is reachable by an unauthenticated remote request.

## Findings, ranked

### F1. Medium. Detection fails open for lost-record stores. CONFIRMED (code path); the trigger is environmental

- **Where:**
  - `migrate.rs:652-658`: `legacy_traces` returns `Err`, then a warning and `Detected::None`;
  - `migrate.rs:478-488`: runtime build failure or a panic on the migration thread means "legacy stores stay as they are".
- **State:** a legacy store whose record was deleted but whose traces remain (decision 26's live exposure). It is unmarked and has no record, so its only gate is the journal.
- **What happens:** if any of these fails at boot, nothing is journalled and `unpublished_mirror` (`lib.rs:617-632`) lets the store through:
  - the alternates `symlink_metadata` (any error other than not-found, such as `EACCES`);
  - `for-each-ref` or `config --list` (15 s `quick` timeout on a cold or large store under boot I/O load; a spawn failure such as `EMFILE` or `EAGAIN`);
  - building the runtime.

  No marker and no record means not gated, so held objects are served with 200 on raw, HTML, JSON, archive and smart-HTTP until some later boot detects the store.
- **Also:** the checks run in sequence, so an error in `for-each-ref` hides a positive `hideRefs` signal that the config read would have found (`migrate.rs:677-703`).
- **Who can reach it:**
  - Anyone, anonymously, once the environment has made detection fail. A request cannot cause the failure: detection runs before the listener (`start_forge`), and the lazy path runs before the service exists.
  - The only party who can change these stores at runtime is the 64-hex owner, by pushing. I found no push that makes `for-each-ref --count=1 refs/mirror` or `config --local --list` fail.
- **Fix:** on a detection error, write a gate journal in its own state (for example `undetermined`, with `attempts` unchanged). On the next boot, re-run `detect` under the lock and delete that journal if detection returns `None`. This answers deviation 5's concern, a plain owner store stuck at 503 for good, without failing open.
  - Evaluate the three signals independently, and treat any positive as `Rebuild`.
  - If the runtime cannot be built or the thread panics, gate every unmarked 64-hex store that has no journal, or fail `new`.
  - Today, `ForgeService::new`'s rustdoc ("stays refused (503)") and the CHANGELOG's "closes this live exposure" both overstate.

### F2. Medium. No held filter and no held verify when `refs/mirror` is absent. CONFIRMED (code path); PLAUSIBLE (store shape)

- **Where:**
  - `migrate.rs:1209-1211`: `held_set` returns empty without `refs/mirror` tips;
  - `migrate.rs:1167`: `reaches_held` is skipped when H is empty;
  - `migrate.rs:1371`: `verify` skips the object check when H is empty.
- **State:** a store detected through hideRefs, alternates, a flagged or `isolated:false` record, an unreadable record, or an earlier report, whose `refs/mirror/*` refs are gone (partial cleanup), and which holds an owner non-root ref (`refs/misc/*`, notes, stash) reaching formerly held commits. This is C15's "non-mirror ref reaching held" shape after cleanup.
- **What happens:** every non-root ref is copied unfiltered. The verify step passes trivially, so the previously held commits and blobs are published in the rebuilt, **marked** store and served with 200.
- **Same class, with `refs/mirror` present:** H covers only the *current* tips. Objects of a held branch that upstream force-pushed away (C17, inside the mirror itself) are not in H, so a non-root ref reaching them is copied too.
  - The forge writes no non-root refs into a mirror store except `refs/forge/heads/*`, which is dropped, so only the owner can create such a ref.
  - Risk 4 accepts this for forks. The B1 claim and the CHANGELOG ("a ref outside branches and tags that reaches a held object is left out") state it for mirror stores without that qualification.
- **Fixtures:** every B1 fixture keeps `refs/mirror`, so no test covers this.
- **Who can reach it:** anonymous readers, once the shape exists. Creating the shape needs the owner key or an operator's partial cleanup.
- **Fix:** when the store is a target and H is empty, or more simply always for mirror class, copy only the roots (`refs/heads/*`, `refs/tags/*`). Route every other ref whose closure leaves the root closure (`rev-list --objects R --not --branches --tags` is non-empty) to recovery, with the reason `outside-roots`. Add a fixture with no `refs/mirror` and hideRefs only.

### F3. Medium (availability; fails closed). A crash never uses up the retry budget

- **Where:** `migrate.rs:894-904`: `attempts` is incremented only in `fail()`. `migrate.rs:947-952` writes `Building` without incrementing it.
- **What happens:** if the process dies mid-build (OOM, SIGKILL, a service-manager start timeout), the journal stays at `Building` with the same `attempts`. Every later boot rebuilds the same store and dies again, so `MAX_ATTEMPTS` never trips and the server never starts.
- **Why OOM is plausible on large stores:** `dropped_objects` holds both id sets with no cap (`1461-1483`), and the held set may reach 16 M `String`s (on the order of 1 to 2 GB) before the cap fires (`374`, `1225`).
- **Fails closed:** nothing is served.
- **Fix:** increment `attempts` when an attempt starts, and apply the `MAX_ATTEMPTS` check to `Building`, `Verified` and `Failed` alike.

### F4. Low. The smart-HTTP gate assumes one path shape. PLAUSIBLE

- **Where:** `lib.rs:785-801`. The gate checks only path segments 1 and 2 (stripping one `.git`). `solid-pod-rs-git` resolves the whole prefix before the service suffix (`service.rs:295-299`, `guard.rs`), and a missing directory on a write is **auto-initialised** (`service.rs:337-349`).
- **Checked and closed:**
  - `..` is stripped and then rejected by component (`guard.rs` `extract_repo_slug` and `path_safe`), so `mirror-recovery/` and `mirror-work/` cannot be reached.
  - A `.`, empty or dot-led segment at depth 1 or 2 now returns 404 (decision 22). This works.
  - `/<o>/<n>/…` (no `.git`) resolves `repos/<o>/<n>` and never `<n>.git` (`config.rs:35-59`), so the suffix-less form cannot bypass the gate.
- **Open:** a request at depth 3 or more (`/<o>/<d>/<x>.git/…`) is gated against `<o>/<d>.git` only.
  - Through auto-init, an owner can create such a repository by pushing to their own namespace, so a producer exists. That repository holds only what the owner pushed.
  - Nothing places held content at those depths: staging and recovery are outside `repos/`, and `.fork-*.tmp` sits at depth 2.
  - B1 does not make this worse. B2 adds fork paths, so close it then.
- **Fix:** require exactly `/<owner>/<name>[.git]/<suffix>` and return 404 otherwise. Consider answering 404 when the canonical `<name>.git` is absent, rather than handing the request to the CGI.

### F5. Low. An owner push in flight across instances can be lost. PLAUSIBLE

- **Where:** `lib.rs:762`. The tombstone `stat` is taken without the per-store lock, and the CGI never takes that lock.
- **What happens:** suppose instance X is running a receive-pack into a lost-record store when instance Y boots and migrates it. Y may:
  - snapshot refs mid-push, in which case the ref mismatch at `migrate.rs:1368` fails it closed; or
  - rename the directory after receive-pack has opened it, in which case the push lands in the recovery copy, the client sees success, and the update is lost from what is served.
- **Impact:** integrity of the owner's own push. No held content is exposed.
- **Fix:** document it alongside runbook step 0 (every instance must be stopped or running B), or take the file lock around receive-pack.

### F6. Low. A fork can be created onto a journalled name while served is absent

- **Where:** `forks.rs:283-290`. `h_api_fork` checks `refuse_mirror_write` (record only) and whether the destination exists, but not the tombstone.
- **State:** a lost-record store whose journal is `moved`, or `failed` after the move, so `repos/<o>/<n>.git` is absent. B1 creates this window.
- **What happens:** the 64-hex owner forks some readable repository onto `<n>`. Readers stay gated, because the journal gates. On the next boot, `migrate_one` either gives up ("a store appeared", `migrate.rs:853-864`) or rebuilds from the fork and fails to rename into the existing recovery directory. Either way the migration is stuck.
- **Impact:** availability, owner only. No exposure.
- **Fix:** refuse the destination when `migrate::journal_exists`.

### F7. Low. Boot time is unbounded, and a timed-out build can leave orphan processes

- **Unbounded lock wait:** `take_lock` (`migrate.rs:734-749`) blocks with no bound. A hung peer instance that holds the engine lock stalls `ForgeService::new`, and with it server start, indefinitely. `STORE_BUDGET` wraps only `build` (`migrate.rs:959`).
- **Orphan processes:** when that outer timeout fires, the build future is dropped. `kill_on_drop` (`migrate.rs:216`) kills only the direct child, not its process group, so the `receive-pack` and `pack-objects` processes that `push` spawned can outlive it. The inner runners' `kill_group` does not run on drop.
- **Fix:** wait for the lock with a deadline (`try_lock` in a loop); on timeout, write an attempts-0 gate journal. Make the drop path kill the group.

## Brief items with no finding (stated, because the brief asks)

### 1. Decision 21 gate

**Reach.** Every reader and writer reaches it:
- **readers:** `resolve_repo` (`lib.rs:599`) and `gated_git_dir` (`lib.rs:699`);
- **listing summaries:** `api.rs:322`;
- **smart HTTP:** `lib.rs:794`;
- **web edit:** `web_edit.rs:598`;
- **merge:** `merge.rs:~294`;
- **compare base:** `compare.rs:~499`;
- **head release:** `compare.rs:~643`;
- **fork source:** `forks.rs:271`.

`refuse_mirror_write` checks only the record, but every caller resolves through the gate first.

**States.** `journal_gates` (`migrate.rs:438-452`):
- gates on any parse failure, any read error other than not-found, and every state except `published`;
- treats `published` exactly like "no journal" and falls through to the marker check, and `published` is written only after the rebuilt, marked store has been placed.

**Forging the journal.**
- Only the server's own user can write `mirror-work/`:
  - LDP never serves `.forge` (`DotfileGuard`);
  - the CGI is confined to `repos/` (`..` is rejected);
  - auto-init cannot leave `repos/`.
- A forged journal can therefore only add denial.
- Minor inconsistency: `journal_gates` follows symlinks (`tokio::fs::read`), but `journal_exists` does not (`symlink_metadata`). A dangling-symlink journal therefore blocks pushes but not reads. A symlink to a `published` file elsewhere would open the gate, but planting it needs local write access as the server's user.
- Suggest using `symlink_metadata` and gating on any non-regular file.

### 2. Failure matrix (fails closed except F1)

- **Gated by a `failed` journal:** fsck reject (t21), verify hit (M1 makes t4 go red), timeout, the held-set cap, report or marker write errors.
- **Lock I/O error:** a `failed` gate journal with `attempts: 0`. It is written without the lock, so it can race a peer's `finish` and leave a clean store gated until the next boot rebuilds it: availability only.
- **Oversize:** not a failure mode. Migration passes no size cap (`cap` is `None` throughout).
- **After `MAX_ATTEMPTS`:** stays gated.
- **Unreadable journal:** stays gated, and `migrate_one` returns `Failed`.

### 3. The swap

- Staging is fully built, verified and marked under `mirror-work/staging/` before the second rename (`migrate.rs:959-1013`), so `repos/` never holds a half-built store.
- Between the renames, served is absent and readers get 404. The tombstone stops the engine's `materialise` and `prepare` (`mirror.rs:1243-1249`, 1785, 2079), and stops smart-HTTP writes, so auto-init cannot run there.
- Recovery is at `plugin_dir/mirror-recovery/` (`<data_root>/.forge/…`). That is outside `GIT_PROJECT_ROOT`, and LDP never serves it.
- Every path is server-derived (`valid_name_segment` owner and name, a UUID staging name). None is attacker-chosen.

### 4. Ref filter and verify

**`safe_ref`.** It is stricter than `check-ref-format` for push syntax: no `:`, glob characters, `..`, `@{`, control bytes, or a missing `refs/` prefix.
- A dropped ref's objects are never pushed, because the push names only copied SHAs by refspec and git packs only their closure.
- `verify` then requires the exact ref set (`migrate.rs:1368`).

**`held_set`.**
- It subtracts the full root closure (C16), and `--glob=refs/mirror` implies `/*`.
- `reaches_held` uses the same `--not --branches --tags` boundary and stops at the first hit.
- Both fail closed on a git error or timeout.

**`verify`.** It counts **objects** in staging (`cat-file --batch-check` over H, `migrate.rs:1396-1416`), not refs. Correct, apart from F2.

**Tags.** Upstream tags cannot enter `refs/tags/*`: the legacy fetch has used `--no-tags` since `mirror.rs` was created (`4e97893`).

**Deviation 7 (`Detected::Finish`).**
- `finish` only clears the flag, wipes the candidate and deletes the journal. It changes nothing that is served, because the read gate is the marker, and no push can write a file at the store root.
- A stale report can only skip a rebuild the operator intended (runbook §5.3 run without step 2). That is runbook safety, not exposure.

### 5. Concurrency

- **Locks:** one `flock`-style file lock per store, never two at once.
- **Cross-instance (t16) and engine against migration:**
  - same-process locks on separate file descriptors exclude each other;
  - the engine's in-memory mutex is never taken by migration, so there is no lock-order cycle.
- **`refuse_while_migrating`:** it covers `materialise` and `prepare`, and `publish` is reached only after `prepare` under the same lock.
- **`clear_staging`:** its `"{name}.git."` prefix also matches the staging of a repository named `<name>.git.<x>`. Across instances, this can delete a peer's staging mid-build. That fails closed (Info).

### 6. Runtime

`run_at_boot` joins its scoped thread explicitly, so `scope` does not re-panic. There is no `block_in_place` and no nested `block_on`.
- Inside actix's current-thread main it blocks the boot thread, before the listener starts, as intended.
- In `handle_forge` it runs under `web::block`, holding `REGISTRY` in the blocking pool. Other forge requests wait there, never on a worker.
- No deadlock found. Boot time is unbounded only through F7.

### 7. The five `fix_a` tests

Confirmed: only the `forge_at` construction line moved, the compare test's inline call became `forge.handle(`, and no assertion changed.

### 8. CHANGELOG

- **Part 1's sentence is now false.** "Repositories mirrored before this change are not migrated yet and stay refused (503) until the migration change lands" (`CHANGELOG.md:67-68`) sits in the same Unreleased section as part 2.
- **Part 2 overstates.** Because of F1 and F2, its "closes this live exposure" and its unqualified held-ref filter sentence both say more than the code does.
- **Forks:** "Forks of legacy mirrors are not swept yet" is present and accurate.
