# Commit B design: migration of pre-release served stores (SP-GIT-02)

- **Code:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, at `d6133e7`. Read-only: no edits, no cargo, no worktrees.
- **Line anchors** are in `crates/solid-pod-rs-forge/src/` unless stated otherwise. `server/` means `crates/solid-pod-rs-server/src/`.
- **Scope:** `commit-b-brief.md` obligations 1–6 and its acceptance list, nothing more.

## Evidence: scratch controls (git 2.56.0)

The scripts and their output are in `/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/3bacf5e1-eb41-460b-9c59-dcbf9490e95d/scratchpad/b-ctl/` (`control.sh` and `control.txt`, `control2.sh` and `control2.txt`, `control3.sh` and `control3.txt`).

The fixture is a legacy store built in today's pre-release layout:
- `hideRefs=refs/mirror`;
- `+refs/heads/*:refs/mirror/*` fetched from an upstream with `main` (approved) and an off-lineage `side` (held, with a unique blob);
- `refs/heads/main` set to the approved commit;
- an annotated owner tag `v1`;
- a note under `refs/notes/commits`;
- a foreign PR head under `refs/forge/heads/bob/feat`;
- `refs/misc/stash-held` pointing at the held commit;
- everything packed into one pack by `repack -a -d`.

| # | Control | Result |
|---|---|---|
| C1 | Fork made the way `build_fork` makes it (`clone --local --bare`, forks.rs:344) | Held commit and blob **present**; the fork's pack has the **same inode** as the legacy pack (link count 2) |
| C2 | Fork of that fork (`clone --local` twice) | Held blob present; same inode again. Forks must be found **transitively** |
| C3 | **Chosen method.** Fresh `init --bare`, then `git -C <source> push --no-verify --receive-pack=<const> -- <staging> <sha>:<ref>…` for the chosen refs | Held commit and blob **absent**. Approved commit, annotated tag (`cat-file -t` = `tag`) and note survive. The ref reaching held is dropped by the filter. No alternates. `fsck --unreachable` reports 0 lines |
| C4 | `clone --local --bare` of the old store | Held present (hardlinked; link count 3). **Not acceptable** |
| C5 | `clone --bare <path>` (local is implicit for a path) | Held present. **Not acceptable** |
| C6 | `clone --mirror --no-local` | Held present (it copies `refs/mirror`). **Not acceptable** |
| C7 | `clone --shared` | Held present, through `objects/info/alternates` pointing at the old store. **Not acceptable** |
| C8 | `clone --bare --no-local` | Held absent, but it copies heads and tags only: notes and other owner refs are lost. **Not acceptable** under decision 2 |
| C9 | Fork rebuilt by C3 from the fork | Held absent; no file in the rebuilt store shares the legacy pack's inode. The fork owner's branch `republished`, pointing at the held commit, is dropped by the filter |
| C10 | In-place `repack -a -d` plus `prune --expire=now` on the fork | Held blob **still present**, because a fork ref reaches it. In-place sweeping is rejected |
| C11 | A ref pointing **directly** at a held blob or tree | `for-each-ref --contains` does not report it, and copying it lands the held blob in staging. Object-level filtering is required (see §2.4) |
| C12 | `for-each-ref --contains=<commit absent from store>` | Exit 129, "no such commit". The held set must be intersected with the objects present before any commit-level query |
| C13 | Source fork with `uploadpack/transfer.hideRefs=refs/forge` | Push from the source still copies `refs/forge/heads/carol/x`; hidden refs do not block push |
| C14 | Executable `pre-receive` planted in the destination, pushed into with the constant receive-pack | Hook **not run** |
| C15 | Mutant: delete `refs/mirror/*`, then `gc --prune=now`, in place | Simple fixture: held blob **gone**, so a simple fixture passes the rejected method. With a `.keep` pack: held **present**. With a non-mirror ref reaching held: held **present** |

| C16 | Approved blob reappearing in a held commit: upstream `M1` sets `f=old`, `M2` sets `f=new`, then `side` off `M2` sets `f=old` again | `rev-list --objects refs/mirror/side --not --branches --tags` **includes** M1's approved blob, because `--not` excludes only the boundary trees. Subtracting the full root closure (`rev-list --objects --no-object-names --branches --tags`) removes it: \|H_approx\| = 4, \|H\| = 3 |
| C17 | Fork taken while `side=X1`; upstream then force-pushes `side` to an unrelated `X2`; the legacy store refetches | The fork still holds X1's blob. X1 is **not** in the parent's current `H`, so an `H` trigger has 0 hits and would skip the fork. The dropped set `D` (all parent objects minus the rebuilt store's objects) contains X1, and the `D` trigger has 3 hits |

C15 governs the test plan. Every migration test fixture must include a `.keep` pack, a non-mirror ref reaching held, or a hardlinked fork, so that a delete-plus-gc implementation goes red.

---

## 1. Constraint set

1. **`repos/` is exported whole.** The CGI runs with `GIT_PROJECT_ROOT=repos` and export-all (publication-design §1). Any bare repository anywhere under `repos/` is anonymously cloneable, so staging, recovery and journals must live outside it.
2. **Migration must not depend on the `mirror` feature** (design §5; acceptance test 12).
   - `pub mod mirror` is `#[cfg(feature = "mirror")]` (lib.rs:69–70).
   - Ungated today: `PUBLICATION_MARKER` (lib.rs:117) and the reader gate (`unpublished_mirror` lib.rs:590, `has_mirror_record` lib.rs:611).
   - Gated, and so unusable from migration code: `SERVED_CONFIG` (mirror.rs:1540), `RECEIVE_PACK` (mirror.rs:1474), `hides_mirror_refs` (mirror.rs:1496), `MirrorStore::lock_file` (mirror.rs:1121), `MirrorStore::load` (mirror.rs:1179), and the process-group runner `GitCli::run_output`/`first_line`/`kill_group` (mirror.rs:1788, 1991, 2063).
3. **`ForgeService::new` is synchronous** (lib.rs:232). Its server caller `forge_service` (server/lib.rs:5008) is also synchronous and holds the `REGISTRY` std mutex (server/lib.rs:5013) across `ForgeService::new` (server/lib.rs:5030) and `Mirror::start` (server/lib.rs:5038–5040).
   - It is called from `start_forge` (server/lib.rs:5057), in `#[actix_web::main]` (server/main.rs:260), before `HttpServer::new` (server/main.rs:460–463).
   - The lazy path is `handle_forge` (server/lib.rs:5086), inside an actix worker.
   - actix-rt runs a **current-thread** tokio runtime, so `block_in_place` panics there. The forge crate's tokio has no `rt-multi-thread` (forge Cargo.toml:27).
   - `spawn_blocking` needs an `.await`, and `new` cannot await.
4. **Engine lock.** `Mirror::lock_repo` (mirror.rs:2865) takes `File::lock` on `mirror-work/locks/<owner>/<name>.lock` across `apply` and `on_event`. Migration must take the same file, so that a running engine in another instance is excluded. `lock_file` accepts only 64-hex owners (mirror.rs:1122), but forks live under any `valid_owner` (compare.rs:32).
5. **The engine reads the flag before the marker.** `prepare_served` returns `AwaitsMigration` from the record flag before it looks at the store (mirror.rs:3233). Clearing the flag is therefore part of finishing a migration, not optional.
6. **Marker semantics (decision 19).** The marker means "publication allowed; legacy layout not detected", so it cannot drive the fork sweep. The marker is trusted for mirror stores (decision 20).
7. **Approved roots are `refs/heads/*` and `refs/tags/*`** (decisions 13 and 18; mirror.rs:1711–1720). Copied refs must be every ref except `refs/mirror/*` (decision 2).
8. **A mirror's served store takes only publication writes** (Codex High 1). A foreign PR head is never fetched into a mirror (compare.rs:525), and `release_head_ref` never deletes one there (compare.rs:614).
9. **Volume.** At most `MAX_SERVED_BYTES` per store. Boot blocks once per legacy store (decision 5). Fork discovery must read every repository's config, so it cannot use `count_forks`, which is capped at `FORK_SCAN_CAP` = 400 (forks.rs:35, 120).
10. **`plugin_dir` is `<data_root>/.forge`** (server/lib.rs:4851). LDP never serves it: `DotfileGuard` (server/lib.rs:3769) allows only `.acl`, `.meta` and `.account` (`solid-pod-rs/src/security/dotfile.rs:24`). The same guard protects `mirror-work/` today, so recovery stores under `.forge/mirror-recovery/` are exposed only if an operator adds `.forge` to the dotfile allowlist.

## 2. Recommendation

Add **one ungated module, `migrate.rs`,** called synchronously at the end of `ForgeService::new`. For each store that needs it, the module:
1. builds a fresh served store in `mirror-work/staging/` by **pushing exactly the chosen refs, by SHA, from the old store**;
2. verifies at **object level** that no held object came across;
3. swaps the stores with two renames under the engine's file lock, behind a **migration journal** that also acts as a tombstone;
4. clears `awaitsMigration`.

Forks reached transitively through `forge.parent` are rebuilt the same way.

### 2.1 Legacy detection (obligation 1, obligation 5)

**Scan scope.** Every `repos/<owner>/<name>.git` for which `repo::list_all` (repo/mod.rs:39) and `valid_name_segment` (ownership.rs:101) accept the owner and name. The scan is uncapped.

**Mirror-class targets.** Only 64-hex owners. A store is a target when the marker is absent **and** at least one of these holds:

| Signal | Source | Decidable? |
|---|---|---|
| Record `Valid` with `awaitsMigration: true` | `MirrorStore::load`, mirror.rs:1179; field mirror.rs:1043 | Yes |
| Record `Valid` with `isolated: false` (pre-release record) | mirror.rs:1038 | Yes. Rebuilding also clears S3's permanent 503 |
| Record `Unreadable` (obligation 6) | mirror.rs:1098 | Yes |
| `uploadpack.hideRefs` or `transfer.hideRefs` naming `refs/mirror` (config read with `config --local --list -z`, as in `legacy_traces`, mirror.rs:1880–1884) | config | Yes. Every legacy engine wrote hideRefs before its first fetch (review-a5 S3) |
| Any `refs/mirror/*` ref (`for-each-ref --count=1`, mirror.rs:1864–1879) | refs | Yes. Use git, not a file read, so reftable stores are covered |
| `objects/info/alternates` or `http-alternates` (`has_alternates`, mirror.rs:2122) | store | Yes |
| A completed recovery entry exists for `(o, n)` and the served store has no marker (a legacy store restored from backup) | `mirror-recovery/` | Yes |

**Not targets:**
- a marked store, which is trusted (decision 20);
- an unmarked store whose record is `Valid` with `isolated: true` and no flag. That is an A-era owner store whose `prepare` errored; the engine retries it (S6/S7).

**Lost record (obligation 5).**
- **Record deleted, traces kept:** decidable from the traces and migrated, no record needed.
- **Record deleted and traces removed:** not decidable from store state.
  - If A has already followed the store, it marked it without a scan as a first follow (decision 18), so it looks exactly like an owner store.
  - The candidate signal adds nothing: A creates a candidate only after the store is marked (`materialise_inner` mirror.rs:2295–2305; `fetch_inner` after `prepare`). "Unmarked plus candidate" arises only through tampering.
  - This case gets an **operator runbook** (§5.3).

**Which forks are swept.** Any repository whose `forge.parent` chain (`read_fork_parent`, forks.rs:108; transitive, cycle-safe, depth bounded by the repository count) reaches a mirror-class store that is a target now, or that has a completed recovery entry. The trigger is object presence, never the marker (decision 19): some object of any migrated ancestor's **dropped set `D`** (§2.4) is present in the fork (`cat-file --batch-check`, streamed). A fork with no `D` object present is left untouched.
- The trigger is `D`, not the held set `H`. Control C17 shows that `H`, computed from the *current* `refs/mirror` tips, misses objects of a held branch that upstream later force-pushed away, while a fork made earlier still hardlinks them.
- `D` over-triggers on dropped PR heads and approved orphans. That is harmless: the rebuild keeps the closure of every fork ref.

**A fork that also has a mirror record** (a followed fork) is both. It gets mirror-class config and the marker (§2.2), and its refs are filtered against its own `H` plus the inherited one.

**Partially built forks.** `repos/<owner>/.fork-*.tmp` directories older than one hour, left by a crashed `h_api_fork` (forks.rs:295), are moved to recovery.
- They may hold hardlinked held objects.
- `valid_name_segment` rejects their leading dot, so `gated_git_dir` returns `PathTraversal` rather than `Busy`, and `h_git_smart` (lib.rs:750) passes the request on to the CGI.

### 2.2 Ref selection (decisions 2 and 14)

**Mirror-class stores**, with the source being the old store:

| Namespace | Copied? | Why |
|---|---|---|
| `refs/heads/*`, `refs/tags/*` | Always | The approved roots (decisions 13 and 18). Tags carry owner releases (decision 2) |
| `refs/mirror/*` | Never | Decision 2 |
| `refs/forge/heads/*` (PR heads) | **Never** | Foreign content that never passed a verdict, which A refuses to write into a mirror (High 1, compare.rs:525). See the cost in §3 |
| `refs/notes/*`, `refs/stash`, any other namespace | Yes, **subject to the held filter** (§2.4) | Owner data (decision 2). A ref that reaches a held object is dropped and listed in the report |
| `HEAD` | Symbolic `HEAD` copied if it names a copied branch; otherwise `refs/heads/main` if copied, otherwise the first copied branch | A detached `HEAD` is never a root (mirror.rs:1711) |

**Forks:** every ref except `refs/mirror/*`, including `refs/forge/heads/*` (PR heads into the fork are not mirror content). Every ref, heads included, is subject to the held filter against the inherited held set.

**Config of the rebuilt store:**
- **Mirror class:** fresh `init --bare --initial-branch=main`, then `SERVED_CONFIG`. Hoist only the init and config half of `build_served` (mirror.rs:1691–1693); `build_served` also writes the marker and renames into `repos/`. The marker is written at step 7b, **after** verification.
- **Fork:** fresh init plus an allow-list copied from the old config: `forge.parent`, `http.receivepack`, and `uploadpack.hideRefs`/`transfer.hideRefs` entries equal to `refs/forge/`.
  - **No `SERVED_CONFIG`:** `receive.denyDeletes` would stop the owner deleting branches.
  - The marker is written only if the old fork had one.
- The old config is never copied whole: it is legacy-written and may name hooks or alternates.

### 2.3 Rebuild mechanism (obligation 1)

The steps run per store. The lock is taken first and detection is re-checked under it (§2.5).

1. **Journal.** Write the journal `mirror-work/migrations/<o>/<n>.json` (tmp, then rename) with `{id, ts, class, state: "building", staging, recovery}`. It is also the tombstone (§2.5).
2. **Clear staging.** Remove any stale `mirror-work/staging/<o>/<n>.git.*`. This is safe under the lock: `materialise` builds staging under the same lock.
3. **Initialise.** Init staging at `mirror-work/staging/<o>/<n>.git.<uuid>` (`GitCli::staging`, mirror.rs:1630) and write its config (§2.2).
4. **Compute the held set** in the source (§2.4) and filter the refs.
5. **Push by SHA:** `git --git-dir=<source> push --porcelain --no-verify --receive-pack=<RECEIVE_PACK> -- <abs staging> <sha>:<ref> …` (control C3).
   - Every `<ref>` passes `git check-ref-format`, and every `<sha>` is 40 hex.
   - The receive-pack argument is the hoisted fixed constant. No `+`: staging is empty, so every update is a creation.
   - Batch up to 512 refspecs per push to stay inside argv limits. Atomicity comes from the rename in step 9, not from `--atomic`.
   - Spawn through the hoisted process-group runner with a migration budget of 600 s per store.
   - Use `GIT_ALLOW_PROTOCOL=file`, and remove `HOME` and `XDG_CONFIG_HOME`.
   - Only receive-pack runs in staging. Push from the source runs no hooks: `--no-verify`, and the hardened `core.hooksPath=/dev/null` applies on the sending side.
6. **HEAD.** Set it per §2.2 with `symbolic-ref`.
7. **Verify, failing closed on any miss:**
   - every staging ref equals the source SHA chosen for it;
   - `cat-file --batch-check` of the full held set in staging returns **only `missing`** (catches C11);
   - staging has no `objects/info/alternates` and no `http-alternates`;
   - no file under staging `objects/` shares an inode with a file under the source `objects/`. This is a cheap guard against a future switch to `clone --local`.

   Then (7b) write the marker for mirror-class stores, and for forks that had one, and write `state: "verified"` to the journal.
8. **Report.** Write `mirror-recovery/<o>/<n>.git.<ts>.migration.json`, which `held.json` in design §5 becomes. It holds:
   - the refs copied, and the refs dropped with their reasons;
   - the held-set file name and count;
   - a copy of the bytes of any unreadable record.

   Write `<ts>.held` beside it: the held object ids, one per line. Also write `<ts>.dropped`: every object id in the source (`cat-file --batch-all-objects --batch-check`) that is absent from staging. Forks need both full sets, not tip SHAs: `.dropped` is the fork trigger and `.held` is the fork ref filter.
9. **Swap.**
   - `rename(repos/<o>/<n>.git → mirror-recovery/<o>/<n>.git.<ts>)`; journal `state: "moved"`.
   - `rename(staging → repos/<o>/<n>.git)`; journal `state: "published"`.
   - All of these are on one filesystem under `plugin_dir`.
10. **Clear the flag.** If the record is `Valid` and has `awaitsMigration`, edit it as a `serde_json::Value`: remove `awaitsMigration` (camelCase, mirror.rs:985), keep every other field unchanged, and save with tmp then rename, as `MirrorStore::save` does (mirror.rs:1210). A test pins the key against `MirrorRecord` under feature `mirror`. An `Unreadable` record is left in place (§5.2).
11. **Wipe the candidate** (mirror class): `remove_dir_all(mirror-work/candidates/<o>/<n>.git)`.
12. **Finish.** Delete the journal, then release the lock.

**Why push and not fetch.** With fetch, `upload-pack` runs under the source's legacy config, where hideRefs blocks fetching hidden refs by name. Push runs receive-pack only under the fixed constant in a store we just created (C13, C14). Push is also the same mechanism, and the same `RECEIVE_PACK`, that A's `publish_inner` uses (mirror.rs:2453–2467), so there is one code path to harden.

### 2.4 The held set and the ref filter

- **Held set of a mirror-class store:** in the source, `H = rev-list --objects <every refs/mirror/* tip> --not --branches --tags`, minus the **full closure** `rev-list --objects --no-object-names --branches --tags`. Control C16 shows the subtraction is needed: `--not` alone keeps an approved blob that reappears in a held commit, and the step-7 check would then fail a correct rebuild for ever. Both sets are streamed into `HashSet`s of ids, bounded by the store size.
  - `H` includes commits, trees and blobs. Control C11 shows why object level is needed.
  - It is empty when there are no `refs/mirror` refs. Unreachable remnants are then excluded by construction, since staging holds only the closure of the copied refs, but they do not appear in `H` (§8 risk 4).
- **Filter for a non-root ref `R` (mirror class):** drop `R` if any line of `rev-list --objects R --not --branches --tags` is in `H`. `H` is already disjoint from the root closure, so boundary-tree leakage cannot cause a false drop.
- **Inherited held set of a fork:** the union of the `.held` files of every migrated mirror-class ancestor.
- **Filter for a fork ref `R`:** drop `R` if any line of `rev-list --objects R --not <ancestor root SHAs present in the fork>` is in the inherited `H`.
  - This is sound because the full closure of the ancestors' roots was subtracted from `H`.
  - "Present" is checked with `cat-file --batch-check` first, because of C12.
  - The filter does not see historical held objects that upstream force-pushed away before migration (C17). Those are not in `H`. The fork rebuild still removes them unless a fork ref reaches them (risk 4).
- **Fork trigger:** any id of an ancestor's `.dropped` set present in the fork (C17). Once a complete pass finds every fork of a parent clean, `forks_clean_at` is stamped in the parent's report and the check stops for that parent.

### 2.5 Ordering, startup and concurrency (obligation 2)

**Synchronous, in `ForgeService::new`, after the layout `create_dir_all` (lib.rs:237–239) and before `Ok(Self{..})`.** A 503-until-done gate is rejected:
- The reader gate already returns 503 for record-bearing legacy stores (lib.rs:575, 665, 750).
- A lost-record legacy store has no record, so it is **ungated** today and would be served during a background migration.
- Backgrounding also reopens the materialise race (design §5 "Ordering hazard").

**Runtime.**
- `new` calls `migrate::run(&plugin_dir)` on a **dedicated `std::thread` (scoped) that builds its own `tokio` current-thread runtime and is joined before `new` returns.**
- This works the same from plain sync code, from actix's current-thread runtime and from `#[tokio::test]`.
- It never panics, unlike `block_in_place` or a nested `Runtime::block_on` on the caller's thread.
- It reuses the hoisted async process-group runner, so no second, unhardened spawn path appears.
- At boot it blocks `start_forge` on purpose, before any listener exists (server/main.rs:460–463).

**Lazy path.**
- Wrap the `forge_service(plugin_dir)` call in `handle_forge` (server/lib.rs:5086) in `actix_web::web::block`, so a worker thread is never blocked.
- The `REGISTRY` mutex is still held across migration, so concurrent forge requests wait in the blocking pool until it finishes. That is intended: no request reaches a legacy store.

**Failure isolation.** A per-store failure is logged, with the recovery path redacted to `<plugin>/…` in any public text, and does not fail `new` (design §5). Only layout I/O errors fail `new`, as today.

**Locks.**
- Per store, take `File::lock` on `mirror-work/locks/<o>/<n>.lock`, inside the migration thread, so blocking is fine there.
- Hoist `MirrorStore::lock_file` to an ungated `lock_path(plugin_dir, owner, name)`. It accepts `valid_name_segment` owners, so forks get the same scheme; for 64-hex owners the path is byte-identical to the engine's.
- **Never hold two locks.** A parent's lock is released before its forks are processed, so there is no lock order and no deadlock.
- **Re-detect after taking the lock.** Another instance may have finished the store while this one waited.

**Tombstone.** While `mirror-work/migrations/<o>/<n>.json` exists, the following refuse. This is one `stat` each, ungated:
- `GitCli::materialise_inner` (mirror.rs:2295), with a retryable error;
- `prepare` (mirror.rs:2588), which returns `Err`, so it is retried rather than recorded (S6);
- the smart-HTTP write path (lib.rs:710–740), with 503.

This closes the window the advisor raised: served is absent after a crash between the two renames, and a running engine in another instance would otherwise `materialise` a fresh store over the rename target.

**Two instances, same release.**
- B's `ForgeService::new` blocks on A's lock, then re-detects and skips.
- B's running engine blocks in `lock_repo` until A releases the lock, then sees the flag cleared and the marker present, so `prepare` returns `Ready`.
- B's readers see the old gated store (503) or the new one; between the renames they get 404. Every outcome fails closed.

**Interruption and resumption.** On boot, before detection, every journal is resumed under its lock:

| Journal state | Served | Action |
|---|---|---|
| `building` | old store present | Delete staging and restart from step 2 |
| `verified` | old store present | Restart from step 2 (cheap enough, and simpler than trusting a partial staging) |
| `moved` | absent | If staging exists, re-run step 7 against the recovery entry as source, then step 9b. Otherwise rebuild from the recovery entry (steps 2–9b). Idempotent |
| `moved` | present (something created it in the gap despite the tombstone, e.g. an operator) | Do not overwrite. Move staging to `mirror-recovery/…<ts>.rebuilt`, log an error, leave the journal, and run the runbook |
| `published` | new store present | Steps 10–12 |
| none | marked, record `awaitsMigration: true`, newest recovery report complete | Steps 10–12. Covers a crash after the journal was deleted, or an engine write in between |

## 3. What it costs

- **Boot latency.** One local push per legacy store and per affected fork, up to `MAX_SERVED_BYTES` each, once (decision 5). The steady-state scan at every `ForgeService::new` is:
  - one `stat` for the marker and one for the record, per repository;
  - one config file read per repository, for fork lineage and hideRefs;
  - one `for-each-ref` spawn per unmarked 64-hex store.
- **Disk.** The recovery copy is never deleted automatically, so each migrated store occupies about twice its size until the operator prunes `mirror-recovery/`.
- **Commits orphaned by approved force moves become 404** after migration. Decision 7 kept them readable under the engine, but migration cannot tell them from held remnants. They survive in recovery.
- **Open PRs against legacy mirrors lose their cached head view.**
  - `refs/forge/heads/*` is not copied, and compare refuses to refetch into a mirror (compare.rs:525).
  - `HeadAccess::Cached`/`ExistingPr` then resolve no head.
  - These PRs were already unmergeable, because merges into a mirror are refused (lib.rs:639).
  - The heads stay in recovery, and the report lists them.
- **Fork owners can lose refs from what is served.** A fork owner's ref that reaches a held object (control C9, `republished`) is dropped. It stays in recovery and is listed in the report (open decision D1).
- **A public semver change.** The hoisted items move out of `mirror.rs`; `mirror.rs` re-exports or uses them, with no behaviour change. The tombstone adds checks to `materialise` and `prepare`.
- **Ruled out:**
  - serving any store during its own migration;
  - migration without the lock;
  - copying hidden PR heads into mirrors;
  - trusting the marker for forks.

## 4. Rejected alternative: in-place sweep (ref deletion plus `gc`/`repack`/`prune` in `repos/`)

It loses on evidence, not taste:
- Control C10: `repack -a -d` plus `prune` keeps held objects that any surviving ref reaches.
- Control C15: `gc --prune=now` keeps held objects behind a `.keep` pack or a non-mirror ref.
- Readers race the sweep in the served namespace.
- A failed sweep leaves a half-swept served store with no recovery copy.
- The acceptance list says it is not evidence.

The boot-time `fsck --unreachable` of every 64-hex store (to catch lost provenance) is not part of the design either: it is decision 18's overturned rule and would 503 owner stores again (R1).

## 5. Migration path, PR heads and unreadable records (obligations 4–6)

### 5.1 No flag day

- **Both shapes run side by side.** Stores already marked by A, and stores the engine created, are untouched. Legacy stores keep today's 503 (A's gate) until the first boot of B migrates them.
- **Rollout is a restart.** No configuration change. The migration is idempotent across restarts and across instances.
- **Rolling back to A** leaves rebuilt stores marked and clean, and A serves them. An A binary ignores journals; resuming an interrupted journal needs B.
- **Mixed instances (A and B on one `plugin_dir`).** An A instance does not honour the tombstone. Upgrade every instance before the first B boot (runbook step 0).

### 5.2 Unreadable records (obligation 6)

- **Migration's treatment.** An unmarked store beside an `Unreadable` record is a mirror-class target. It is rebuilt and marked. The record file is never rewritten or removed; its bytes are copied into the recovery report.
- **Afterwards:**
  - readers serve the store, since the marker is present (lib.rs:591);
  - the engine still skips it without saving, so the file stays (mirror.rs:3350–3363, 3123–3126; test mirror.rs:8959);
  - with feature `mirror`, pushes stay refused (lib.rs:724–740, `#[cfg(feature = "mirror")]`). Without the feature, only the namespace guard applies (lib.rs:716), so the owner key can push, as for any marked store with no record.
- **Operator.** Read the report, then do one of:
  - fix the file's permissions;
  - repair the JSON, using camelCase keys and removing `awaitsMigration`;
  - delete the file.

  On the next event the engine re-creates the record or reads the repaired one. `prepare` then finds the marker and returns `Ready`.

### 5.3 Lost provenance (obligation 5): runbook

This applies only when the operator knows a store had mirror history but both its record and its traces are gone.

0. Make sure every instance on the `plugin_dir` runs B.
1. Stop the server.
2. Delete `<plugin>/repos/<o>/<n>.git/mirror-publication-v1`.
3. Write `<plugin>/mirror/<o>/<n>.json` as `{"pubkey":"<o>","name":"<n>","events":0,"awaitsMigration":true}`.
   - `events` is the only field without a serde default (mirror.rs:~994).
   - Any unparseable content also works, as `Unreadable`, but then step 6 needs the §5.2 operator step.
4. Start the server. The reader gate returns 503 from step 3 onwards; B rebuilds the store and clears the flag.
5. Check the report under `<plugin>/mirror-recovery/<o>/`.
6. Delete the record if the repository is not followed.

### 5.4 PR heads already fetched into legacy stores

- **Mirror class:** not copied (§2.2). They stay in recovery and are listed in the report.
- **Forks:** copied, subject to the held filter.

## 6. Failure modes and semantics

The invariants:
- **I1:** no served path shows a held object;
- **I2:** no approved ref is lost; it is either served or in recovery;
- **I3:** the served namespace never holds a half-built store.

| Failure | State left | Detection | Recovery |
|---|---|---|---|
| Push fails, times out (600 s) or is killed | Old store served (gated if it has a record) and staging partial | Journal `building`, error log | Next boot restarts the store from scratch |
| Verify finds a held id in staging, or an inode is shared | Old store served; staging deleted | Error log with the count | Bug: fails closed. A record-bearing store stays at 503; a lost-record store stays as today |
| Disk full during staging | Same as push failure | `ENOSPC` in the log | Free space, restart |
| Crash between the renames | Served absent (404); tombstone present, so materialise, prepare and push refuse | Journal `moved` | §2.5 resume table |
| The store reappears in the gap | Both stores present | Journal `moved` with served present | Rebuilt store set aside; runbook |
| Staging `receive.fsckObjects=true` rejects an old object (e.g. `badTimezone`) anywhere in the copied history | Old store served (gated if it has a record); staging deleted | Push error naming the fsck message id | Permanent until the operator acts. Runbook: inspect the object, then re-run that store's migration with `receive.fsck.<msg-id>=ignore` added to a second fixed constant (never interpolated). This is an operator decision recorded in the report |
| Lock cannot be taken (I/O) | Store skipped this boot | Warning | Next boot |
| Record edit fails after publish | Served clean and marked; engine still skips | Journal `published` stays | Next boot finishes steps 10–12 |
| Rename fails with `EXDEV` (operator split `plugin_dir` across filesystems) | Old store served | Error log | Unsupported; documented |
| Recovery exposed | Held objects readable as files | Operator added `.forge` to the dotfile allowlist | Documented: never allow `.forge` |

**What breaks first under load:** boot time on large `plugin_dir`s with many legacy stores (serial, one push each). The log shows `migrated <o>/<n> in <ms>` per store, plus a final count line.

**Under partition:** only the shared filesystem matters. `flock` on a network filesystem that ignores it breaks cross-instance exclusion. This is already accepted for the engine (design §3).

## 7. Test plan (written first; each must fail before the fix)

All fixtures use the C15-hard shape: a held blob kept by a `.keep` pack, **and** `refs/misc/stash-held` reaching the held commit, plus a direct blob ref for test 4. Test 1's fixture also includes C16's case: a held commit re-adding an approved blob from older history. "All readers" means:
- raw `/raw/<held>/held.txt`;
- HTML `/commit/<held>` and `/tree/<held>`;
- JSON API commit and tree;
- archive by held SHA;
- smart-HTTP `fetch <held>`, plus clone-then-`cat-file`.

Each must return 404 or 4xx with no held bytes in the body. Approved content must return 200 with exact bytes.

| # | Test | Fails before the fix because |
|---|---|---|
| 1 | **Restart migrates a record-bearing legacy store** (C16 blob present: the migration must succeed and the approved blob must stay readable). Record `isolated: false`; build `ForgeService::new`. Expect: held denied on all readers; approved 200; tag `v1` annotated and resolvable; note readable; marker present; `awaitsMigration` cleared; recovery under `mirror-recovery/` and not under `repos/`; `.held` lists the held blob | Today the store stays 503 (A gate), so the approved 200 assertion fails |
| 2 | **Lost-record legacy store** (hideRefs plus `refs/mirror`, no record) | Today the held blob is served with 200 on raw (no gate) |
| 3 | **Feature off.** Test 2 under default features | Migration module absent, so held is served |
| 4 | **Direct held blob or tree ref** (`refs/misc/blob`) is dropped, and the object-level verify would catch it | A commit-level-only filter lands the blob (C11) |
| 5 | **Mutant guard.** A test-only alternative implementation (delete `refs/mirror` plus `gc --prune=now`) run on the fixture must leave held present. This pins the fixture's hardness | Guards against a weak fixture |
| 6 | **Fork of a legacy mirror**, made with the production `build_fork`, then marked by hand (decision 19): rebuilt; held denied on the fork's readers; fork branches and `forge.parent` survive; `receive.denyDeletes` is not set | Today the fork serves held with 200 |
| 7 | **Fork of a fork** (transitive) is rebuilt | A direct-parent-only scan misses it (C2) |
| 7b | **Fork taken before an upstream force-push of the held branch** (C17): the fork is rebuilt and X1's held blob is denied on the fork's readers | An `H`-based trigger skips the fork |
| 8 | **Fork owner's ref reaching held** is dropped and listed in the report; other fork refs survive | Without the filter, `republished` serves held (C9) |
| 9 | **Owner-pushed repository at a 64-hex owner that became a mirror**, with tag `v1`: the tag survives, held is denied (decision 2 test) | Heads-only copying would drop the tag |
| 10 | **Mirror PR head not copied**; existing PR view degrades without a 5xx; the report lists the head | Copying would serve the foreign head's objects |
| 11 | **Unreadable record.** The store is rebuilt and marked, the record bytes are unchanged, and readers serve approved content | Today: 503 for ever |
| 12 | **Crash after the first rename** (simulate the journal at `moved`, served absent): `new` publishes; a `GitCli::materialise` called meanwhile refuses | No resume; materialise wins the race |
| 13 | **Crash after publish, before the flag is cleared** (journal `published`): the next `new` clears the flag; an engine `apply` then follows the store | Flag never cleared, `prepare_served` mirror.rs:3233 |
| 14 | **Restart idempotence.** Second `new`: no new recovery entry, no rebuild (recovery directory count unchanged) | Repeated rebuilds |
| 15 | **After migration and a restart, the redelivered newest 30618 is applied** (not refused as a replay) and a new approved state publishes | Guards the A behaviour this relies on (mirror.rs:3466–3481) |
| 16 | **Concurrent instances:** two processes run `new` on one `plugin_dir` (server-crate integration test or script). Exactly one recovery entry; no `ENOTEMPTY`; held never served | Missing lock or missing re-check |
| 17 | **Runtime:** `ForgeService::new` called inside a `#[tokio::test]` current-thread runtime, and inside `actix_rt::System`, completes without panic | `block_in_place` or nested `block_on` panics |
| 18 | **Smart-HTTP push refused while a journal exists** (503) | No tombstone check |
| 19 | **Stale `.fork-*.tmp`** older than an hour is moved out of `repos/` | Today it is left, and is CGI-reachable |
| 20 | **Constants:** hoisted `RECEIVE_PACK` and `SERVED_CONFIG` are equal to what `mirror.rs` uses; `awaitsMigration` key pinned against `MirrorRecord` serde | Drift |

Test 1 doubles as the per-reader matrix. Smart-HTTP is defence in depth only (decision 15), so the raw, HTML, JSON and archive assertions carry acceptance.

## 8. Risks and open decisions for the lead

- **D1, fork owner refs reaching held (C9).** Recommendation: drop the ref (it stays in recovery and is reported). The alternative is to leave the whole fork out of `repos/` (404) for the operator. The ref is the fork owner's own act, built from content readable before the fix.
- **D2, mirror PR heads.** Recommendation: do not copy (High 1). The alternative is to copy them subject to the held filter, which keeps PR views but puts unvetted foreign objects in a marked mirror store.
- **D3, marked mirror-class store with legacy traces** (only reachable by tampering). Recommendation: trust the marker (decision 20). Including it costs one `for-each-ref` per marked 64-hex store per boot.
- **D4, partial-fork sweep (§2.1).** This touches non-mirror code paths. The one-hour age guard is a heuristic against a concurrent fork build in another instance.
- **Risk 1: tombstone in A code.** `materialise` and `prepare` gain a `stat`. Forgetting either reopens the post-crash race. Tests 12 and 18 pin them.
- **Risk 2: held-set memory.** About 50 bytes per id; a 2.5 GiB store with roughly 10 M objects is about 500 MB in the worst case. Mitigation: a sorted on-disk list from `rev-list | sort` with a streamed merge-join against `batch-check`, if measurement demands it. Start with a `HashSet` and a count cap that fails the store closed.
- **Closed: replay after migration.** The `AwaitsMigration` skip in `apply_locked` (mirror.rs:3466–3481) leaves `last_event` unset and arms no retry. Its comment says the same state, redelivered after migration, still passes the replay check. Step 10 therefore needs to remove only `awaitsMigration`. Test 15 pins this.
- **Risk 4: historical held objects in fork refs.**
  - `H` lists only what the *current* `refs/mirror` tips reach.
  - Not covered: objects of a held branch that upstream force-pushed away before migration (C17, ordinary upstream behaviour), and remnants of a failed legacy gc.
  - Every such object is in `D`. The parent rebuild drops them, and any fork holding them is rebuilt.
  - What escapes is a fork owner's **own ref** that reaches one: it is not filtered, so the object stays served from that fork.
  - That needs a deliberate ref at content read before the fix. The fix would be to filter fork refs against `D` minus the parent's PR-head closure, at the cost of dropping fork branches that reach the parent's approved orphans.
  - Accepted unless the lead wants D1 applied to `D`.
- **Risk 5: boot time.** No progress UI. Documented, with the per-store log line.
- **Risk 6: `.forge` allowlisting** would expose recovery stores. Add a line to the operator documentation.

---

## Locked decisions (lead, 2026-11-02, after advisor)

21. **Failure gating (advisor item 1).** Every reader (`unpublished_mirror`, `gated_git_dir`, `h_git_smart`) returns 503 while `mirror-work/migrations/<o>/<n>.json` exists in any state other than `published`. Any failure leaves the journal at `state: "failed"` with the reason and an attempt count. If the lock cannot be taken, a lock-free gate journal is still written, since it only adds denial. After 3 failed attempts the store is no longer retried at boot and stays gated. This edits A's reader gate; name that hunk in every review brief. New test: a lost-record legacy store with an fsck-bad object in its copied history; after `ForgeService::new`, held objects return 503/404 on raw, HTML, JSON and archive, and the journal persists.
22. **A defect found in B design review (advisor item 2).** `h_git_smart` (lib.rs:750) stops the request only on `Busy`. A `PathTraversal` from an invalid name (for example `repos/<owner>/.fork-x.tmp`) falls through to the git CGI (`GIT_PROJECT_ROOT=repos`, export-all), with no marker gate. Fix: any `gated_git_dir` error other than `NotFound` stops the request, and `PathTraversal` returns 404. Test first: a smart-HTTP fetch of a bare repo at `repos/<owner>/.fork-x.tmp` is refused. Reported as Commit A's defect. The D4 sweep then becomes cleanup, not the security control.
23. **Runtime.** The lazy path reaches migration only if `start_forge` (server/main.rs:460, same `forge_plugin_dir`, before the listener) failed. `web::block` stays as a guard. The forge crate's tokio already has `rt` and `process` (forge Cargo.toml:27).
24. **Split.** B1 = (a) a pure hoist of the shared items into an ungated module, with no behaviour change, all nine gates green and test 20; (b) decision 22; (c) mirror-class migration with decision 21: tests 1–5, 9–18, 20 and the 21 failure test. B2 = forks, the transitive sweep, the `.dropped` trigger and `.fork-*.tmp` cleanup: tests 6–8, 7b and 19. Each is green and reviewed before the next. Handoff checkpoint after B1.
25. **D1–D4 and Risk 4.** As the architect recommended: drop a fork ref that reaches held (report it); do not copy PR heads into mirrors; trust the marker; keep the `.fork-*.tmp` sweep as cleanup. Risk 4 accepted as written.
26. **Live exposure on main today (predates A, not a regression).** A legacy store whose record is deleted but whose traces are intact has no gate, and is served with 200, held objects included. Only record-bearing legacy stores serve 503. B1 closes this.
