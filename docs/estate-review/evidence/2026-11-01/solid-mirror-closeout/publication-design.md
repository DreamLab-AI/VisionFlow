# Mirror publication isolation: design

Worktree: /home/devuser/workspace/wt/solid-pod-rs-mirror-publication (Solid main 31fe898 plus the regression test). Read-only design; nothing in the worktree was edited.

## Verdict

The lead's direction holds, with four changes:

1. The hardening flags don't reach the served side of a local push.
2. Receive-pack quarantine is weaker than the brief assumed.
3. Every staging, candidate and recovery path must live outside `repos/`.
4. Migration must move legacy stores out before the mirror engine starts.

## Evidence (scratch Git controls, git 2.56)

Scratch location: /home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/a1b3b8e8-7072-4498-9afd-e0eee40aa872/scratchpad/ctl

- **Hooks.** `git -c core.hooksPath=/dev/null push <path>` still ran a planted `served/hooks/pre-receive`.
  - Cause: local transport strips `GIT_CONFIG_PARAMETERS` before spawning receive-pack.
  - `--receive-pack="git -c core.hooksPath=/dev/null receive-pack"` suppressed the hook.
  - `GIT_CONFIG_GLOBAL` and `GIT_CONFIG_NOSYSTEM` do propagate: a global file setting hooksPath suppressed the hook.
- **Stale lease, client side.** A stale `--force-with-lease` (including inside `--atomic`) is rejected before any pack is sent. Objects were confirmed absent from served.
- **Ref failure, receiver side.** With a planted `refs/heads/main.lock`, receive-pack migrates the quarantine into the served object DB *before* the ref transaction. The objects stay in served, with no ref pointing at them. Quarantine protects only against unpack, fsck and connectivity failure.
- **New branch.** An empty lease (`--force-with-lease=refs/heads/x:`) works for a must-not-exist new branch.
- **Earlier control.** publication-control.sh: an approved-only push into a fresh store excludes held objects. Ref deletion in an old store leaves them readable.

## 1. Store layout and GitCli method mapping

```
plugin_dir/repos/<owner>/<name>.git                         served; approved closure only; marker file `mirror-publication-v1` inside
plugin_dir/mirror-work/candidates/<owner>/<name>.git        private fetch/validation store
plugin_dir/mirror-work/staging/<owner>/<name>.git.<uuid>    new served stores under construction (renamed into repos/)
plugin_dir/mirror-recovery/<owner>/<name>.git.<unix-ts>     legacy stores moved out; operator recovery; never deleted automatically
plugin_dir/mirror-work/locks/<owner>/<name>.lock            cross-process advisory lock
```

**Why nothing else may live under `repos/`.**
- The Git CGI runs with `GIT_PROJECT_ROOT=repos` and `GIT_HTTP_EXPORT_ALL` (solid-pod-rs-git service.rs spawn_cgi).
- `h_git_smart` (forge/src/lib.rs:569) passes `rel_path` straight through.
- So any bare repo anywhere under `repos/` can be cloned anonymously.
- The interrupted draft's `.{name}.git.migrate-*` inside `repos/<owner>/` is therefore wrong.

**Constructor.** `GitCli::new(repo_root)` (mirror.rs:1226) keeps its signature. It derives the sibling paths from `repo_root.parent()`, as the draft did. Rustdoc must state that `repo_root` is `plugin_dir/repos`.

| Method (mirror.rs) | Store | Notes |
|---|---|---|
| `exists` L1358 | served | |
| `local_heads` L1365 | served | Also the lease snapshot for publish |
| `materialise` L1385 | both | Served: init bare in staging, write fixed config (below) and marker (tmp then rename), rename into `repos/<o>/<n>.git`. Candidate: init bare if absent. |
| `fetch` L1400 | candidate | `+refs/heads/*:refs/mirror/*`, add `--prune`, size cap and timeout measured on the candidate dir |
| `has_commit` L1475 | candidate | |
| `is_ancestor` L1493 | candidate | |
| `read_trail` L1510 | candidate | |
| `update_ref` L1541, `set_head` L1562 | — | Replaced by `publish` (§2, §7) |

**Fixed served config,** written at creation:
- `core.hooksPath=/dev/null`
- `receive.fsckObjects=true`
- `receive.denyDeletes=true`
- No `uploadpack.hideRefs` needed: served never has `refs/mirror`.

**Optional efficiency.**
- Give the candidate `objects/info/alternates` pointing at served `objects/`.
- Fetch negotiation and push then become incremental, and `dir_size(candidate)` counts only candidate-only objects.
- Never put an alternate in the other direction. Test that served has no `objects/info/alternates`.
- If served is rebuilt, the candidate is disposable: delete it.

## 2. Publication mechanism

### Command line

Spawn through `GitCli::run(..., Some("file"), ...)`, so it still uses `hardened_git_async`, `GIT_ALLOW_PROTOCOL=file`, a process group and a timeout.

```
git -C <candidate> push --porcelain --atomic --no-verify
    --receive-pack=<RP>
    --force-with-lease=refs/heads/<b1>:<served_old_sha_or_empty>
    --force-with-lease=refs/heads/<b2>:<...>
    -- <absolute served path>
    +<sha1>:refs/heads/<b1> +<sha2>:refs/heads/<b2>

RP (compile-time const &str):
"git -c core.hooksPath=/dev/null -c core.fsmonitor=false -c core.sshCommand=false -c commit.gpgSign=false -c receive.fsckObjects=true -c receive.denyDeletes=true receive-pack"
```

### Rules

- **No interpolation into `RP`.** Git runs it via a shell for local transport, so no path, owner, name or SHA may ever appear in it. Its content must equal `HARDENED_CONFIG` plus the receive keys; consider building it from `HARDENED_CONFIG` in a `const`, or test equality.
- **Arguments.** Validate the served path through `dir()` (hex64 owner, `is_repo_name`). It goes after `--`. Refs pass `is_branch_ref`; SHAs pass `is_sha40`.
- **Publication invariant (blocking).** The refspec set is *exactly* the SHAs that passed `branch_verdict` in this apply.
  - Push by SHA, never by candidate ref name.
  - Held branches never appear in the push.
  - Pack generation sends only the closure of `<approved shas> --not <served refs>`.
  - Hence on every failure path, including a receiver-side ref failure after quarantine migration, the most that can land in served is approved closure.
- **Force and non-fast-forward approved moves.** `+` keeps today's semantics, where `update_ref` moves to any approved SHA unconditionally. The per-ref lease against the `local_heads` snapshot taken at the start of `apply` (mirror.rs:1915) is the compare-and-swap. A new branch gets an empty lease.
- **HEAD.** Inside `publish`, run `symbolic-ref HEAD <ref>` on served only after the push succeeds, and only if that ref was in the pushed set. This is today's L2069–2072 rule. A `symbolic-ref` failure is logged and does not undo the push.

### Atomicity (a semantic change)

- Today each branch moves independently (L2051–2058).
- With `--atomic`, all approved branches in one state move together or none do.
- On failure, every approved branch is recorded `moved=false`, with the verdict plus `; publish failed: <first stderr line>`. `seen.action=FetchFailed` or `Held`, `failures += 1`, and the next state retries.
- Held branches are excluded either way, so mixed approved/held states still publish the approved set.
- **Alternative if per-branch independence is preferred:** one push per approved ref, each with its own lease. The invariant is unchanged; a partial set may land.

### Failure semantics

| Failure | Effect |
|---|---|
| Stale lease (client side) | Nothing sent, nothing moves |
| Unpack, fsck or connectivity failure | Quarantine discarded, nothing lands |
| Receiver ref-transaction failure | Approved closure objects present but unreferenced, refs unchanged |
| Killed or timed out | Possibly `objects/tmp_objdir-incoming-*` left in served. Readers never consult it, and it holds approved objects only. Swept at startup under the lock. |

No candidate object outside approved closure can enter served on any path.

## 3. Concurrency

- **Readers vs publication.** No lock.
  - Served only ever gains approved objects.
  - Ref updates are atomic (lockfile and rename).
  - Objects appear slightly before refs, but they are approved.
  - Reads during fetch and validation touch a store the mirror isn't writing (served is untouched until publish).
- **Smart-HTTP.**
  - Reads see served only.
  - Pushes to mirrored repos are already refused (lib.rs ~L600–617, keyed on the MirrorStore record).
  - Residual window: the served store exists before the first record is saved. It is bounded by `can_write_namespace(owner)`, so only the owner key. Optional fix: save a stub record before `materialise`.
- **In-process.**
  - `Mirror::lock_for` (mirror.rs:1699) already serialises apply per repo.
  - The server's `forge_service` REGISTRY (server/src/lib.rs ~L5005) holds one engine per `plugin_dir`.
  - `solid_pod_rs_git::push_guard::repo_lock` (push_guard.rs:535) is `pub(crate)`, in-process, and used only for guarded pushes. It adds nothing here; do not widen its visibility.
- **Across server instances on one `plugin_dir`.**
  - Publication: leases give compare-and-swap on served refs. A candidate race between instances (concurrent fetch, or one wiping the candidate) makes a fetch or push fail, which is safe. No file lock is required for publication correctness.
  - Migration does require a cross-process lock: two instances would otherwise rename and rebuild the same store.
  - Use an advisory exclusive lock on `mirror-work/locks/<o>/<n>.lock`. Take the same lock across `apply` too; it is cheap and serialises the candidate.
- **Lock implementation.**
  - `std::fs::File::lock` was stabilised in Rust 1.89; workspace MSRV is 1.88 (Cargo.toml:22).
  - Either bump the MSRV to 1.89 or add `fs4`. No raw `flock`: the crate forbids `unsafe`.
  - Take the lock in `spawn_blocking`, or use `try_lock` with retry, so the runtime never blocks.
- **Residual cross-instance issue (pre-existing).** The MirrorStore JSON record is read-modify-write with no cross-process lock, so replay state (`newest_seen_at`) can race. Holding the file lock across `apply` fixes this as a side effect.

## 4. Candidate lifecycle

**Persistence.**
- The candidate persists between events, so fetches stay incremental.
- `--prune` on fetch stops branches deleted upstream from pinning objects indefinitely.

**Size accounting.**
- `MAX_REPO_BYTES` (512 MiB, mirror.rs:107) applies to the candidate via the existing `run_capped` watcher (L1253–1338) and the post-check.
- Served grows only by approved closure, which is at most the candidate's closure.
- Worst case on disk is about twice the cap per repo, or about the cap plus a delta with the alternates option.

**Failure, timeout, oversize or cancel.**
- Replace `drop_mirror_refs_and_gc` (L1185) with removing the whole candidate directory (`remove_dir_all`). It is private and recreated on next use.
- This is stronger and cheaper than `gc --prune=now`, and the size cannot ratchet.

**Cancellation.**
- `kill_on_drop`, the process-group kill (L1288, L1344) and `MirrorHandle::stop` already end git.
- A partially written candidate is harmless: it is never served. The next fetch reuses it, or it is wiped if over the cap.
- Served leftovers (`tmp_objdir-incoming-*`) are swept by the startup pass (§5) under the lock.

**Timeouts.**
- Fetch keeps `FETCH_TIMEOUT` (60 s).
- Publish: 120 s with the candidate size cap reused. The push writes into served, so cap the *served* growth too: run `publish` under `run_capped` measuring served with `MAX_REPO_BYTES`.

## 5. Migration of pre-repair served stores

**Detection.** A store is legacy when the marker `mirror-publication-v1` is absent AND either:
- a MirrorStore record exists for `(owner, name)`, or
- the store's config has `uploadpack.hideRefs=refs/mirror` / `transfer.hideRefs=refs/mirror`, or
- `refs/mirror/*` exists (loose or in packed-refs).

Scan scope and timing:
- Only `repos/<64-hex owner>/*.git`. Mirror owners are pubkeys, so non-mirror pushed repos are untouched.
- Must NOT be behind `cfg(feature = "mirror")` and must not depend on mirror config being present this boot. A `plugin_dir` mirrored once and booted without `forge-mirror` would otherwise keep serving held bytes. The detection and move-out code therefore lives in plain forge code.

**Where it runs.** In `ForgeService::new` (forge/src/lib.rs:204), synchronously, using `solid_pod_rs_git::hardened_git` (std). It runs after the directory layout is created and before returning. The server's `forge_service` constructs this before any request is dispatched, and `Mirror::start` comes after (server/src/lib.rs ~L5030–5045).

**Algorithm,** per legacy store, under the file lock:

1. **Move out.** Rename `repos/<o>/<n>.git` to `mirror-recovery/<o>/<n>.git.<ts>`. Same filesystem, atomic. From this instant readers get 404 (fail closed), so "no read before migration" holds by construction for raw, HTML, JSON, archive and smart-HTTP.
2. **Clean up.** Remove stale `mirror-work/staging/<o>/<n>.git.*`.
3. **Initialise staging.** `git init --bare --initial-branch=main` in `staging/<o>/<n>.git.<uuid>`, plus the fixed served config (§1).
4. **Copy heads.** List `refs/heads/*` in the recovery store. Heads only: the mirror fetched `--no-tags` and only ever wrote heads. Do not copy tags or notes (the draft did).
5. **Push by SHA.** Run the §2 command from the recovery store into staging: `+<sha>:refs/heads/<b>` for each head, `--atomic`, the same `RP`, empty leases.
6. **HEAD.** Copy it: read `symbolic-ref HEAD` in recovery and set it in staging if it names a pushed branch.
7. **Verify.**
   - Every staging head equals its recovery head.
   - For each recovery `refs/mirror/*` tip not an ancestor of any head, `cat-file -e <tip>` must fail in staging.
   - Record those tips in `mirror-recovery/<o>/<n>.git.<ts>.held.json`, the known held-SHA list used by the restart tests.
8. **Mark.** Write the marker (tmp then rename) in staging.
9. **Publish.** Rename staging into `repos/<o>/<n>.git`.
10. **Wipe the candidate.** Delete `mirror-work/candidates/<o>/<n>.git` if present. It may hold alternates or be stale; it is recreated by the next fetch.

**Crash safety.**
- Recovery entry exists with no served store: rebuild from the newest recovery entry (steps 2–9). This is idempotent.
- Crash after step 1: the served store is absent, so 404, which is safe.
- Staging leftovers are deleted.
- Recovery data is never an alternate and never auto-deleted.
- Also sweep `objects/tmp_objdir-incoming-*` in every marked served store.

**Ordering hazard (blocking).**
- If the rebuild were backgrounded, `Mirror::apply` would see `exists() == false` (mirror.rs:2007) and `materialise` a fresh store.
- The rebuild rename would then fail (target exists) and approved history would be stranded.
- So the rebuild must be synchronous before `Mirror::start`. If it is ever moved off the startup path, `materialise` must refuse while a recovery entry without a completed rebuild is pending.

**Cost.** Boot is blocked by about one local push per legacy repo, up to about 512 MiB. This happens once per store, and failing closed is the accepted price.

**Rebuild failure.** Leave the served store absent (404). Log an error naming the recovery path, and do not fail `ForgeService::new` for other repos.

## 6. Reader gates

**Required: none,** once §5 is in place. Every object in a marked served store is approved closure: commits approved by a past or current verdict.

**Optional defence in depth: a marker gate.**
- In `resolve_repo` (lib.rs:536) and in `h_git_smart` (lib.rs:569), which bypasses `resolve_repo`, return 503 when a mirror record exists but the marker is absent.
- Cost: one `stat` plus one small JSON read per request. It also covers a legacy store restored from backup while running.
- Recommended because it is cheap, but not part of acceptance.

**Rejected: a reachability gate** (is the requested SHA reachable from served heads).
- It costs a history walk per read, and harder still for blob or tree SHAs.
- After isolation it protects nothing new.
- It would also change semantics for commits orphaned by approved force moves (see §9).

## 7. Trait and API changes

`GitCli` is the only `impl MirrorGit` in the workspace (mirror.rs:1357), so an in-place trait change is free internally. It is a public semver break: record it in the changelog.

```rust
/// One approved branch move. `expected` is the served sha the move was
/// planned against (`None` = the branch must not exist yet).
pub struct RefUpdate { pub git_ref: String, pub sha: String, pub expected: Option<String> }

/// Publish exactly these approved commits into the served store,
/// atomically, then point HEAD at `head` if it was among them.
async fn publish(&self, owner: &str, name: &str, updates: &[RefUpdate], head: Option<&str>) -> Result<(), String>;
```

- Remove `update_ref` and `set_head`.
- `apply` (L2030–2073): collect the verdicts first, then call one `publish`, then fill `BranchOutcome.moved` from the result.
- If `prepare_existing` is kept (draft idea): give it a default of `Ok(())` for test fakes, and document that the default is fail-open and that real adapters must override it. Migration itself lives in `ForgeService::new`, not on the trait, so the trait method is unnecessary. Prefer not adding it.
- Add `pub const PUBLICATION_MARKER: &str = "mirror-publication-v1";` with rustdoc.

**Rustdoc to update.**
- L1108 `fetch`: fetch goes into the private candidate, not the served store.
- L1123–1128: the verdict reads evaluate the candidate.
- L1167–1175: GitCli layout, adding the candidate, staging and recovery paths and the "nothing private under repos/" rule.
- L1905 `apply`: verdict, then publish only approved.
- `ForgeService::new` (lib.rs:201): migration and fail-closed behaviour.
- `cargo doc --no-deps` must stay warning-free.

## 8. Test plan

**Design rule.** Held commits must sit OFF the approved lineage, on a side branch as in publication-control.sh. A held commit that is an ancestor of a later approved tip is legitimately published with it.

The current test `held_mirror_commit_is_not_published_through_raw_route` (mirror.rs:2846) builds held on top of approved, which is fine: held is a descendant, not an ancestor. It is the red-to-green anchor.

| # | Acceptance case | Test | Where |
|---|---|---|---|
| 1 | Approved positive controls | Approved SHA returns 200 with exact content on raw, `/commit/<sha>` HTML, tree HTML, JSON API (commit/tree), archive | in-crate (mirror.rs tests via `ForgeService::handle`, feature `mirror`) |
| 2 | Held denial, raw | Existing regression: 404 | in-crate |
| 3 | Held denial, HTML/JSON/archive | `/commit/<held>`, `/tree/<held>`, API commit/tree, archive by held SHA: 404/4xx and no held bytes in the body | in-crate |
| 4 | Held denial, object store | `git cat-file -e <held>` and `<held>:held.txt` fail in served; no `objects/info/alternates` in served | in-crate |
| 5 | Held denial, smart-HTTP | `git fetch <served-url> <held>` and clone-then-`cat-file` fail | in-crate when `git-http-backend` is locatable (skip-with-log otherwise); also server crate |
| 6 | Reads during fetch/validation | Upstream with a large file (or a `file://` upstream behind a slow pack) while fetch runs; concurrent raw reads of the held SHA return 404 and approved returns 200 throughout | in-crate |
| 7 | Mixed approved/held branches | One state: main approved (no mark) plus side held (marked, no trail), or two branches with differing verdicts; only main published; side objects absent | in-crate |
| 8 | Repeated updates | approved1, held (side), approved2 (non-FF force onto another line); served never gains held objects; `local_heads` ends at approved2; non-FF move works | in-crate |
| 9 | Hooks | Plant executable `hooks/pre-receive`, `post-receive` and `update` in served that touch a sentinel file; publish; the sentinel never exists | in-crate |
| 10 | Stale lease | Move a served ref behind the engine's back between plan and publish (fake or a direct `update-ref` in the test); publish fails; no refs move and the new objects are absent | in-crate |
| 11 | Pre-repair migration | Build a legacy store with today's layout (fetch into served `refs/mirror`, heads at approved, no marker, record present) → `ForgeService::new` → held 404 on all readers, approved 200, recovery dir outside `repos/`, `held.json` lists the held tip, marker present | in-crate |
| 12 | Migration with mirror feature off | Same as 11, with forge built without `mirror` (config-only detection) | in-crate (run under default features) |
| 13 | Crash mid-migration | Recovery entry present and served absent (simulate after step 1) → `ForgeService::new` rebuilds; staging leftovers removed | in-crate |
| 14 | Restart | After 11, drop and rebuild `ForgeService` and `Mirror`; each SHA in `held.json` still 404s; a new approved state publishes | in-crate |
| 15 | Cancellation/timeout/oversize | Adapt mirror.rs:2529 (size cap) and :3383 (slow walk): candidate dir removed, served refs and objects unchanged; abort an `apply` future mid-fetch and assert the same | in-crate |
| 16 | Concurrent instances | Two processes running `ForgeService::new` (migration) and two engines applying the same state on one `plugin_dir`: no ENOTEMPTY/lost history, held never served, final heads correct | server crate integration test or a script; in-process threads cannot prove the cross-process lock |
| 17 | `RP` constant | Unit test: `RP` contains every `HARDENED_CONFIG` pair and `receive.fsckObjects=true`, and no format arguments | in-crate |

Also run: the mirror and forge suites, `cargo fmt --check`, clippy with `-D warnings`, and `cargo doc --no-deps` for the forge crate (features `mirror` and default).

## 9. Risks and open questions, ranked

1. **Hook boundary and the `RP` string.**
   - The served side's own config plus `RP` is the only hook boundary on local push; `-c` on the pusher is stripped (verified).
   - A future refactor that interpolates into `RP` is shell injection.
   - Mitigation: a constant, test #9, test #17.
2. **Receiver-side ref failure leaves unreferenced objects.** This is safe only because of the push-by-approved-SHA invariant. Any later change that pushes candidate ref names (`refs/mirror/*`) or a wider refspec silently reopens the defect. Pin it with tests #7 and #10 and a code comment at `publish`.
3. **Boot latency.** The synchronous rebuild in `ForgeService::new` blocks start-up, once per legacy store (up to about the cap per repo). The alternative, backgrounding, needs the `materialise` refusal described in §5. Accept the boot cost.
4. **Old approved commits stay readable.** Commits orphaned by an approved force move remain readable by SHA. They were approved once. Does policy require un-publishing on a force move? If so, that is a separate rebuild-on-force (like migration), not GC.
5. **Lock dependency.** Bump the MSRV to 1.89 (`File::lock`) or add `fs4`. This is an operator or maintainer decision.
6. **Atomic vs per-branch publication.** `--atomic` lets one failing ref delay the others. Per-branch pushes keep today's independence at the cost of partial sets. The recommendation is atomic.
7. **`--prune` on the candidate** changes when held objects leave the private store. It has no effect on what is served.
8. **Window before the first record is saved.** A served store exists before the record is saved, so the owner key could push in that window (existing behaviour). Optional fix: save the record before `materialise`.
9. **Smart-HTTP held-denial test depends on `git-http-backend`** being present in CI. Otherwise it needs to run in the server crate.

## Rejected alternative

Deleting refs plus `gc --prune=now` in the existing shared store:
- Held objects are readable by SHA between fetch and GC.
- Concurrent readers race it.
- publication-control.sh shows legacy objects survive ref deletion.
- It also does nothing for migration.

## Locked decisions (lead, after advisor review, 2026-11-01 ~23:00)

1. **No `+` on publish refspecs.** Lead control on git 2.56:
   - `+<sha>:refs/heads/main` with a stale `--force-with-lease` force-updated anyway and landed the object in served.
   - Without `+`: a stale lease is rejected and the object is absent.
   - Without `+`: a correct lease allows a non-fast-forward move.
   - Without `+`: an empty lease rejects a branch that already exists.
   - Publish uses `<sha>:refs/heads/<b>` with one explicit `--force-with-lease=<ref>:<expect|empty>` per ref.
   - Tests #8 and #10 must run the real production argv through the real `GitCli`.
2. **Migration copies every ref except `refs/mirror/*`,** not heads only. Owner-pushed repos at a 64-hex owner may have become mirrors, so their tags and release refs must survive. New test: an owner-pushed repo with a tag is followed, then the server restarts; the tag survives and the held SHA returns 404.
3. **Marker for existing clean stores.** When `publish` runs against an existing served store with no marker, no `refs/mirror` and no hideRefs, write the marker.
4. **Cross-process lock:** bump workspace `rust-version` 1.88 → 1.89 and use `std::fs::File::lock`.
   - `fs4` is not in `Cargo.lock`, so adding it would break `--locked`.
   - The toolchain is 1.99.1.
   - Record the bump in the changelog.
5. **Migration runs synchronously in `ForgeService::new`.** `start_forge` (`server/src/main.rs:460`) runs before `HttpServer::new`. The lazy request path (`server/src/lib.rs:5086`) goes through the REGISTRY, so a request blocks at most once, before the first service exists. Accepted, documented in rustdoc.
6. **Atomic publication accepted** as a semantic change.
7. **Commits orphaned by an approved force move stay readable** (recorded policy question, not in scope).
8. **Marker gate included:** `resolve_repo` and `h_git_smart` return 503 when a mirror record exists and the marker is absent.
9. **Two commits, each green before the next:**
   - A: candidate store and publication (tests 1–10, 15, 17).
   - B: migration (tests 11–14, 16, and the tag-survival test).
10. **Old-layout tests are inverted, not deleted.** `mirror.rs` ~2570 asserts hideRefs on served; the tests at ~2983 and ~3131 read the served dir. They become positive checks: served has no `refs/mirror`, and the candidate is under `mirror-work/`.
11. **One cargo-running worker at a time.** `CARGO_TARGET_DIR` is shared.

### Fix-round decisions (lead, 2026-11-02, after advisor)

12. **Listings omit unmarked mirrors.** Readers of an unmigrated mirror's refs or content return 503: pages, raw, archive, JSON API, smart and dumb HTTP, pulls, compare (including as head), merge, `resolve_head` and `release_head_ref`. `/api/repos` and the home index leave it out instead of failing as a whole. This exposes no content, and one legacy mirror cannot take the index down for every repository before Commit B. Recorded in the `PUBLICATION_MARKER` rustdoc (forge `lib.rs`).
13. **Shutdown cancellation is not fixed in Commit A** (security I3). `MirrorHandle::stop` does not abort drain tasks. The candidate is private and is re-checked on the next fetch, and quarantine leftovers are swept at the next publish (item 5), so nothing leaks.
14. **Approval publishes history.** Every ancestor of an approved tip is published. "Held" means not reachable from any approved tip (Codex gap 1, security I2); the rustdoc states this.
15. **Evidence classes.** The smart-HTTP test also passes on code without the fix, because the earlier Git HTTP fix refuses unadvertised wants. It is defence in depth, not proof. The tests for items 11 and 12 were green before the fix, and mutations M1–M3 show what they catch. The tests for items 5 and 7 went red only against an intermediate tree. MSRV 1.89 has not been checked with a 1.89 toolchain; only rustc 1.99.1 is available offline.

### Commit B brief additions
- Forks cloned with `--local` from a legacy mirror hardlink held objects and carry no marker. Sweep them or repack them to reachable objects only.
- Decide explicitly whether migration copies `refs/forge/heads/*` (PR heads fetched into legacy stores by compare).
- Legacy stores stall at 503 until migration, because `prepare`/`mark_clean_store` refuses them. This fails closed, as intended.
- Rebuild copies every ref except `refs/mirror/*` (locked decision 2).

### Rounds 2-4 decisions (lead, 2026-11-02)

13. Reachability roots for marking a store clean are `refs/heads/*` and `refs/tags/*` only (round 4). Any followed owner-pushed repository holding objects reachable only through `HEAD`, `refs/forge/heads/*` (open PRs), `refs/notes/*`, `refs/stash` or other namespaces, or dangling objects after a force push, is left unmarked: its readers return 503 and it is not followed until Commit B migration rebuilds it. This is the fail-closed default. *Superseded by decision 18 (round 5); see decision 19 for the marker's meaning. Round 6 narrows "the verdict is persisted" in 18 to verdicts backed by evidence (a trace, an alternates file, an unreachable object, an unreadable record); a check that could not be made is retried and not recorded.*
14. Commit B must decide explicitly whether migration copies `refs/forge/heads/*` (open PR heads) and notes into the rebuilt served store, since decision 13 now routes such repositories through migration.
15. `pull_merge_info` runs `merge-tree --write-tree` in the base store. On a mirror it can only combine already-approved refs, so it writes no unapproved content; accepted and noted in `compare.rs` rustdoc.
16. Accepted limitation: age cannot prove an owner push has stopped, so a quarantine older than `PUBLISH_TIMEOUT` belonging to a stalled push may be swept. Availability only; documented in rustdoc.
17. `MAX_SERVED_BYTES` is checked before a push; one push may overshoot it by at most the per-push growth cap. CHANGELOG wording to say so.
18. Decision 13 is REVISED (round 5, after security review R1): scanning for unreachable objects every owner-pushed store made nearly every followed owner repo 503 permanently, with no owner recovery. The scan now applies only to stores with pre-release mirror history (a record already present while unmarked, or legacy config/refs/mirror traces); a store first followed under the new code is the owner's, and its unreachable objects are the owner's. The verdict is persisted; the scan runs after the replay check.
19. Marker semantics (round 5/6): for a store first followed after this release, the marker means "publication allowed; legacy layout not detected", not "every object reachable". Owner leftovers are the owner's own content. Commit B must NOT use the marker to decide what to sweep: it sweeps forks of legacy mirrors (found through `forge.parent`) whether or not they are marked.
20. Accepted for Commit A, measure later: verification CPU per relay event and scan memory have no hard limit; the marker is trusted once written; an `isolated: true` record is reused if an operator replaces the served directory by hand.

### Commit B1 decisions (lead, 2026-11-02)

27. Every migration journal gates its store, in every state including `published`; cleanup removes it. Chosen over checking the migration id on each request: same closure, no extra read.
28. An incomplete boot pass (enumeration error, journal write failure, migration thread failure) gates every 64-hex owner, marked stores included, until a restart completes the pass. Availability cost accepted.
29. Notes are not copied to the migrated served store; they stay in the recovery copy.
30. A legacy store whose roots-only rebuild exceeds 512 MiB stays gated (503) for operator action.
31. Smart-HTTP accepts exactly `/<owner>/<name>[.git]/<suffix>`; a suffix-less path opens `<name>.git`.
32. Threat model: anyone able to write the plugin directory, recycled process-group IDs, and several instances sharing one plugin directory are out of scope. The B1 claim is stated for remote attackers.
33. Review coverage: Opus security and code reviews covered B1 and fix round 1; fix rounds 2 and 3 had the Codex cross-vendor audit only.
