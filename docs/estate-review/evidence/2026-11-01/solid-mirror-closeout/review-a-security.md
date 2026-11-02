# Security review — Commit A (SP-GIT-02 candidate isolation and publication)

- Worktree: `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-publication-isolation`, base `31fe8985`, uncommitted diff (8 files).
- Method: read-only review of the diff, `publication-design.md` ("Locked decisions") and the surrounding forge, git and server code. Nothing was edited, built or run.
- Line numbers refer to the working tree at review time.

## Summary

No Critical or High issues. In every path traced, a held or unapproved object never reaches the served object store. There are two Medium findings, three Low, and three informational notes.

## Findings

### Medium

**M1. The announcement handler updates the mirror record without the cross-process lock** (`crates/solid-pod-rs-forge/src/mirror.rs:2039-2040`, `2160-2177`)

- **Problem:** `on_event` sends kind-30617 announcements to `on_announce`, which reads, changes and saves the record holding only the in-process mutex (`lock_for`). It never takes `lock_repo`. The `apply` rustdoc (`2225-2228`) says the lock covers the record's read-modify-write; for announcements it does not.
- **Attack path:** two server instances share one `plugin_dir`.
  1. Instance A reads the record for an announcement.
  2. Instance B applies state S2 and saves `newest_seen_at=t2`.
  3. A saves its stale copy, which still has `t1`.
  4. A relay replays an older, owner-signed state S1, and it passes the replay check (`2281`).

  Branches roll back to S1's commits. Those commits still need a valid trail, so this is a rollback, not a held-object leak. `clone`/`clones` can be reverted the same way.
- **Fix:** take `lock_repo` in `on_event` before branching on kind, and hold it for both paths (it can sit under the in-process mutex).

**M2. The publish size cap measures the whole served store, not the growth** (`mirror.rs:1833-1842`, together with `1488-1517`)

- **Problem:** `run_capped(&served, …, Some(MAX_REPO_BYTES))` kills the push once the total size of served passes 512 MiB. The comment at `1833-1834` says publication may grow served "by at most the cap". Served is never pruned:
  - commits orphaned by approved force moves stay (locked decision 7);
  - an owner-pushed repository may have been large before it became a mirror.
- **Attack path:** the upstream owner, or ordinary history churn, pushes served past the cap. From then on every publish is killed and the mirror stops updating.
- **Second effect:** when a push is killed (timeout or cap), receive-pack's quarantine directory (`served/objects/tmp_objdir-incoming-*`) stays in served.
  - It holds only approved objects.
  - Neither `cat-file` nor http-backend's dumb-HTTP paths look inside it, because its path does not match `objects/xx/yyy` or `objects/pack/pack-*`.
  - It still adds to the size that keeps the mirror stuck.
- **Fix:**
  - Measure served before the push and cap the increase (`size - baseline > cap`).
  - On any publish error, while still holding the lock, remove `served/objects/tmp_objdir-incoming-*`.
  - Correct the comment.

### Low

**L1. Git finds the store by searching from the working directory, not by explicit path** (`mirror.rs:1465-1467`, `1337`, `1808-1812`)

- **Problem:** every `run` relies on `current_dir`, and `publish` relies on `-C candidate`. Nothing passes `--git-dir` or sets `GIT_CEILING_DIRECTORIES`. `ensure_candidate` only checks that the directory exists.
- **Effect:** if `remove_store` fails part-way (`1558-1563` only logs a warning), the next round finds a directory that is no longer a valid repository. `fetch`, `has_commit`, `read_trail` and the push source would then search upward for a parent repository (`data_root/.forge/…` and above). In such a deployment the fetch writes `refs/mirror/*` into an unrelated repository, and the verdict and push read from it.
- **Fix:** pass `--git-dir=<abs>` on every invocation, or set `GIT_DIR`. In `ensure_candidate`, re-initialise (after a full remove) when `HEAD`, `objects/` or `refs/` is missing.

**L2. Error text in the public mirror record leaks internal paths** (`mirror.rs:1845`, `2365`, `2453`)

- **Problem:** only the served path is replaced with `<served>`. Two kinds of text are saved verbatim into `last_seen.reason` and the per-branch verdicts, which the public endpoint `api/repos/<o>/<n>/mirror` serves (`lib.rs:883`):
  - the candidate or plugin-directory absolute path, for example from a `-C` failure or a fetch error on the candidate;
  - the upstream server's `remote:` text, which it controls. This part is not new.
- **Fix:**
  - Also replace `candidate_s` and the `plugin_dir` prefix, in both `fetch` and `publish`.
  - Ideally store a fixed category ("fetch failed", "publish rejected: stale lease") in the public record and keep raw stderr in the tracing log only.

**L3. The marker gate does not cover every reader** (`lib.rs:98-101` says "refused (503) by every reader")

- **Problem:** these call `repo::repo_git_dir` directly and skip `resolve_repo`, so they skip the gate:

  | Reader | Location |
  |---|---|
  | `repo_summary` | `api.rs:321`, reached from the listing at `api.rs:359` and from `api.rs:434` |
  | pulls | `pulls.rs:542` (closed PR page), `602`, `1162` (`pull_diff_page`) |
  | merge | `merge.rs:225` (`pull_merge_info`), `307` |
  | compare | `compare.rs:500`, `526`, `616`, `649` |

- **Effect:**
  - `compare_data` → `resolve_base` accepts a raw SHA, but it is only reached after `resolve_repo` (`compare.rs:697`, `731`). I found no ungated way to read a caller-supplied SHA.
  - The pulls and merge paths read SHAs taken from refs or the PR index, not from the caller.
  - In the new layout this is harmless, because served holds only approved objects.
  - In the legacy window, `repo_summary` runs `for-each-ref` with no pattern, so it includes `refs/mirror/*` and reveals a held commit's committer date.
- **Fix:**
  - Either route these through the gate (a `gated_git_dir` helper), or narrow the rustdoc to "every reader of commit content".
  - In `repo_summary`, limit `for-each-ref` to `refs/heads refs/tags`.

### Informational

**I1. Some repositories get stuck behind the 503 gate (availability, not security)** (`mirror.rs:1791` vs `1807`; `lib.rs:567`, `673`)

- `publish` returns early when `updates` is empty, so `mark_clean_store` never runs.
- A record can exist from an announcement or from an apply where every branch was held.
- A clean owner-pushed repository at a followed key then returns 503 for every reader and for the owner's own smart-HTTP pushes, because the gate runs on writes too. This lasts until some branch passes.
- **Fix:** run the clean-store marking in `apply` once the served store exists (or in `materialise`), not only inside `publish`.

**I2. Publishing an approved commit also publishes its ancestors**

- A held branch whose tip H is an ancestor of an approved tip X becomes readable once X is published, by SHA through any reader.
- More generally, every ancestor of an approved commit is published, whether or not the trail lists it.
- Git cannot avoid this. The `RefUpdate`/`MirrorGit::publish` rustdoc and the design doc should say that approval publishes the whole history, so "held" is not read as "never served".

**I3. Shutdown cancellation**

- `MirrorHandle::stop` (`mirror.rs:2572`) waits only for the relay sessions; drain tasks are not aborted.
- On runtime shutdown or task drop, `kill_on_drop` (`1473`) kills only the git process itself. Its helpers (`git-remote-https`, `index-pack`) keep running until EOF.
- The candidate is not removed. It is private, and its size is re-checked on the next fetch, so nothing leaks.
- A publish dropped mid-push can leave the same quarantine leftovers as M2.

## Not covered by Commit A (legacy window until Commit B)

These are not Commit A defects.

1. **Legacy served stores still hold held objects.** They are gated only by the 503 in `resolve_repo` (`lib.rs:567`) and `h_git_smart` (`lib.rs:665-678`). The ungated readers in L3 still reach them; at minimum `repo_summary` leaks `refs/mirror` tip dates.
2. **Forks made from a legacy mirror before Commit A carry held objects.**
   - `forks.rs:~342` runs `git clone --local --bare`, which hardlinks the source's whole object store, including unreferenced held objects.
   - The fork has no mirror record and no marker, so the gate never applies.
   - Held commits are readable by SHA through raw, HTML, JSON, archive and dumb-HTTP readers.
   - Commit B needs to sweep forks of mirrored repositories, or repack them to reachable objects only, not just the mirrors.
3. **Legacy mirrors stall.** `mark_clean_store` correctly refuses them; legacy code always set `uploadpack.hideRefs`/`transfer.hideRefs=refs/mirror`, at materialise and before every fetch (`HEAD:mirror.rs:1397`, `1418`). They stay at 503 and unpublished until migration. This fails closed, as intended.
4. **Fetched PR heads land in legacy mirror stores.** `compare.rs:~545` fetches `refs/forge/heads/*` into a mirror's served store. Commit B's "every ref except `refs/mirror/*`" copy will carry them over; confirm that is intended.

## Checked and found sound

- **Refspecs:** SHA-only, no `+`, one explicit `--force-with-lease=<ref>:<expect|empty>` per ref (`mirror.rs:1825-1831`).
  - With `--atomic`, a stale lease on the client side rejects every ref before any pack is sent.
  - If served moves between advertisement and receive-pack's ref update, the transaction fails. Only objects from the approved history will have migrated out of quarantine.
- **Push contents:** the push packs only what the approved SHAs reach, minus what served advertises.
  - Thin-pack bases are completed from served's own store.
  - Pack reuse from candidate bitmaps sends wanted objects only.
  - Served has no alternates in either direction, and nothing writes `objects/info/alternates`.
- **Lease snapshot:** `local_heads` reads served after `lock_repo` (`2256`), and the verdict reads come only from the candidate (`1707-1771`). Since SHAs are content-addressed, the window between verdict and publish cannot change what an approved SHA reaches.
- **Input revalidation:** `publish` re-checks `is_branch_ref`, `is_sha40` and the lease, and rejects duplicate branches (`1794-1803`).
  - An empty update list never runs a bare `git push`, which would fall back to `push.default`.
  - `HEAD` is set only to a ref that was pushed, after success, with argv only (`1851`).
- **`porcelain_rejection`:** fails closed on a `!` line or a missing ref line, and a non-zero exit is already an error.
- **`RECEIVE_PACK`** (`1223-1225`):
  - It is a fixed literal that is never interpolated, and git shell-quotes the path it appends.
  - `served_s` follows `--`, and `candidate_s` is a separate, absolute `-C` argument. No ref, SHA, owner or name reaches a shell.
- **Deletions:** refused by `receive.denyDeletes` both on the receive-pack command line and in the served config (`SERVED_CONFIG`). The mirror never generates a deletion.
- **Protocol allow-list:** `GIT_ALLOW_PROTOCOL` is set per call: `http:https` for remote clones, `http:https:file` for operator clones, `file` only for publish.
  - Redirects are off for remote clones, the URL comes after `--`, and the DNS pin from the SSRF check is applied (code not changed in this diff).
- **Hooks:**
  - Client side: `--no-verify` plus the hardened `-c core.hooksPath=/dev/null`.
  - Server side: the `RECEIVE_PACK` `-c` flags plus the served config.
  - `HOME` and `XDG_CONFIG_HOME` are removed.
  - Unverified from source: from git's `local_repo_env` list I believe `GIT_CONFIG_NOSYSTEM` and `GIT_CONFIG_GLOBAL` survive the local transport into receive-pack. Rely on `publication_never_runs_hooks_planted_in_the_served_store` as the evidence.
- **Store placement:**
  - A new served store is initialised, configured and marked in staging, then renamed into `repos/` (`1362-1374`). A rename onto an existing non-empty directory fails closed, and staging is removed on error (`1623-1627`).
  - Candidate, staging and lock paths are siblings of `repo_root`, never under it, by construction (`1275`).
  - The server builds `GitCli` on `plugin_dir/repos` (`server/src/lib.rs:5037`), which matches the CGI root (`forge/src/lib.rs:228`), and the lock path matches `MirrorStore`'s `plugin_dir`.
- **Alternate routes to `.forge/`:**
  - `plugin_dir` is `data_root/.forge` (`server/src/lib.rs:4852`), but `DotfileGuard` checks every path component of the raw path (`solid-pod-rs/src/security/dotfile.rs:108-123`).
  - The LDP handlers use the same raw `uri().path()` and do no percent-decoding, so `%2Eforge` does not resolve to `.forge`.
  - The pod-level git route (`server/src/lib.rs:4724-4740`) takes the pod name from the first segment, which the same guard covers.
  - So no LDP or pod-git route reaches `mirror-work/`. This matters because the candidate deliberately has no `hideRefs`.
- **`h_git_smart` marker gate parsing** (`forge/src/lib.rs:665-678`):
  - It strips `.git` the same way as the write guard and maps to the same directory http-backend's `enter_repo` suffix tries find.
  - `//`, `.` and `..` fail `valid_name_segment` here, and http-backend's `daemon_avoid_alias` (with `GIT_PROJECT_ROOT` set) rejects them too.
  - Dumb-HTTP object paths go through the same gate.
  - The gate and the CGI see the same string.
- **Gate details:** the marker is written by temp file then rename, so a reader never sees a partial one. The record path check works without the `mirror` feature.
- **Fetch failures:** a failed, oversize or timed-out fetch removes the whole candidate (`1696-1700`), so its size cannot ratchet. The candidate is capped as a whole, and the process-group kill takes down the fetch helpers on timeout or cap.
- **Lock:** `File::lock` runs on the blocking pool and is released when the file is dropped or the process dies. A failure to lock leaves the record unwritten (`2232-2249`). The in-process mutex serialises callers within one instance.
- **`browse.rs` `commit_log`:** a full object ID that is not in the store now returns 404 instead of an empty list, matching the other readers.
- **Legacy detection:** `mark_clean_store` detects legacy stores reliably, because every legacy store carries `hideRefs`.
