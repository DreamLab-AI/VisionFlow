# Security review: Commit B2 (SP-GIT-02 fork sweep)

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-fork-sweep`, uncommitted on `8ba3372` (`b2/after-b2.patch`).
- **Method:** read-only. No cargo was run, nothing was edited, and no worktree was created.
- **What I read:**
  - the brief, `b2/report.md` and `b2-brief.md`;
  - all of `migrate_fork.rs`;
  - the B2 hunks of `migrate.rs` (`run`, `served_stores`, `migrate_store`, `build`, `select_mirror`, `keep_parent`, `Input::Files`), `lib.rs` and `forks.rs`;
  - `compare.rs` `valid_owner`/`valid_repo`, `mirror_record.rs` `is_repo_name`, and where `mirror.rs` takes its follow names from.
- **Anchors:** paths are under `crates/solid-pod-rs-forge/src/`, with line numbers from the working tree.
- **Threat model:** decision 32. Plugin-directory writers, recycled process-group IDs and several instances sharing a plugin directory are listed separately at the end.

## Verdict

The claim does **not fully hold**. One remote-reachable gap remains:
- **F1 (Medium):** a fork whose `forge.parent` the forge's own lineage parser rejects is invisible to the sweep, to the incomplete-pass rule and to stamping. Forks of a legacy mirror whose name ends in `.git` are one real producer.

Within forks the sweep does see, the claim holds:
- **Trigger:** object presence against `.dropped`, never the marker.
- **Ref filter:** the inherited-held filter, with `verify` checking staging against the inherited H as a backstop.
- **Fail-closed paths:** the gated-plan guard, class-mismatched journals, `Ungated`, and an incomplete pass.
- **Order:** the order is independent.
- **Partial forks:** the `.fork-*.tmp` sweep only ever moves directories, and only within `repos/<owner>/`.

F2 is a documented residual that I think is more likely than the CHANGELOG suggests. F3 is a narrow stamping gap.

## Findings, ranked

### F1. Medium. A fork whose `forge.parent` fails `valid_owner`/`valid_repo` is never swept or gated, and its ancestor gets stamped. CONFIRMED (code path); PLAUSIBLE (configuration)

**Where:**
- `forks.rs:91-104`: `parse_fork_parent` accepts a lineage only if `compare::valid_owner` and `compare::valid_repo` pass (`compare.rs:32-55`). `valid_repo` refuses a trailing `.git`. `valid_owner` refuses an owner over 64 bytes, a reserved owner, and a leading `-`, `_` or `.`.
- Repository names under `repos/` follow the looser `valid_name_segment`. Mirror names follow `is_repo_name` (`mirror_record.rs:25-28`), which allows `foo.git`. Follow entries are validated only by `is_repo_name` (`mirror.rs:337-338`), so an operator may follow a repository named `foo.git` (served as `repos/<hex>/foo.git.git`).
- `forks.rs:290`: a fork without a `name` field takes the source name unvalidated, so `carol/foo.git` is a valid fork destination. `forks.rs:398` writes `forge.parent = <hex>/foo.git` into the fork.
- `read_fork_parent_checked` returns `Ok(None)` ("not a fork") for that config (`forks.rs:122-130`). So:
  - the sweep classes it `Lineage::Root` and it is never a target (`migrate_fork.rs:127-129`, `141-145`);
  - the incomplete-pass rule treats it as not a fork (`lib.rs:665-677`);
  - it never blocks stamping, so the legacy parent's report is stamped `forksCleanAt` on the first complete pass (`migrate_fork.rs:189-191`, `604-611`). After that no boot re-examines the parent's forks.

**Both ends are reachable, verified on the routes:**
- `parse_route` strips `.git` only once, both for the fork route (`router.rs:555-558`, the same at `423e8c6`, where the forks in question were made) and for the reader routes (`router.rs:595-597`, e.g. `Route::Raw`).
  - `POST /api/repos/<hex>/foo.git.git/fork` forks the store named `foo.git`.
  - `/<prefix>/carol/foo.git.git/raw/<sha>/…` reaches `resolve_repo("carol", "foo.git")`, which uses `valid_name_segment`.
- Only the JSON action API refuses such a name (`ok_repo`, `api.rs:546`). Raw, HTML and archive do not.

**Scenario:**
1. Before B, the operator followed a repository named `foo.git` under a 64-hex owner, and a user forked it, by default to `carol/foo.git`.
2. The fork is a `clone --local` hardlink of the legacy store, so it holds every held object.
3. After B2's boot, the mirror is rebuilt but the fork is untouched. Any held commit, tree or blob is served with 200 by SHA through carol's raw, HTML and archive readers. Carol's JSON action API refuses the name.

Test 7b shows that the readers serve unreachable objects. A descendant of carol inherits the same unparseable lineage (`carol/foo.git`).

**Who can reach it:** anonymous readers. The producers are an operator name ending in `.git` plus any fork made before migration. For forks of forks, an owner id the forge accepts as a path segment but `valid_owner` rejects is another producer.

**Fix:** for migration and for the incomplete-pass rule, parse lineage with the validator the readers use (`valid_name_segment` for both parts). Treat a `forge.parent` key that is present but unparseable as a fork of unknown lineage: give it the orphan treatment (check it against every mirror-class report) and gate it while the pass is incomplete. Add a fixture with a mirror named `<x>.git`, whose test is red now. Optionally, make `h_api_fork` refuse to write a lineage `parse_fork_parent` would reject.

### F2. Low (documented residual, but realistic). Objects pruned from a legacy parent but kept in a fork's hardlinked packs are never triggered. PLAUSIBLE

**Where:** the trigger is presence of a `.dropped` object (`migrate_fork.rs:374-386`). D is computed against the parent as it was at migration time.

**Why it is realistic:**
- Before Commit A, the legacy engine ran `gc` on the served store after a failed fetch (CHANGELOG part 1).
- That pruned unreachable held objects from the mirror: held tips that upstream force-pushed away, and orphans of approved force moves.
- A fork taken earlier keeps them in its own hardlinked pack, because `gc` repacks the parent and leaves the fork's pack alone.
- If the parent's current held set was fetched after the fork, the fork holds no D object. It is left byte-identical and keeps serving the pruned, formerly held objects by SHA.

**Within the formal claim, but not the intent:** these objects are outside "the held set" as it is computed at migration, but they are content that was never approved.

**Who can reach it:** anonymous readers who know or can guess the SHA. Upstream PR commits are often public.

**Fix (option):** for a fork whose lineage reaches a mirror-class ancestor with a report, also trigger when the fork holds any object its own refs do not reach. That means comparing `cat-file --batch-all-objects` with `rev-list --objects --all` in the fork, streamed and capped like `outside_roots`. A rebuild copies only ref closures, so it removes them. The cost is an occasional rebuild of a fork that only has its own garbage.

### F3. Low. A gate partway up the lineage blocks only the reports collected so far. PLAUSIBLE (transient errors only)

**Where:** `plan_for` returns at the first gate with the reports gathered up to that point (`migrate_fork.rs:225-252`). `sweep` then blocks only `plan.report_ids()`, and blocks all reports only when none were collected (`migrate_fork.rs:176-181`).

**Scenario:**
- Take `dave → F' (64-hex followed fork, with a report) → carol → M`, where checking carol's directory fails transiently at plan time but not during listing.
- M's report is not blocked, so it is stamped.
- On later boots M's dropped set is no longer a trigger for dave. Dave is still triggered through F''s report, and the rebuild filter still uses every plan report, stamped ones included (`inherited_held` iterates `plan.reports`).
- So the window needs a lineage where M is the only trigger and the transient error hits exactly between listing and planning.

**Fix:** on any `plan.gate`, block everything (`blocked.insert(String::new())`), not just the ids collected.

## Brief items with no finding

**1. Trigger completeness.**
- **Missing or unreadable report on a 64-hex ancestor:** gates (`migrate_fork.rs:230-235`, and `read_report` requires both list files, `329-333`).
- **Unreadable report anywhere:** makes `every_mirror_report` fail, which also stops all stamping (`597-602`).
- **Stamped report:** stops only the trigger, never the filter.
- **Fork created after stamping:** its source has either been checked or rebuilt, or is refused by `journal_gates` or the incomplete-pass rule. B1 already refuses forking a journalled source.
- **`forge.parent` naming something other than a store, or a store that is gone:** gets the orphan treatment (every mirror report).
- **Renamed parent:** there is no repository delete or rename route, so only an operator can break lineage.
- **Depth:** bounded by the store count, `visited.len() > lineage.len() + 1`.
- **`.dropped` read:** read completely. `copy_files` lstat-checks the file and streams it whole, and a read or write error turns into an error, which gates (`migrate.rs:2830-2843`, `2906-2911`). The report is created only after `.held` and `.dropped` are synced, so a report never sits beside a half-written list.
- **Gap:** F1.

**2. Inherited-held filter.**
- **Held objects reachable from ancestor roots:** cannot happen for the ancestor's own roots, because H excludes the full root closure.
- **Foreign roots:** roots from a different report (orphan mode, or a followed fork's stale roots) can shield a held object from the `--not` walk. `verify_with_roots` still runs `present_count` of the inherited H against staging (`migrate.rs:2137-2159`, `2178`), so such a rebuild fails closed. That is the worker's stated residual, and I agree.
- **Grafts and replacements:** off for the walks and the push (`MIGRATION_ENV`, `migrate.rs:2792-2795`). `refs/replace/*` is dropped.
- **Pull-request heads (`refs/forge/heads/*`):** filtered like any other ref. The fork's `transfer.hideRefs` is written after the push.
- **Followed fork:** verified against its own H plus the inherited H, with its filtered roots.

**3. Fail-open paths.**
- Every early return in `migrate_store` after a journal exists returns `Failed` with that journal on disk. `Rebuild` writes the journal before any branch below (`migrate.rs:1520-1530`).
- A journal of the other class gates and is left alone.
- The gated-plan guard runs before `rebuild` (`1631-1639`), so a journalled fork with a gated ancestor is never rebuilt with an empty H.
- `Ungated` clears `listed`, which sets the flag.
- A panic in the sweep panics the thread, which also sets the flag.
- `fork::detect` never returns `Finish`.
- **Gap:** F1 (the store is never a target in the first place).

**4. Incomplete-pass fork rule (`lib.rs:665-677`).**
- **Gated now:**
  - every store under a 64-hex owner, unchanged from B1;
  - any repository, under any owner, whose config names a **parseable** `forge.parent` or cannot be read.
- **Not gated:** a non-64-hex user's fork is refused while the pass is incomplete, unless its lineage is unparseable (F1).

**5. Cycles and transitivity.**
- Plans read only mirror-class reports, and all of those exist before the fork pass starts (`migrate.rs:834-837`).
- A grandchild is planned against the mirror ancestor directly, never against its parent fork's rebuild. Detection is therefore order-independent.
- Cycles stop on the `visited` set. The cycle test covers this.

**6. `.fork-*.tmp` sweep (`migrate_fork.rs:634-676`).**
- It only renames, never deletes.
- The source is a direct entry of a `valid_name_segment` owner directory. Symlinked entries are skipped, because `symlink_metadata` reports them as not directories.
- The destination is `mirror-recovery/<owner>/<entry>.<ts>`, which no report pattern can collide with, because of the leading dot.
- The sweep runs at boot, before any fork build in this process.

**7. CHANGELOG.** Accurate apart from two points:
- "reads every served repository's `forge.parent` … follows each fork's lineage" is not true for an unparseable lineage (F1).
- The pruned-parent residual is stated, but its likelihood is understated given the legacy `gc` (F2).

## Out of scope under decision 32 (recorded only)

- **Hand-rolled config parser:** `forks::config_entries` takes the *first* `forge.parent` and ignores subsections, where git takes the last. Only a plugin-directory writer can make the two disagree.
- **`.held` reads:** `inherited_held` opens a `.held` file after `lstat`, not through `read_regular`'s same-inode check (`migrate_fork.rs:453-457`).
- **Separators between streamed lists:** `copy_files` concatenates the lists with no separator. Only a list missing its final newline would merge two ids, and `write_report` always ends a list with one.
- **Symlinked owner directory:** the tmp sweep follows a symlinked `repos/<owner>` directory.
- **No mid-fork journal re-check:** this only matters with several instances.
