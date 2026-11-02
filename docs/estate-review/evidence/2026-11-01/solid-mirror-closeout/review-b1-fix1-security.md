# Security re-review: Commit B1, fix round 1 (SP-GIT-02)

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-migration`, on `d6133e7`, uncommitted (`b1/fix1/after-fix1.patch`).
- **Method:** read-only. No cargo was run, nothing was edited, and no worktree was created. I read `b1/fix1/report.md`, the current `migrate.rs`, `lib.rs`, `forks.rs` and `router.rs`, the relevant tests in `migrate_tests.rs`, `solid-pod-rs-git` `config.rs`, `repo/mod.rs`, and my earlier `review-b1-security.md`.
- **Anchors:** paths are under `crates/solid-pod-rs-forge/src/` unless stated otherwise. Line numbers are from the working tree.

## Verdict

Most of the round holds up:
- The held definition, unconditional verify, smart-HTTP path shape, crash resume, attempt counting, bounded lock wait and process-group kill are sound.
- No new path lets an anonymous remote request reach a held object.

Two gaps remain, both on the fail-open side of the claim:
- **N1 (Medium, environmental):** detection errors now gate, but **repository enumeration errors are still silent**. A store that is never listed is never examined, and the pass still reports itself complete. This is the rest of F1.
- **N2 (Low, needs a forged or stale journal):** three early-return branches in `migrate_one` leave a `published` journal untouched. The reader gate treats `published` as open, so the round's Codex #1 fix ("a forged `published` journal does not open a store") holds only for the shape the test covers.

## Earlier findings: status

| # | Earlier finding | Status | Where |
|---|---|---|---|
| F1 | Detection fails open for lost-record stores | **PARTLY FIXED.** Signals are evaluated independently. Any error with no positive signal writes a `detection-error` gate journal. A runtime failure, a panic, or an ungateable store sets `migration_incomplete`. Enumeration errors are not covered: see N1 | `migrate.rs:946-959`, `1246-1251`, `685-714`, `753-759`; `lib.rs:286`, `659` |
| F2 | No held filter or verify when `refs/mirror` is absent | **FIXED** (details below) | `migrate.rs:1789-1828`, `2031-2047`, `2068-2112` |
| F3 | A crash never consumes the retry budget | **FIXED.** The attempt is counted and persisted before the build, and `MAX_ATTEMPTS` applies to every state except a store already placed | `migrate.rs:1473-1488`, `1301-1304` |
| F4 | Smart-HTTP gate checked segments 1–2 only | **FIXED.** `smart_http_path` accepts exactly two `valid_name_segment` segments, so `%`, `/` and leading `.` are all refused. The CGI is always handed `/<o>/<n>.git<suffix>`, and `find_git_dir` (`solid-pod-rs-git/src/config.rs:34-58`) opens that same directory, or its `.git` child, under the same gate | `lib.rs:156-166`, `772-784` |
| F5 | An owner push in flight across instances can be lost | **ACCEPTED** and documented (runbook step 0). Integrity of the owner's own push only | report "Accepted" |
| F6 | Fork onto a journalled name | **FIXED.** A fork onto a journalled name is refused with 503 before the destination check. A cross-instance race remains, in F5's class | `forks.rs:286-292` |
| F7 | Unbounded lock wait; orphaned helpers | **FIXED.** `try_lock` is polled until `lock_wait`, and a timeout writes an exclusive gate journal. A budget timeout kills every registered process group. Total boot time is still unbounded (accepted, documented) | `migrate.rs:1069-1098`, `1225-1238`, `1499-1531` |
| minor | `journal_gates` followed symlinks | **FIXED.** `read_journal` uses `symlink_metadata`, and any non-regular file is `Bad` and gates | `migrate.rs:628-659` |
| minor | `clear_staging` prefix matched `<name>.git.<x>` repositories | **FIXED.** The leaf must now be all alphanumerics | `migrate.rs:1408-1413` |
| minor | CHANGELOG part 1's stale sentence | **FIXED.** It now reads "migrated at boot (part 2)". Two small imprecisions remain (N6) | `CHANGELOG.md` |

**F2 in detail.** `refs/mirror` no longer matters:
- The roots R are the safe-named branches and tags.
- H is closure(`--all`) minus closure(R).
- A ref outside R is copied only when it reaches nothing in H.
- `verify` runs `present_count` whatever H holds, then `outside_roots`. That check requires every object in staging to lie in closure(R) as computed in the source. With no roots it returns the staged count, so it fails closed.

So staging is held to closure(R) by the verify step, independently of how complete H is. That also covers C17: objects no ref reaches are neither in H nor pushed.

## New findings, ranked

### N1. Medium. Enumeration errors leave legacy stores unexamined, and the pass reports complete. CONFIRMED (code path); the trigger is environmental

- **Where:**
  - `migrate.rs:735` takes its candidates from `repo::list_all`.
  - `repo/mod.rs:39-58`: `list_all` returns an empty list when `read_dir(repos)` fails, and `while let Ok(Some(..))` silently ends the scan on the first iteration error.
  - `repo/mod.rs:62-80`: `list_owner` does the same per owner, and skips an entry whenever `entry.path().is_dir()` is false, which includes a transient `stat` error.
  - `run` then returns `true` (`migrate.rs:759`), so `migration_incomplete` is not set.
- **What happens:**
  - A lost-record legacy store that the listing missed gets no journal and no flag.
  - The readers do not enumerate. `resolve_repo` and `gated_git_dir` `stat` `repos/<o>/<n>.git` directly (`lib.rs:617-628`, `694-703`), so the store is served with 200, held objects included, for the life of the process.
  - Triggers include:
    - `EMFILE` or `ENFILE` at boot;
    - an I/O error while iterating the directory;
    - `repos/` or `repos/<owner>/` having search but not read permission, so `stat` of a known path works but `read_dir` fails;
    - a symlinked store whose target is briefly unavailable while the scan runs.
- **Who can reach it:** anonymous readers, once the environment has made the scan fail. No request can cause it: the scan runs before the service exists.
- **Fix:** give migration its own enumeration that returns `Err` on any `read_dir`, `next_entry` or `metadata` error other than `NotFound`. On `Err`, make `run` return `false`, so the existing service-wide flag gates every unmarked 64-hex store. Also, in `detect`, return `Detected::Error` rather than `None` when `is_dir(&p.served)` fails with anything other than `NotFound` (`migrate.rs:915-917`; `is_dir` at `846-848` maps every error to false). That is the same transient-`stat` class. The CHANGELOG's "if detection cannot examine a store" would then be true of enumeration too.

### N2. Low. A `published` journal left in place by an early return opens the reader gate. CONFIRMED (code path); the trigger needs a forged or stale journal

- **Provenance:** this is pre-existing and only partly addressed by item 5. It is not introduced by this round. In `after-c`, `journal_gates` also let `Published` through, and a forged one was finished and removed. Item 5 closed the finish path but not the reader gate.
- **Correction:** this corrects my B1 review (§1, "Forging the journal"), which said a forged journal "can only add denial". That was wrong for a regular-file `published` journal.

- **Where:**
  - `journal_gates` treats `Published` as "not gated" (`migrate.rs:656`) and falls through to the marker and record checks (`lib.rs:656-664`).
  - `migrate_one` returns `Outcome::Failed` **without rewriting the journal** in three places that a `published` journal can reach:
    1. `contained` refuses the journal (`migrate.rs:1255-1260`);
    2. `attempts >= MAX_ATTEMPTS` (`migrate.rs:1301-1304`);
    3. the served store and its recovery copy both exist, the "a store appeared" branch (`migrate.rs:1305-1317`).
- **Scenario:** a `published` journal exists for a lost-record legacy store that is not "ours" (unmarked, or without the journal's `migration-id`), and one of those three conditions holds:
  - its `recovery` names an existing `mirror-recovery/<o>/<n>.git.<ts>` entry, such as one from an earlier migration;
  - its `attempts` is 3 or more;
  - its `recovery` is a path that `contained` refuses.

  The pass logs and moves on, and the journal stays `published`. `unpublished_mirror` then finds no gating journal, no marker and no record, and serves the store with 200.
- **Test coverage:** `f3_a_forged_published_journal_does_not_open_a_store` (`migrate_tests.rs:1966-1982`) covers only the branch where recovery is absent and `fail()` rewrites the state (`migrate.rs:1318-1327`).
- **Who can produce the state:**
  - Anyone with write access to `mirror-work/` as the server's user. Nothing remote can write there: `DotfileGuard` covers it, and the CGI is confined to `repos/`.
  - An operator restoring `repos/` from a pre-migration backup while a `published` journal remains from a `finish` that failed. A stale `published` journal is a real state: `finish` can fail after `publish` and is retried at the next boot.
- **Why it matters despite needing local access:** the round lists this exact property as must-fix item 5 (Codex #1). The CHANGELOG states that "a journal in any state, `published` included, never finishes a store that does not carry its migration id". That is true, but not finishing is different from not opening the store.
- **Fix (either one):**
  - In `journal_gates`, gate a `Published` journal unless `served_is_ours(served, j.id)`. This costs one extra `lstat` and a small read on the reader path.
  - In `migrate_one`, rewrite any non-`Failed` journal to `Failed` before each early `Outcome::Failed` return. The first option is the more robust of the two, because it does not depend on boot having run.

  Add the three shapes to f3_a.

### N3. Info. Legacy `info/grafts` is not neutralised on the migration's walks. PLAUSIBLE (not probed)

- **Where:** `migration_command` (`migrate.rs:2308-2312`) sets `GIT_NO_REPLACE_OBJECTS` only. The push (`push_refs`, `migrate.rs:1958`) runs through `git_command`.
- **What happens:** if a source store carried an `info/grafts` file that adds a parent to a root commit, the closure computed for H and for `outside_roots`, and the objects `pack-objects` sends, would all follow the graft. A held commit grafted in that way would then count as inside closure(R), be copied, and pass verify. Grafts only *adding* parents keep staging's connectivity check happy.
- **Who can produce it:** no remote party. Neither the legacy engine nor any push writes `info/grafts`, so creating it needs filesystem write access. I did not verify whether current git honours grafts in every one of these commands.
- **Fix:** treat an `info/grafts` file (and a `shallow` file) in the source as a failed detection or build, which fails closed. Alternatively, point `GIT_GRAFT_FILE` at an empty file in `migration_command` and in the push.

### N4. Info. `read_regular` has two remaining gaps, both reachable only with server-user write access

- **Where:** `migrate.rs:595-625`.
- **Same-inode check:** `lstat`, then `open`, then `fstat` with a dev/inode comparison closes the symlink swap. Two gaps remain:
  - A plain `open` (no `O_NONBLOCK`, no `O_NOFOLLOW`) of an entry swapped to a FIFO between the `lstat` and the `open` blocks forever. The reader path runs this in `spawn_blocking`, so it hangs a blocking-pool thread, and that request is never answered (fails closed). At boot it hangs `new`.
  - Parent components (`mirror-work/migrations/<owner>`) are followed through symlinks.
- **Fix:** open with `custom_flags(O_NOFOLLOW | O_NONBLOCK)` through `OpenOptionsExt`. Optionally, check the parent directories against the canonical plugin directory, as `contained` does.

### N5. Info (availability, fails closed). The containment check is inconsistent when an operator relocates `mirror-recovery/` with a symlink

- **Where:** `migrate.rs:1189-1203`. The canonical comparison runs only when `mirror-recovery/<owner>` already exists.
- **What happens:** the first migration under an owner goes through a symlinked `mirror-recovery` (`create_dir_all` follows it). A later migration under that owner, or the resume of the same one after a crash, then fails closed, because the parent now canonicalises elsewhere. A cross-device rename into it fails with `EXDEV` anyway.
- **Impact:** nothing exposed. The store stays gated and needs an operator.
- **Fix:** document that `mirror-recovery/` and `mirror-work/` must be real directories on the plugin filesystem, or apply the check whether or not the parent exists.

### N6. Info. CHANGELOG precision

- **"The held set … includes the history of a held branch that upstream force-pushed away":** the held set includes that history only if some ref still reaches it. Unreferenced history is kept out by `outside_roots`, not by H. The safety property holds; only the sentence is imprecise.
- **"A legacy store larger than that is not migrated":** the cap applies to the rebuilt (roots-only) copy, not to the legacy store.
- The `published` sentence is discussed under N2.

## Specific checks from the brief

**1. Held definition.** H and the copy use the same R:
- R comes from one `for-each-ref` listing (`migrate.rs:1774-1816`).
- H, `reaches_held`, the push refspecs and `outside_roots` all use those SHAs.

Ways an object outside R could get into staging:
- **Replace refs:** off for every walk, through `migration_command`. `pack-objects` turns replacements off itself, so the push sends the original objects, as the report's probe shows. `refs/replace/*` lands in H and is dropped as `outside-roots`.
- **Alternates:** the source's alternates affect only which objects are available, not which refs count. Staging alternates fail verify.
- **Shallow:** a shallow source can only shrink the closure. A shallow push into staging is refused by receive-pack's default, so it fails closed.
- **A ref added between listing and push:** the push is by the listed SHAs. The `Building` journal, written before `build`, makes smart-HTTP refuse writes, and the per-store lock excludes the engine.
- **Grafts:** see N3.

**Empty H with `refs/mirror` gone:** verify still runs both object checks.

**2. Detection-error journal.**
- **Removal:** it is removed only under the lock, only when its reason starts with `detection-error`, and only when a fresh `detect` returns `None`. That needs every signal clean, or the served store gone. Smart-HTTP writes are refused while the journal exists, so the owner cannot change the store in between.
- **Prefix collision:** for a `fail()` journal to collide, a stored git error would have to begin with `detection-error`. Git's first stderr line begins `fatal:`, `error:` or `warning:`. Even then, `None` needs a clean or absent served store, so nothing would be exposed.
- **The flag:**
  - every reader reaches it through `unpublished_mirror`;
  - it is set once in `new` and never cleared during the service's life;
  - the server's `REGISTRY` caches one service per plugin directory.
- **The 64-hex condition:** the reader names that matter (`valid_name_segment`) are a superset of detection's (`is_repo_name`). `is_hex64` and `is_repo_name` are the legacy engine's own validators (`d6133e7:mirror.rs:155-172`), so no legacy mirror store lies outside them. The flag check is broader still: it tests the owner only.
- **Gap:** see N1 for the enumeration gap.

**3. `served_is_ours` and `migration-id`.**
- **Forging:** forging a match needs write access to a file at the root of a store under `repos/`. Receive-pack, auto-init, fork (a `clone`) and web edit or merge (git) cannot write one. The id is a v4 UUID that appears only in the journal, the report and the store root, and none of those is served.
- **Exposure:** `migration-id` is never served:
  - the forge routes only `/info/refs`, `git-upload-pack` and `git-receive-pack` to the CGI (`router.rs:371`, `400-412`), so there are no dumb-HTTP file paths;
  - the raw, HTML and JSON readers and archive read git objects only;
  - LDP never serves `.forge`.
- **Gap:** see N2 for the `published` journal gap.

**4. `read_regular` and containment.**
- Symlink swaps are closed.
- The leftovers in N4 and N5 need server-user write access, or are availability only.
- Containment checks every component (plain components, the exact base, the owner, the leaf pattern).

**5. `smart_http_path`.**
- Mapping a suffix-less path to `<name>.git` opens nothing new: the gate checks `repos/<o>/<n>.git`, and the CGI opens exactly that.
- An owner's suffix-less push now auto-initialises `<n>.git` under the same tombstone and record checks.
- The engine works on filesystem paths and is unaffected.
- Old bare `repos/<o>/<n>` directories become unreachable over smart HTTP. That is availability only, and they were never served by the forge's other readers.

**6. Process-group killing.**
- Only pids noted inside the build's `GROUPS` scope are killed. Each is a git child spawned with `process_group(0)`, so it leads its own group.
- On normal completion a pid is forgotten before its group could be reused.
- On the budget path, the group stays alive while any helper lives, so its id cannot be reused meanwhile.
- If none live, the signal hits only a new group that took the same pid as its leader in the gap of a few milliseconds, and only within the server's uid.
- The engine runs outside the scope.
- No finding.

**7. Notes dropped.** Accepted. A notes commit can reference any blob, so copying `refs/notes/*` would bring back F2. The notes are kept in recovery and reported as `outside-roots`.

**8. The lock-timeout residual.** Fails closed:
- The gate journal is `failed` and gates readers and writers.
- `create_exclusive` never replaces a peer's journal.
- If a peer's journal already exists, no gate is written, and the peer's journal gates until the peer finishes.
- The cost is one redundant rebuild at the next boot, with the old copy kept in recovery. No exposure.
