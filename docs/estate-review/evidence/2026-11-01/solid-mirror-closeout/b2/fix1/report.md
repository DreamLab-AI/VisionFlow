# Commit B2 fix round 1 report (SP-GIT-02)

- **Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-fork-sweep`, `HEAD` `8ba3372`. Nothing is committed or staged; `migrate_fork.rs` and `migrate_fork_tests.rs` are intent-to-add (from B2). No new files this round, and no worktree was created.
- **Patch:** `after-fix1.patch` = `git diff 8ba3372`, sha256 `1997038475c93f31…`, 7 files, +2607 / −98. **Interdiff:** `interdiff-vs-after-b2.patch` (`git diff --no-index` of `../after-b2.patch` and `after-fix1.patch`).
- **Gates:** all nine exit 0 (`gates.txt`, run with `OUT=b2/fix1/gates.txt bash b1/gates.sh`).

| # | Gate | Exit | Result |
|---|------|------|--------|
| 1 | `cargo fmt --all --check` | 0 | |
| 2 | forge `--features mirror` | 0 | lib 414 passed (B2: 405) |
| 3 | forge (default features) | 0 | lib 293 passed (B2: 284) |
| 4 | forge `--features mirror,anchoring,announce` | 0 | lib 475 passed (B2: 466) |
| 5 | clippy forge, all features, `-D warnings` | 0 | |
| 6 | server `--features forge-full forge` | 0 | |
| 7 | clippy forge `--features mirror`, `-D warnings` | 0 | |
| 8 | clippy server `forge-full`, `-D warnings` | 0 | |
| 9 | `cargo doc --no-deps` forge `--features mirror` | 0 | |

## Method

1. **Seam** (no behaviour change): the stamping-block rule moved out of `sweep` into a pure `stamp_blocks(plan, outcome)`, so F3 has a direct unit test. The build was clean.
2. **Tests first:** eight new tests (a ninth came in the review round, step 4), all red on the B2 tree as assertion failures (`after-seams-and-tests.patch`, `red.txt`). One is a unit test in `migrate_fork.rs` (where `Plan` can be built); seven are integration tests in `migrate_fork_tests.rs`.
3. **Fixes:** all eight green (`green.txt`). The full forge suites passed under both feature sets with **no B2 or B1 test changed** (`full-run*.txt`). Then mutations, the CHANGELOG and module docs, and the gates (`gates-round1.txt`, all nine green: 413 / 292 / 474).
4. **Review round** (advisor, before handoff). It found:
   - **An F2 hole:** the reachability walk counted refs the rebuild drops (`refs/mirror/*`, `refs/replace/*`), so a pruned object reached only through one counted as reachable and stayed served.
   - **A wrong C15 claim:** the incomplete-pass fixture never asserted its hardlink.
   - **A weak F2 test:** it did not show that a rebuild happened.

   I added a ninth test, `b2f1_a_pruned_object_reached_only_by_dropped_refs_is_removed`.
   - **First run green** (`red-2.txt` before the fixture change): the fixture was weak, not the code safe. Carol's hardlinked `.keep` pack carries other unreachable objects (the legacy notes, stash and pull-request objects), which kept the counts unequal whatever X1 did.
   - **Fixture fixed:** it points `refs/mirror/dangling-*` at every `fsck --dangling` tip, and asserts as a precondition that every object is then reachable from some ref.
   - **Red:** with that, it failed on behaviour (`X1, reached only by refs the rebuild drops, served by the fork: …/raw/<X1>/held.txt -> 200`, `red-2.txt`, `after-tests-red-2.patch`).
   - **Fix and green:** the walk now has `--exclude=refs/mirror/* --exclude=refs/replace/*` (probed first in a scratch store: the globs match nested names such as `refs/mirror/a/b`). Green in `green-2.txt`; mutation M2b, then the final gates.

   I also added `assert_hardlinked` to the incomplete-pass test, and gave the F2 test a recovery-entry assertion plus a precondition that the fork still has a `.keep` pack.

Before writing F2, I probed git (scratch store): `cat-file --batch-all-objects` is deduplicated across loose objects and several packs (14 listed, 14 unique), and `rev-list --objects --all` prints each reachable object once, annotated tags included. So "holds an object its refs do not reach" is exactly `count(all) > count(reachable)`. Both are streamed and counted, never held, so the check needs no cap.

## Fixes

### 1. F1: lineage the sweep could not parse (Must fix)

- **Parser.** New in `forks.rs`: `StoredLineage { None, Parent, Unknown(raw) }`, `parse_lineage_value` (both parts checked with `valid_name_segment`, the readers' rule) and `read_lineage_checked`, which replaces `read_fork_parent_checked` (removed; its three callers moved). `<hex>/foo.git` now parses.
- **Unknown lineage.** A `forge.parent` that is present but still does not parse (`x/y/z`):
  - is a sweep target with the orphan treatment (checked against every mirror-class report). An ancestor of unknown lineage does the same for its descendants (`Plan::unknown`);
  - is refused by the incomplete-pass rule in `lib.rs` (only `StoredLineage::None` passes);
  - never lets a report be stamped (`stamp_blocks` returns the block-all id);
  - keeps its raw value through a rebuild (`write_fork_config`, and `keep_parent` on a mirror-class rebuild), so it stays unknown on later boots. Dropping it would make the store a root, and its descendants' plans would stop there with no reports, which is the empty-H shape.
- **`h_api_fork`** (in `forks.rs`, not `lib.rs`) refuses with 422 when `!(valid_owner(owner) && valid_repo(name))`, after `resolve_repo`, so a missing source still answers 404.

### 2. F2: objects pruned from the parent (Must fix)

`fork::detect`: when the fork holds no dropped object, `holds_unreachable` runs. It counts `cat-file --batch-all-objects --batch-check=%(objectname)` and `rev-list --objects --exclude=refs/mirror/* --exclude=refs/replace/* --all`, both through `migration_command` (replace objects and grafts disabled), streamed with `STORE_BUDGET`. The walk counts only refs the rebuild keeps (`select` drops `refs/mirror/*` and `refs/replace/*`). More objects than reachable → `Rebuild`. Either walk failing → `Detected::Error` → a gate journal, never clean.

**Scope:** it runs only while the fork has unstamped triggers (its lineage reaches a mirror-class ancestor with an unstamped report), after `rebuilt_against`. Once a parent is stamped, the check stops, as the stamp means. Running it on every boot would rebuild any fork with ordinary garbage (a force-pushed branch) on every boot, forever.

### 3. F3: a gate blocks every report (Must fix)

`stamp_blocks` returns the block-all id for any plan with a gate, whatever the outcome. The unit test is the red: on the B2 tree, a plan gated after collecting `r1` blocked only `["r1"]`. **Why a unit test:** I found no deterministic fixture that reaches this state. Any ancestor gate that a descendant hits is also hit by the intermediate target's own plan, and that plan starts *after* the intermediate store, so it has no reports at the gate and already blocked everything on the B2 tree. The review agrees the window needs a transient error between listing and planning.

### 4. Codex Medium 2: a journalled ancestor under any owner (Must fix)

`plan_for`: the `journal_exists` check moved out of the `is_hex64` block, so it runs for every ancestor.

### 5. Codex fail-open 3: the gate check before the resume path (Must fix)

`migrate_store`: the gated-plan check (B2's review-round guard) moved from just before `rebuild` to just before the `served_is_ours` → `publish` path. It now covers both: a gated fork is neither rebuilt nor published. This changes B2 deviation 11, which said the resume path was unaffected.

### 6. Claim and CHANGELOG (Must fix)

- **Claim:** "the sweep covers objects present when it runs. A fork owner who later pushes held bytes they kept back into their own fork is re-publishing content they hold, which is outside the claim, as risk 4 is." That closes Codex High 1 by narrowing the claim, as the brief directs.
- **"Is dropped" corrected:** the CHANGELOG now says a ref reaching held *beyond the ancestors' roots* is dropped. Where an ancestor root itself reaches held (a followed fork's stale root), the filter cannot drop the ref, verification finds held in staging, and the whole rebuild fails and stays refused (Codex Medium 3, an availability residual).
- **Also added:** lineage parsing and unknown lineage, the F2 trigger, the any-owner journal gate, the resume-path gate, the gate-blocks-all stamping rule, the 422 fork refusal, and the pruned-parent residual narrowed to its reachable case.
- **Module docs:** `migrate_fork.rs` steps 1–3 updated to match.

## Tests

| Test | Item | Shows | Red on the B2 tree (`red.txt`) |
|------|------|-------|--------------------------------|
| `b2f1_a_fork_of_a_dot_git_named_mirror_is_swept` | 1 | The brief's fixture: a legacy mirror moved to `repos/<hex>/foo.git.git`, forked to `carol/foo.git` with the production `build_fork_at`. Preconditions: the hardlink (C15), `forge.parent = <hex>/foo.git`, `parse_fork_parent` rejects it, carol holds held. Held is denied on carol's raw, blob, tree, commit and archive readers (`/forge/carol/foo.git.git/…`); approved is served (200); held is gone from the fork. | `…/carol/foo.git.git/raw/<held>/held.txt -> 200 (held bytes: true)` |
| `b2f1_a_fork_of_unknown_lineage_is_checked_against_every_report` | 1 | A hardlinked fork with `forge.parent = x/y/z` is rebuilt (held denied, approved 200), keeps `x/y/z`, and the parent's report has no `forksCleanAt` | `…/carol/sample/raw/<held>/held.txt -> 200` |
| `b2f1_a_fork_of_unknown_lineage_is_gated_while_the_pass_is_incomplete` | 1 | Erin's config at mode 000 (EACCES precondition; mode restored before drop) makes the pass incomplete. Carol (`x/y/z`) answers 503 | `left: 200` |
| `b2f1_a_fork_whose_lineage_the_parser_rejects_is_refused` | 1 | `POST /forge/api/repos/alice/foo.git.git/fork` answers 422 and creates no store | `left: 201` |
| `b2f1_a_fork_keeping_an_object_pruned_from_the_parent_is_rebuilt` | 2 | The brief's fixture. C17's fork holds X1; then, in the parent, the `.keep` is removed, `reflog expire` and `gc --prune=now` run. Preconditions: X1 is gone from the parent, still in the fork, the fork still has its `.keep` pack, and (after boot) X1 is **not in the parent's `.dropped`**. X1 is denied on carol's readers and gone from the fork, and carol has one recovery entry (it was rebuilt). | `X1, pruned from the parent, served by the fork: …/raw/<X1>/held.txt -> 200` |
| `b2f1_a_pruned_object_reached_only_by_dropped_refs_is_removed` | 2 (review round) | The same fixture, plus fork refs that the rebuild drops: `refs/mirror/old-side` and `refs/replace/<id>` at X1, and `refs/mirror/dangling-*` at every other dangling tip. Precondition: every object is reachable from some ref. X1 is denied and removed, and carol was rebuilt | `X1, reached only by refs the rebuild drops, served by the fork: …/raw/<X1>/held.txt -> 200` |
| `migrate::fork::tests::a_gate_partway_up_the_lineage_blocks_every_report` | 3 | A plan gated after collecting report `r1` blocks every report, for all four outcomes | `a gated plan blocked only ["r1"]` |
| `b2f1_a_journalled_ancestor_fork_under_any_owner_gates_its_descendants` | 4 | dave → carol → M; carol has a planted `failed` fork journal (`attempts: 3`). Carol's journal stays; held is denied on dave; dave's approved answers 503; dave has a journal | `a descendant of a journalled fork is served: left: 200` |
| `b2f1_a_crash_resumed_fork_with_a_gated_ancestor_keeps_its_journal` | 5 | Boot 1 rebuilds carol. Then carol is rewound (a `published` journal with carol's real migration id and recovery entry), and M is gated (a planted mirror journal that has given up). Boot 2: M stays gated, carol's journal stays, carol's approved answers 503 | `a crash-resumed fork with a gated ancestor had its journal removed` |

**C15:** every fork fixture except the API test (which makes no fork) asserts the hardlink precondition (the incomplete-pass test since the review round). The F2 fixture is C17's hardlinked fork with the parent's `.keep` removed afterwards, so the old pack survives only in the fork.

## Mutations

Each mutation copies the file, mutates it, runs one test, restores (`cp`, `touch`, sha256 match), and re-runs green. All ten were caught, every red is an assertion failure (no compile errors), and every restore matched with the re-run green. Files: `mutate.py`, `mutation-M*.txt`, `mutation-M*-restored.txt`, `mutation-summary.txt` (M1a–M5), and `mutation-summary-M2b-walk-counts-dropped-refs.txt` (M2b, run alone in the review round).

| Mutation | Fix | Test | Red |
|----------|-----|------|-----|
| M1a: an unparseable lineage is "not a fork" (the B2 semantics) | 1 | dot-git fork | `held commit served by the fork of a mirror named foo.git` |
| M1b: unknown lineage is a root | 1 | unknown lineage swept | `held commit served by a fork of unknown lineage` |
| M1c: unknown lineage allows stamping | 1 | unknown lineage swept | `a fork of unknown lineage let the parent's report be stamped` |
| M1d: the incomplete-pass rule ignores unknown lineage | 1 | unknown, incomplete | `left: 200` |
| M1e: `h_api_fork` writes any lineage | 1 | API refusal | `left: 201` |
| M2: the unreachable-object trigger off | 2 | pruned object | `X1, pruned from the parent, served by the fork` |
| M2b: the walk counts refs the rebuild drops (excludes removed) | 2 | dropped refs | `X1, reached only by refs the rebuild drops, served by the fork` |
| M3: a gate blocks only the collected reports | 3 | unit test | `a gated plan blocked only ["r1"]` |
| M4: ancestor journals checked for 64-hex owners only | 4 | journalled ancestor | `a descendant of a journalled fork is served` |
| M5: the gate check disabled before the resume path | 5 | crash-resumed fork | `…had its journal removed` |

**Informational, M1x (survives, by design):** a strict parser alone (`valid_owner`/`valid_repo` added back to `parse_lineage_value`), with the unknown-lineage fallback left in place. The dot-git test stays green (`mutation-M1x-strict-parser-only-informational.txt`), because `<hex>/foo.git` then becomes unknown lineage and is still swept, against every report. The two layers are independent; only M1a, which reverts both, reopens F1.

## Deviations

1. **F3's test is a unit test**, not a fixture (see fix 3).
2. **F2 runs only while the fork has unstamped triggers** (see fix 2). The brief says "for a fork whose lineage reaches a mirror-class ancestor with a report"; I read that as the unstamped reports the trigger already uses, so the check stops with the stamp.
3. **Unknown lineage is kept on rebuild**, written back as read (a `git config` argument, a single line). The alternative, dropping it, turns the store into a root and empties its descendants' plans.
4. **B2 deviation 11 is superseded.** The gated-plan guard now also stops the resume path (fix 5).
5. **`h_api_fork` is in `forks.rs`**; the brief's wording put it in `lib.rs`.
6. **B2's `mutate.py` anchors are stale.** Two of B2's mutations no longer apply as written: M8 (ancestor journals not gating) is superseded by this round's M4, since the check moved out of the `is_hex64` block; and M6's anchor string (the tail of `detect`) changed when F2 was added. Re-running `b2/mutate.py` would stop at those two anchor-string assertions. I left B2's driver unchanged, as the record of that round.
7. **`read_fork_parent_checked` removed**, replaced by `read_lineage_checked`; it was `pub(crate)`, so this is no API change. The public `parse_fork_parent` and `read_fork_parent` (used for display and fork counts) keep their strict rule.

## Residuals (recorded, not fixed)

- **Codex Medium 3 (stale roots, availability):** a followed fork's stale root in a descendant's `--not` list can make the descendant's rebuild fail verification and stay refused. In the CHANGELOG.
- **Pruned parent, reachable:** a pruned object that a fork's own ref reaches is kept (no report lists it). Same class as risk 4. In the CHANGELOG.
- **Stamping can stay blocked for good, which keeps F2 armed.** A fork of unknown lineage, or a gate anywhere in a lineage (with item 4, a fork that has given up at `attempts: 3` gates its descendants for good), blocks stamping for every report as long as it exists. Every fork of an unstamped parent is then re-checked on every boot, including F2's unreachable-object check. A fork that later accumulates ordinary unreachable objects (a force-pushed or deleted branch) is then rebuilt once into recovery, possibly long after migration; `rebuilt_against` keeps it from being rebuilt again. That is a real availability and disk cost, not only I/O.
- **One extra boot to clear.** A descendant that sorts before its gated ancestor gets a gate journal on the boot where the ancestor is still gated, and is re-examined only on the next boot.
- **Unsafe-named refs:** a ref whose name is unsafe on a push command line is dropped by the rebuild (`unsafe-name`) but cannot be excluded from the reachability walk by a glob, so it still counts as reaching. In the CHANGELOG.
- **Filesystem writers, outside the threat model (decision 32):** report deletion or truncation, `.dropped`/`.held` counts not checked against the report on read, symlinked owner or recovery directories in the tmp sweep, timestamp collisions in recovery names, the first-versus-last `forge.parent` disagreement with git, and `.held` read after `lstat` rather than through `read_regular`.

## Evidence (`b2/fix1/`)

| File | What |
|------|------|
| `after-seams-and-tests.patch`, `red.txt` | The seam plus tests, and the red run |
| `green.txt`, `full-run*.txt` | The new tests green; full forge suites under both feature sets |
| `after-tests-red-2.patch`, `red-2.txt`, `green-2.txt` | Review round: the ninth test red, then green |
| `gates-round1.txt`, `gates-round1.txt.patch` | The first full gate run, before the review round |
| `mutate.py`, `mutation-*.txt`, `mutation-summary*.txt` | Mutations M1a–M5 and M2b, plus M1x (informational) |
| `gates.txt`, `gates.txt.patch` | The nine gates |
| `after-fix1.patch`, `interdiff-vs-after-b2.patch` | The final patch and the interdiff |
| `SHA256SUMS` | Checksums of everything above |
