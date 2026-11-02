# Commit B1 brief (rust-engineer; tests first)

You are implementing Commit B1 of SP-GIT-02 in solid-pod-rs. Your report goes to the delegating lead, who re-runs the gates, has the work reviewed and commits.

**Do not:** commit, push, or create or remove worktrees. If you decline, begin your report with `DECLINED:`.

**Tree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, on branch `fix/mirror-migration`, at `d6133e7` (Solid main with Commit A merged).

**Build:** use only `CARGO_HOME=/home/devuser/workspace/.cargo CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target CARGO_BUILD_JOBS=4`, with `--locked --offline`. No other cargo runs at the same time. After any mutation run, restore the file and then `touch` it: `mv` keeps the old mtime, so cargo would not rebuild and would run the mutated binary.

**Evidence:** put everything under `VisionFlow/docs/estate-review/evidence/2026-11-01/solid-mirror-closeout/b1/`. Write the red output for every new test before its fix, an interdiff per step, and `b1-report.md`.

## Read first

All in the same evidence folder:
- `b-design.md`: the whole design, especially §§1–2.5, 5.2, 5.3, 6 and 7, and **locked decisions 21–26 at the end, which override the body where they differ**;
- `publication-design.md`: decisions 1–20;
- `b-ctl/control*.txt`: the git controls C1–C17.

## Steps

Do them in order, with all gates green at the end of each.

**(a) Pure hoist, with no behaviour change.**
- Move the items listed in b-design §1 point 2 into a new ungated module, re-exported or used from `mirror.rs`: `RECEIVE_PACK`, the init and config half of `build_served` with `SERVED_CONFIG`, `hides_mirror_refs`, a `lock_path(plugin_dir, owner, name)` that accepts `valid_name_segment` owners and is byte-identical for 64-hex owners, record loading, and the process-group runner.
- Add test 20, which pins the constants and the `awaitsMigration` serde key.
- Run all nine gates and save the step-(a) interdiff. The existing tests must be untouched.

**(b) Decision 22, a live defect in Commit A.**
- Test first: a smart-HTTP fetch of a bare repo at `repos/<owner>/.fork-x.tmp` (and a raw or `cat-file` path, if one exists) must be refused. Capture it red.
- Then change `h_git_smart` (lib.rs:750) so that any `gated_git_dir` error other than `NotFound` stops the request, with `PathTraversal` returning 404.
- Show the test green.

**(c) Mirror-class migration (b-design §§2.1–2.5, minus forks), with the decision 21 failure gating.**
- Write tests 1–5, 9–18 and the decision 21 failure test (a lost-record legacy store with an fsck-bad object), all red first.
- Every fixture must be C15-hard: a `.keep` pack, **and** a non-mirror ref reaching held.
- **Forks are out of scope.** Tests 6–8, 7b and 19, the transitive fork sweep, the `.dropped` trigger and the `.fork-*.tmp` sweep belong to B2. You still write the `.held` and `.dropped` report files, because B2 consumes them.
- Mutation-check the key tests:
  - verification switched off: test 4 or 1 goes red;
  - journal gate switched off: the decision 21 test goes red;
  - tombstone switched off: test 12 or 18 goes red.

## Gates

Run these on the final tree, plus once after step (a):
1. `cargo fmt --all --check`
2. forge tests with `--features mirror`
3. forge tests with default features
4. forge tests with `--features mirror,anchoring,announce`
5. server tests with `--features forge-full` (filter `forge`)
6. clippy `-D warnings` on forge with `mirror`
7. clippy `-D warnings` on server with `forge-full`
8. clippy `-D warnings` on forge with `mirror,anchoring,announce`
9. `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps -p solid-pod-rs-forge --features mirror`

The script `scripts/lead-gates-a8.sh` in the evidence folder shows the exact commands.

## Also

- Update the CHANGELOG for every public change: hoisted items, the new module, the journal, the 404 on `PathTraversal`, and the boot migration.
- Write rustdoc on every new public item.
- Write a report that lists, per test, its red message and where the fix is. Include the mutation results, the gate exits and any deviation from the design, with your reasons.

When you finish, reply with only the path of `b1-report.md`.
