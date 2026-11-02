# Commit B2 brief (rust-engineer; tests first): forks of legacy mirrors

Your report goes to the delegating lead, who reruns the gates, has the work reviewed and commits. **Do not commit, push, merge, or create or remove worktrees.** If you decline, start your report with `DECLINED:`.

## Where you are

- **Worktree:** `/home/devuser/workspace/wt/solid-pod-rs-mirror-publication`, branch `fix/mirror-fork-sweep`, at local main `8ba3372`. That is Commit A (`30135fec`) plus Commit B1 (`bcaf9cf6`), both merged.
- **Design:** `b-design.md` in the evidence folder. The fork sections are §2.4 (the `.held` and `.dropped` reports), §3 "Which forks are swept", the fork refs rule, steps 7b and 162, the definitions at lines 179–184, the lock rules at 208–209, tests 6, 7, 7b, 8 and 19, decisions D1 and D4, and risk 4. The controls C1, C2, C9, C10, C13 and C17 are in `b-ctl/`.
- **Locked decisions:** `publication-design.md`, decisions 1–33. Decisions 19, 20, 24 (D1: drop the ref and report it), 25 (D4) and 27–32 constrain this work.

## Scope (B2 = forks only)

1. **Fork discovery.** Read every repository's `forge.parent`, transitively, cycle-safe, with depth bounded by the repository count. Don't use `count_forks` (capped at 400). Any error reading a config, or listing an owner, makes the pass incomplete (decision 28), so it fails closed.
2. **Fork trigger.** Some object from any migrated ancestor's `.dropped` set is present in the fork, checked with streamed `cat-file --batch-check`. Never use the marker (decision 19), and never use `H` alone (C17). Stamp `forks_clean_at` once a complete pass finds every fork of a parent clean.
3. **Fork rebuild** uses the B1 machinery: journal (every journal gates), staging, verify, seal, swap, recovery. Copy every ref except `refs/mirror/*`, including `refs/forge/heads/*`. Drop any fork ref that reaches an object in the inherited `H` (the union of the ancestors' `.held` files), and list it in the report (D1). Grafts and replace objects stay disabled. Keep `forge.parent`. Write the marker only if the old fork had one. A followed fork gets both treatments (§3).
4. **Locks.** The fork lock uses `lock_path` for `valid_name_segment` owners. Never hold two locks: release the parent's lock before processing its forks.
5. **`.fork-*.tmp` directories** under `repos/<owner>/` older than one hour move to recovery (D4). Smart-HTTP already returns 404 for them, so this is cleanup.
6. **Fork creation against a journal.** `h_api_fork` must refuse a source or target that has any journal. B1 added this; confirm it covers a fork created while a parent's journal appears, within one instance. Several instances are out of scope (decision 32).

## Tests first (capture each one red)

- design tests 6, 7, 7b, 8 and 19;
- a fork whose config can't be read (the pass must be incomplete, and it must gate);
- a fork cycle (`forge.parent` loop) terminates and is handled safely;
- a fork with no `.dropped` object present is left untouched.

Every fixture must make a delete-plus-gc implementation fail (C15): give each one a hardlinked fork, a `.keep` pack, or a non-mirror ref that reaches held. Run at least one mutation per security fix (the trigger, the inherited-held filter, transitivity, the tmp sweep), show the test going red, then restore the code and `touch` it.

## Gates

Run all nine gates in your own target directory using `b1/gates.sh` (`OUT=` a file under `b2/`). Write the report to `b2/report.md`, with:
- each item's red and green evidence;
- the mutations;
- the gate results;
- every deviation from this brief.

Save `b2/after-b2.patch` (`git diff` against `8ba3372`). Before you diff, run `git add -N` on every new file.

## Out of scope (B3, record only)

- operator recourse for stores that stay gated;
- the engine's race between process exit and the size check;
- ancestor-directory fsync;
- descendant-process cleanup.
