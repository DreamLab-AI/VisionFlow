# Decisions timeline

The register's sibling view: the decision records in the order they were
written, and the code changes that moved a decision after it was recorded.
Hand-written; the generator does not produce it. Dates are commit dates from
`git log` up to the source revision `815f963` assessed on 1 November 2026.

## 2026-10-15

| When | Commit | Decision | Later |
|---|---|---|---|
| 12:15 | `1a16ad2` | Workspace skeleton with the two leaf crates, `pairwise-gate` and `failure-fingerprint`. | — |
| 12:18 | `1d5394c` | ADR-DE-001 to ADR-DE-006 recorded: standalone permissive engine, licence boundary, protected-path guard, pairwise selection and fingerprints, five seams, automerge off by default. | The README's "Planned" status for `dream-gate` was corrected in `3202102` (20:09), before the re-stamp (DE-11.1). |
| 12:30 | `79df0ba` | `dream-gate` built: verdict, manifest, receipts, guard, per-test evidence and the gate (DE-10). | — |
| 12:55 | `d2e8f48` | `candidate::run` fixes the order refuse, guard, worktree, build, evaluate (DE-10.3). | — |
| 13:04 | `a89bfc0` | Replay capture and verify with a stable on-disk schema (DE-08). | `ce4811e` (13:56) bumps it to schema 2 so the live policy is recorded; schema 1 is then refused. |
| 13:16 | `2f8e496` | Automerge policy, default off (DE-09). | `cd40f83` (13:23) reads protection of the PR's real base branch and un-drafts before `--auto`. |
| 13:19 | `2662980` | `dream` command-line binary over the pure checks. | — |
| 20:00 | `4f5ac53` | ADR-DE-007: a night is identified by its directory, not by the run id; persist before external effects; `reset_night` (DE-07). | Hardened the same evening, see the rows below. |
| 20:20 | `b721824` | The gate derives required evaluators and command identity from the frozen manifest, not the receipts; a forged or mismatched receipt is a harness veto (DE-10.4). | — |
| 20:23 | `e089375` | The guard bounds patch size, file count and header lines (`TooLarge`), reads evaluator commands as shell words, and refuses commands it cannot read when frozen (DE-10.5). | — |
| 20:23 | `63143d9` | Evaluator-named files are resolved against the repository into recorded `PolicyExtras`, so replay protects what the live night did (DE-10.5, DE-08.1). | — |
| 20:24 | `029a711` | The verified patch is fed to `git apply` on stdin; no predictable patch file in the temp dir (DE-10.3). | The worktree directory path is still predictable (DE-10.3 debt). |
| 20:27 | `1026013` | Typed journal errors, night-level attempt budget, `NightLock`, non-overwriting two-step `reset_night`; MSRV 1.89 (DE-07). | The MSRV is declared but no CI job builds on it (DE-11.2 debt). |
| 20:28 | `c93897f` | ADR-DE-007 records the hardening and the at-most-once publish limit (DE-07.1, DE-09.2). | The limit stands: a publish can be lost, not repeated. |
| 20:36 | `d86aa4a` | Test that no patch file reaches the temp dir; ADR wording on the idempotency key corrected. | `0f6a669` (20:39) makes the assertion real. This was the previous source pin. |

## Later source and audit checkpoints

| Date | Commit | Change | Audit disposition |
|---|---|---|---|
| 2026-10-17 | `8b59c04` | ADR-DE-008 and requested-source prioritisation, core 0.2.0. | Source gathering also calls the host LLM; DE-12 corrects the former host-only claim. Source-budget and denylist limits are recorded in the estate recovery assessment. |
| 2026-10-22 | `a3f4938` | ADR-DE-009 and candidate formatting before evaluation, core 0.3.0. | DE-10.3 now includes formatting on the control host before relocation. A worktree is not a command sandbox. |
| 2026-11-01 review | `815f963` source pin | Eight topics and all 24 panels source-assessed. | Corrected recovery, replay, receipt, licence and automerge claims. New fixtures reproduce fingerprint separator aliases, normalisation limits, unchecked manifest fields and repeated-call streaks. These findings remain open implementation work. |
