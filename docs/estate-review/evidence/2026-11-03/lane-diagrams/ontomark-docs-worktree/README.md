# ontomark/docs worktree — archived review artefacts (2026-11-03)

Source: worktree `wt-ontomark/VisionFlow/docs`, branch `ontomark/docs` at `c0fdb59`
(two local-only commits over `33ece38`, 11 Oct 2026). These files were untracked there.

| File | What it is |
|---|---|
| `2026-10-13-glm.{md,json}`, `2026-10-14-glm.{md,json}` | Nightly GLM review reports written into the worktree. Main holds different files under the same names (reviews of main), so these are not duplicates. |
| `ontomark-docs-review-ledger-rows.jsonl` | 94 ledger rows, 13-18 Oct, none present in main's `docs/diagrams/review-ledger.jsonl`. Triage rows record the worktree path as their only repository, so they reviewed the branch head, not main; archived as evidence, not appended. |

The branch itself was parked, not merged: its terminology crate table predates
`ontomark-ingest`, `ontomark-source`, `ontomark-jev`, `rdfs-certificate` and `packages/graph`,
and its ADR-2014/2016 notes ("proposed with no implementation") contradict main's later
`implementation_status: partial` on both (ADR-2015 is still `none` on main). See the diagrams-lane phase 1 report.
