# Worktree-only review-ledger rows

Recovered on 2026-11-03 before removing three merged worktrees. The nightly
dream job had reviewed each worktree's own branch head, so these rows record
reviews of branch revisions rather than of main:

| File | Worktree | Branch head reviewed | Rows |
| --- | --- | --- | --- |
| `jev-visionflow-review-ledger-rows.jsonl` | `wt/jev-visionflow` (`feat/jev-option-p1`) | VisionFlow `592cbe2` | 16 (13 review, 3 triage; 26–29 Oct) |
| `dream-engine-release-review-ledger-rows.jsonl` | `wt/dream-engine-release` (`release/0.4.0`) | dream-engine `d81fc71` | 3 triage |
| `jev-dream-engine-review-ledger-rows.jsonl` | `wt/jev-dream-engine` (`feat/jev-option-p1`) | dream-engine `a5f4628` | 3 triage |
| `2026-10-26-glm.{json,md}` | `wt/jev-visionflow` | VisionFlow `592cbe2` | GLM review for 26 Oct |

They are not appended to `docs/diagrams/review-ledger.jsonl`. Main already has
its own triage rows for the same days. The ledger is in append order, not
timestamp order, so appending older triage rows that name branch commits would
make the latest triage record point at a branch revision. Both branches are
merged into their mains.
