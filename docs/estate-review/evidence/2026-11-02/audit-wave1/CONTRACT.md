# Audit wave 1: worker contract (read this first, follow it exactly)

You are one Opus reviewer in a ruflo mesh finishing the VisionFlow estate audit. Your report goes to the delegating lead, who verifies your work, computes the hashes, enters claims in the ledger and commits. A `DECLINED:` report is passed up unchanged.

## Where you work

- **Write only inside** `/home/devuser/workspace/wt/VisionFlow-audit-wave1` (a VisionFlow worktree on branch `audit/wave1`), and **only the files named in your assignment** below. Other workers are editing other files in the same tree at the same time: never touch a file not assigned to you, never run any git command that writes (no add, commit, stash, checkout, reset, worktree), never run `cargo`.
- **Read product repositories at their pinned revision only**, with `git -C <repo> show <sha>:<path>` or `git -C <repo> grep ... <sha>`, never from their working trees (several are dirty). Pinned revisions for this wave:
  - solid-pod-rs `9b6143ce55b2`, at `/home/devuser/workspace/solid-pod-rs`
  - agentbox `c57d9c0508f4`, at `/home/devuser/workspace/agentbox`
  - VisionClaw `86cdd7635582`, at `/home/devuser/workspace/project`
  - loom `ba21a689a57d`, at `/home/devuser/workspace/loom`
  - dream-machine `b77093e564ef`, at `/home/devuser/workspace/dream-machine`
  - VisionFlow: the worktree itself.
- **Never edit product code** in any repository. A wrong thing found is a finding in your report, not a repair.

## What a review is (evidence bar)

Your evidence class is **source tracing at the pinned revision**. State that plainly. Do not claim runtime, test or deployment evidence you did not produce; do not run product tests. A citation check passing is not a review: you must read each cited line and say whether the claim it supports is true.

### For a diagram panel

For every panel in your assigned topics:

1. Read the panel's Mermaid, its "What it shows" and "Why it is this way" paragraphs, and every `path:line` it cites, at the pinned revision.
2. Give it **one disposition**:
   - **accurate**: every mechanism claim is true at the pinned revision.
   - **corrected**: something was wrong or stale, and you fixed the topic file (say what changed and why, with the new citations).
   - **accurate-with-limits**: true as far as source shows, but it asserts behaviour only runtime could confirm. Name the limit.
   - **inaccurate-unfixed**: wrong, and you could not fix it confidently. Say why.
3. You **may correct** your assigned topic files, and you **must** if a citation no longer supports its claim. When you edit a topic, set its `verified_commit` to the pinned revision of its repository and re-resolve **every** citation in that topic at that revision (derive line numbers from the symbol with `git grep -n`, never by shifting). Then the topic must pass, from the worktree root:

   ```sh
   node scripts/diagram-index-gen.cjs docs/diagrams --only <topic-file-name> --check --cite-check --strict-citations --render
   ```

   Exit 0 required. If you did not edit a topic, still run that command and record the result.
4. Use register markers (`**Tension:**`, `**Debt:**`, `**Drift:**`, `**Open:**`, `**Invariant:**`) in the topic for what you find, per the diagrams-as-code rules.

### For an ADR

For each assigned ADR (read at the pinned revision):

1. Classify it: operative decision, superseded/historical, imported copy, or support (README, template, preamble).
2. For an operative or historical decision, give **every section heading** (exactly as written) a disposition in a table: what the section claims, and whether the code at the pinned revision implements it, contradicts it, or cannot be checked from source. Cite `path:line` for each check.
3. Support files only need a classification line and a sentence on whether they are accurate.

## What you write

1. **One dated report** at `docs/estate-review/2026-11-02-<your-slug>.md`, in this shape (match existing reports such as `docs/estate-review/2026-11-01-solid-identity-boundaries.md`):

   ```markdown
   ---
   title: <short name>
   date: 2026-11-02
   status: assessed
   type: explanation
   ---

   # <short name>

   <One paragraph leading with the outcome: what you assessed, how many accurate / corrected / with limits / inaccurate, and the most important finding.>

   ## Findings
   <Each finding: trigger, consequence, evidence (path:line at the pinned sha), and what would close it.>

   ## <ADR-NNNN> section assessment      (one per ADR, if any)
   | Section | Disposition |

   ## Panel dispositions                  (if any)
   | Panel | Disposition | Evidence |

   ## Verification
   <Evidence class, pinned revisions, the exact generator commands and their exit codes, what you did not verify.>
   ```

2. **One claim proposal** at `docs/estate-review/evidence/2026-11-02/audit-wave1/claims-<your-slug>.json`: a JSON array. Do **not** compute hashes; the lead does. One entry per ADR you assessed in full, and one per topic:

   ```json
   [
     {"kind": "adr", "repo": "solid-pod-rs", "path": "crates/solid-pod-rs/docs/adr/ADR-2009-....md",
      "report": "docs/estate-review/2026-11-02-<your-slug>.md", "coverage": "whole-document",
      "basis": "Source tracing at solid-pod-rs 9b6143c; every section assessed; <one-line limits>."},
     {"kind": "topic", "repo": "VisionFlow", "path": "docs/diagrams/solid-pod-rs/NN-slug.md",
      "report": "docs/estate-review/2026-11-02-<your-slug>.md", "coverage": "selected-panels",
      "panels": ["SP-NN.1", "SP-NN.2"],
      "basis": "Source tracing at <repo> <sha>; strict cite-check and render pass; <limits>."}
   ]
   ```

   Only list a panel you actually gave a disposition. A panel marked **inaccurate-unfixed** is still listed: the claim records that it was assessed, and the report says it is wrong.

3. **Your final reply** to the lead is only the report path, the claims path, and one line of counts. Nothing else.

## Never

- Credit a panel or ADR you did not read line by line.
- Copy an assessment from an older report without re-checking it at the pinned revision.
- Edit a file outside your assignment, or any product code.
- Promote source tracing to "tested", "verified at runtime" or "deployed".
