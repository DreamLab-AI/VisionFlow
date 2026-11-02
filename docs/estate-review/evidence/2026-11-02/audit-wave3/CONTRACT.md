# Audit wave 3: worker contract (read this first, follow it exactly)

You are one Opus reviewer in a ruflo mesh finishing the VisionFlow estate audit. Your report goes to the delegating lead, who verifies your work, computes the hashes, enters claims in the ledger and commits. A `DECLINED:` report is passed up unchanged.

## Where you work

- **Write only inside** `/home/devuser/workspace/wt/VisionFlow-audit-wave3` (a VisionFlow worktree on branch `audit/wave2`), and **only the files named in your assignment** below. Other workers are editing other files in the same tree at the same time: never touch a file not assigned to you, never run any git command that writes (no add, commit, stash, checkout, reset, worktree), never run `cargo`.
- **Read product repositories at their pinned revision only**, with `git -C <repo> show <sha>:<path>` or `git -C <repo> grep ... <sha>`, never from their working trees (several are dirty). Pinned revisions for this wave:
  - nostr-rust-forum `c39f41f104b1`, at `/home/devuser/workspace/nostr-rust-forum`
  - dreamlab-ai-website `0e1572801d03`, at `/home/devuser/workspace/dreamlab-ai-website`
  - ontomark `4b9d15df4805`, at `/home/devuser/workspace/ontomark`
  - visionGraph `556a0156b5c3`, at `/home/devuser/workspace/visionGraph`
  - WasmVOWL `51a148430190`, at `/home/devuser/workspace/WasmVOWL`
  - agentbox `c57d9c0508f4`, at `/home/devuser/workspace/agentbox`
  - VisionClaw `86cdd7635582`, at `/home/devuser/workspace/project`
  - solid-pod-rs `9b6143ce55b2`, at `/home/devuser/workspace/solid-pod-rs`
  - loom `ba21a689a57d`, at `/home/devuser/workspace/loom`
  - dream-machine `b77093e564ef`, at `/home/devuser/workspace/dream-machine`
  - knowledgeGraph `4ed9ac159daf`, at `/home/devuser/workspace/knowledgeGraph`
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

2. **One claim proposal** at `docs/estate-review/evidence/2026-11-02/audit-wave3/claims-<your-slug>.json`: a JSON array. Do **not** compute hashes; the lead does. One entry per ADR you assessed in full, and one per topic:

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

## Wave 2 additions (lessons from wave 1)

- **A restamp voids every claim on that topic.** If you change a topic's `verified_commit` or any of its text, you must give a disposition to **every** panel in that topic, including panels an earlier report already credited, and list them all in your claim. A topic left with uncovered panels loses their credit.
- **Topic claims always use `"coverage": "selected-panels"`** with an explicit `panels` list. The ledger collector rejects any other value for a topic, including `"all-panels"`.
- **A topic citing a file in another repository** resolves it through `/home/devuser/workspace/wt/<repo>` aliases, which already exist. If one is missing, ask the lead; do not create it.
- **Cross-repository estate topics** carry a per-repository `verified_commit` map. Restamp each repository you re-resolve to its pinned revision above, and leave the others as they are only if you did not re-read their citations.
- **Generated or uncommitted source paths** (build receipts, logs) do not exist at any revision. If a topic's `sources:` names one, record it as a **Drift** finding and say so in your report; do not invent a citation for it.


## Wave 3 additions (read these; they override anything above)

1. **No forks.** Do the whole assignment yourself. Do not spawn sub-agents or forks: the session's pane budget is fixed, and the lead needs every disposition to come from the reviewer named in your report.
2. **Pins are unchanged from wave 1 and 2:** agentbox `c57d9c0508f4`, VisionClaw (`/home/devuser/workspace/project`) `86cdd7635582`. Read with `git -C <repo> show <pin>:<path>` or `git grep`, never the working tree.
3. **agentbox topics cite `../project/agentbox/...`.** That path is VisionClaw's agentbox submodule. It holds the same commit `c57d9c0508f4` with the identical tree as `/home/devuser/workspace/agentbox`, so read those paths at the agentbox pin in either checkout.
4. **Generator advisories to settle by hand:** AB-01.3 (`agentbox.sh:11-12`) and AB-01.5 (`lib/npm-cli.nix:184`) were flagged by the citation reviewer as possibly not holding their labels. If they are in your assignment, check those two by hand and say what you found.
5. **ADR workers:** your ADR list and section counts are in `ADR-GROUPS.md` beside this file. Assess every headed section of every ADR in your group, one row per section, exactly as the ADR section of this contract says.
6. **Imported ADR copies** (`skills/ruvector-catalog/docs/adr/*`, `skills/bhil-methodology/examples/*` in agentbox) are vendored, not agentbox decisions. Classify each first: name its upstream owner and whether agentbox adopts any of its obligations. Still give every headed section a row, with disposition `imported, not adopted` unless agentbox code or an agentbox ADR adopts that section, in which case assess it against agentbox code like any other section. Do not count an imported copy as an adopted decision in your summary.
