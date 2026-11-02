# Audit wave 4: worker contract (read this first, follow it exactly)

You are one Opus reviewer finishing the VisionFlow estate audit. The completion ledger (every diagram panel and every numbered decision ADR) is already complete. Wave 4 covers the obligations that the ledger does not measure. Your report goes to the delegating lead, who checks it before anything is committed.

## Where you work

- **Write only** inside `/home/devuser/workspace/wt/VisionFlow-audit-wave4`, and only the files named in your row of `ASSIGNMENTS.md`. Do not edit diagram topics, ADRs, claims or the ledger.
- **Read product repositories at their pinned revision only**: `git -C <repo> show <sha>:<path>`, `git -C <repo> ls-tree -r --name-only <sha> <dir>`, `git -C <repo> grep -n <pattern> <sha> -- <path>`. Never read working trees, several of which are dirty. Pins:

  | Repository | Path under /home/devuser/workspace | Pin |
  |---|---|---|
  | prose-sanitiser | prose-sanitiser | `20ad8b33ab7d` |
  | diagram-ir | diagram-ir | `15cba163da70` |
  | loom | loom | `ba21a689a57d` |
  | WasmVOWL | WasmVOWL | `51a148430190` |
  | codescape (standalone) | codescape | `58108c0357bd` |
  | visiontouch-rs | visiontouch-rs | `7b95fb20d50a` |
  | vowl-wasm | vowl-wasm | `28af4365a68d` |
  | VisionClaw | project | `86cdd7635582` |
  | agentbox | agentbox | `c57d9c0508f4` |
  | solid-pod-rs | solid-pod-rs | `9b6143ce55b2` |
  | dream-engine | dream-engine | `bfe8c8885a58` |
  | ontomark | ontomark | `4b9d15df4805` |
  | ontomark-obsidian | ontomark-obsidian | `2a6c44e02412` |
  | visionGraph | visionGraph | `556a0156b5c3` |
  | dream-machine | dream-machine | `b77093e564ef` |
  | nostr-rust-forum | nostr-rust-forum | `c39f41f104b1` |
  | dreamlab-ai-website | dreamlab-ai-website | `0e1572801d03` |
  | sidestr-rs | sidestr-rs | `814b9e54cd16` |

- **Never edit product code.** Something wrong is a finding in your report, not a repair.
- **No cargo, no builds, no tests, no git writes**, and no worktrees.
- **Excluded, do not read:** RuView, Logseq history, the RuVector repository, and the Codescape and Unmute copies consumed inside other repositories. Standalone `codescape` is in scope.
- **Work alone.** Do not split the job across forks or sub-agents.

## Evidence bar

Your evidence class is **source tracing at the pinned revision**. Say so plainly. Never claim runtime, test or deployment evidence you did not produce. Every factual claim cites `repo:path:line` at the pin, and you must have read the cited line. Never infer that something is closed from a commit message or a document. Show it in the code, or mark it not checkable.

## Your output

Write exactly two files, at the paths in your row:

1. **Report** `docs/estate-review/2026-11-02-w4-<name>.md` with:
   - frontmatter: `title`, `date: 2026-11-02`, `status`, `type: explanation`;
   - a one-paragraph outcome first;
   - your section table or tables, as your row specifies;
   - **Findings**: each with an ID `<NAME>-F<n>`, class (security / defect / drift / debt / gap), severity, citation, consequence;
   - **Verification**: what you read and at which pin, and what you could not check.
2. **Data** `docs/estate-review/evidence/2026-11-02/audit-wave4/w4-<name>.json`: the same rows as JSON (`{"rows": [...], "findings": [...]}`), so the lead can merge them.

When you have finished, reply with **only three lines**: the report path, the data path, and a one-line count summary.

## Deliverable types

### Component coverage (`cov` workers)

`coverage-detail.md` in this folder lists, for each repository, every **uncited component**: a top-level directory that holds tracked source files that no diagram topic cites. Give **every uncited component in your repositories** exactly one disposition:

- **covered-elsewhere**: an existing topic or dated report already describes this component's behaviour. Name it. A citation that the coverage script missed counts.
- **out-of-scope**: generated output, vendored third-party code, test fixtures or result data, benchmarks with no product role, or dead or legacy code that nothing builds or imports. Show why: the build file that excludes it, the absence of importers, or a generated-file header.
- **assessed**: you read its entry points and record what it does, how it is reached from product paths, and anything wrong. Cite `repo:path:line`.
- **gap**: it is product code with behaviour worth recording, and you could not assess it fully. Say what is missing.

Row columns: `| Repository | Component | Files | Disposition | Evidence | Note |`. Also say whether the repository's existing topics still describe what it is for, and record anything material that the corpus misses.

### Re-check of open findings (`sidestr` worker)

For **every** finding in your listed reports that the report marks open, unresolved or unverified, give one status at the pin:

- **still-open**: the defect is still present. Cite the line.
- **closed-in-source**: fixed. Cite the line, and the commit if `git log` at the pin shows it.
- **not-checkable**: only runtime or external state could settle it.
- **superseded**: the code path no longer exists. Cite what replaced it.

Row columns: `| Report | Finding | Was | Now | Evidence |`.

### Cross-repository contract assessment (`contracts` worker)

List every cross-repository wire contract the estate relies on: event envelopes, HTTP APIs between repositories, Nostr kinds, shared schemas, and pinned dependency versions. Start from the estate topics (`docs/diagrams/estate/`), the confirmed finding W2-01, and each repository's integration points. For each contract:

- the producer side and the consumer side, each cited `repo:path:line` at the pins;
- whether they **agree**, **disagree** (state how), or are **unverified**;
- whether a fixture or test exercises both sides, cited, or `none`.

Row columns: `| Contract | Producer | Consumer | Agreement | Fixture | Note |`. Do not write fixtures. This is an assessment.

### Comparison and recommendation index (`compare` worker)

Build the cross-repository comparison and recommendation index.

1. Collect every **comparison of alternatives** in the dated reports under `docs/estate-review/` and in the ADRs. Tie each to the product path it affects.
2. Collect every **recommendation**, deduplicated across reports, with its source.
3. For each product area, state what the user gets today, judged from source, and what measurable outcome a recommendation would change.
4. Name the product areas that have **no** primary-source comparison yet.

Row columns: `| Product area | Comparison or recommendation | Source | Product path | User outcome | Status |`. Primary sources only: a report, an ADR or code at a pin. Use no web material unless you cite it and it is a primary source.
