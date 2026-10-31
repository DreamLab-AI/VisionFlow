---
title: Whole-corpus input drift and estate-health review
status: in-progress
date: 2026-10-31
type: explanation
---

# Whole-corpus input drift and estate-health review

At VisionFlow `64ef3af`, all 158 topics were compared with their declared input revisions, including **governing** files as well as sources. The [input-level receipt](evidence/2026-10-31/diagram-input-drift.json) records the owning repository, declared commit, Git blob identities and comparison result for each declaration. The [topic queue](2026-10-31-diagram-review-queue.md) makes every topic visible without equating unchanged input bytes with a correct diagram.

There are **109 topics with changed input bytes** and 49 without detected byte changes. Across 4,371 topic/input relationships, 543 changed, 3,822 matched, two were directories, two lacked an owning-repository revision, and two could not resolve a declared object. Repeated files in different topics count separately. Forty-five topics include changed files declared only as governing inputs. A source-only scan would miss those relationships.

The sweep uses the corpus generator's repository-prefix ownership rules and the topic's declared revisions. `git cat-file --batch-check` resolves revision/path object identities; current file bytes are hashed with Git's blob header, while symlinks use their target text. No unrelated workspace repository is searched. Two VisionFlow examples, one changed and one identical, were independently checked through `git show` and direct byte comparison. This is a non-atomic observation across moving checkouts, not a semantic verification or assurance about uncited dependencies. The receipt retains current repository HEADs for context. It precedes this pass's VF-03 wording changes.

## Inputs requiring explicit handling

| Topic | Input | Limit |
|---|---|---|
| AB-05 | `project/agentbox/config/instructions/` | Directory was not recursively byte-compared. Its children need a separate inventory. |
| VC-36 | `project/xr-client/tests/unit/` | Directory was not recursively byte-compared. Test additions/deletions remain outside this sweep. |
| OM-01, OM-02 | VisionFlow `docs/engineering/sovereign-corpus-contracts.md` | Declared revisions do not pin the VisionFlow owner for this governing input. Add a verified owning-repository revision after reading the document; do not silently borrow another topic's pin. |
| VF-02 | `website/build-receipt.json` | Present locally but absent from the declared commit. Treat as generated working-tree evidence, not revision-backed source. |
| VF-07 | `texput.log` | Present locally but absent from the declared commit. Same generated-evidence limitation. |

The 49 topics without changed bytes still require semantic assessment of existing claims and completeness. Conversely, changed bytes may be comments or unrelated implementation: the 109 count is a re-reading queue, not 109 proven diagram defects. Today's GLM timeout triage remains a separate, weaker queue.

## ADR-2008 and the health reader

The record was read in full against `scripts/estate-health.mjs`, the roster, workflow and CI-state tests. It declares complete/live at a September commit; that is its recorded status, not a deployment conclusion from this audit.

| Section | Source-grounded assessment |
|---|---|
| Context | Fourteen repositories/seven registries describes September. The current roster contains eighteen repositories and twenty-two registry entries. Keep the historical context dated; do not use those counts as current coverage. |
| Decision: collection and publication | The workflow collects at 02:30, reports the check non-blockingly, stages and commits the snapshot, then dispatches deployment when changed. This supports the bot-data path in source. No live workflow run or published snapshot equality was verified here. |
| Decision: offline dream reader | Default checking reads the snapshot. Optional Jev advisory runs before the verdict (`estate-health.mjs:1005`, `:1035`) and may make configured model requests. An unconditional “no network” description is too broad when the option is enabled. The probes explicitly set `VISIONFLOW_JEV=off`; they did not contact a model. |
| Consequences: historical table and private visibility | The roster/snapshot split exists and unreadable rows can be represented. Unreadable/amber rows do not make the final check non-zero. OK is therefore not proof that every repository was readable and green. This is consistent with a reporting monitor, but must be clear to consumers. |
| Consequences: red still deploys | The workflow deliberately catches a non-zero verdict and proceeds. A red monitor must remain publishable; this review does not recommend turning sibling health into a publication veto. |
| Consequences: ordinary gates and staleness | Source dispatches the ordinary deployment workflow. The reader rejects invalid/old timestamps after red precedence, but does not reject future timestamps. Full deployment gates were not rerun. |
| Verification | The existing CI-state fixture passed. Prior dispatch/browser results remain historical receipts. The malformed-snapshot probes below identify coverage missing from the current reader and its stated verdict contract. |

## Reproduced health-reader counterexamples

The actual `node scripts/estate-health.mjs check --file TEMP.json` was run with temporary synthetic JSON files and Jev disabled. The committed snapshot and remote services were untouched. [Full inputs and outputs](evidence/2026-10-31/health-boundary-probes.json) are retained.

| Fixture | Actual result | Why it matters |
|---|---|---|
| Correct schema tag and current timestamp, no `repos` or `surfaces` | Exit 0, `ESTATE-HEALTH-OK` | `snap.repos ?? []` and `snap.surfaces ?? []` turn missing measurements into empty scans. The schema tag is not structural validation or roster coverage. |
| Correct schema, empty arrays, timestamp `2099-01-01` | Exit 0, negative age, `ESTATE-HEALTH-OK` | The freshness test only rejects age above the maximum, not implausible future dates. Clock or data errors can suppress stale detection. |
| Correct schema, current timestamp, `repos: {}` | Exit 1 with TypeError and no verdict | Callers expecting OK/STALE/RED receive an unclassified failure. The workflow has an UNKNOWN fallback, but the reader itself does not honour its classified-verdict contract. |

Source: `estate-health.mjs:956` only checks the schema identifier, `:967`/`:984` iterate optional arrays, and `:1012` applies the one-sided age test. These source bytes are unchanged between VF-03's declared `4d361b58` and `64ef3af`.

**Impact:** the monitor can report success without establishing a measured estate. This is an integrity/reliability finding, not evidence that the current published snapshot is malformed. **Proposed owner:** VisionFlow health collector/reader maintainers. **Acceptance:** reject or explicitly classify missing/wrong-type collections, validate expected roster membership without hiding unreadable members, reject timestamps beyond an agreed clock-skew tolerance, and preserve RED/STALE policy and non-blocking publication. Test malformed fixtures through the CLI, rather than only testing CI-state folding helpers.

## Diagram corrections and validation

VF-03.1 reused node ID `P15` for both sidestr-rs and visiontouch-rs, causing the latter label to replace the former. The haptic node now has its own ID. The Ontomark label now points to its existing diagram area. VF-03.6 qualifies optional network advisory, includes invalid-age handling, removes the false claim that every early-return path prints INFO, and records missing-array/future-date limits.

`node tests/gates/estate-health-ci.test.mjs` returned `ESTATE-HEALTH-CI-TEST-OK`. The nine VF-03 diagrams passed structural/render validation with `--check --no-source-paths --render --only visionflow/03-`. This is not a full citation or semantic refresh of VF-03; its remaining dated snapshot narrative and source drift remain on the queue. No topic revision was advanced and no product code was changed.
