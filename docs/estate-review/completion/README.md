---
title: Estate audit completion ledger
date: 2026-11-01
status: in-progress
type: reference
---

# Estate audit completion ledger

This ledger makes the review backlog explicit. It currently reconciles **80 ADR assessments and 264 diagram panels** with named reports. Those are counted evidence records, not a percentage of the whole audit. Earlier reports still need reconciliation, candidate decisions still need classification, and substantial source review remains.

[Scope](scope.json), [review claims](claims.json), the [dated snapshot](snapshot-2026-11-01.json) and [validation](validation-2026-11-01.json) are separate files. The [Rust collector](../../../scripts/estate-review-ledger/README.md) can rebuild the snapshot without scanning unrelated workspace repositories. This replaces the old census's undifferentiated pending list as a working queue; earlier audit findings remain valid to their stated evidence boundaries.

## Current reconciliation

| Measure | Recorded result | Meaning |
| --- | ---: | --- |
| Explicit checkout targets | 25 | The existing roster/map scope, including separate standalone/consumed Codescape observations |
| Available checkouts | 24 | Missing RuVector remains visible; it is not silently removed from the denominator |
| ADR-like candidate files | 1,148 | Includes support, imported copies, history and retained audit evidence |
| Numbered decision candidates | 1,055 | A naming-based queue, not a count of operative or unique adopted decisions |
| Parsed heading units | 15,150 | Excludes headings in fenced examples; a heading is not an exhaustive requirement decomposition |
| ADR assessments reconciled | 80 | Explicit whole-document assessments linked to unchanged source-document and report bytes |
| Heading units with those assessments | 560 | Inherit the bounded whole-document report; implementation claims still need its evidence |
| Diagram topics / panels | 197 / 1,694 | Exact panel-ID set matches the existing generated coverage index |
| Panel assessments reconciled | 264 | Selected panels only; 48 topics have all panels recorded and six have partial coverage |
| Stale claim records | 0 | None of this manifest's source-document/report bindings changed during this observation |

This snapshot retains all 1,129 records from the 31 October census and adds nineteen: fifteen retained audit-source copies and four ADR-named evidence/review files discovered by compact identifier matching. Copies are classified separately from decision candidates. The census does not adopt imported decisions merely because they are present, and it does not treat a content page named ADR as an operative policy. Those classifications remain explicit review work.

This update records ADR-2016 and superseded ADR-2026, NF-05.16 and five new NF-27 panels. Three retained ADR evidence copies are added to the candidate census; consulted ADR-2028 remains unassessed as a whole document. A concurrent VisionClaw update changed eleven unassessed ADRs and added fourteen heading units; their queue entries were refreshed without review credit. Previously assessed NF-05 panel bodies were checked byte-for-byte unchanged before rebinding their whole-file hashes.

## What completion must prove

| User requirement | Evidence needed | Current disposition |
| --- | --- | --- |
| Limit work to the linked VisionFlow estate | Explicit repository identity, link/ownership basis, nested-owner handling and unavailable-source record | Scope is enumerated; further dependencies found through in-scope references require a recorded inclusion decision |
| Check the systems against implementation | Per-repository component and cross-repository contract assessments, exact source revisions, relevant adversarial checks and clearly bounded runtime observations | Many reports exist; whole-system path coverage remains open and cannot be inferred from ADR counts |
| Inspect and update every diagram | A source-backed disposition for every existing panel, corrections where needed, revision citations and rendering/visual evidence | 264 panel assessments reconciled; 1,430 remain unreconciled, some already reviewed in earlier reports |
| Examine every ADR section | Classify each candidate; reconcile operative, historical, imported and support obligations; link every material section/requirement to a disposition | 80 complete-document assessments reconciled; partial and earlier assessments still need explicit mapping |
| Gap analysis, adversarial review and sense checks | Findings with triggers, consequences, evidence strength, counterexamples or bounded non-reproduction, and acceptance criteria | Available throughout dated reports; repo-by-repo completeness and cross-system synthesis remain open |
| Compare alternatives and develop user utility | Primary-source comparisons tied to product paths, distinct recommendations and measurable user outcomes | Bounded comparisons exist; remaining product areas and final estate positioning still need review |

Open implementation findings do not require every proposed feature to be built before the audit can finish. The audit must accurately establish what exists, what fails, what is proposed and what remains unverified. Conversely, writing a finding about one path does not complete the rest of its repository.

## Reading the states

`assessment-recorded` links a scoped report to exact document bytes. It does not certify deployed behaviour or remediation. A topic with some recorded panels is `partial-assessment-recorded`. `unreconciled` means the manifest has no qualifying entry yet; it must not be reported as “never reviewed”. `stale-evidence` requires rechecking a changed source document or report before counting its claim again.

No review credit comes from ADR mentions, historical extension markers, passing tests without scope, or rendered sibling panels. The collector rejects unmatched claims and unknown panel IDs. Filename-based categories remain candidates, and byte-identical imported copies retain separate ownership paths until their obligations are reconciled. The source implementation manifest behind each report still controls its freshness; matching an ADR and report hash alone does not revalidate changed code.

## Remaining reconciliation work

First, map earlier whole-document reviews and partial section assessments without promoting a consumer-only review to whole-ADR coverage. Preserve distinctions such as the dream-only reading of agentbox ADR-2155 and the explicitly incomplete host assessment of ADR-2141. Next, classify the large historical/imported corpora and route surviving obligations, rather than counting every copy as a new adopted decision. Continue source review where that process exposes missing assessments.

Component/path coverage and a final cross-repository comparison/recommendation index still need to be added. Until those dimensions and candidate classifications are reconciled, an overall percentage remains a judgement estimate. Neither the raw ADR ratio nor the diagram ratio measures total effort or task completion.

## Repository and diagram queues

Counts below come from the dated snapshot. Zero means no qualifying claim entered in this first reconciliation, not zero previous audit work.

| Checkout | Candidate files | Numbered decision candidates | ADR assessments reconciled |
| --- | ---: | ---: | ---: |
| VisionFlow | 54 | 30 | 1 |
| VisionClaw | 287 | 273 | 10 |
| agentbox | 239 | 227 | 7 |
| solid-pod-rs | 19 | 15 | 0 |
| nostr-rust-forum | 64 | 58 | 23 |
| dreamlab-ai-website | 47 | 43 | 3 |
| loom | 10 | 8 | 0 |
| knowledgeGraph | 9 | 6 | 4 |
| sidestr-rs | 5 | 3 | 0 |
| ontomark | 12 | 10 | 1 |
| ontomark-obsidian | 5 | 5 | 5 |
| WasmVOWL | 1 | 1 | 0 |
| visionGraph | 12 | 10 | 8 |
| vowl-wasm | 2 | 1 | 1 |
| prose-sanitiser | 0 | 0 | 0 |
| diagram-ir | 0 | 0 | 0 |
| dream-engine | 10 | 9 | 9 |
| visiontouch-rs | 9 | 6 | 6 |
| dream-machine | 9 | 8 | 0 |
| Logseq history | 10 | 10 | 0 |
| RuVector | unavailable | unknown | 0 |
| RuView | 336 | 330 | 0 |
| Codescape standalone | 4 | 1 | 1 |
| Codescape consumed | 4 | 1 | 1 |
| unmute consumed | 0 | 0 | 0 |

| Diagram area | Panels | Assessments reconciled |
| --- | ---: | ---: |
| agentbox | 396 | 24 |
| dream-engine | 14 | 14 |
| dreamlab-ai-website | 64 | 17 |
| estate | 135 | 0 |
| knowledgegraph | 16 | 16 |
| nostr-rust-forum | 214 | 94 |
| ontomark | 49 | 0 |
| sidestr-rs | 20 | 0 |
| solid-pod-rs | 133 | 0 |
| visionclaw | 467 | 0 |
| visionflow | 87 | 0 |
| visiongraph | 27 | 27 |
| visiontouch-rs | 39 | 39 |
| vowl-wasm | 33 | 33 |
