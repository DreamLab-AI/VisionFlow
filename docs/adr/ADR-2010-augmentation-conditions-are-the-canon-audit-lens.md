---
id: ADR-2010
title: The six augmentation conditions are the canon's audit lens for human judgement
date: 2026-09-14
decision_status: accepted
implementation_status: complete
activation_status: staged
supersedes: []
superseded_by: []
verified_commit: 03db67106e8d5aab6e6a992095107f14df3fc886
owner: jjohare
review_trigger: any substrate row in the augmentation-conditions matrix changing status, or a second longitudinal measurement cycle completing (first is 2026-12-14)
repo: visionflow
domain: BASELINE-visionflow.md
lineage: extends ADR-2006 (canon owns the cross-repo view) by giving the view a rubric; complements the judgment-broker PRD/DDD, which define the loop this ADR grades.
---

# ADR-2010 — The six augmentation conditions are the canon's audit lens for human judgement

## Context

The Judgment Broker loop grades itself by protocol completeness: kinds published, receipts committed, provenance minted. Audited against arXiv 2609.12482 on 2026-09-14, that framing hides three facts. The decision surface withholds the proposal and fabricates the human's rationale (`nostr-rust-forum` `pages/governance.rs:463-466`; VisionClaw `AcspCaseQueue.tsx:26`). The escalation boundary is the requesting agent's own `risk_tier` (`governance.rs:129-176`). No metric anywhere measures the humans doing the reviewing. The canon has asserted since the coordination-collapse essay that passive oversight decays, but had no rubric to grade the substrates against, so CP-05 "human judgement and governance" could not be closed by evidence.

## Decision

The six conditions in arXiv 2609.12482 — C1 durable net value, C2 meaningful human control, C3 accountability and recovery, C4 deepening learning, C5 career pathways, C6 job purpose — are the canon's audit lens for every surface where a human decides on an agent's behalf. Concretely:

1. `docs/architecture/compatibility-matrix.md` carries an **Augmentation conditions** table: one row per condition per substrate, status `absent | partial | measured`, each cell cited to `file:line`. A status may only be raised by a citation to code or a runtime receipt, never by a document.
2. CP-05 and CP-07 in the closeout programme name the conditions they discharge by id. CP-05 cannot exit while C2 or C3 is `absent` on any substrate that publishes or consumes kind 31403. CP-07 cannot exit while C4 is `absent` for humans.
3. The paper's *workflow record* is the ACSP chain plus PROV-O plus receipts. The canon lists the record's required fields (agent authority level, review and override points, fallback, unapproved actions, effort including verification, effects on human capability) and marks which substrate owns each; a field with no owner is a matrix defect.
4. No surface may fabricate a human's rationale, an agent's intent or a confidence value. Absence renders as absence. This generalises the D7 honesty rule from intent to every field a human might read as a judgement.

Implementation of the surface changes belongs to the substrates (ADR-2006) and is specified in [PRD-augmentation-conditions](../PRD-augmentation-conditions.md) FR2–FR7.

## Consequences

- The estate can be graded C1–C6 from code; the matrix table becomes the honest ledger for human judgement, and "human in the loop" claims on the website must cite a `measured` row or be softened.
- Augmentation Ratio is reported as provisional until C1 has a measured verification-effort denominator; the KPI page states this.
- Longitudinal conditions (C4–C6) require a second measurement; the first `measured` status on those rows is by construction a baseline, not a pass.
- Cost: reviewer telemetry, calibration sampling and rationale capture add friction to the approve path by design (the coordination-collapse Part 4 position). The forum client and VisionClaw case queue each gain a mandatory text input on high and critical tiers.
- Follow-on: ADR-2011 (task-property triple), forum ADR-2011, agentbox ADR-2087, VisionClaw ADR-2110 own the mechanics.

## Verification

- `grep -n "Augmentation conditions" docs/architecture/compatibility-matrix.md` returns the table header; each cell carries a `file:line` citation checked by `scripts/check-citations.cjs` where the path exists.
- `grep -n "C[1-6]" docs/estate-review/closeout/README.md` shows the CP-05 and CP-07 rows naming conditions.
- Expectation EXP-AC-001 evidence file in `.claude/evidence/` with executed commands and git SHA.
- `verified_commit` is set when the matrix table and closeout references land; `implementation_status` moves to `complete` when every substrate row cites code rather than a plan.

### Re-grade at merged mains (2026-09-15)

The three substrate branches merged to `main` — nostr-rust-forum `11b674c`, agentbox `b859b37d2`, VisionClaw `8f9affe49` — and the matrix table was re-graded against those heads (`docs/architecture/compatibility-matrix.md` §Augmentation conditions). `implementation_status` moves to `complete`: every one of the 18 cells now cites code at a merged main, not a plan or an unmerged branch. `activation_status` stays `staged`: nothing in this re-grade was deployed. `verified_commit` is left `pending` for the queen to set against the actual merge/deploy commit on this repo. `scripts/check-augmentation-citations.cjs` and `tests/gates/augmentation-citations.test.cjs` both pass against the merged sibling checkouts.
