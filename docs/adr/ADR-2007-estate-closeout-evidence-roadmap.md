---
id: ADR-2007
title: Bind estate closeout to decision lineage and system evidence
date: 2026-09-04
decision_status: proposed
implementation_status: partial
activation_status: staged
supersedes: []
superseded_by: []
verified_commit: 8cf1a1bf9e4ef2ef98ee6c7bf56ef361aa8d0304
owner: estate architecture maintainers
review_trigger: completion of ADR inventory review or a change to the estate roster
repo: visionflow
---

# ADR-2007 — Bind estate closeout to decision lineage and system evidence

## Context

The user requested upgrades and extensions across the estate's ADR documents to form a complete-system closeout roadmap. Existing packs distinguish decision, implementation and activation, while older records are frozen as history. Current review evidence shows that component completion labels do not establish cross-repository behaviour.

## Decision

Propose the [estate closeout roadmap](../estate-review/closeout/README.md) as the coordinating contract. Inventory every candidate record, preserve lineage, amend operative decisions and their governing documents together, and require system-journey evidence before closeout. Declared status is not a substitute for verified implementation or activation. Repository owners retain authority over their decisions.

## Consequences

The roadmap exposes dependencies and acceptance conditions without rewriting history or silently declaring untested capabilities complete. Work increases initially because imported, archived and duplicate records require explicit disposition. The proposal remains staged while inventory classification, per-record amendment and system verification continue.

## Verification

The collector `python3 docs/estate-review/evidence/adr-inventory.py` ran on 2026-09-04 against the working tree on top of commit `8cf1a1bf9e4ef2ef98ee6c7bf56ef361aa8d0304` (the `verified_commit` above); the `docs/estate-review/` corpus it inventories was not yet committed at that revision. It records candidate paths, source hashes, declared axes and missing fields across the initial 14 repository identities. Its inventory is provisional. The roadmap links established review receipts and marks unreviewed areas open. This verifies documentation scaffolding only; it does not verify full estate closeout or adoption of this proposed decision.

## Closeout extension — 2026-09-04

Retain proposed/partial/staged. The inventory, thematic findings, historical maps and ordered execution sequence are implemented documentation. They are not owner ratification or system acceptance. Imported and historical semantic review and remaining claim reconciliation are still incomplete.

**Closeout (CP-01–09):** Complete every in-scope record disposition and material claim assessment before declaring the documentation objective complete. Implement and verify the six system journeys before declaring system closeout. Confirm accountable ownership without inventing dates or converting helper-test success into deployment evidence.

## Acceptance progress — 2026-09-05

Retain proposed/partial/staged. This record is the roadmap, not the work; what
changed is that five of the roadmap's operative gates moved from *documented* to
*executed and self-tested*, and a dated evidence directory now exists for them.

**Evidence produced.** `docs/estate-closeout/2026-09-05/` holds
`gate-closeout-receipt.json` (11 gates, each with command, exit code, result and
SHA-256 over the 20 governed sources it covers),
`website-browser-receipt.json` (26 browser checks), nine screenshots across
three rendering scenarios, `release-manifest.local-draft.json`, and per-gate
logs under `logs/`. Every figure in the sibling ADRs' acceptance annexes is
traceable to a file there.

**Records advanced.** ADR-2002 (asset inventory, build receipt, browser
verification), ADR-2003 (six blocking publication gates plus one reported),
ADR-2004 (full render gate executed with real Chrome, zero drift), ADR-2005
(`partial` → `complete`), ADR-2006 (roster six → fourteen, fixture parity now
requires a compared revision set), and engineering ADR-004 (source resolution,
edge de-duplication, distinct-source capping). Engineering ADR-005 is annexed to
state explicitly that it remains Speculative and that adjacent work is not
progress on it — the failure mode this ADR exists to prevent.

**The roadmap's own thesis held.** "Component completion labels do not establish
cross-repository behaviour" was borne out three times: a build reporting
`BUILD-COMPLETE` while every image copy could fail; an audit scoring 100%
source-backed on fabricated paths and 200% coverage on duplicated edges; a
fixture gate that exited green because it had nothing to compare. Each was a
green signal with nothing behind it, and each is now backed by a test that
proves the gate can still fail.

Receipts: [gate closeout receipt](../estate-closeout/2026-09-05/gate-closeout-receipt.json),
[browser receipt](../estate-closeout/2026-09-05/website-browser-receipt.json).

**Remaining.** Unchanged in substance. This is local execution on a working
tree: no hosted CI run, no deployment, no owner ratification. The six system
journeys are not implemented or verified. Imported and historical semantic
review and the remaining claim reconciliation are still incomplete, and none of
the above is system acceptance.

Governed paths changed: none in this record; `docs/estate-closeout/2026-09-05/`
added as dated evidence.

[Canon assessment](../estate-review/canon-and-verification.md), [source hashes and local check receipt](../estate-review/evidence/canon-operative-closeout.json), [execution sequence](../estate-review/closeout/execution-sequence.md). Historical verification above is preserved; this annex assesses the current working tree.
