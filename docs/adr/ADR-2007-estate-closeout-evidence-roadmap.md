---
id: ADR-2007
title: Bind estate closeout to decision lineage and system evidence
date: 2026-09-04
decision_status: accepted
implementation_status: complete
activation_status: live
supersedes: []
superseded_by: []
verified_commit: 493e683237c941e3ed858315d2e571279fba496a
owner: estate architecture maintainers
review_trigger: completion of ADR inventory review or a change to the estate roster
repo: visionflow
evidence_class: source+test
acceptance_boundary: local runs in all five ledger repos (VisionFlow, VisionClaw, agentbox, nostr-rust-forum, solid-pod-rs) of the evidence and evidence_class tests, adr-index-gen --check and the ratchet; nothing pushed, so no hosted CI run of the new gates observed; the gate checks field presence and enum membership, not the truth of the named evidence
---

# ADR-2007 — Bind estate closeout to decision lineage and system evidence

## Context

The user requested upgrades and extensions across the estate's ADR documents to form a complete-system closeout roadmap. Existing packs distinguish decision, implementation and activation, while older records are frozen as history. Current review evidence shows that component completion labels do not establish cross-repository behaviour.

## Decision

**Accepted, narrowed to the evidence rule (owner decision 2026-10-02, Q8).** An item closes only on evidence that names its repository, the source revision it was checked at (plus source hashes when the tree was dirty), its evidence class (source, test, CI, deployment or runtime receipt) and its acceptance boundary. Declared status is not a substitute for verified implementation or activation: a bare ADR number, a green index or a component completion label closes nothing. Repository owners retain authority over their decisions. *Overtaken, not accepted:* the original text proposed the [estate closeout roadmap](../estate-review/closeout/README.md) as the coordinating contract, with inventory, amendment and six-journey closeout duties; that coordination is overtaken by the 2026-09-21 planning cycle and the TODO board (see Acceptance below).

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

## Estate audit — 2026-09-07

The [2026-09-07 estate audit](../estate-review/2026-09-07-estate-audit.md), [workspace/ADR census](../estate-review/evidence/2026-09-07/estate-inventory.md) and [closeout sprint](../estate-review/closeout/2026-09-07-sprint.md) extend this programme. The row-level board remains VisionClaw `docs/TODO-unified.md`; this canon owns cross-repository dependencies and acceptance evidence, not a duplicate completion counter.

Every claim must name its repository, record/path, source revision plus dirty-tree source hashes, evidence class, and acceptance boundary. Bare ADR numbers and a green index are insufficient. Historic/imported records retain lineage and consumer scope; unadopted decisions do not become estate implementation promises through inventory membership. The sprint prioritises public-data exclusion, generation publication, truthful transactional receipts, custody, consumer identity and runtime verification. No proposal is ratified, no feature is declared deployed and no implementation gap is closed merely by this audit. Retain proposed/partial/staged.

## Closeout execution — 2026-09-07

The [execution report](../estate-review/2026-09-07-estate-closeout-execution.md) links tested production repairs, the 61-item master reconciliation, source-hashed ADR graph, current diagram checks and real-browser receipts. It distinguishes closed source obligations from live activation and explicit excluded profiles. RuView is excluded by user instruction. Actual release and GPU checks are recorded alongside original failures; the seven-surface public/candidate browser run is not the six complete system journeys.

Retain proposed/partial/staged for this coordinating roadmap. The remaining custody, credential, runtime identity, device, elapsed observation and historical disposition problems are carried in the master register. Source closure cannot erase them or turn a held profile on.

## Disposition — 2026-10-02

- **Suitability:** fits, needs revision
- **Priority:** P2 — next cycle (decide at the cycle-exit reconciliation on 20 October, when the 61-item closeout register is either retired into the cycle board or kept)
- **Why:** The evidence discipline this record proposed now lives in accepted mechanisms: the three-axis status contract (TODO row G-15), ADR-2008 (estate health collected by CI), ADR-2010 (augmentation conditions as the audit lens), and the `adr-index-gen --check` gate and ratchet in all five ledger repos (CY-A4). Its specific claim no longer holds. The closeout roadmap (`docs/estate-review/closeout/README.md`, last touched in `ec7f353` on 15 September) is not the coordinating contract any more. Since 22 September the planning cycle (`dreamlab-cumbria/docs/planning/audit-2026-09-21/planning-cycle.md` §2, §9, §10) and `project/docs/TODO-unified.md` coordinate the work. Its acceptance bar, all six system journeys under CP-09, includes federation work that plan §3 parks for 12 weeks, so it cannot close as written. Verified against VisionFlow `main` at `7096f2b`.
- **Next:** Owner choice at cycle exit. Either narrow this record to "every status claim names its repository, revision, evidence class and acceptance boundary", which is already met and so ready to accept, or reject it as overtaken by the cycle plan.

## Acceptance — 2026-10-02

Owner decision 2026-10-02, Q8: accepted, narrowed to its evidence rule. The accepted text is the Decision paragraph above; nothing else in this record is a live obligation.

- **Accepted:** the evidence rule. An item closes only on evidence naming its repository, revision, evidence class and acceptance boundary. Status declarations, a green index and component completion labels close nothing.
- **Overtaken by the cycle plan and the TODO board:** the closeout roadmap as coordinating contract, the inventory and per-record amendment programme, CP-01 to CP-09 and the six-system-journey acceptance bar. The planning cycle (`dreamlab-cumbria/docs/planning/audit-2026-09-21/planning-cycle.md` §2, §9, §10) and `project/docs/TODO-unified.md` coordinate the work now. The annexes above stay as dated history and are not reopened by this acceptance.
- **`implementation_status: partial`.** The machine-checkable half of the rule is built: `scripts/adr-index-gen.cjs --check` requires `repo`, `verified_commit` and the three status axes on every record (`REQUIRED_FIELDS`), and the three-axis contract (`docs/architecture/adr-status-contract.md`) separates decision from implementation and activation. Evidence class and acceptance boundary are enforced only at review, not by a gate.
- **`activation_status: live`.** That half runs on every push and pull request through `.github/workflows/adr-index.yml` (index check and ratchet), in this repo and the other ledger repos (CY-A4).
- **`verified_commit`** moves to `8c5acc2efa1af7994288aba2e38a88cf8afaf2f9`, the VisionFlow `main` this acceptance was checked against.
- **Diagram citations:** VF-01 cites `:7` (activation axis) and `:24` (the Decision paragraph). Both lines keep their roles; the frontmatter and Decision were edited in place, and this annex was appended. VF-01 resolves those citations at its stamped revision, so its "proposed/partial/staged" wording describes that revision until the topic is next re-verified.

## Close-out 2026-10-31

The machine half of the evidence rule is built in VisionFlow only, so `implementation_status` stays `partial`: the Decision is the estate rule, and the other four ledger repos do not yet have these gates.

- **Gate.** `scripts/adr-index-gen.cjs` validates `evidence_class` (one or more of `source`, `test`, `ci`, `deployment`, `runtime`, joined by `+`) and `acceptance_boundary` (non-empty text). Both are **required on records dated on or after 2026-10-31** and validated whenever present; that date gate is a backstop and binds nothing yet (the newest record is dated 2026-10-30). The binding gate is the transition check in `scripts/adr-ratchet.sh` (run by `adr-index.yml`): over BASE..HEAD, any record whose `implementation_status` becomes `complete` or whose `activation_status` becomes `live` must carry both fields, however old the record. Earlier records are frozen history: none was backfilled with an invented class. This record carries both, filled from its own annexes. `TEMPLATE.md` carries the keys, and `docs/architecture/adr-status-contract.md` states the rule.
- **Evidence.** `bash tests/gates/adr-ratchet-evidence.test.sh` runs six cases for the transition gate (closure without fields fails, with both passes, activation to `live` without fields fails, empty boundary fails, an unrelated edit and an already-closed record pass), and `bash tests/gates/adr-evidence-field.test.sh` runs seven cases for the date backstop (pre-cut-over record passes without the fields, post-cut-over without them fails, missing boundary fails, unknown class fails, joined classes pass, an invalid class on an old record still fails) and is registered in `tests/gates/run-all.sh`. `node scripts/adr-index-gen.cjs docs/adr --check` reports `ok: 21 ADR(s) valid`.
- **Boundary.** The gate checks presence and enum membership, not whether the named evidence is true; truth stays a review matter. Run locally only. `activation_status: live` is unchanged (the existing `adr-index.yml` job runs `--check`); the other four ledger repos carry their own copies of the generator and ratchet and do not have these gates. Closing ADR-2007 to `complete` needs them ported, or an owner decision that narrows the rule to VisionFlow (the owner's call).
- `verified_commit` moves to `f85e6bedc72d5564eba03ad8cda3816edc3f2a21`.

## Close-out 2026-10-31 (wave 2): gate in all five ledgers

The machine half of the evidence rule now runs in every ledger repo the record's CY-A4 list names, so `implementation_status` moves to `complete`. `activation_status` stays `live` for the pre-existing `--check`; the new gates are activated only when each repo's merge reaches its hosted CI.

- **Ported (wave 2, W10; reviewed by Opus R9, fixed by F8).** The transition check in each repo's ratchet, the `evidence_class`/`acceptance_boundary` validation in each generator, an eight-case evidence test and a four-case `evidence_class` value test, registered in each repo's CI. Local `main` merge commits:
  - VisionClaw `b84ac5ddb26c425e57b941674f9ff3b2781509f5`
  - agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf`
  - nostr-rust-forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`
  - solid-pod-rs `83cdda68bc5834f7e5520637bf6e1224efc6f095`
- **VisionFlow.** The ratchet's `awk … exit` killed `git show` with SIGPIPE on records over 64 KiB under `pipefail` (a false red on VisionClaw's and agentbox's real ranges). Fixed and regression-tested in `89385c6ebaabd2f17450dd89f41e23ace0660f74`: the large-record cases exit 141 on the old ratchet and pass on the new.
- **Backfills.** The seven closures this sprint took to `complete` (agentbox 2041, 2090, 2105, 2107, 2109, 2139; forum 2013) now carry both fields, `source+test`, with boundaries naming the revision their tests ran at. No older record was backfilled with an invented class.
- **agentbox CI masking.** `check-declared-vs-running` had failed on `user-cron` since the 2026-10-31 15:28 push, so hosted `invariants` runs skipped every ADR step after it. The runtime fixture now lists `user-cron`; the step exits 0.
- **Evidence (local).** All five: evidence test 8/8, `evidence_class` test 4/4 (the `vibes` and `source+` cases fail on the pre-W10 generators), `adr-index-gen --check` ok, ratchet OK. agentbox's ratchet over the whole sprint range `2a36c42d8..HEAD` reports `ADR-RATCHET-OK`.
- **Boundary.** Nothing is pushed, so no hosted run of the new gates has been seen. visionGraph, ontomark and knowledgeGraph run only the pinned checker and are outside the five-repo list; whether they should carry the gate is an owner decision. `verified_commit` is VisionFlow `main` at `493e683` before this annex.
