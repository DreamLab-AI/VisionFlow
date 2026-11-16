---
title: Earlier review reconciliation
date: 2026-11-02
status: assessed
type: explanation
---

# Earlier review reconciliation

This read-only sweep looked through the dated reports under `docs/estate-review/*.md`, excluding `2026-11-02-*`. It looked for panels and ADRs that a report had already given an explicit disposition but that are still `unreconciled` in the wave-1 snapshot (`evidence/2026-11-02/audit-wave1/snapshot-wave1-start.json`).

Ten units qualify: seven whole-document ADR assessments and three panels. Every one is in a 2026-10-31 or 2026-11-01 report, and its source bytes have not changed since that report was committed.

Most of the 323 panel-ID mentions do not qualify. They fall into four groups:

- 275 are hypothesis anchors in the parallel-implementations report.
- Most of the rest are narrow corrections. Their own reports say these give no whole-panel credit.
- A few are render receipts.
- The 2026-09-07 mentions are for panels whose bodies have changed since.

The main finding is that 2026-10-31 reports hold complete section assessments that never entered the ledger: agentbox ADR-2136, VisionFlow ADR-2006 and ADR-2008, forum ADR-2013 and Ontomark ADR-OM-007. Their ADR bytes match the wave-1 snapshot. The claims below name the earlier reports. This report only records the reconciliation.

## Findings

**R1: 2026-10-31 whole-document ADR assessments were never claimed.**

- **Trigger:** an earlier ledger pass entered only 2026-10-31 claims whose source matched "the earlier census hash". It gave 7 claims to 2026-10-31 reports and 202 to 2026-11-01 reports.
- **Consequence:** five complete section assessments show as `unreconciled`, and the completion README counts them as queue work.
- **Evidence:** each report's section table maps every parsed heading (tables below). The ADR bytes at the report's own revision equal the snapshot hash:
  - agentbox ADR-2136: `3257df2a8fb2` at `9dfc27b` and at `c57d9c0`.
  - forum ADR-2013: `36919877874e` at `c39f41f`.
  - ontomark ADR-OM-007: `35619bfa78de` at `0caae5d` and at `4b9d15d`.
  - VisionFlow ADR-2008: `1d7b84b53eba` at report commit `de1fd11` and at HEAD. The last change to the file was `1f43d89` on 2026-09-06.
  - VisionFlow ADR-2006: `e7f50fc13264` at report commit `38323b0` and at HEAD. The last change to the file was `7cb004d` on 2026-10-11.
- **Close:** the lead enters the proposed claims after hashing.

**R2: archived knowledgeGraph ADR-008 and ADR-012 have a full section assessment.**

- **Evidence:** `2026-11-01-knowledgegraph-inventory-and-live-receipt.md` § "Archived content, section by section" applies one nine-row table to each page independently. It covers the banner and metadata (which map to the page heading), Definition, Semantic Classification, Relationships, Content and Provenance.
- **Hashes:** the report's inventory (`evidence/2026-11-01/kg-inventory/decision-inventory.json`) records `afce7473…` and `4fa1e11b…` at knowledgeGraph `4ed9ac1`. These equal the snapshot hashes, and the files were last changed in `721d807` on 2026-09-22.
- **Classification:** historical content, explicitly not governance. That limit belongs in the claim basis.

**R3: three VisionFlow panels were reworked against source in the reports that assessed their subject.**

- VF-08.1 (`2026-10-31-corpus-consumer.md`): redrawn, with the Agentbox revision slice advanced and strict citation checking passed.
- VF-03.6 (`2026-10-31-diagram-drift-and-health.md`): corrected in four places against `scripts/estate-health.mjs`. The same report reads ADR-2008 against that file in full.
- VF-07.3 (`2026-10-31-release-evidence.md`): corrected to the digest, commit-binding and missing-fixture behaviour that the report reproduces.

Each panel body at the report commit equals the body at HEAD. Each topic's `verified_commit` is unchanged, because the report commit is still the last commit to touch the topic (`ed5e8eb`, `de1fd11`, `38323b0`).

**Limit:** the VF-03.6 and VF-07.3 reports ran structural and render validation only (`--no-source-paths` for VF-03). They read cited source at the declared `4d361b58` (VF-03 says those bytes are unchanged up to `64ef3af`), but they did not run a strict citation check.

**R4: three ADRs were assessed in full but have changed since.**

| ADR | Report | Bytes the report read | Snapshot hash | Changed after the report |
| --- | --- | --- | --- | --- |
| agentbox ADR-2109 | `2026-10-31-proposal-application.md` | `2c21a280eedc` at `9dfc27b` | `2ffd53c5d69e` | Yes: `7900a3d1a` (2026-11-01) and later amendments, including "Re-verification — 2026-11-01 (cc949bb79)" |
| VisionClaw ADR-2116 | `2026-10-31-proposal-construction.md` | `7c581e18a530` at `75b7241` | `267ff8a61b2b` | Yes: "Scoped re-verification — 2026-11-01, ADR-2168" added |
| VisionClaw ADR-2147 | `2026-10-31-proposal-construction.md` | `8a5d9f58ac41` at `75b7241` | `a50501457a30` | Yes: two 2026-11-01 scoped re-verification sections added |

- **Consequence:** these three get no claim. A re-review needs to assess the added sections, and confirm that the earlier sections have not been edited.
- **Close:** a targeted section re-check in a later wave.

**R5: the narrow-correction precedent holds.** The existing claims say "Narrow subsequent corrections are separately linked". They exclude OM-04.1, AB-13.7 and the NF-05 prose correction by name. Where a 2026-11-01 report already has panel claims, the panels it left unclaimed were deliberately excluded, and this sweep does not reverse that. The panel table below gives each exclusion and the report wording it rests on.

## Proposed ADR claims (earlier report is the named report)

| Repo | ADR | Named report | Heading mapping |
| --- | --- | --- | --- |
| agentbox | `docs/adr/ADR-2136-pin-the-shared-ontology-toolchain.md` | `2026-10-31-corpus-consumer.md` | All 10 headings mapped. Title is covered by the table. Context; Decision 1–5 cover Decision; Consequences; Verification; the 10-11, 10-12 and 10-18 continuations; Close-out 2026-10-30; "Re-verification — 2026-10-31" covers § 10 (heading text truncated). § 8 matches apart from backtick markup |
| VisionFlow | `docs/adr/ADR-2008-estate-health-collected-by-ci-read-by-the-dream-cycle.md` | `2026-10-31-diagram-drift-and-health.md` | Context; Decision (two rows); Consequences (three rows); Verification. The report says the ADR was "read in full" |
| VisionFlow | `docs/adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md` | `2026-10-31-release-evidence.md` | All 7 body headings, including 09-04, 09-05 and the 10-11 roster continuation |
| nostr-rust-forum | `docs/adr/ADR-2013-ontology-governance-panel-promote-demote-expiry.md` | `2026-10-31-governance-ingress.md` | Context; Decision 1–5; Consequences; Verification. The rows "30 October close-out" and "31 October close-out" cover § 6–7 |
| ontomark | `docs/adr/ADR-OM-007-vault-root-is-the-content-root.md` | `2026-10-31-proposal-application.md` | Context; Decision 1–5 cover Decisions; Evidence; Consequences |
| knowledgeGraph | `archive/logseq-era-2026-09-22/pages/ADR-008.md` | `2026-11-01-knowledgegraph-inventory-and-live-receipt.md` | Title (via the banner and metadata rows); Definition; Semantic Classification; Relationships; Content; Provenance |
| knowledgeGraph | `archive/logseq-era-2026-09-22/pages/ADR-012.md` | same | same |

## Panel dispositions

These are reconciliation dispositions, not new semantic reviews. "Qualifies" means the earlier report assesses the panel and the panel is unchanged since. "Body" compares the panel body at the report commit with the body at HEAD.

| Panel | Report | Disposition | Evidence |
| --- | --- | --- | --- |
| VF-08.1 | 2026-10-31-corpus-consumer | qualifies | Redrawn against source; 22 strict revision citations; body same; topic-last = report commit `ed5e8eb` |
| VF-03.6 | 2026-10-31-diagram-drift-and-health | qualifies, with limits | Four source-grounded corrections tied to the full ADR-2008 / `estate-health.mjs` reading; body same; render-only validation |
| VF-07.3 | 2026-10-31-release-evidence | qualifies, with limits | Reworked to the reproduced digest and parity findings; body same; render-only validation |
| VF-03.1 | 2026-10-31-diagram-drift-and-health | dropped | Node-ID and label fix only; the report says "not a full citation or semantic refresh of VF-03" |
| VF-06.1, VF-06.14 | 2026-10-31-market-and-audit-gaps | dropped | Gap observations; "full-topic semantic refresh remains open" |
| NF-06.16 | 2026-10-31-diagram-review-queue, governance-ingress | dropped | SQL wording corrected only; "other historical NF-06 panels remain separately queued" |
| OM-06.4 | 2026-10-31-obsidian-decisions | dropped | A single host-hash claim corrected; "no whole-topic revision stamp was advanced" |
| VC-20.15 | 2026-10-31-proposal-construction | dropped | A single blocker claim corrected at the historical pin |
| NF-03.4, NF-03.10, NF-03.12, NF-03.14 | 2026-11-01-forum-read-cache-and-activity | dropped | ADR reference corrections and "narrow boundary notes". The same report's NF-17 panels are claimed, so these were excluded deliberately |
| NF-09.3, NF-12.3, NF-10.2, NF-10.7 | 2026-11-01-forum-subkey-derivation | dropped | Only the parity and compromise claims were corrected; the report has ADR claims and no panel claims for these |
| NF-05.1, NF-10.7 | 2026-11-01-forum-wallet-foundations | dropped | "This is not a new semantic assessment of every other node in those topics". The NF-10.7 body has also changed since `b6bee87` |
| AB-36.4 | 2026-11-01-mesh-signer-and-custody | dropped | A single receipt-failure statement corrected; "earlier source pin is retained" |
| OM-04.1 | 2026-11-01-ontomark-reasoning-and-certificates | dropped | Excluded by name: "without whole-panel credit" |
| AB-13.7 | 2026-11-01-parallel-remediation | dropped | Excluded by name: "earns no additional assessment credit" |
| DW-04.5, DW-04.9, DW-07.3, DW-07.8 | 2026-11-01-website-information-and-inbox | dropped | Admission and deletion corrections; "not newly declared semantically complete" |
| DW-01.1, DW-01.2 | 2026-11-01-website-poker | dropped | DREAM-only label fix; "not given a fresh semantic assessment" |
| SP-06.3, SP-09.7, SP-14.4 | solid-git-export, 2026-09-07-federation, 2026-11-01-handoff | dropped, out of scope | A JEV advisory note, a stale-claim correction (body since changed) and a draft. All SP-* panels belong to w-solid-* this wave |
| AB-16.2, AB-16.6, AB-11.3, AB-11.14, AB-06.9, ES-08.9, ES-08.11, ES-10.10, NF-10.8, VC-10.10, VC-12.6, VC-22.1, VC-26.13, VC-26.14, VC-36.9 | 2026-09-07 audits | dropped | Mostly narrow corrections or render and count receipts. Every body except AB-16.2 and ES-08.9 has changed since the report commit `00217a5`. AB-16.2 is a seccomp observation and ES-08.9 rebuts an external interpretation; neither is a whole-panel disposition |
| 275 IDs (AB-, VC-, NF-, …) | 2026-11-01-parallel-implementations | dropped | Diagram markers were used as hypothesis sources ("a map, not as truth"); the intake report says imported findings earn no panel credit |

## ADR candidates considered and dropped

- **Changed since assessment:** agentbox ADR-2109, VisionClaw ADR-2116 and ADR-2147 (R4).
- **Record-level tables, not section-level:** the 2026-09-07 reports' "Per-record source disposition", "Every current ADR disposition", "Agentbox per-record disposition", "ADR reconciliation" and "Per-pack disposition" give one row per record or pack. No qualifying whole-document section claim follows from them.
- **Same number, different repository:** ADR-2001, ADR-2006, ADR-2008, ADR-2013, ADR-2109 and ADR-2136 copies elsewhere (for example VisionClaw ADR-2136 "memory-cloud query response" and VisionFlow ADR-2008 "bbs-signer" copy) were not the assessed records.
- **Evidence snapshots:** `docs/estate-review/evidence/2026-11-01/*/source/…` copies of ADR-OBS-001–005, ADR-OM-004, ADR-OM-007, ADR-2136 and ADR-2166 are retained audit evidence (`audit-evidence-or-support`), not decision records.
- **Not assessed:** visionGraph `knowledge/ADR-008.md` and `knowledge/ADR-012.md`. The knowledgeGraph report says its findings "are not inherited by today's published pages".
- **Not assessed:** WasmVOWL `ADR-001-explorer-consumer-contract-closeout.md`. The vowl reports assess vowl-wasm ADR-001, which is already claimed.
- **Owned by w-stale:** VisionClaw ADR-2143, ADR-2164 and ADR-2166, which are `stale-evidence`.
- **Already claimed:** the ID-less headings checked ("Decision dispositions" in visiontouch, "ADR section dispositions" in dream-candidate, "Migration guide section dispositions", and "Contract review by section" in forum-passkey) cover ADRs already claimed or files that are not ADR candidates.
- **Support files, classified but no claim proposed:**
  - knowledgeGraph `docs/adr/PREAMBLE.md`, `README.md` and `TEMPLATE.md` (hashes match the report's inventory).
  - Codescape standalone `README`, `PREAMBLE` and `TEMPLATE` (assessed at `58108c0`, the current HEAD).

  Both reports describe and classify these files. A whole-document claim would also credit TEMPLATE's generic Context and Decision headings, which neither report assesses as decisions. `claims.json` has no support-file claims.

## Verification

**Evidence class:** documentary reconciliation of earlier reports against committed bytes. This pass did not re-trace product source and makes no runtime, test or deployment claim. Every implementation claim stays with its named report and that report's stated limits.

**Snapshot repository heads:**

| Repository | Head |
| --- | --- |
| VisionFlow | `9e5c90a` (worktree HEAD) |
| agentbox | `c57d9c0` |
| VisionClaw | `86cdd76` |
| nostr-rust-forum | `c39f41f` |
| ontomark | `4b9d15d` |
| knowledgeGraph | `4ed9ac1` |
| Codescape standalone | `58108c0` |

**Method:**

1. Matched every panel ID against the snapshot's unreconciled set (323 mentions, 309 distinct IDs).
2. Read the context of each panel mention.
3. Compared each panel body between the report's commit and HEAD. VF topics are clean in the worktree, so the worktree equals HEAD for them.
4. Swept ADR section headings and tables in every report and matched them to snapshot records lacking a claim from that report.
5. Hashed `git show <rev>:<path>` at the report's stated revision and compared it with the snapshot `sha256`.
6. Mapped every parsed heading to a report row by hand.

No generator command was run: this assignment edits no topic, so no `diagram-index-gen` check applies.

**Not verified:**

- The implementation files behind each earlier report may have changed. ADR and panel byte identity does not revalidate the code that a report traced, as the ledger README notes.
- Hashes for the claims are left to the lead.
