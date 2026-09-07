---
id: VG-05
title: Invariants register — vault contract facts vs the proposed, inactive closeout ADRs
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-001, ADR-VG-002]
sources:
  - ../visionGraph/docs/PUBLICATION-contract.md
  - ../visionGraph/docs/adr/ADR-VG-001-publication-policy-boundaries.md
  - ../visionGraph/docs/adr/ADR-VG-002-generation-and-consumer-identity.md
  - ../visionGraph/CLAUDE.md
  - ../visionGraph/README.md
verified_commit: 9e308164c
---

## VG-05.1 Live invariants — what actually holds today

```mermaid
flowchart TB
    I1["1 · public: true is the KG inclusion gate;<br/>non-empty owl-class ingests unconditionally<br/>CLAUDE.md:10-11"]
    I2["2 · NO frontmatter means private — the gate<br/>fails CLOSED, not open"]
    I3["3 · Page identity is the vault-relative path under<br/>pages/ WITHOUT .md, / as namespace separator"]
    I4["4 · Journals #40;YYYY-MM-DD.md#41; are excluded<br/>from graph ingest"]
    I5["5 · knowledge/assets is a SYMLINK to<br/>../working/assets — never replace it<br/>#40;961 MB stored once, shared#41;"]
    I6["6 · VAULT_ROOT is the single path authority —<br/>no consumer hard-codes a corpus path"]
    I7["7 · agents NEVER push this repo — a push<br/>deploys to narrativegoldmine.com #40;CLAUDE.md:22-25#41;"]
    note1["Sources: README.md 'vault contract' + 'How consumers bind to it';<br/>CLAUDE.md 'Vault contract' + 'Agents never push this repo'"]
```

## VG-05.2 Proposed, inactive — ADR-VG-001 and ADR-VG-002 acceptance requirements

```mermaid
flowchart TB
    subgraph VG001["ADR-VG-001 — publication inclusion + inferred visibility<br/>decision_status: proposed · activation_status: inactive<br/>ADR-VG-001-publication-policy-boundaries.md:24, one dense paragraph"]
        A1["every input: explicit included/excluded/invalid<br/>disposition under a NAMED policy"]
        A2["require BOOLEAN publication metadata"]
        A3["expose conflicting inclusion fields"]
        A4["prevent malformed input disappearing<br/>before validation"]
        A5["deliberate, TESTED private-ancestor<br/>disclosure policy across API/scaffold/Turtle"]
    end
    subgraph VG002["ADR-VG-002 — generation and consumer identity<br/>decision_status: proposed · activation_status: inactive<br/>ADR-VG-002-generation-and-consumer-identity.md:24, one dense paragraph"]
        B1["ONE release manifest: source/corpus/pipeline/<br/>explorer/dependency/policy identity + output hashes"]
        B2["demonstrate deletion, equal-count substitution,<br/>namespace moves, interrupted activation, rollback"]
        B3["verify embedded explorer schema + browser<br/>behaviour INDEPENDENTLY"]
        B4["preserve pre-split provenance and frozen /notes<br/>WITHOUT presenting as current production"]
    end
    note1["Both records: 'this record does not ratify the proposal or certify<br/>a deployment' — ADR-VG-001/002 closeout extension, verbatim"]
```

## VG-05.3 What is NOT yet established — the gap between behaviour and contract

```mermaid
flowchart LR
    CURRENT["current behaviour #40;PUBLICATION-contract.md:11#41;:<br/>omitted malformed fences · truthy string flags ·<br/>private-ancestor leakage · asserted/inferred IRI divergence"]
    NOTESTABLISHED["explicitly NOT established:<br/>'no real private-page disclosure is established' —<br/>these are reproducible BOUNDARY CASES, not incidents"]
    STALE["'earlier local tests reported 57 passed and one<br/>corpus-size assertion failure' — that dated result is<br/>NOT a fresh census or remote deployment receipt"]
    REOPEN["review_trigger #40;both ADRs, identical#41;: change to authoring,<br/>publication, inference, consumer or deployment contracts"]
    CURRENT --> NOTESTABLISHED
    CURRENT --> STALE
    NOTESTABLISHED --> REOPEN
    STALE --> REOPEN
    note1["DIVERGENCE: this is a PROPOSED closeout contract #40;2026-09-04#41; that<br/>'separates current behaviour from proposed acceptance requirements' —<br/>it does not replace the existing authoring format or claim production<br/>activation #40;PUBLICATION-contract.md:3#41;"]
```
