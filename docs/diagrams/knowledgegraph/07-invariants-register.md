---
id: KG-07
title: Historical invariants — obligations, dated evidence and publication limits
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
adrs: [ADR-2001, ADR-2002, ADR-2003, ADR-2004]
sources:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
  - ../knowledgeGraph/docs/adr/ADR-2001-corpus-consolidation.md
  - ../knowledgeGraph/docs/adr/ADR-2002-corpus-is-the-ontology.md
  - ../knowledgeGraph/docs/adr/ADR-2003-expected-classes-tripwire.md
  - ../knowledgeGraph/docs/adr/ADR-2004-corpus-page-identity-immutable.md
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/README.md
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/pipeline/manifest.py
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/pipeline/release_gate.py
  - ../knowledgeGraph/scripts/adr-index-gen.js
  - ../knowledgeGraph/docs/adr/PREAMBLE.md
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/pages/ADR-008.md
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/pages/ADR-012.md
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/quartz/scripts/check-urls.sh
verified_commit: {knowledgegraph: 4809ea9772e61766058cf20d38e7527082e78fed, visiongraph: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6}
---

## For developers

These eleven rules belong to the historical baseline and its archived implementation. [The dated review](../../estate-review/2026-11-01-knowledgegraph-publication-gates.md) assesses all three panels and knowledgeGraph ADR-2002, ADR-2003 and ADR-2004 without treating their live status fields as fresh runtime evidence.

## For the business

Archiving a publisher does not remove the need to preserve identity, visibility and provenance. Those promises need acceptance evidence for the replacement publication path; historical test counts cannot supply it.

## KG-07.1 Eleven historical obligations, grouped by concern

```mermaid
flowchart TB
    PUB["1: boolean publication flag and protected projection<br/>5: disclose synthetic corpus provenance<br/>8: account for every parsed, rejected or excluded input<br/>9: exclude private identifiers from public outputs<br/>Historical baseline"]
    %% PUB source: BASELINE-narrativegoldmine.md:184-186,195-197,202-208
    SHAPE["2: NGG1 24-byte record and golden fixture<br/>3: bridges.json carries multi-category membership<br/>7: SharedArrayBuffer disabled pending synchronisation<br/>Historical baseline"]
    %% SHAPE source: BASELINE-narrativegoldmine.md:187-192,200-201
    IDENT["6: independent expected class count<br/>10: count and committed identity set change together<br/>Historical baseline"]
    %% IDENT source: BASELINE-narrativegoldmine.md:198-199,209-212
    PROV["4: verify each generation against its own manifest<br/>11: source revision, generation and artefact SHA-256<br/>Historical baseline"]
    %% PROV source: BASELINE-narrativegoldmine.md:193-194,213-215
    SCOPE["INVARIANT: these are scoped historical obligations<br/>The Logseq source and pipeline are archived<br/>logseq-era-2026-09-22/README.md:1-4"]
    PUB --> SHAPE --> IDENT --> PROV --> SCOPE
```

**What it shows:** All eleven baseline obligations remain visible, with explicit historical scope. The diagram does not claim current enforcement of every obligation by the replacement publisher.

**Why it is this way:** The archive marker retires the implementation; the ADRs preserve its decisions and dated acceptance history. Scope must be stated separately from those status fields.

## KG-07.2 Historical closeout is not successor acceptance

```mermaid
flowchart TB
    OLD["September acceptance reports census, strict flags,<br/>visibility fixtures and export generation checks<br/>ADR-2002, September acceptance"]
    %% Source: ADR-2002-corpus-is-the-ontology.md:100-148
    SET["Archived gate compares exact class identities<br/>release_gate.py:127-148"]
    HASH["Archived manifest inventories and re-hashes output<br/>manifest.py:87-135,173-197"]
    ARCH["Pipeline retired to historical archive<br/>logseq-era-2026-09-22/README.md:1-4"]
    NEW["Current workflow checks minimum volume, domain contracts,<br/>copy hashes, compressed output, smoke and final gates<br/>publish.yml:264-324,341-373,492-520"]
    LIMIT["DEBT: final notes gate compares counts, not identities<br/>check-urls.sh:115-126<br/>Equal-count replacement passes the scoped probe"]
    OLD --> SET --> HASH --> ARCH --> NEW --> LIMIT
```

**What it shows:** The archived identity set and generation verifier are distinct from the current workflow's checks. A notes-page count can agree even when the intended page is absent.

**Why it is this way:** The successor has a different assembly path. Retirement neither proves that all previous obligations transferred nor makes their unresolved acceptance conditions moot.

**Open (ADR-2002 reconciliation, 2026-11-02 and 2026-11-03):** the account above was purely historical at the previous stamp (`4ed9ac1`). ADR-2002 now records that a legacy explorer is retained independently of the authored-vault publisher and pinned to the vowl-wasm v0.1.1 release, and asks for an explicit choice between retiring it and a tested consumer upgrade (`../knowledgeGraph/docs/adr/ADR-2002-corpus-is-the-ontology.md:26`, `../knowledgeGraph/docs/adr/ADR-2002-corpus-is-the-ontology.md:28`). Its 2026-11-03 repair entry says the retained copy's DoubleBuffer now counts outstanding transfers, and claims neither a renderer upgrade nor a remote publication (`../knowledgeGraph/docs/adr/ADR-2002-corpus-is-the-ontology.md:22`). ADR-2002 records no retire-or-upgrade decision yet.

## KG-07.3 Historical gaps and continuing decision tensions

```mermaid
flowchart TB
    LINEAGE["ADR-NG-001 remains historical-absent<br/>Locator map is evidence, not reconstructed authority<br/>ADR-2001, historical-absent mapping"]
    %% Source: ADR-2001-corpus-consolidation.md:108-128
    COUNTS["Historical baseline: 7,874 prose figures were stale,<br/>three uncategorised classes and 4,383+ undeclared targets<br/>Historical baseline<br/>OPEN: these are not a current active-vault census"]
    %% COUNTS source: BASELINE-narrativegoldmine.md:143-168
    REPRO["UUID/time and blank-node labels prevent blanket byte equality<br/>Historical baseline<br/>manifest.py:149-150<br/>Byte receipts and semantic RDF equivalence are distinct"]
    %% REPRO source: BASELINE-narrativegoldmine.md:193-194
    MOVE["TENSION: ADR-2004 forbids relocating corpus pages<br/>ADR-2004, decision<br/>The whole source tree was later archived<br/>logseq-era-2026-09-22/README.md:1-4"]
    %% Source: ADR-2004-corpus-page-identity-immutable.md:34-39
    LINEAGE --> COUNTS --> REPRO --> MOVE
```

**What it shows:** Historical counts and missing lineage stay dated; a later whole-tree archive conflicts with an unqualified forever-in-place reading of ADR-2004. The section does not declare old corpus defects present in the current vault.

**Why it is this way:** A migration needs explicit succession and identity evidence. Comparing raw RDF triple strings is insufficient when blank-node identifiers change; the byte manifest answers a different question.

## KG-07.4 Decision inventory and the validator boundary

```mermaid
flowchart TB
    %% knowledgeGraph/docs/adr/PREAMBLE.md:3
    LEDGER["Baseline plus four operative ledger records<br/>Separate decision, implementation and activation status"]
    %% knowledgeGraph/archive/logseq-era-2026-09-22/pages/ADR-008.md:3
    %% knowledgeGraph/archive/logseq-era-2026-09-22/pages/ADR-012.md:3
    CONTENT["ADR-008 and ADR-012 are frozen corpus pages<br/>Their labels do not confer decision authority"]
    %% knowledgeGraph/scripts/adr-index-gen.js:113
    CHECK["Validator checks required fields and enums<br/>List types and duplicate IDs also reject errors"]
    %% knowledgeGraph/scripts/adr-index-gen.js:148
    GRAPH["DEBT: missing or nonreciprocal supersession warns only<br/>A reciprocal supersession cycle passes"]
    %% knowledgeGraph/scripts/adr-index-gen.js:205
    LIMIT["--check validates metadata, leaving README untouched<br/>It does not establish body evidence, valid revisions,<br/>index freshness or deployed activation"]
    LEDGER --> CHECK --> GRAPH --> LIMIT
    CONTENT --- LEDGER
```

**What it shows:** Decision authority comes from the operative ledger, while the generator establishes a limited metadata contract; the [inventory and eight probes](../../estate-review/2026-11-01-knowledgegraph-inventory-and-live-receipt.md) make that limit reproducible.

**Why it is this way:** The preamble separates corpus content from governance. Required-field validation helps maintain the ledger, but cannot substitute for section review, source verification or a consumer's activation receipt.
