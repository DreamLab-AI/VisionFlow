---
id: KG-07
title: Invariants register — the 11 baseline invariants and the open items they qualify
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
adrs: [ADR-2001, ADR-2002, ADR-2003, ADR-2004]
sources:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
  - ../knowledgeGraph/pipeline/census.py
  - ../knowledgeGraph/pipeline/emit_graph_tiers.py
  - ../knowledgeGraph/pipeline/visibility.py
verified_commit: 75a5c1f1acda50bfe9d66a92a5343cf12f8b84ef
---

## KG-07.1 The 11 baseline invariants

```mermaid
flowchart TB
    I1["1 · vc:public is the ONLY publication gate — strictly boolean,<br/>re-checked in every output stage"]
    I2["2 · NGG1 node record is 24 bytes, one u16 category —<br/>writer + both readers pinned to a 183-byte golden fixture"]
    I3["3 · Bridged membership recoverable ONLY from bridges.json —<br/>binary category is nearest, not membership"]
    I4["4 · Pipeline deterministic within a day for every artefact<br/>EXCEPT ontology.ttl — compare that one by triple set"]
    I5["5 · Corpus is synthetic-AI-generated-under-human-direction,<br/>surfaced from data #40;ATTRIBUTED_TO, CORPUS_NATURE#41;, never rebranded"]
    I6["6 · EXPECTED_CLASSES in build.yml must equal the true class<br/>count, moved in lockstep with any corpus change"]
    I7["7 · SharedArrayBuffer transport stays disabled until a<br/>double-buffered SAB with Atomics-gated flip lands"]
    I8["8 · Every input file accounted for — input_files == parsed<br/>+ rejected + excluded; a release has zero rejected"]
    I9["9 · No public artefact contains a private identifier —<br/>visibility.py at every output, release gate re-scans"]
    I10["10 · Count and identity move together — EXPECTED_CLASSES<br/>and class-identity.txt agree, same commit"]
    I11["11 · Every export carries a generation manifest — source rev,<br/>generation id, counts, SHA-256 per artefact"]
    note1["BASELINE-narrativegoldmine.md:212-245 — numbered 1-11 verbatim"]
```

## KG-07.2 Estate closeout — four 2026-09-04 qualifications, closed 2026-09-05

```mermaid
stateDiagram-v2
    [*] --> Qualified: 2026-09-04 estate closeout review<br/>knowledge-production.md + authored-vault-transition.md
    Qualified --> ParserCensus: pipeline/census.py — balanced, 0 rejected
    Qualified --> StrictFlags: PUBLIC_FLAG_NOT_BOOLEAN / MISSING_PUBLIC_FLAG
    Qualified --> InferenceVis: pipeline/visibility.py — per-format fixture tests
    Qualified --> ImmutableExport: pipeline/manifest.py — versioned, CI re-verified
    ParserCensus --> Closed: 8,138/8,138 balanced, 0 rejected
    StrictFlags --> Closed: non-boolean/absent flag is now an ERROR
    InferenceVis --> Closed: consulted by every exporter
    ImmutableExport --> Closed: generation id + SHA-256 per artefact
    Closed --> [*]
    note right of Closed: DIVERGENCE #40;still open, BASELINE §Estate closeout#41;<br/>equal counts cannot prove equal identities or intended<br/>visibility — release_gate.py's identity SET check is the<br/>mitigation, not a closure of the concern
```

## KG-07.3 Open items the baseline flags — what remains unresolved, by design or not

```mermaid
flowchart LR
    ADRNG["ADR-NG-001 classified historical-absent<br/>#40;resolved via ADR-2001#41; — 33 files cite it, document never existed<br/>in this tree; each cited section mapped to a real in-tree surface"]
    STALE["docs/architecture/pipeline.md + explorer.md still write<br/>against a 7,874-page corpus — corpus is now 8,138 —<br/>every derived count needs re-verification"]
    TTLNODET["ontology.ttl is the one non-byte-reproducible artefact —<br/>rdflib mints fresh blank-node ids per run; compare by triple set"]
    UNCAT["3 classes resolve to NO category — electric-vehicle,<br/>ethan-mollick, urban-planning — a corpus ancestry gap"]
    STUBS["4,383+ object-property targets referenced but never<br/>declared — ship as skos:Concept stubs, deliberately NOT redacted"]
    note1["INVARIANT vs DIVERGENCE: I5/I8/I9 above are enforced invariants;<br/>STALE and UNCAT are acknowledged, unfixed drift — the baseline<br/>distinguishes 'must not silently change' from 'known and open'"]
```
