---
id: VG-02
title: Authoring → pipeline → published corpus — the 9-stage build and its swarm-authoring gates
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-002]
sources:
  - ../visionGraph/pipeline/jsonld_parser.py
  - ../visionGraph/pipeline/build.py
  - ../visionGraph/pipeline/public_projection.py
  - ../visionGraph/pipeline/conflicts.py
  - ../visionGraph/pipeline/iri_integrity.py
  - ../visionGraph/pipeline/reason.py
  - ../visionGraph/pipeline/gate.py
  - ../visionGraph/pipeline/scaffold_index.py
  - ../visionGraph/pipeline/prose_index.py
  - ../visionGraph/pipeline/validate.py
verified_commit: worktree-2026-09-07
---

## VG-02.1 build() — 9 stages, two more than knowledgeGraph's pipeline

```mermaid
flowchart TD
    S1["Stage 1 · Parse<br/>build.py:31"]
    S2["Stage 2 · Validate<br/>build.py:38"]
    S3A["Stage 3a · Turtle<br/>build.py:48"]
    S3B["Stage 3b · WebVOWL<br/>build.py:56"]
    S4["Stage 4 · Reasoning closure<br/>EL-profile transitive subClassOf<br/>build.py:64"]
    S5["Stage 5 · Page API<br/>closure-enriched<br/>build.py:72"]
    S6["Stage 6 · Search index<br/>build.py:78"]
    S7["Stage 7 · Graph tiers NGG1<br/>build.py:86"]
    S8["Stage 8 · Scaffold index<br/>compact one-file class index<br/>build.py:91"]
    S9["Stage 9 · Prose index<br/>definitions + Current Landscape excerpts<br/>build.py:105"]
    PRE["Input census: reject malformed and ambiguous publication flags"] --> S1
    S1 --> S2 --> PUB["Public projection and safe markdown mirror"] --> S3A --> S3B
    PUB --> S4 --> S5
    PUB --> S6
    PUB --> S7
    S4 --> S8
    S4 --> S9
    note1["DIVERGENCE from knowledgeGraph#39;s 7-stage build #40;by the SAME '# Stage N'<br/>code-comment count — see KG-02.1#41;: VG adds Stage 4 EL-closure reasoning<br/>#40;pipeline.reason#41; feeding BOTH the closure-enriched Page API #40;5#41; and two<br/>NEW agent-consumer indexes #40;8, 9#41; that knowledgeGraph does not emit"]
    note2["DOC-DRIFT #40;this repo's OWN build.py#41;: its print#40;#41; progress markers run<br/>#91;1/9#93; through #91;8/9#93; for stages 1-7, then flip denominator to<br/>#91;9/10#93; and #91;10/10#93; for stages 8-9 #40;build.py:95,104#41; — the scaffold/prose<br/>indexes were added without updating the earlier steps' /9 to /10. A reader<br/>watching stdout sees an inconsistent total; the comment count above #40;9#41;<br/>is the only self-consistent one"]
```

## VG-02.2 pipeline.conflicts — semantica-style pre-merge conflict detection for swarm authoring

```mermaid
sequenceDiagram
    autonumber
    participant SWARM as multi-agent authoring swarm
    participant CONF as pipeline.conflicts<br/>conflicts.py:222 main
    participant DET as detect_conflicts<br/>conflicts.py:187

    SWARM->>CONF: python -m pipeline.conflicts knowledge/pages --severity high
    CONF->>DET: analyse pages
    DET->>DET: detect_duplicate_concepts — distinct IRIs, same normalised label<br/>conflicts.py:103
    DET->>DET: detect_subclass_cycles — a cycle in subClassOf<br/>conflicts.py:122
    DET->>DET: detect_relation_contradictions — subClassOf AND contrasts_with same target<br/>conflicts.py:162
    DET->>DET: detect_type_conflicts — subClassOf parent declared an Individual<br/>conflicts.py:174
    DET-->>CONF: ConflictReport — DUPLICATE_CONCEPT/SUBCLASS_CYCLE high,<br/>RELATION_CONTRADICTION/TYPE_CONFLICT medium
    alt any conflict at or above --severity
        CONF-->>SWARM: exit 1 — gate, resolve highs before write proceeds
    else clean
        CONF-->>SWARM: exit 0
    end
    Note over CONF: pairs pipeline.validate #40;structural well-formedness#41; with SEMANTIC<br/>conflicts a multi-agent swarm creates — adopted from semantica#39;s<br/>ConflictDetector pattern, natively, no live dependency
```

## VG-02.3 pipeline.iri_integrity — baseline-aware referential integrity gate

```mermaid
flowchart TB
    REF["urn:ngm:class: IRI referenced in subClassOf or relations"]
    RESOLVE["should resolve to a class DECLARED by some page#39;s json-ld block"]
    ORPHAN["collect_orphans — iri_integrity.py:271<br/>Orphan#123;referencing_page, ref_iri, nearest_declared_label#125;"]
    LEV["nearest_label — bounded Levenshtein, stdlib only<br/>iri_integrity.py:83,193 — a 'did you mean' candidate"]
    BASELINE["iri_integrity_baseline.json — committed snapshot<br/>of the CURRENT orphan set (load_baseline — iri_integrity.py:298)"]
    NEW["gate hard-fails ONLY on NEW orphans absent from<br/>the baseline — pre-existing gap is non-blocking"]
    REF --> RESOLVE
    RESOLVE -->|"dangling"| ORPHAN --> LEV
    ORPHAN --> BASELINE --> NEW
    note1["Named incident: a historic rename #40;tax → corporate-tax-compliance-<br/>framework#41; also matched the substring inside 'taxonomy', corrupting<br/>the class IRI to '...frameworkonomy' in 10 pages — build succeeded,<br/>validation reported clean #40;iri_integrity.py header, ADR-NG-002 Problem 1#41;"]
```

## VG-02.4 pipeline.reason — pure-Python EL-profile closure

```mermaid
sequenceDiagram
    autonumber
    participant BUILD as build.py stage 4
    participant COMP as compute_closure<br/>reason.py:82
    participant BFS as ancestor BFS<br/>visited set, cycle-safe
    participant EMIT as emit_inferred_ttl<br/>reason.py:171

    BUILD->>COMP: compute_closure(pages)
    COMP->>BFS: for each class, walk subClassOf transitively
    Note over BFS: a class in a subClassOf cycle receives the closure of its<br/>strongly connected component MINUS itself — cycle-safe by construction
    BFS-->>COMP: Closure — inferred #40;non-direct#41; superclass set<br/>+ relations inherited from ancestors
    COMP->>EMIT: emit_inferred_ttl#40;pages, closure, www/data/ontology-inferred.ttl#41;
    Note over COMP: reason.py — no external reasoner dependency #40;pure Python#41;,<br/>distinct from VisionClaw#39;s Whelk-rs OWL 2 EL reasoner — EXTERNAL: VC-20
```

## VG-02.5 pipeline.gate — domain-true autonomous continuation gate for swarm loops

```mermaid
flowchart LR
    QUICK["quick tier<br/>pipeline.validate errors == 0<br/>gate.py:110 check_validate — fast, in-process"]
    FULL["full tier<br/>quick + OWL/turtle BUILD #40;no build errors#41;<br/>+ RuVector recall band — gate.py:131,149"]
    VERDICT["run_gate — gate.py:182<br/>GateVerdict#40;checks#91;#93;#41;"]
    LOOP["autonomous_loop — gate.py:214<br/>pre-gate #40;line 234#41; before EVERY iteration"]
    QUICK --> VERDICT
    FULL --> VERDICT
    VERDICT --> LOOP
    note1["INVARIANT #40;adapted from prime-agent, kept verbatim#41;: a PASSED gate<br/>verifies ONLY that the graph stays logically well-formed — it does<br/>NOT prove the enrichment is correct; hitting a budget limit is NOT<br/>success, the loop stops 'out of budget', not 'done' #40;gate.py header#41;"]
```

## VG-02.6 Agent-consumer indexes — scaffold_index and prose_index, split by layer

```mermaid
classDiagram
    class ScaffoldEntry {
        t: Title
        d: definition, truncated 400 chars
        dom: domain
        q: qualityScore
        m: maturity
        sup: direct parent slugs
        isup: inferred #40;non-direct#41; ancestor slugs
        rel: relation map, empty lists omitted
        bl: backlink slugs, max 20
    }
    class ProseEntry {
        dfull: full definition, ONLY when it exceeds<br/>the scaffold-index 400-char truncation
        cl: Current Landscape section, capped
    }
    ScaffoldEntry "1" ..> "0..1" ProseEntry : complements, no duplication
    note for ScaffoldEntry "emit_scaffold_index — scaffold_index.py:55<br/>schema contract version 1 — www/data/scaffold-index.json"
    note for ProseEntry "emit_prose_index — prose_index.py:64<br/>pages contributing NEITHER field are omitted, keeping the file small.<br/>Consumers treat absence of a slug as 'no prose beyond structural'"
```

## VG-02.7 Shared public boundary in the actual and extracted publishers

```mermaid
flowchart TB
    VAULT["visionGraph actual producer"] --> CENSUS["Inspect every input before parse can discard failures"]
    EXTRACT["knowledgeGraph extracted producer"] --> CENSUS
    CENSUS --> VALID["Malformed, non-boolean or conflicting flags block publication"]
    VALID --> PUBLIC["Copy public pages; remove known private graph and prose references"]
    PUBLIC --> REASON["visionGraph closure sees public nodes only"]
    PUBLIC --> MIRROR["Title-form markdown generated from projected JSON-LD and body"]
    REASON --> STAGE["Build into fresh sibling staging directory"]
    MIRROR --> STAGE
    STAGE --> REPLACE["Successful build replaces generated trees; failed build preserves previous bundle"]
    REPLACE --> LIMIT["Build-time guarantee only: deployed activation and consumer acknowledgement are separate"]
```

Execution qualification, 2026-09-07: both canonical builders use `pipeline/public_projection.py`. The earlier private-grandparent and raw-markdown bypass findings are retained in the [audit](../../estate-review/2026-09-07-federation-audit.md); their regressions now pass. visionGraph's workflow no longer re-copies raw Markdown after the build. The actual dirty corpus produces 8,432 public pages; authored deletions are preserved. See the [execution receipt](../../estate-review/closeout/2026-09-07-execution-federation.md) for tests and deployment limits.
