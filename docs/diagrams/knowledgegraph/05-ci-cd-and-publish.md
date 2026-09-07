---
id: KG-05
title: CI/CD — the six build.yml gates, and the deploy that happens in a different repo
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/ci-cd/build-and-gates.md
adrs: [ADR-2003]
sources:
  - ../knowledgeGraph/.github/workflows/build.yml
  - ../knowledgeGraph/docs/ci-cd/build-and-gates.md
  - ../knowledgeGraph/docs/ecosystem.md
  - ../knowledgeGraph/pipeline/release_gate.py
  - ../knowledgeGraph/pipeline/jsonld_to_page_api.py
verified_commit: 2791111fc
---

## KG-05.1 build.yml — six gates, cheapest first, no deploy step

```mermaid
flowchart TD
    CO["Checkout — build.yml:94"]
    G1["GATE 1 · Secret scan<br/>8-pattern anchored alternation over ontology/<br/>build.yml:109-130 — 0 hits at 8,138 pages"]
    G2["GATE 2 · pytest pipeline/tests -q<br/>build.yml:147-148"]
    BUILD["python -m pipeline.build ontology/pages dist-ci --strict<br/>build.yml:160-163"]
    G3["GATE 3 · Corpus contract<br/>EXPECTED_CLASSES=8138 vs stats.json + ontology.json<br/>build.yml:83,185-213"]
    G4["GATE 4 · Validate<br/>python -m pipeline.validate — 0 errors required<br/>build.yml:225-226"]
    G5["GATE 5 · Release contracts<br/>release_gate.py — identity set, schema, visibility<br/>build.yml:239-244"]
    G6["GATE 6 · Manifest verify<br/>pipeline.manifest re-hashes the tree<br/>build.yml:252-253"]
    ART["Upload dist-ci artefact<br/>if: always#40;#41;, retention 14 days — build.yml:263-270"]
    CO --> G1 --> G2 --> BUILD --> G3 --> G4 --> G5 --> G6 --> ART
    note1["INVARIANT: permissions: contents: read at workflow AND job level —<br/>this workflow cannot write to the repository even if a step tried<br/>(build.yml:65-66,89-90)"]
```

## KG-05.2 Gate 5 — release_gate.py catches what the count alone cannot

```mermaid
flowchart LR
    COUNT["GATE 3: a COUNT — 8138 classes"]
    SUB["equal-count identity substitution<br/>delete one class, add another — count UNCHANGED"]
    COUNT -.->|"cannot see this"| SUB
    IDENT["check_identity — release_gate.py:127<br/>vs pipeline/contracts/class-identity.txt sorted IRI SET"]
    SCHEMA["check_schema — release_gate.py:151<br/>consumer-shaped class#91;#93;/property#91;#93; arrays present"]
    VIS["check_publication_visibility — release_gate.py:183<br/>re-derives private IRIs from SOURCE, scans built dist-ci<br/>for their presence in any public artefact"]
    SUB --> IDENT
    IDENT & SCHEMA & VIS --> RUN["run_gate — release_gate.py:245"]
    note1["INVARIANT: count and identity move TOGETHER — EXPECTED_CLASSES and<br/>class-identity.txt must agree and change in the same commit<br/>(BASELINE-narrativegoldmine.md:239-242)"]
```

## KG-05.3 Publication topology — deploy lives in visionGraph's workflow, not here

```mermaid
sequenceDiagram
    autonumber
    participant VG as visionGraph publish.yml<br/>EXTERNAL: see VG-03.1
    participant PAGES as gh-pages branch<br/>DreamLab-AI/knowledgeGraph
    participant SITE as narrativegoldmine.com
    participant THIS as this repo's build.yml

    Note over VG: build.yml here has NO peaceiris/actions-gh-pages step,<br/>no secrets.* reference, no CNAME write (ecosystem.md:254-256)
    VG->>PAGES: push publish_dir=www, external_repository=knowledgeGraph<br/>commit_message references jjohare/visionGraph @ sha
    PAGES->>SITE: GitHub Pages: html_url narrativegoldmine.com,<br/>cname narrativegoldmine.com, status built
    THIS->>THIS: reproducible half only — secret scan, pytest,<br/>7-stage build, class-count gate, validate — ecosystem.md:254-259
    Note over THIS,SITE: this repo's root CNAME (narrativegoldmine.com) is read by<br/>GitHub Pages branch publishing — no workflow HERE writes it<br/>(build-and-gates.md §4, the 'one clarification' note)
```

## KG-05.4 Gates added because something broke — the markdown-mirror incident

```mermaid
flowchart TB
    BUG["shell grep filter: 'vc:public: true' WITH a space<br/>corpus carries compact JSON-LD too: 'vc:public:true' — no space"]
    IMPACT["890 of 7,874 pages silently dropped from the title-form<br/>markdown mirror — front end 404'd, build reported SUCCESS<br/>(fewer files copied is not an error)"]
    FIX1["regex made whitespace-tolerant"]
    FIX2["CONTRACT GATE added: mirror file count vs<br/>independently re-parsed count via parse_corpus"]
    BUG --> IMPACT --> FIX1
    IMPACT --> FIX2
    note1["INVARIANT #40;the general lesson#41;: a filter that under-publishes emits NO<br/>error signal — a contract gate must assert a count against an<br/>INDEPENDENTLY COMPUTED expectation over the same population<br/>the emitter writes (build-and-gates.md 'lesson generalises')"]
    note2["Runnable form: md_count#61;ls dist/api/markdown/*.md | wc -l vs<br/>parse_corpus#40;#41; count of #40;is_public and body#41; — both read 7,823<br/>#40;7,874 public minus 51 empty-body pages, jsonld_to_page_api.py:33#41;"]
```
