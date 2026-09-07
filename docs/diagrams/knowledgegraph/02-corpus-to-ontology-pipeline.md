---
id: KG-02
title: Corpus → ontology build pipeline — census to graph tiers to manifest
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
adrs: [ADR-2001, ADR-2002, ADR-2003, ADR-2004]
sources:
  - ../knowledgeGraph/pipeline/build.py
  - ../knowledgeGraph/pipeline/public_projection.py
  - ../knowledgeGraph/pipeline/census.py
  - ../knowledgeGraph/pipeline/jsonld_parser.py
  - ../knowledgeGraph/pipeline/validate.py
  - ../knowledgeGraph/pipeline/visibility.py
  - ../knowledgeGraph/pipeline/manifest.py
  - ../knowledgeGraph/pipeline/emit_graph_tiers.py
  - ../knowledgeGraph/pipeline/jsonld_to_turtle.py
  - ../knowledgeGraph/pipeline/jsonld_to_webvowl.py
  - ../knowledgeGraph/pipeline/jsonld_to_page_api.py
  - ../knowledgeGraph/pipeline/jsonld_to_search.py
  - ../knowledgeGraph/pipeline/backlinks.py
  - ../knowledgeGraph/docs/architecture/pipeline.md
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
  - ../knowledgeGraph/README.md
  - ../knowledgeGraph/.github/workflows/build.yml
verified_commit: worktree-2026-09-07
---

## KG-02.1 Canonical build — safety boundary before every exporter

```mermaid
flowchart TD
    START["pipeline.build.build — build.py:69"] --> PRE["inspect_publication_inputs — public_projection.py"]
    PRE --> CENSUS["take_census — build.py:81"] --> VALID["validate_corpus — build.py:103"]
    VALID -->|"errors"| BLOCK["BuildBlocked; previous destination preserved"]
    VALID --> PUB["public_projection before all export calls"]
    PUB --> TTL["Turtle — build.py:133"]
    PUB --> VOWL["WebVOWL and explorer projection — build.py:143"]
    PUB --> API["Page API — build.py:169"]
    PUB --> SEARCH["Search index — build.py:176"]
    PUB --> TIERS["Graph tiers — build.py:186"]
    PUB --> MD["Projected title-form Markdown and namespace aliases"]
    TTL & VOWL & API & SEARCH & TIERS & MD --> MANIFEST["Manifest hashes staged output — build.py:223"]
    MANIFEST --> PROMOTE["Replace generated trees after successful build"]
```

## KG-02.2 Input accounting and authoring diagnostics

```mermaid
flowchart LR
    FILES["Recursive Markdown inputs"] --> PRE["Typed preflight counts every input outcome"]
    PRE -->|"malformed or ambiguous flag"| STOP["Block build; aggregate error only"]
    PRE --> PARSE["take_census and parse_page_outcome"]
    PARSE --> LOCAL["Detailed Census remains available to local authoring tools"]
    PARSE --> PUBLIC["Released census contains aggregate counts only"]
    PUBLIC --> CHECK["input_files equals included plus excluded; zero rejected inputs"]
```

## KG-02.3 Parse — JSON-LD block extraction and the PageData model

```mermaid
classDiagram
    class PageData {
        path: Path
        page_iri: str
        slug: str
        title: str
        is_public: bool
        schema_version: int
        body: str
        wikilinks: list~WikilinkRef~
        ontology_class: OntologyEntity | None
        raw_page_block: dict
    }
    class OntologyEntity {
        iri: str
        label: str
        entity_type: str
        domain: str
        definition: str
        quality_score: float
        maturity: str
        sub_class_of: list~WikilinkRef~
        relations: RelationSet
    }
    class RelationSet {
        has_part
        requires
        enables
        depends_on
        implements
        contrasts_with
        bridges_to
        uses
        related_to
        supports
        standardized_by
        part_of
    }
    PageData "1" --> "0..1" OntologyEntity : ontology_class
    OntologyEntity "1" --> "1" RelationSet : relations
    note for PageData "JSONLD_BLOCK_RE = re.compile(...) — jsonld_parser.py:124<br/>parse_page_outcome keeps AT MOST TWO blocks:<br/>first @type==Page, first typed Class/Individual/OntologyClass<br/>jsonld_parser.py:264-271"
    note for PageData "is_public=(public_raw is True) — STRICT boolean gate<br/>jsonld_parser.py:291; a non-true value publishes nothing"
```

## KG-02.4 Whole-page permission and reference filtering

```mermaid
flowchart LR
    FLAG["Literal JSON boolean true — jsonld_parser.py:291"] --> PROJECTION["Known private references removed from typed graph, prose and raw JSON"]
    PROJECTION --> EXPORT["All canonical exporters receive public copies"]
    EXPORT --> FILTER["Exporter public filters remain defence in depth"]
    EXPORT --> BACKLINK["Backlinks computed from public pages only — jsonld_to_page_api.py:36"]
    PROJECTION --> LIMIT["Unknown dangling concepts remain; this is not arbitrary secret detection"]
```

## KG-02.5 Visibility — redacting references to private entities, not just private pages

```mermaid
sequenceDiagram
    autonumber
    participant EXP as an exporter (Turtle/WebVOWL/Page API/Search/Tiers)
    participant POL as VisibilityPolicy<br/>pipeline/visibility.py:75
    participant BLD as VisibilityPolicy.from_pages<br/>visibility.py:91

    Note over BLD: builds private_iris/private_slugs and public_iris/public_slugs<br/>from every page's is_public flag (visibility.py:98-121)
    EXP->>POL: is a reference private?
    alt reference resolves to a known private entity
        POL-->>EXP: redact
    else reference resolves to nothing (dangling, e.g. 4,383+ SKOS stubs)
        POL-->>EXP: leave alone — dangling is not private
    end
    Note over POL: INVARIANT: whole-page filtering is the floor, not the ceiling —<br/>subClassOf parents, 12 relation kinds and wikilinks all pass through<br/>this one decision procedure (BASELINE-narrativegoldmine.md:104-107)
```

## KG-02.6 Generation manifest — every artefact SHA-256'd against the source revision

```mermaid
classDiagram
    class Manifest {
        manifest_version
        generation_id: uuid4
        generated_at: ISO8601
        strict: bool
        source: dict
        counts: dict
        census: dict
        validation: dict
        visibility: dict
        artefacts: list~ArtefactEntry~
    }
    class ArtefactEntry {
        path
        sha256
    }
    Manifest "1" --> "*" ArtefactEntry : artefacts
    note for Manifest "build_manifest — manifest.py:137-162<br/>write_manifest → dist/api/generation-manifest.json — manifest.py:166"
    note for ArtefactEntry "verify_manifest re-hashes the tree and reports every drift —<br/>missing, sha256 mismatch, or present-but-unrecorded — manifest.py:173-196"
```

Execution qualification — 2026-09-07: `public_projection.py` adds an input preflight in every build mode, public-only graph/prose projection, safe title-form markdown and fresh staging. Detailed census diagnostics remain available to local authoring callers; released census/validation artefacts contain aggregate counts. Successful rebuilds remove stale generated private pages, while rejected builds preserve the prior destination. See [execution evidence](../../estate-review/closeout/2026-09-07-execution-federation.md).
