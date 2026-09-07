---
id: KG-02
title: Corpus → ontology build pipeline — 8 stages, census to graph tiers
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
adrs: [ADR-2001, ADR-2002, ADR-2003, ADR-2004]
sources:
  - ../knowledgeGraph/pipeline/build.py
  - ../knowledgeGraph/pipeline/census.py
  - ../knowledgeGraph/pipeline/jsonld_parser.py
  - ../knowledgeGraph/pipeline/validate.py
  - ../knowledgeGraph/pipeline/visibility.py
  - ../knowledgeGraph/pipeline/manifest.py
  - ../knowledgeGraph/pipeline/emit_graph_tiers.py
  - ../knowledgeGraph/docs/architecture/pipeline.md
verified_commit: 2791111fc
---

## KG-02.1 build() — 8 stages plus manifest, in fixed order

```mermaid
flowchart TD
    START["python -m pipeline.build ontology/pages dist<br/>build.py:68 def build"]
    S1["1/8 Census<br/>census.take_census — build.py:78"]
    S2["2/8 Validate<br/>validate_corpus — build.py:97"]
    S3["Turtle<br/>jsonld_to_turtle.build_graph — dist/data/ontology.ttl"]
    S4["WebVOWL JSON<br/>jsonld_to_webvowl.build_webvowl — dist/data/ontology.json"]
    S5["Page API + backlinks<br/>jsonld_to_page_api + backlinks.py"]
    S6["Search index<br/>jsonld_to_search.build_search_index"]
    S7["Graph tiers NGG1<br/>emit_graph_tiers.emit_graph_tiers"]
    S8["Generation manifest<br/>manifest.build_manifest + write_manifest"]
    START --> S1 --> S2
    S2 -->|"strict and errors: raise BuildBlocked<br/>before ANY artefact written — build.py:115-119"| BLOCK["BuildBlocked<br/>no artefact emitted"]
    S2 --> S3 --> S4
    S2 --> S5
    S2 --> S6
    S2 --> S7
    S3 --> S8
    S4 --> S8
    S5 --> S8
    S6 --> S8
    S7 --> S8
    note1["INVARIANT: is_public re-checked independently in Turtle, WebVOWL, Page API,<br/>Search and Graph-tiers — no stage inherits a filtered list from another (KG-02.4)"]
```

## KG-02.2 Census — every input file accounted for, or the build refuses

```mermaid
sequenceDiagram
    autonumber
    participant BUILD as build()<br/>pipeline/build.py:68
    participant CENSUS as take_census<br/>pipeline/census.py:86
    participant OUTCOME as parse_page_outcome<br/>pipeline/jsonld_parser.py:218
    participant CE as Census<br/>pipeline/census.py:37

    BUILD->>CENSUS: take_census(pages_dir)
    loop every *.md under ontology/pages/
        CENSUS->>OUTCOME: parse_page_outcome(path)
        alt page parses
            OUTCOME-->>CENSUS: PageData, None
        else rejected or excluded
            OUTCOME-->>CENSUS: None, PageRejection(status, code, detail)<br/>codes: UNREADABLE · NO_JSONLD_FENCE ·<br/>MALFORMED_JSONLD · NO_PAGE_BLOCK (jsonld_parser.py:229-274)
        end
    end
    CENSUS->>CE: input_files, parsed, rejected[], excluded[]
    BUILD->>CE: assert_census_clean(census, strict)<br/>census.py:100
    CE->>CE: balanced == (input_files == parsed + len(rejected) + len(excluded))<br/>census.py:61
    alt strict and not balanced, or rejected non-empty
        CE-->>BUILD: raise CensusError → BuildBlocked<br/>build.py:91-93
    end
    Note over CE: INVARIANT: input_files == parsed + rejected + excluded, always —<br/>a release contains zero rejected entries (census.py:61)
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
    note for PageData "is_public=(public_raw is True) — STRICT boolean gate<br/>jsonld_parser.py:284-288; a non-true value publishes nothing"
```

## KG-02.4 is_public — one gate, checked independently in every stage

```mermaid
flowchart LR
    GATE["vc:public strictly boolean<br/>jsonld_parser.py:284"]
    TTL["jsonld_to_turtle.py:258<br/>if public_only and not page.is_public: continue"]
    VOWL["jsonld_to_webvowl.py:48<br/>[p for p in pages if p.is_public and p.ontology_class]"]
    API["jsonld_to_page_api.py:33<br/>public_pages = [p for p in pages if p.is_public]"]
    SRCH["jsonld_to_search.py:30<br/>if not page.is_public: continue"]
    TIERS["emit_graph_tiers.py:538<br/>[p for p in pages if p.is_public and p.ontology_class]"]
    GATE --> TTL & VOWL & API & SRCH & TIERS
    BL["build_backlink_index — backlinks.py:13<br/>runs over the FULL page list, filtered<br/>separately after (BASELINE §Current State)"]
    GATE -.->|"one deliberate exception, documented"| BL
    note1["INVARIANT: no stage may filter once and let another inherit the list<br/>(BASELINE-narrativegoldmine.md:214-216)"]
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
