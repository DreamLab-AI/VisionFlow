---
id: KG-02
title: Corpus → ontology build pipeline — census to graph tiers to manifest
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
  - ../knowledgeGraph/pipeline/jsonld_to_turtle.py
  - ../knowledgeGraph/pipeline/jsonld_to_webvowl.py
  - ../knowledgeGraph/pipeline/jsonld_to_page_api.py
  - ../knowledgeGraph/pipeline/jsonld_to_search.py
  - ../knowledgeGraph/pipeline/backlinks.py
  - ../knowledgeGraph/docs/architecture/pipeline.md
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
  - ../knowledgeGraph/README.md
  - ../knowledgeGraph/.github/workflows/build.yml
verified_commit: 2791111fc
---

## KG-02.1 build() — the pipeline's OWN two stage counts disagree

```mermaid
flowchart TD
    START["python -m pipeline.build ontology/pages dist<br/>build.py:68 def build"]
    S1["#91;1/8#93; Census<br/>census.take_census — build.py:78"]
    S2["#91;2/8#93; Validate<br/>validate_corpus — build.py:100"]
    S3["#91;3/8#93; Turtle — code comment 'Stage 3a'<br/>jsonld_to_turtle.build_graph — build.py:126,128"]
    S4["#91;4/8#93; WebVOWL — code comment 'Stage 3b'<br/>jsonld_to_webvowl.build_webvowl — build.py:136,138"]
    S5["#91;5/8#93; Page API + backlinks — comment 'Stage 4'<br/>jsonld_to_page_api + backlinks.py — build.py:161,163"]
    S6["#91;6/8#93; Search index — comment 'Stage 5'<br/>jsonld_to_search.build_search_index — build.py:169,171"]
    S7["#91;7/8#93; Graph tiers NGG1 — comment 'Stage 6'<br/>emit_graph_tiers.emit_graph_tiers — build.py:179,181"]
    S8["#91;8/8#93; Generation manifest — comment 'Stage 7'<br/>manifest.build_manifest + write_manifest — build.py:203,205"]
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
    note2["DOC-DRIFT: build.py's OWN two numbering schemes disagree. Its print#40;#41;<br/>progress markers count 8 STEPS #91;1/8#93;..#91;8/8#93; #40;splitting Turtle/WebVOWL and<br/>counting the manifest#41;, shown above; its '# Stage N' code COMMENTS instead<br/>collapse Turtle+WebVOWL into one 'Stage 3' #40;3a/3b#41; and count 7 top-level<br/>stages #40;1,2,3,4,5,6,7 — comment 'Stage 7' is the manifest#41;. README.md:173<br/>'pipeline/ — 7 stages', README.md:418 'the seven-stage build' and<br/>build.yml:14 'this same seven-stage pipeline' all follow the COMMENT<br/>count, not the PRINT count a reader running the build actually sees"]
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
