---
id: KG-03
title: Ontology architecture — page anatomy, Turtle mapping, taxonomy resolution, bridging
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
adrs: [ADR-2002, ADR-2003]
sources:
  - ../knowledgeGraph/pipeline/jsonld_parser.py
  - ../knowledgeGraph/pipeline/jsonld_to_turtle.py
  - ../knowledgeGraph/pipeline/emit_graph_tiers.py
  - ../knowledgeGraph/pipeline/validate.py
  - ../knowledgeGraph/static/ns/v2.jsonld
  - ../knowledgeGraph/docs/reference/jsonld-schema.md
  - ../knowledgeGraph/docs/architecture/pipeline.md
  - ../knowledgeGraph/docs/ecosystem.md
  - ../knowledgeGraph/docs/BASELINE-narrativegoldmine.md
verified_commit: 2791111fc
---

## KG-03.1 Page anatomy — two-to-three fenced json-ld blocks per Logseq page

```mermaid
flowchart TB
    FILE["ontology/pages/A Star Algorithm.md"]
    P1["logseq page properties<br/>public:: true, alias::"]
    H1["H1 title line"]
    B1["fenced json-ld block — Page block<br/>vc: namespace, publication metadata<br/>jsonld-schema.md:21"]
    B2["fenced json-ld block — Class block<br/>@context v2.jsonld, urn:ngm:class:SLUG<br/>jsonld-schema.md:21"]
    B3["fenced json-ld block — vc:LinkResolutionsAnnotation<br/>optional, authoring provenance ONLY<br/>read by NO pipeline stage"]
    OUT["Logseq outline body<br/>never parsed by the pipeline"]
    FILE --> P1 --> H1 --> B1 --> B2 --> B3 --> OUT
    note1["4,173 pages: Page+Class only. 3,684: +vc:LinkResolutionsAnnotation.<br/>17: +legacy LinkResolutionsAnnotation. 19,449 blocks total, 0 json.loads failures<br/>(docs/reference/jsonld-schema.md:21, 32-36)"]
```

## KG-03.2 v2.jsonld context — term mapping onto owl/rdfs/skos/prov/vc

```mermaid
classDiagram
    class JsonLdTerms {
        Class → owl:Class
        Individual → owl:NamedIndividual
        label → rdfs:label
        definition → rdfs:comment
        domain → vc:sourceDomain
        maturity → vc:maturity
        qualityScore → vc:qualityScore #40;xsd:float#41;
        subClassOf → rdfs:subClassOf #40;@type:@id, @container:@set#41;
        instanceOf → rdf:type
        sameAs → owl:sameAs
        relations → scoped @context #40;12 predicates#41;
    }
    note for JsonLdTerms "static/ns/v2.jsonld, served at /ns/v2.jsonld<br/>docs/reference/jsonld-schema.md:66-77"
    note for JsonLdTerms "DOC-DRIFT: only 7,531 of ~19k Page/Class blocks cite v2.jsonld;<br/>9,546 cite /context/v1.jsonld (does not dereference on the site),<br/>1,538 cite /ns/v1 (ecosystem.md:93-96) — parser never fetches remote,<br/>so this affects only an external JSON-LD consumer, not the build"
```

## KG-03.3 Turtle emission — per-class triples plus four structures with no JSON-LD source

```mermaid
flowchart TB
    CLS["per Class: rdf:type owl:Class, rdfs:label, rdfs:comment,<br/>vc:sourceDomain, vc:qualityScore, vc:slug, vc:hasMaturity,<br/>rdfs:subClassOf per parent — jsonld_to_turtle.py:94 build_graph"]
    SKOS["1 · SKOS taxonomy marking<br/>TAXONOMIC_SLUGS #40;6 domain + 34 category#41; get skos:Concept;<br/>a subClassOf edge into one ALSO emits skos:broader<br/>jsonld_to_turtle.py:50,279-284 — 3,458 triples"]
    EXIST["2 · Existential restrictions<br/>every requires/hasPart edge → owl:Restriction BNode<br/>#40;onProperty + someValuesFrom#41; as extra subClassOf<br/>19,751 restrictions, jsonld_to_turtle.py:359"]
    DISJ["3 · Domain-root disjointness<br/>owl:AllDisjointClasses over the 6 domain roots<br/>jsonld_to_turtle.py:404"]
    STUB["4 · Dangling-target stubs<br/>class/ IRI referenced but never declared → skos:Concept<br/>+ slug-derived label; 4,383 stubs<br/>jsonld_to_turtle.py:380-387"]
    CLS --> SKOS
    CLS --> EXIST
    CLS --> DISJ
    CLS --> STUB
    note1["DIVERGENCE #40;resolved#41;: AllDisjointClasses once made 5,881/5,951 classes<br/>#40;98.8%#41; unsatisfiable via EL's ∃R.⊥≡⊥ propagation; fixed by single-domain-<br/>normalising the taxonomy #40;903 clashes/370 pages remediated, 9 cycles broken#41;<br/>jsonld_to_turtle.py:391-401"]
```

## KG-03.4 Category resolution — breadth-first walk to the NEAREST ancestor category

```mermaid
sequenceDiagram
    autonumber
    participant EGT as build_graph_model<br/>emit_graph_tiers.py:535
    participant RES as _build_category_resolver<br/>emit_graph_tiers.py:483
    participant WALK as BFS over subClassOf/instanceOf ancestry

    EGT->>RES: resolve_category = _build_category_resolver(public_pages)
    RES->>WALK: frontier = parents_of.get(slug, ()) — declared order, tuple not set
    loop depth < MAX_DEPTH #61; 12
        WALK->>WALK: for ps in frontier: CATEGORY_INDEX.get(ps) — first hit wins
        alt category found at this depth
            WALK-->>RES: found = cid — nearest wins, stop, don't go deeper
        else
            WALK->>WALK: frontier = next depth's parents, seen-guarded against cycles
        end
    end
    RES-->>EGT: category_id or CATEGORY_NONE #40;0xFFFF#41;
    Note over WALK: 3 classes reach no category root at all: electric-vehicle,<br/>ethan-mollick, urban-planning — a corpus ancestry gap<br/>#40;emit_graph_tiers.py, hop table in pipeline.md:385-393#41;
    Note over WALK: INVARIANT: parents visited in DECLARED ORDER as a tuple — makes<br/>category assignment reproducible across runs — the NGG1 binary tiers<br/>are byte-compared in CI and a set-ordered walk would break that
```

## KG-03.5 Bridging — multiple inheritance the 24-byte NGG1 record cannot hold

```mermaid
flowchart LR
    MP["MULTI_PARENT info #40;validate.py:204#41;<br/>1,401 classes declare #62;1 subClassOf"]
    STATS["stats.json bridging<br/>#123;multiParent:1401, crossCategory:454, crossDomain:153#125;"]
    BRIDGES["bridges.json — 542 entries<br/>#123;iri,label,categories#91;#93;,domains#91;#93;,parents#91;#93;#125;"]
    OVW["overview.json edges — 124 total<br/>34 backbone #40;type 0#41; + 90 weighted<br/>category↔category bridges #40;type 1#41;"]
    NODE["NGG1 node record: ONE u16 category field<br/>#40;24-byte record, offset 14#41; — keeps only the NEAREST"]
    MP --> STATS --> BRIDGES
    BRIDGES --> OVW
    BRIDGES -.->|"full membership recoverable ONLY here"| NODE
    note1["INVARIANT: a consumer that treats the binary category field as complete<br/>membership is wrong about all 542 bridging classes — join bridges.json<br/>by IRI to recover full membership (BASELINE-narrativegoldmine.md:220-222)"]
    note2["FLAG_BRIDGE #40;0x08#41; in the node record is set by a resolved bridges_to<br/>RELATION, a different fact from bridges.json multi-parenting —<br/>3,235 nodes carry it (pipeline.md:447-449)"]
```
