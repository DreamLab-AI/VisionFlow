---
id: OM-04
title: Artefact emitters — Turtle, reasoner, indexes, NGG1 tiers, WebVOWL, OKF, against contract C3
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-004, ADR-2124, ADR-2128]
sources:
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/turtle.rs
  - ../ontomark/crates/ontomark-compile/src/build/indexes.rs
  - ../ontomark/crates/ontomark-compile/src/build/page_api.rs
  - ../ontomark/crates/ontomark-compile/src/build/graph_tiers.rs
  - ../ontomark/crates/ontomark-compile/src/build/ngg1.rs
  - ../ontomark/crates/ontomark-compile/src/build/webvowl.rs
  - ../ontomark/crates/ontomark-compile/src/build/okf.rs
  - ../ontomark/crates/ontomark-compile/src/reason.rs
  - ../ontomark/crates/ontomark-compile/src/whelk.rs
  - ../ontomark/crates/ontomark-compile/src/ir/model.rs
  - ../ontomark/crates/ontomark-compile/src/ir/closure.rs
  - ../ontomark/crates/ontomark-compile/src/ir/projection.rs
  - ../ontomark/crates/ontomark-compile/src/check/validate.rs
  - ../ontomark/crates/ontomark-ngg1/src/lib.rs
  - ../ontomark/crates/ontomark-cli/src/main.rs
verified_commit: ffe06e45bfebd388a1d85849b24a8810e4d0d610
---

## OM-04.1 Emitter by emitter — corpus selector, output, C3 artefact

```mermaid
flowchart LR
    PROJ["Corpus handed to every emitter is already public-only<br/>records = public.iter clean_record<br/>projection.rs:426<br/>both projection paths, mod.rs:365-370"]
    TTL["turtle::serialise of build_graph_attributed<br/>selector: records filtered by public_only AND has_ontology<br/>turtle.rs:231-234<br/>call mod.rs:379,416"]
    INF["reason::inferred_turtle<br/>selector: none, takes the Reasoning not the corpus<br/>reason.rs:685<br/>call mod.rs:417-420"]
    SCA["scaffold_index<br/>selector: public_classes<br/>indexes.rs:102<br/>call mod.rs:422-423"]
    PRO["prose_index<br/>selector: public<br/>indexes.rs:255<br/>call mod.rs:424-425"]
    SRCH["search_index<br/>selector: public<br/>indexes.rs:310<br/>call mod.rs:427-429"]
    PG["page_api::build_attributed + domain index<br/>selector: public<br/>page_api.rs:77<br/>call mod.rs:431-440"]
    CEN["census + validation-report<br/>selector: none, inputs are census and report<br/>mod.rs:441-445"]
    CTX["okf::context to three paths<br/>selector: none, vocabulary only<br/>okf.rs:31<br/>mod.rs:451-455"]
    VOWL["webvowl::build<br/>selector: records filtered r.public AND has_ontology<br/>webvowl.rs:60-63<br/>call mod.rs:458-461"]
    TIER["graph_tiers::emit<br/>selector: records filtered r.public AND has_ontology<br/>graph_tiers.rs:195-198<br/>call mod.rs:465-468"]
    MIR["markdown mirror, only with the flag<br/>selector: public<br/>mod.rs:471,487,497"]
    OKF["okf::bundle<br/>selector: public_classes<br/>okf.rs:74<br/>call mod.rs:571-573"]
    GEN["two .generation.json markers<br/>mod.rs:607-608"]
    PROJ --> TTL & SCA & PRO & SRCH & PG & VOWL & TIER & MIR & OKF
    TTL --> C1["C3 data/ontology.ttl<br/>mod.rs:416"]
    INF --> C2["C3 data/ontology-inferred.ttl<br/>mod.rs:418"]
    SCA --> C3["C3 data/scaffold-index.json<br/>mod.rs:423"]
    PRO --> C4["C3 data/prose-index.json<br/>mod.rs:425"]
    SRCH --> C5["C3 api/search-index.json<br/>mod.rs:429"]
    PG --> C6["C3 api/pages/SLUG.json<br/>mod.rs:435<br/>plus api/pages/_domain-index.json, not in C3<br/>mod.rs:440"]
    CEN --> C7["C3 api/census.json, api/validation-report.json<br/>mod.rs:441-443"]
    CTX --> C8["C3 ns/v2.jsonld, api/schema/context.jsonld, context/v1.jsonld<br/>mod.rs:453-455"]
    VOWL --> C9["C3 data/ontology.json<br/>mod.rs:459"]
    TIER --> C10["C3 data/graph/ overview.json, domain-*.bin, full.bin, stats.json, bridges.json<br/>mod.rs:467"]
    MIR --> C11["C3 api/markdown/, off by default<br/>mod.rs:546"]
    OKF --> C12["C3 okf/ index.md and concept files<br/>mod.rs:572"]
    GEN --> C13["C3 .generation.json, plus data/.generation.json not in C3<br/>mod.rs:607-608"]
    MISS["C3 data/ontology-corpus.records.jsonl, --with-rvdb<br/>has no emitter: absent from the staged list<br/>mod.rs:416-608"]
```
- **DIVERGENCE:** C3 lists `ontology-corpus.records.jsonl` (only `--with-rvdb`); `run` stages no such file and no `rvdb` option exists in `build/mod.rs:302-613`. Emitted but not in C3: `api/pages/_domain-index.json` (`mod.rs:440`), `context/v1.jsonld` (`mod.rs:453`), `data/.generation.json` (`mod.rs:607`).
- **INVARIANT:** the selector split is `public()` for per-page artefacts and `public_classes()`, which also requires `EntityType::Class` (`ir/model.rs:456-460`), for scaffold, OKF and `class_count`; Turtle, WebVOWL and NGG1 use the inline `public AND has_ontology AND iri` filter (`turtle.rs:234`, `webvowl.rs:63`, `graph_tiers.rs:198`), so they include individuals and the others do not. The corpus is already public-only, so each `r.public` test is a second filter (`turtle.rs:168-172`).
- **DIVERGENCE:** `class_count` counts `public_classes()` un-deduplicated (`mod.rs:405`), `page_count` is `records.len()` (`mod.rs:406`), scaffold `counts.classes` keeps the first record per class slug (`indexes.rs:110-111`) and `stats.json` `pages` counts distinct page ids (`graph_tiers.rs:201-210`); they can differ only if two records share a slug, which `DUPLICATE_IRI` refuses (`check/validate.rs:561`) and `Staged` write-once refuses for OKF paths (`mod.rs:164-167`), so the gap is theoretical.

## OM-04.2 Turtle graph build — one function, header last, digest feeds the version IRI

```mermaid
flowchart TD
    CALL["run calls build_graph_attributed public_only=true<br/>mod.rs:379"]
    DECL["Declare skos:Concept, annotation props, object props, maturity<br/>turtle.rs:204-228"]
    SEL["records = corpus.records filtered<br/>public_only or r.public AND has_ontology AND iri not empty<br/>turtle.rs:231-234"]
    DECLSET["declared = set of every selected iri_to_uri<br/>turtle.rs:237-240"]
    CLS["Per record: owl:Class or NamedIndividual, label, comment,<br/>sourceDomain, qualityScore, hasMaturity, slug<br/>turtle.rs:260-335"]
    SUBC["rdfs:subClassOf per parent, plus skos:broader when the parent is a taxonomy node<br/>turtle.rs:272-284"]
    REL["One property per relation key, every target kept, declared or not<br/>turtle.rs:337-342"]
    RESTR["Existential restrictions, only both endpoints declared<br/>turtle.rs:358-385"]
    DISJ["owl:disjointWith, only when vocabulary registers the key<br/>turtle.rs:393-416"]
    DEFS["owl:equivalentClass from defines-as, all or nothing<br/>turtle.rs:426-442"]
    STUB["Hardcoded 13-name object-property list: undeclared ngm targets become skos:Concept stubs<br/>turtle.rs:446-483"]
    HDR["Header added last: versionIRI = ontology_iri / ontology_digest of every non-header triple<br/>turtle.rs:487-508"]
    DIG["ontology_digest: sorted N-Triples lines, header subject skipped, sha256_12<br/>turtle.rs:611-641"]
    SER["serialise: prefixes sorted, subjects in graph order<br/>turtle.rs:933-981"]
    GENM["Same digest recomputed for the marker<br/>mod.rs:590"]
    CALL --> DECL --> SEL --> DECLSET --> CLS --> SUBC --> REL --> RESTR --> DISJ --> DEFS --> STUB --> HDR --> SER
    HDR -.-> DIG
    SER --> GENM
```
- **INVARIANT:** the header is built last so `owl:versionIRI` addresses everything above it and none of it addresses itself (`turtle.rs:485-488`); `run` recomputes `ontology_digest` on the finished graph (`mod.rs:590`), the same value because the header subject is skipped (`turtle.rs:614-616`), a duplicated computation rather than a shared one.
- **DIVERGENCE:** the stub list is a hardcoded array of 13 property names (`turtle.rs:446-460`), a third copy of the relation set beside the vocabulary-driven `declare_object_properties` list (`turtle.rs:678-691`); a relation the vocabulary adds is declared but never stubbed.
- **DIVERGENCE:** `build_graph` with `public_only=false` is called by the CLI `gate` command on the raw, unprojected corpus (`ontomark-cli/src/main.rs:341`), so that graph holds private classes; it is a gate-only graph and is never serialised.

## OM-04.3 Reasoner — mode decision, transitive cap, consistency gate

```mermaid
flowchart TD
    ENTRY["reason graph, classifier<br/>reason.rs:554"]
    CAPD["reason_capped with RELEVANT_TRANSITIVE_CAP = 2000<br/>reason.rs:473,555-556"]
    EXTR["extract: classes, sub-properties, transitive, existentials, definitions<br/>reason.rs:283"]
    NODEF{"axioms.definitions empty?<br/>reason.rs:570"}
    SKIP["Skip: no existentials given to the reasoner<br/>reason.rs:270"]
    KEEP["Relevant: keep existentials over a sub or super property of a definition property<br/>reason.rs:272-278,573"]
    OVER{"kept over transitive properties > cap?<br/>reason.rs:574-578"}
    ERR["Err RelevantOverCap, reported under code WHELK_RELEVANT_CAP<br/>reason.rs:477,507,580"]
    CLASSIFY["classifier.refusals, then named_subsumptions<br/>reason.rs:628-632"]
    SPLIT["Drop asserted pairs, Thing, Nothing, reflexive: remainder is inferred<br/>reason.rs:647-651"]
    ROOTS["root_causes = unsatisfiable minus those under an unsatisfiable asserted parent<br/>reason.rs:653-663"]
    GATE["ensure is_consistent else build fails, nothing written<br/>mod.rs:395-399"]
    CONS["is_consistent = no unsatisfiable AND no refusals<br/>reason.rs:78-80"]
    INFTTL["inferred_turtle: one rdfs:subClassOf per inferred pair, inferenceMethod whelk-el++-closure<br/>reason.rs:685-711"]
    WH["WhelkEl implements Classifier over ontomark-reason, refusals plus subsumptions<br/>whelk.rs:98-127"]
    ENTRY --> CAPD --> EXTR --> NODEF
    NODEF -- yes --> SKIP
    NODEF -- no --> KEEP --> OVER
    OVER -- yes --> ERR
    OVER -- no --> CLASSIFY
    SKIP --> CLASSIFY
    CLASSIFY --> SPLIT --> ROOTS --> CONS --> GATE
    GATE --> INFTTL
    CLASSIFY -.-> WH
```
- **INVARIANT:** the mode is derived from the graph (`Relevant` iff a class has a definition, else `Skip`), never configured; `Include` is a test-only mode that does not finish on the full corpus (`reason.rs:143-145,546`).
- **INVARIANT:** `WHELK_RELEVANT_CAP` is an error-code string only (`reason.rs:477`); the cap is the constant 2000 and `vault.toml` has no reasoner section (`reason.rs:473`).
- **DIVERGENCE:** `reason_with` (`reason.rs:604`) and `Restrictions::Include` (`reason.rs:145`) have no caller outside `reason.rs` tests; `reason` is called by `run` (`mod.rs:383`) and by the migrate compile path.

## OM-04.4 NGG1 tiers — where relation names collapse and where ids and positions are baked

```mermaid
flowchart TD
    EMIT["graph_tiers::emit<br/>graph_tiers.rs:720"]
    SEL["public = records where r.public AND has_ontology AND iri not empty<br/>graph_tiers.rs:195-198"]
    NODE["Node: domain_id = resolve_domain with domain_aliases, category_id by BFS ancestry<br/>x and y initialised 0.0<br/>graph_tiers.rs:233-266"]
    DOMID["resolve_domain folds aliases then domain_index, else DOMAIN_NONE<br/>graph_tiers.rs:81-90"]
    CATID["CategoryResolver walks sub_class_of and instance_of up to depth 12<br/>graph_tiers.rs:50,135-186"]
    BACKB["Backbone edge: subClassOf, or instance-of for individuals, type EDGE_SUBCLASS<br/>graph_tiers.rs:290-301"]
    REL["Every one of the 12 relation keys becomes the SAME type byte EDGE_RELATION<br/>graph_tiers.rs:307-316"]
    DEDUP["edge_set keyed src, tgt, type: two different relations between one pair collapse to one edge<br/>graph_tiers.rs:283,313"]
    BRIDGE["bridges-to is only a node flag FLAG_BRIDGE, its name is gone<br/>graph_tiers.rs:318-320,332"]
    CONST["EDGE_SUBCLASS = 0, EDGE_RELATION = 1<br/>ontomark-ngg1/src/lib.rs:142,144"]
    UNDECL["Edge to a target that is not a selected node is silently dropped<br/>graph_tiers.rs:297-298,309-310"]
    FULL["full.bin: bake_positions then build_tier uncapped<br/>graph_tiers.rs:737-738"]
    DOM["domain-SLUG.bin: filter domain_id, cap 1500 nodes, bake_positions, relations top-8<br/>graph_tiers.rs:45,47,751-776"]
    BAKE["bake_positions: domain ring 900, category ring 240, leaf ring 95<br/>ngg1.rs:302-312"]
    PACK["pack narrows x,y to f32 and writes domain, category, flags, degree<br/>ngg1.rs:76-88"]
    OVR["overview.json: force_layout seed 42 over domain and category roots<br/>graph_tiers.rs:594"]
    ORDER["order_source_edges: backbone whole, relations by target degree then truncate topk<br/>graph_tiers.rs:420-451"]
    STATS["stats.json and bridges.json: multi-parent classes, since the node record holds one category<br/>graph_tiers.rs:355-380,825-829"]
    EMIT --> SEL --> NODE
    NODE --> DOMID
    NODE --> CATID
    SEL --> BACKB --> REL --> DEDUP
    REL --> BRIDGE
    REL --> UNDECL
    DEDUP --> CONST
    NODE --> FULL --> BAKE
    NODE --> DOM --> BAKE
    DOM --> ORDER
    BAKE --> PACK
    EMIT --> OVR
    EMIT --> STATS
```
- **DIVERGENCE:** the force-projection gap (ADR-OM-005): NGG1 keeps one bit of relation information (backbone vs relation, `graph_tiers.rs:299,313`) where `ontology.ttl` keeps a named property per relation (`turtle.rs:337-341`), existential restrictions (`turtle.rs:358-383`), `owl:disjointWith` (`turtle.rs:409-413`), `owl:equivalentClass` (`turtle.rs:436-440`), `skos:broader` (`turtle.rs:278-282`), maturity and quality (`turtle.rs:312-334`) and `skos:Concept` stubs for targets outside the graph (`turtle.rs:475-483`).
- **DIVERGENCE:** `declared_backbone` and `declared_relations` count edges the tiers then discard when the target is not a selected node (`graph_tiers.rs:285-286,297,309`), so `stats.json` reports declared against shipped.
- **DIVERGENCE:** the module note explains the single `u16` category limit (`graph_tiers.rs:22-25`); relation identity is lost by a separate, deliberate collapse that no comment explains (`graph_tiers.rs:313-315`).

## OM-04.5 Scaffold, prose and search indexes, backlinks and the domain index

```mermaid
flowchart TD
    BL["backlink_index: slug_set and loop over ALL records, not public<br/>indexes.rs:51-55"]
    BLSLUG["target slug recomputed: ref_slug of iri if it has a colon, else slugify of label<br/>indexes.rs:57-60"]
    CLO["closure::compute: loop over ALL records with ontology, class slug first wins<br/>closure.rs:99-106<br/>call mod.rs:372"]
    SC["scaffold_index: public_classes sorted by class slug, first definition wins<br/>indexes.rs:102-110"]
    SCREL["rel lists in SCAFFOLD_REL_ORDER, empty lists omitted<br/>indexes.rs:30,115-127"]
    SCTITLE["title = label, else record title<br/>indexes.rs:129"]
    SCSUP["sup and isup read from the structural closure, bl capped at 20<br/>indexes.rs:23,145-160"]
    PR["prose_index: public, has_ontology, dfull over 400 chars, cl capped at 1500<br/>indexes.rs:21,25,255-275"]
    SE["search_index: public, domain_name by title_case of the domain, wikilinks first 20<br/>indexes.rs:308-350"]
    PGS["page_api: backlinks looked up by page slug, label from slug_to_title<br/>page_api.rs:78,175-190"]
    PGD["Domain index: groups by domain, empty domain bucket named unclassified<br/>page_api.rs:147-161"]
    PGREL["relationships in PAGE_API_REL_ORDER, every key present even when empty<br/>page_api.rs:18-31,111-119"]
    BL --> BLSLUG
    BL --> SCSUP
    BL --> PGS
    CLO --> SCSUP
    CLO --> PGREL
    SC --> SCREL --> SCTITLE --> SCSUP
    PR -. same cap constant .-> SC
    PGS --> PGD
```
- **DIVERGENCE:** `backlink_index` (`indexes.rs:53-55`) and `closure::compute` (`closure.rs:99`) iterate `records`, while the artefacts consuming them iterate `public()` or `public_classes()`; safe today only because the corpus is pre-filtered (`projection.rs:426`), unenforced here.
- **DIVERGENCE:** three emitters compute the same domain and label differently: scaffold label falls back to title (`indexes.rs:129`), closure to title then slug (`closure.rs:109-113`), NGG1 to the slug (`graph_tiers.rs:254-256`); the domain is a title-cased name in search (`indexes.rs:328`), `unclassified` in the page API (`page_api.rs:148`), alias-folded in NGG1 (`graph_tiers.rs:81-90`) and exact-match in WebVOWL (`webvowl.rs:91`).
- **DIVERGENCE:** `SCAFFOLD_REL_ORDER` (`indexes.rs:30`), `PAGE_API_REL_ORDER` (`page_api.rs:18`), `RELATION_KEYS` (`ir/model.rs:36`) and WebVOWL `REL_MAP` (`webvowl.rs:30`) are four hand-kept orderings of the same twelve relations; `page_api::build` has no caller outside tests, `build_attributed` is the live one (`page_api.rs:55,71`).

## OM-04.6 WebVOWL graph and OKF bundle — second and third projections of the same records

```mermaid
flowchart TD
    VW["webvowl::build<br/>webvowl.rs:59"]
    VWSEL["public = records where r.public AND has_ontology AND iri not empty<br/>webvowl.rs:60-63"]
    VWDECL["declared = remapped IRIs of the selection, edges outside it not drawn<br/>webvowl.rs:66,118"]
    VWCOL["Colour: exact slug match on d.slug == record.domain, else DEFAULT_COLOUR<br/>webvowl.rs:25,88-93"]
    VWREL["Relations iterate the hardcoded REL_MAP, with related-to mapped to skos:related<br/>webvowl.rs:30-42,150"]
    VWPROP["One property node per edge, id prefix prop-rel, counter shared<br/>webvowl.rs:157-161"]
    VWHDR["Header: node count and distinct domain count<br/>webvowl.rs:177-197"]
    OK["okf::bundle<br/>okf.rs:68"]
    OKSEL["classes = public_classes sorted by class slug<br/>okf.rs:74-75"]
    OKIDX["index.md with generation id, vocabulary_version, concept_count<br/>okf.rs:84-90"]
    OKCON["concepts/SLUG.md per class: description via truncate_definition, is-a, 12 relation keys<br/>okf.rs:125-165"]
    OKCTX["okf::context: every vocabulary relation and scalar becomes a context term<br/>okf.rs:31-64"]
    SHARED["truncate_definition shared with the scaffold index<br/>indexes.rs:77-79"]
    VW --> VWSEL --> VWDECL --> VWCOL --> VWREL --> VWPROP --> VWHDR
    OK --> OKSEL --> OKIDX --> OKCON
    OKCON --> SHARED
    OKCTX -.-> OKCON
```
- **DIVERGENCE:** WebVOWL labels `related-to` as `skos:related` and `part-of` as `vc:isPartOf` from a hardcoded table (`webvowl.rs:30-42`), while Turtle takes each relation IRI from the vocabulary (`turtle.rs:656-673`); the two agree only while the vocabulary does.
- **INVARIANT:** `iri_to_uri` is deliberately not reused by WebVOWL, which has its own `remap_iri` that passes foreign IRIs through (`webvowl.rs:44-53`); NGG1 imports that function so tiers and WebVOWL share node ids (`graph_tiers.rs:39`).
- **DIVERGENCE:** the OKF concept file carries only `is-a` and relation IRIs plus scalars (`okf.rs:152-165`); it drops provenance, restrictions and disjointness, and `okf::bundle` has the single `public_classes` selector in this file set, the same as scaffold (`indexes.rs:102`).

## OM-04.7 Duplicated computation — what each emitter recomputes instead of sharing

```mermaid
flowchart LR
    SUBC["Subclass closure"]
    SUBC1["Structural BFS, byte-frozen ordering<br/>closure.rs:1-15"]
    SUBC2["Whelk EL++ closure<br/>reason.rs:554"]
    SUBC3["NGG1 ancestry walk for category only<br/>graph_tiers.rs:154-186"]
    SLUG["Slug of an IRI"]
    SLUG1["ref_slug via class_slug<br/>ir/model.rs:188-190"]
    SLUG2["slug_of local copy, splits on last colon or slash<br/>graph_tiers.rs:68-77"]
    SLUG3["slug_from_uri in Turtle<br/>turtle.rs:882"]
    BLK["Backlinks"]
    BLK1["backlink_index, shared by scaffold and page API<br/>indexes.rs:51<br/>mod.rs:373"]
    BLK2["Search wikilinks first 20 from record.links<br/>indexes.rs:346-352"]
    DOMN["Domain id"]
    DOMN1["alias-folded index, NGG1<br/>graph_tiers.rs:81-90"]
    DOMN2["raw string compared to taxonomy, WebVOWL<br/>webvowl.rs:91"]
    DOMN3["raw string, Turtle sourceDomain<br/>turtle.rs:302-304"]
    SUBC --> SUBC1 & SUBC2 & SUBC3
    SLUG --> SLUG1 & SLUG2 & SLUG3
    BLK --> BLK1 & BLK2
    DOMN --> DOMN1 & DOMN2 & DOMN3
```
- **DIVERGENCE:** the structural closure and the Whelk closure are intentionally two engines; a build test pins that they agree on the taxonomy (`build/mod.rs:1254-1265`), so only the third, NGG1's ancestry walk, is untested for agreement.
- **DIVERGENCE:** Turtle emits the domain verbatim while NGG1 folds aliases (`turtle.rs:302-304` against `graph_tiers.rs:81-90`): a class in an aliased domain lands in the canonical domain tier but carries the unfolded string in `ontology.ttl`.
- **DIVERGENCE:** callable only from tests (no non-test caller): `turtle::version_iri` (`turtle.rs:645`), `reason::reason_with` (`reason.rs:604`), `page_api::build` (`page_api.rs:55`).
