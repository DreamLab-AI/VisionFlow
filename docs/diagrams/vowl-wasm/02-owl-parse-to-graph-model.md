---
id: VW-02
title: OWL parse → graph model
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
adrs: [vowl-wasm:ADR-001]
sources:
  - ../vowl-wasm/src/ontology/mod.rs
  - ../vowl-wasm/src/ontology/parser.rs
  - ../vowl-wasm/src/ontology/model.rs
  - ../vowl-wasm/src/graph/mod.rs
  - ../vowl-wasm/src/graph/builder.rs
  - ../vowl-wasm/src/graph/node.rs
  - ../vowl-wasm/src/graph/edge.rs
  - ../vowl-wasm/src/graph/statistics.rs
  - ../vowl-wasm/src/graph/pinning.rs
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
  - ../vowl-wasm/README.md
verified_commit: 2aff199e2de06f4bba512301acf7023477c492ac
---

## VW-02.1 `loadOntology()` — JSON string to `VowlGraph`
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS: vowl.loadOntology(json)
    participant WV as WebVowl::load_ontology<br/>src/bindings/mod.rs:56
    participant SP as StandardParser<br/>src/ontology/parser.rs:38
    participant OV as StandardParser::validate<br/>src/ontology/parser.rs:485 impl OntologyParser
    participant GB as GraphBuilder::from_ontology<br/>src/graph/builder.rs:20
    participant G as VowlGraph
    JS->>WV: loadOntology(json)
    WV->>SP: parser.parse(json)
    SP-->>WV: OntologyData
    WV->>OV: parser.validate(&ontology_data)
    OV-->>WV: Result<()>
    WV->>GB: GraphBuilder::from_ontology(&ontology_data)
    GB->>G: add_node() per class, EdgeBuilder per property
    GB-->>WV: VowlGraph
    WV->>WV: store metadata and graph on self<br/>src/bindings/mod.rs:69-72
```
- Every stage's `VowlError` is mapped to `JsValue::from_str` before returning to JS (src/bindings/mod.rs:58,62,66) — see VW-01.6. Validation checks domains but permits absent datatype ranges; the builder still requires both endpoints (src/ontology/parser.rs:485-507, src/graph/mod.rs:271-285).

## VW-02.2 `StandardParser` — classes, properties, metadata
```mermaid
flowchart TB
    P["parse(json)<br/>impl OntologyParser src/ontology/parser.rs:468"] --> PC["parse_classes(json)<br/>src/ontology/parser.rs:53"]
    P --> PP["parse_properties(json)"]
    P --> PM["parse metadata + namespaces"]
    PC --> PCN["parse_class_node(json)<br/>src/ontology/parser.rs:73"]
    PCN --> ID["id (required)<br/>src/ontology/parser.rs:75-78"]
    PCN --> IRI["iri (defaults to id)<br/>src/ontology/parser.rs:81-85"]
    PCN --> LABEL["label (defaults to id;<br/>ParserConfig.allow_empty_labels gate)<br/>src/ontology/parser.rs:87-93"]
    PCN --> CT["determine_class_type(json)"]
    PCN --> EQ["equivalent[] (optional)<br/>src/ontology/parser.rs:104-112"]
    PCN --> ATTR["parse_class_attributes(json)"]
    PCN --> OM["parse_ontology_metadata(json)"]
    PCN --> CN["ClassNode { id, iri, label, class_type,<br/>equivalent, attributes, ontology_meta }<br/>src/ontology/parser.rs:114"]
```
- `ParserConfig` (src/ontology/parser.rs:17) has a `max_classes` cutoff enforced at src/ontology/parser.rs:62-64: a `class` array longer than the configured cap is silently truncated, not rejected.
- `StandardParser` is the JSON ingest path: it reads an inline subset of WebVOWL-shaped JSON (`class`/`property` arrays) into `OntologyData` (src/ontology/mod.rs:6-7, src/ontology/mod.rs:24). The second path that existed at the earlier stamp, a feature-gated `MarkdownParser` over DreamLab `### OntologyBlock` markdown with an `OntologyLoader` filesystem driver and an `OWL2Validator`, was deleted together with the `markdown-ontology` feature and the `Domain` and `OntologyBlock` types (docs/adr/ADR-001-shared-ontology-rendering-contract.md:36, README.md:204-205). Parsing Markdown vaults is left to `ontomark-parse` (README.md:205).
- Root `classAttribute`/`propertyAttribute` tables are not joined, labels must be strings, and `validate_iris` has no implementation (src/ontology/parser.rs:17-25, src/ontology/parser.rs:53-114, src/ontology/parser.rs:485-507). Class metadata now accepts caller-supplied domain strings, or none, and the renderer holds no domain registry (README.md:210-212).

## VW-02.3 `GraphBuilder::from_ontology` — class/property to node/edge
```mermaid
flowchart LR
    OD["OntologyData { classes, properties }"] --> LOOP1["for class in data.classes<br/>src/graph/builder.rs:24"]
    LOOP1 --> NB["NodeBuilder::new(id).label().node_type()<br/>.iri().external().ontology_meta()<br/>src/graph/builder.rs:25-31"]
    NB --> ADDN["graph.add_node(node)<br/>src/graph/builder.rs:32"]
    OD --> LOOP2["for property in data.properties<br/>src/graph/builder.rs:38"]
    LOOP2 --> EB["EdgeBuilder::new(id).label().edge_type()<br/>src/graph/builder.rs:39-41"]
    EB --> CHAR["functional/transitive/symmetric/<br/>inverse_functional/cardinality<br/>src/graph/builder.rs:43-55"]
    CHAR --> ADDE["graph.add_edge(source, target, edge)<br/>src/graph/builder.rs:73-75"]
    ADDE --> INV{"inverseOf supplied?<br/>src/graph/builder.rs:78"}
    INV -->|yes| EXTRA["add synthetic reversed edge using inverse ID;<br/>explicit inverse records are not deduplicated<br/>src/graph/builder.rs:80-89"]
    NB -.->|"map_node_type()"| NT["NodeType: Class · Datatype ·<br/>Special(String) · SetOperator(SetOperator)<br/>src/graph/mod.rs:85"]
    EB -.->|"map_edge_type()"| ET["EdgeType: ObjectProperty ·<br/>DatatypeProperty · SubClass · Special(String)<br/>src/graph/mod.rs:176"]
```

- **Semantic loss:** parsed class equivalences/individuals are not copied to nodes (src/graph/builder.rs:24-31); exact cardinality is discarded when only min/max reach the edge (src/graph/builder.rs:59-60). Bounds without a `cardinality` key are ignored and u64 values narrow to u32 unchecked (src/ontology/parser.rs:389-413). `rdfs:subClassOf` misses the case-sensitive substring mapping (src/graph/builder.rs:121-128).

## VW-02.4 `VowlGraph` node/edge model
```mermaid
flowchart TB
    G["VowlGraph: directed graph, ID map, metadata<br/>src/graph/mod.rs:24"] --> N["Node: id, label, type, visual, semantic<br/>src/graph/mod.rs:51"]
    G --> E["Edge: id, type, characteristics<br/>src/graph/edge.rs"]
    N --> V["VisualAttributes: x, y, vx, vy<br/>src/graph/mod.rs:101"]
    N --> S["SemanticAttributes: iri, external,<br/>equivalent IDs, optional individual count<br/>src/graph/mod.rs:129"]
    E --> C["EdgeCharacteristics: functional,<br/>inverse_functional, transitive, symmetric<br/>src/graph/mod.rs:226"]
```
- `GraphStatistics.calculate` density is defined as `0.0` when `node_count <= 1` (src/graph/statistics.rs:240-243), avoiding a divide-by-zero on `max_edges`. Parallel/synthetic edges can still make density exceed one. Component DFS follows outgoing neighbours, so component and isolation counts depend on node insertion order (src/graph/statistics.rs:371-399, src/graph/mod.rs:326-339).

## VW-02.8 `PinManager` — node pin lifecycle
```mermaid
flowchart TB
    M["PinManager owns enabled flag and ID set"] --> P["pin_node(graph, id)<br/>fails when disabled<br/>src/graph/pinning.rs:42"]
    M --> A["pin_node_at(graph, id, x, y)<br/>src/graph/pinning.rs:82"]
    P --> SET["Pinned ID; node visual.fixed = true"]
    A --> SET
    SET --> U["unpin_node removes ID and clears fixed<br/>src/graph/pinning.rs:126"]
    SET --> R["reset clears all managed pins<br/>src/graph/pinning.rs:156"]
    SET --> Q["is_pinned reads manager ID set<br/>src/graph/pinning.rs:151"]
```
- `pin_node` returns `VowlError::InteractionError("Pinning is disabled")` when `self.enabled == false` (src/graph/pinning.rs:43-46), independent of graph state.
- Pin IDs are separate manager state: graph replacement does not reset them, and `pin_node_at` accepts non-finite coordinates (src/bindings/mod.rs:56-72, src/graph/pinning.rs:82-105). A pinned node's `visual.fixed = true` is what `ForceSimulation::apply_forces` reads to skip velocity/position integration (see VW-03.1).
