---
id: VW-02
title: OWL parse → graph model
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
adrs: []
sources:
  - ../vowl-wasm/src/ontology/mod.rs
  - ../vowl-wasm/src/ontology/parser.rs
  - ../vowl-wasm/src/ontology/model.rs
  - ../vowl-wasm/src/ontology/owl2_validator.rs
  - ../vowl-wasm/src/ontology/loader.rs
  - ../vowl-wasm/src/ontology/markdown_parser.rs
  - ../vowl-wasm/src/graph/mod.rs
  - ../vowl-wasm/src/graph/builder.rs
  - ../vowl-wasm/src/graph/node.rs
  - ../vowl-wasm/src/graph/edge.rs
  - ../vowl-wasm/src/graph/statistics.rs
  - ../vowl-wasm/src/graph/pinning.rs
  - ../vowl-wasm/src/bindings/mod.rs
verified_commit: 65e2d1e78
---

## VW-02.1 `loadOntology()` — JSON string to `VowlGraph`
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS: vowl.loadOntology(json)
    participant WV as WebVowl::load_ontology<br/>src/bindings/mod.rs:57
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
    WV->>WV: store metadata and graph on self<br/>src/bindings/mod.rs:71-72
```
- Every stage's `VowlError` is mapped to `JsValue::from_str` before returning to JS (src/bindings/mod.rs:59,63,67) — see VW-01.4.

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

## VW-02.3 `GraphBuilder::from_ontology` — class/property to node/edge
```mermaid
flowchart LR
    OD["OntologyData { classes, properties }"] --> LOOP1["for class in data.classes<br/>src/graph/builder.rs:24"]
    LOOP1 --> NB["NodeBuilder::new(id).label().node_type()<br/>.iri().external().ontology_meta()<br/>src/graph/builder.rs:25-31"]
    NB --> ADDN["graph.add_node(node)<br/>src/graph/builder.rs:32"]
    OD --> LOOP2["for property in data.properties<br/>src/graph/builder.rs:38"]
    LOOP2 --> EB["EdgeBuilder::new(id).label().edge_type()<br/>src/graph/builder.rs:39-41"]
    EB --> CHAR["functional/transitive/symmetric/<br/>inverse_functional/cardinality<br/>src/graph/builder.rs:43-55"]
    CHAR --> ADDE["graph.add_edge(source, target, edge)"]
    NB -.->|"map_node_type()"| NT["NodeType: Class · Datatype ·<br/>Special(String) · SetOperator(SetOperator)<br/>src/graph/mod.rs:85"]
    EB -.->|"map_edge_type()"| ET["EdgeType: ObjectProperty ·<br/>DatatypeProperty · SubClass · Special(String)<br/>src/graph/mod.rs:176"]
```

## VW-02.4 `VowlGraph` node/edge model
```mermaid
classDiagram
    class VowlGraph {
        -DiGraph~Node,Edge~ graph
        -HashMap~String,NodeIndex~ node_map
        -GraphMetadata metadata
        src/graph/mod.rs:24
    }
    class Node {
        +String id
        +String label
        +NodeType node_type
        +VisualAttributes visual
        +SemanticAttributes semantic
        src/graph/mod.rs:51
    }
    class VisualAttributes {
        +f64 x
        +f64 y
        +f64 vx
        +f64 vy
        src/graph/mod.rs:101
    }
    class SemanticAttributes {
        +String iri
        +bool external
        +Vec~String~ equivalent
        +Option~usize~ individuals
        src/graph/mod.rs:129
    }
    class Edge {
        +String id
        +EdgeType edge_type
        +EdgeCharacteristics characteristics
        src/graph/edge.rs
    }
    class EdgeCharacteristics {
        +bool functional
        +bool inverse_functional
        +bool transitive
        +bool symmetric
        src/graph/mod.rs:226
    }
    VowlGraph --> Node
    VowlGraph --> Edge
    Node --> VisualAttributes
    Node --> SemanticAttributes
    Edge --> EdgeCharacteristics
```

## VW-02.5 Markdown ontology parse (feature `markdown-ontology`)
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS: parseMarkdownOntology(md)
    participant WV as WebVowl::parse_markdown_ontology<br/>src/bindings/mod.rs:101
    participant MP as MarkdownParser::parse<br/>src/ontology/markdown_parser.rs:56
    MP->>MP: extract_ontology_block(markdown)
    MP->>MP: extract_all_properties(block_content)
    MP->>MP: get_required_property("term-id" / "preferred-term" / "owl:class")
    MP->>MP: parse_owl_class, then build_full_iri
    MP->>MP: separate_properties → core vs extension
    MP->>MP: extract_owl_axioms(block_content)
    MP-->>WV: OntologyBlock<br/>src/ontology/markdown_parser.rs:89-97
    WV->>WV: MarkdownOntologyData::from_block(&block)
    WV-->>JS: serde_wasm_bindgen JsValue
```
- The always-compiled `StandardParser` reads WebVOWL-shaped JSON (`class`/`property` arrays); the feature-gated `MarkdownParser` reads a distinct DreamLab markdown `### OntologyBlock` shape — two independent ontology-ingest paths into the same `OntologyData`/`OntologyBlock` model family (src/ontology/mod.rs:6-14, src/ontology/markdown_parser.rs).
- `OntologyLoader` (src/ontology/loader.rs:132, feature `markdown-ontology`) is the filesystem batch driver over `MarkdownParser::parse`: `load_file`/`load_directory`/`load_files` (src/ontology/loader.rs:169-245) plus a term index and domain grouping (src/ontology/loader.rs:296-314); it is native-only (no `#[wasm_bindgen]` surface).

## VW-02.6 `OWL2Validator` — DL compliance and antipattern gates
```mermaid
flowchart TB
    VB["validate_block(block)<br/>src/ontology/owl2_validator.rs:88"] --> IRI["validate_iri_format(iri)<br/>src/ontology/owl2_validator.rs:134"]
    VB --> UNIQ["check_iri_uniqueness(iri)<br/>src/ontology/owl2_validator.rs:192"]
    VB --> NS["validate_namespace(domain, iri)<br/>src/ontology/owl2_validator.rs:210"]
    VB --> DL["check_owl2_dl_compliance(block)<br/>src/ontology/owl2_validator.rs:262"]
    VB --> AP["detect_antipatterns(block)<br/>src/ontology/owl2_validator.rs:341"]
    IRI -->|error| RES["ValidationResult { errors, warnings }<br/>src/ontology/owl2_validator.rs:15"]
    UNIQ -->|error| RES
    NS -->|error| RES
    DL -->|warning| RES
    AP -->|warning| RES
```
- INVARIANT: IRI-format and uniqueness failures are hard `errors`; DL-compliance and antipattern findings are `warnings` only (src/ontology/owl2_validator.rs:37-43) — `validate_block` never rejects a block outright.
- `validate_blocks` (src/ontology/owl2_validator.rs:441) folds `validate_block` over a batch via `ValidationResult::merge` (src/ontology/owl2_validator.rs:48).

## VW-02.7 `GraphStatistics::calculate` — the six sub-metrics
```mermaid
flowchart LR
    G["VowlGraph"] --> CALC["GraphStatistics::calculate(graph)<br/>src/graph/statistics.rs:198"]
    CALC --> BM["calculate_basic_metrics<br/>src/graph/statistics.rs:216<br/>node/edge/class/datatype counts, density"]
    CALC --> DS["calculate_degree_statistics<br/>DegreeStatistics/DegreeInfo"]
    CALC --> CA["calculate_component_analysis<br/>ComponentAnalysis"]
    CALC --> O2["calculate_owl2_metrics<br/>Owl2Metrics"]
    CALC --> PD["calculate_property_distribution<br/>PropertyDistribution"]
    CALC --> CD["calculate_class_distribution<br/>ClassDistribution"]
    BM --> OUT["GraphStatistics { basic, degree,<br/>components, owl2, properties, classes }<br/>src/graph/statistics.rs:13"]
    DS --> OUT
    CA --> OUT
    O2 --> OUT
    PD --> OUT
    CD --> OUT
    OUT -->|"getStatistics()"| JS["JS: serde_wasm_bindgen JsValue<br/>src/bindings/mod.rs:440"]
```
- `density` for a graph with `node_count <= 1` is defined as `0.0` (src/graph/statistics.rs:239-243), avoiding a divide-by-zero on `max_edges`.

## VW-02.8 `PinManager` — node pin lifecycle
```mermaid
stateDiagram-v2
    [*] --> Unpinned
    Unpinned --> Pinned: pin_node(graph, id)<br/>src/graph/pinning.rs:42<br/>fails if !enabled
    Unpinned --> Pinned: pin_node_at(graph, id, x, y)<br/>src/graph/pinning.rs:82
    Pinned --> Unpinned: unpin_node(graph, id)<br/>src/graph/pinning.rs:126
    Pinned --> Unpinned: reset(graph)<br/>src/graph/pinning.rs:156 clears all
    Pinned --> Pinned: is_pinned(id) query<br/>src/graph/pinning.rs:151
    state PinManager {
        enabled: bool
        pinned_nodes: HashSet~String~
    }
```
- `pin_node` returns `VowlError::InteractionError("Pinning is disabled")` when `self.enabled == false` (src/graph/pinning.rs:43-46), independent of graph state.
- A pinned node's `visual.fixed = true` is what `ForceSimulation::apply_forces` reads to skip velocity/position integration (see VW-03.1).
