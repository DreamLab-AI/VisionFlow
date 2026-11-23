---
id: WV-05
title: Legacy WebVOWL - bundle, loading paths, parse and notation coverage
area: wasmvowl
governing:
  - ../WasmVOWL/CLAUDE.md
  - ../WasmVOWL/README.md
adrs: []
sources:
  - ../WasmVOWL/CLAUDE.md
  - ../WasmVOWL/README.md
  - ../WasmVOWL/legacy/webpack.config.js
  - ../WasmVOWL/legacy/src/index.html
  - ../WasmVOWL/legacy/src/webvowl/js/entry.js
  - ../WasmVOWL/legacy/src/app/js/entry.js
  - ../WasmVOWL/legacy/src/app/js/app.js
  - ../WasmVOWL/legacy/src/app/js/loadingModule.js
  - ../WasmVOWL/legacy/src/app/js/menu/exportMenu.js
  - ../WasmVOWL/legacy/src/webvowl/js/graph.js
  - ../WasmVOWL/legacy/src/webvowl/js/parser.js
  - ../WasmVOWL/legacy/src/webvowl/js/elements/nodes/nodeMap.js
  - ../WasmVOWL/legacy/src/webvowl/js/elements/properties/propertyMap.js
  - ../WasmVOWL/modern/src/types/graph.ts
  - ../WasmVOWL/modern/src/stores/useGraphStore.ts
  - ../WasmVOWL/modern/src/components/UI/TopMenuBar.tsx
verified_commit: {wasmvowl: af25086a605482c8234580ccd1062cf506e0b181}
---

## WV-05.1 Two bundles, two globals, one boot call

```mermaid
flowchart TB
    WP["webpack entries webvowl and webvowl.app into deploy/js<br/>legacy/webpack.config.js:8-16<br/>assigned to globals of the same name, legacy/webpack.config.js:17-18"] --> D3["d3 is an external global, copied in by Grunt<br/>legacy/webpack.config.js:30-36"]
    WP --> LIB["webvowl: graph, options, util, filter and switch modules, node and property maps<br/>legacy/src/webvowl/js/entry.js:7-46"]
    WP --> APPE["webvowl.app: toolstyle css, browser warning, app<br/>legacy/src/app/js/entry.js:1-4"]
    LIB --> BOOT["index.html loads d3, webvowl, webvowl.app, then onload initialize<br/>legacy/src/index.html:539-542"]
    APPE --> BOOT
    BOOT --> INIT["app.initialize: selection modules focuser, details, pickAndPin<br/>legacy/src/app/js/app.js:179-192"]
    INIT --> FM["filter modules in order: empty literal, statistics, degree, datatype,<br/>object property, subclass, disjoint, set operator, scaling, compact, externals<br/>legacy/src/app/js/app.js:194-205"]
    FM --> START["graph.start<br/>legacy/src/app/js/app.js:279"]
    N1["INVARIANT: legacy is archived, DO NOT MODIFY, CLAUDE.md:33<br/>kept for reference and comparison, WasmVOWL/README.md:17"]
    START --- N1
```

## WV-05.2 Four loading methods, two of them need a converter service

```mermaid
flowchart TB
    P["parseUrlAndLoadOntology reads the location hash<br/>loadingModule.js:214-243"] --> ID["identifyOntologyLoadingMethod: file=, url=, iri=, else preset<br/>loadingModule.js:704-719"]
    ID --> PRE["preset: ./data/name.json, default foaf<br/>loadingModule.js:18,525"]
    ID --> FU["file=: cached upload<br/>loadingModule.js:352"]
    ID --> JU["url=: read?json= through the server, after a serverTimeStamp probe<br/>loadingModule.js:252-271"]
    ID --> IR["iri=: convert?iri= through OWL2VOWL<br/>loadingModule.js:306-321"]
    DROP["file drop: .json read in the browser, any other file sent to the converter<br/>loadingModule.js:325-348"] --> IR
    IR --> FB["fallback POST convert, warns of a deprecated OWL2VOWL service<br/>loadingModule.js:406-422"]
    JU --> EXT["EXTERNAL: no OWL2VOWL or read endpoint is in this repo,<br/>the retired Docker image served the upstream WebVOWL war, see WV-04.3"]
    FB --> EXT
```

## WV-05.3 From owl2vowl JSON to a d3 v3 force layout

```mermaid
sequenceDiagram
    autonumber
    participant G as loadGraphData<br/>graph.js:1505
    participant P as parser.parse<br/>parser.js:130
    participant F as d3.layout.force<br/>graph.js:225
    participant U as graph.update<br/>graph.js:1300
    G->>F: stop, empty nodes and links, clear graph data, graph.js:1509-1518
    G->>P: parse(options.data()), graph.js:1526
    P->>P: merge class with classAttribute and datatype with datatypeAttribute, parser.js:141-143
    P->>P: inject set-operator properties, merge propertyAttribute, parser.js:148-149
    P->>P: build node and property structures, parser.js:157-158
    P-->>G: nodes and properties, kept as unfilteredData, graph.js:1527-1530
    G->>G: run every filter module over a clone, graph.js:1644-1647
    U->>F: force.start, then redrawContent, graph.js:1315-1316
    Note over P: The attribute merge here is what the modern loader skips, see WV-03.1
```

## WV-05.4 Notation and export coverage, legacy against modern

```mermaid
flowchart TB
    subgraph L["legacy"]
        LN["14 node prototypes: external, owl Class, complementOf, deprecated,<br/>disjointUnionOf, equivalent, intersectionOf, Nothing, Thing, unionOf,<br/>rdfs Class, Datatype, Literal, Resource<br/>nodeMap.js:2-15"]
        LP["14 property prototypes incl. allValuesFrom, someValuesFrom, functional,<br/>inverse functional, symmetric, transitive, subClassOf, set operator<br/>propertyMap.js:2-15"]
        LE["export menu: Turtle, SVG, JSON, TeX<br/>exportMenu.js:50,213,761,803"]
    end
    subgraph M["modern"]
        MN["NodeType union has four values, types/graph.ts:5<br/>the loader only creates type class, useGraphStore.ts:191"]
        MP["edges are objectProperty or datatypeProperty only<br/>useGraphStore.ts:211"]
        ME["export: SVG and PNG<br/>TopMenuBar.tsx:21,52"]
    end
    L --> M
    GAP["DEBT: set operators, restrictions, equivalence, disjointness and external<br/>colouring have no modern counterpart, so the modern view is not VOWL-complete"]
    M --- GAP
```

The modern counterpart of each legacy stage is WV-02; the VOWL notation engine itself is the vowl-wasm area (VW-02).
