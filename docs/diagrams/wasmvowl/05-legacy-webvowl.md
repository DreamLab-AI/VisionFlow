---
id: WV-05
title: Legacy WebVOWL - bundle, loading paths, parse, notation, menus and editor
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
  - ../WasmVOWL/legacy/src/app/js/menu/exportTTLModule.js
  - ../WasmVOWL/legacy/src/app/js/menu/configMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/debugMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/filterMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/gravityMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/modeMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/navigationMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/ontologyMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/pauseMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/resetMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/searchMenu.js
  - ../WasmVOWL/legacy/src/app/js/menu/zoomSlider.js
  - ../WasmVOWL/legacy/src/app/js/editSidebar.js
  - ../WasmVOWL/legacy/src/app/js/directInputModule.js
  - ../WasmVOWL/legacy/src/webvowl/js/options.js
  - ../WasmVOWL/legacy/src/webvowl/js/graph.js
  - ../WasmVOWL/legacy/src/webvowl/js/parser.js
  - ../WasmVOWL/legacy/src/webvowl/js/elements/nodes/nodeMap.js
  - ../WasmVOWL/legacy/src/webvowl/js/elements/properties/propertyMap.js
  - ../WasmVOWL/modern/src/types/graph.ts
  - ../WasmVOWL/modern/src/stores/useGraphStore.ts
  - ../WasmVOWL/modern/src/components/UI/TopMenuBar.tsx
  - ../WasmVOWL/modern/src/components/Canvas/Nodes/ClassNode.tsx
  - ../WasmVOWL/modern/src/lib/owl2vowl.ts
verified_commit: {wasmvowl: 2bb54b3ee2dc3a7eda34b60d9f08fdb0fb535aab}
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
    Note over P: CHANGED at 2bb54b3: the modern normaliser now does this attribute merge, see WV-03.1
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
        MN["NodeType union has four values, types/graph.ts:5<br/>the normaliser makes class, datatype or literal, owl2vowl.ts:258-267"]
        MP["five edge kinds incl. subclass, disjoint and set operator<br/>types/graph.ts:6, owl2vowl.ts:177-191"]
        MF["VOWL fills for datatype, literal, Thing and external classes<br/>ClassNode.tsx:34-40"]
        ME["export: SVG and PNG<br/>TopMenuBar.tsx:32,63"]
    end
    L --> M
    GAP["DEBT: restrictions and property characteristics draw as plain object-property edges,<br/>owl2vowl.ts:184-190, equivalent classes stay separate nodes, owl2vowl.ts:44-45,<br/>so the modern view is still not VOWL-complete.<br/>CHANGED at 2bb54b3: set operators, disjointness and external colouring now have a counterpart"]
    M --- GAP
```

The modern counterpart of each legacy stage is WV-02; the VOWL notation engine itself is the vowl-wasm area (VW-02).

## WV-05.5 The menu layer: twelve menus wired by app.js, one export helper

```mermaid
flowchart TB
    REQ["app.js requires twelve menu modules, each handed the graph<br/>legacy/src/app/js/app.js:13-23,27"] --> S1
    subgraph S1["set up before the browser check, app.js:209-218"]
        M1["exportMenu.setup, exportMenu.js:28<br/>gravityMenu.setup, gravityMenu.js:18"]
        M2["filterMenu.setup takes six filter modules, app.js:211, filterMenu.js:45<br/>modeMenu.setup takes pick and pin, scaling, compact, externals<br/>app.js:212, modeMenu.js:40"]
        M3["pauseMenu.setup, pauseMenu.js:16<br/>debugMenu.setup, greys editor helpers unless editing, debugMenu.js:8,16"]
        M1 --> M2 --> M3
    end
    S1 --> S2
    subgraph S2["set up only when the browser is supported, app.js:219-273"]
        M4["resetMenu.setup takes gravity, filter, mode, focuser, details, pause<br/>app.js:237, resetMenu.js:19"]
        M5["searchMenu.setup, searchMenu.js:112<br/>navigationMenu.setup, navigationMenu.js:242<br/>zoomSlider.setup, zoomSlider.js:42, app.js:238-240"]
        M6["ontologyMenu.setup takes loadOntologyFromText, app.js:272, ontologyMenu.js:75<br/>configMenu.setup adds zoom controls and max label width<br/>app.js:273, configMenu.js:6,13-14"]
        M4 --> M5 --> M6
    end
    S2 --> REG["ten of the twelve menus, all but debug and config,<br/>are registered on options for cross-module calls, app.js:243-262"]
    TTL["exportTTLModule is not wired by app.js, exportMenu requires it<br/>exportMenu.js:16, called from the Turtle export, exportMenu.js:51-52"]
    M1 --- TTL
```

Of these, `exportMenu` already appears in WV-05.4, `modeMenu` and `ontologyMenu` lead into WV-05.6.

## WV-05.6 The experimental editor path

```mermaid
flowchart TB
    subgraph ON["turning editing on"]
        CB["Modes menu checkbox Editing, labelled experimental<br/>modeMenu.js:47,109-112, calls graph.editorMode, modeMenu.js:100-104"]
        URL["URL option editorMode in the hash opts<br/>loadingModule.js:651, options.js:614-623"]
        NEWO["new_ontology hash loads an empty ontology and forces editing<br/>app.js:411-426"]
        CB --- URL --- NEWO
    end
    ON --> TOG["graph.editorMode sets editMode and the new-ontology entry state<br/>graph.js:2723-2760"]
    TOG --> ENT["editing on: left edit sidebar shown, prefix UI, double click bound to create<br/>graph.js:2810-2815"]
    ENT --> ACT
    subgraph ACT["editing actions inside graph.js"]
        A1["double click on canvas creates a node of the default class type<br/>graph.js:3480-3487,2835-2841"]
        A2["double click on a node or label raises inline edit<br/>graph.js:732-737,791-795"]
        A3["hover controls: delete a node, add a datatype property<br/>graph.js:3844-3849,3872-3877"]
        A4["class dragger dropped on a node creates an object property<br/>after a sanity check, graph.js:321-332,3113-3119"]
        A1 --- A2 --- A3 --- A4
    end
    ACT --> SIDE["editSidebar: IRI and label editors, element type change<br/>editSidebar.js:769-786,1242-1260<br/>into graph.changeNodeType and changePropertyType, graph.js:2541,2623"]
    SIDE --> OUT["Turtle export walks the edited graph<br/>exportMenu.js:51-52, exportTTLModule.js:15-36"]
    IDS["INVARIANT: after each load the editor id counters start past<br/>the loaded counts, graph.js:1531-1533"]
    DI["direct input: pasted JSON loads locally, anything else is POSTed<br/>to directInput for conversion, directInputModule.js:16-40, ontologyMenu.js:387-398"]
    EXT["EXTERNAL: the directInput converter endpoint is not in this repo, see WV-05.2"]
    A1 --- IDS
    DI --> EXT
```

The editor's starting state is set in two places, `editMode = true` in `graph.js:83` and an `editorMode` default of `"false"` in `options.js:333`; this diagram does not claim which wins at boot.
