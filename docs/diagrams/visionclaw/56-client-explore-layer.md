---
id: VC-56
title: Client explore layer, node table, crossfilter widgets, highlight mask and column encoding
area: visionclaw
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2183, ADR-2184, ADR-2185, ADR-2189]
sources:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/adr/ADR-2183-interactive-filters-grey-out-through-a-highlight-mask.md
  - ../project/docs/adr/ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md
  - ../project/docs/adr/ADR-2185-linked-timeline-histogram-and-search-widgets-write-the-highlight-mask.md
  - ../project/client/src/features/graph/explore/index.ts
  - ../project/client/src/features/graph/explore/ExplorePanel.tsx
  - ../project/client/src/features/graph/explore/GraphLegend.tsx
  - ../project/client/src/features/graph/explore/buildTable.ts
  - ../project/client/src/features/graph/explore/columnEncoder.ts
  - ../project/client/src/features/graph/explore/crossfilter.ts
  - ../project/client/src/features/graph/explore/exploreStore.ts
  - ../project/client/src/features/graph/explore/filterRegistry.ts
  - ../project/client/src/features/graph/explore/nodeFilter.ts
  - ../project/client/src/features/graph/explore/__tests__/nodeTable.test.ts
  - ../project/client/src/features/graph/explore/nodeTable.worker.ts
  - ../project/client/src/features/graph/explore/tableBuilder.ts
  - ../project/client/src/features/graph/explore/tableSchema.ts
  - ../project/client/src/features/graph/explore/useExplore.ts
  - ../project/client/src/features/graph/state/highlightMask.ts
  - ../project/client/src/features/graph/state/nodeEncoding.ts
  - ../project/client/src/features/graph/state/nodeIndex.ts
  - ../project/client/src/features/graph/hooks/useGraphFiltering.ts
  - ../project/client/src/features/graph/hooks/useGraphSelection.ts
  - ../project/client/src/features/graph/components/GraphManager.tsx
  - ../project/client/src/features/graph/overview/useNodeEncodingSync.ts
  - ../project/client/src/features/analytics/store/nodeAnalyticsStore.ts
  - ../project/client/src/features/analytics/analyticsLayout.ts
  - ../project/client/src/features/analytics/__tests__/analyticsLayout.test.ts
  - ../project/client/src/features/graph/state/analyticColours.ts
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/client/src/features/graph/workers/graph.worker.ts
  - ../project/client/src/features/graph/workers/lib/binary-processor.ts
  - ../project/client/src/features/control-center/ControlCenter.tsx
  - ../project/client/src/features/control-center/drawer/RightDrawer.tsx
---
## For developers

The explore layer (ADR-2183, ADR-2184, ADR-2185, landed in the 16d3bb49e suite) is everything between the delivered graph and the two render-indexed buffers that VC-31.13 draws from: a typed-array node table built in a worker, a crossfilter over it, one highlight-mask filter source per widget (`filter:explore:<widget>`) written through `filterRegistry.ts`, and a column encoder that swaps `nodeEncoding`'s encoder for colour-by and size-by. Filters grey nodes out and never remove them; only population scope and hierarchy collapse remove (ADR-2183 D4), and the `hideFiltered` opt-in turns grey back into removal. The render side of the mask and encoding (texel `w`, sprites, gems, labels) is VC-31.13; the drawer and section that host the panel are VC-57. This topic does not cover the provenance `TimelineScrubber` (VC-30) or the server copy of the node-filter predicates (`client_filter.rs`).

The node table is the typed arrays themselves: ADR-2184 D1 was amended on 2026-11-06 to say so, and the unused Arrow conversion and the `apache-arrow` dependency were deleted (`ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:35`, `ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:61`).

## For the business

A visitor can now ask the graph questions by clicking bars: "what changed recently", "which domain", "how central". Everything outside the answer greys out but stays in place, so the shape of the whole knowledge base is never lost, and a legend says what each colour and size means. The figures the panel shows count what is lit on screen, with one known exception: nodes hidden inside a collapsed hierarchy group are still counted.

## VC-56.1 Who mounts the explore lane and what it writes

```mermaid
flowchart TB
    CC["ControlCenter mounts LegendSlot, always<br/>ControlCenter.tsx:135, :195-212"] --> GL["GraphLegend: feeds the table and installs the encoder<br/>GraphLegend.tsx:126-132"]
    RD["RightDrawer explore tab, mounted on first open and kept<br/>RightDrawer.tsx:101, :141, :287"] --> EP["ExplorePanel<br/>ExplorePanel.tsx:314-315"]
    GL --> SRC["useExploreTableSource, ref-counted, one graph subscription<br/>useExplore.ts:97-123"]
    EP --> SRC
    SRC --> ES["exploreStore: table, crossfilter, clauses<br/>exploreStore.ts:38, :178"]
    EP -->|"widget clauses"| ES
    ES -->|"writeFilter filter:explore:widget, include"| FR["filterRegistry, the one write path<br/>filterRegistry.ts:92-114"]
    FR --> HM["highlightMask, by render index<br/>highlightMask.ts:227"]
    GL -->|"useExploreEncoding"| ENC["nodeEncoding.setEncoder<br/>useExplore.ts:166-169, nodeEncoding.ts:188-195"]
    HM --> R31["drawn by VC-31.13: sprites, gems, texel w, labels"]
    ENC --> R31
    UGF["useGraphFiltering: settings grey-out as filter:settings, exclude<br/>useGraphFiltering.ts:130-132"] --> FR
    UGS["useGraphSelection: selection, neighbours, search as emph sources<br/>useGraphSelection.ts:188-189, :215"] --> FR
    IDX["INVARIANT: every buffer is indexed by the node's slot in graphData.nodes<br/>nodeIndex.ts:4-10, :24"] --- HM
```

**What it shows.** Two mounts feed one shared store; only `GraphLegend`, which ControlCenter always mounts, installs the column encoder, so colour-by works with the drawer closed. **Why it is this way.** ADR-2183's implementation notes route every mask write through `filterRegistry.ts` under `filter:*` and `emph:*` names (`ADR-2183-interactive-filters-grey-out-through-a-highlight-mask.md:71`), and ADR-2189 put the panel in the drawer.

## VC-56.2 Node table build: worker, transfer, newest-wins adoption

```mermaid
sequenceDiagram
    autonumber
    participant UE as rebuild<br/>useExplore.ts:55
    participant TB as createTableBuilder<br/>tableBuilder.ts:19
    participant WK as node-table worker<br/>nodeTable.worker.ts:18
    participant BT as buildTable<br/>buildTable.ts:28
    participant ES as setColumns<br/>exploreStore.ts:68
    Note over UE: triggered by a new graph load, the 5 s analytics poll<br/>or a change of includeLinkedPages (useExplore.ts:91-95, :100-104, :122)
    UE->>UE: private copy of the analytics buffer, seq = ++buildSeq (:59, :61)
    UE->>TB: build with nodes, indexMap, analytics, nodeFilter (:62-66)
    TB->>TB: lean nodes, only includeLinkedPages crosses (tableBuilder.ts:67, :70)
    TB->>WK: postMessage request id (tableBuilder.ts:74)
    WK->>BT: buildTable(input) (nodeTable.worker.ts:21)
    BT->>BT: buildNodeColumns, then eligible = present minus linked_page scope (buildTable.ts:30-36)
    BT-->>WK: table and eligible mask
    WK-->>TB: posted with every typed array transferred, text columns cloned (nodeTable.worker.ts:22, buildTable.ts:53-59)
    TB-->>UE: resolve offThread true (tableBuilder.ts:47)
    UE->>ES: adopt only when seq is still the latest (useExplore.ts:68)
    ES->>ES: new Crossfilter, re-apply clauses, re-write widget mask sources (exploreStore.ts:69-73)
    Note over TB: worker error or failure: fallBackAll builds every pending input on the main thread<br/>and never retries (tableBuilder.ts:26-31, :49-53)
    Note over TB: DEBT: dispose terminates the worker without settling pending promises (tableBuilder.ts:80)
```

**What it shows.** How the table is rebuilt off the main thread and why an overlapping build can never overwrite a newer one. **Why it is this way.** ADR-2184 D1 builds the table on the client from data already delivered, with no new read path (`ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:35-38`); the worker is a same-origin module worker so COEP needs no extra headers (`nodeTable.worker.ts:1-4`).

**Open (risk):** releasing the last mount calls `builder.dispose()` (`useExplore.ts:112`), which terminates the worker without resolving the entries in `pending` (`tableBuilder.ts:24`, `tableBuilder.ts:80`), so an in-flight build's promise and its closure never settle.

## VC-56.3 Table schema: three column sources, one render-index row order

```mermaid
flowchart TB
    SCH["NODE_TABLE_SCHEMA, declared types, never guessed per value<br/>tableSchema.ts:57"]
    SCH --> N["node: id, label<br/>tableSchema.ts:58-59"]
    SCH --> M["metadata strings typed by the schema: type, domain, maturity, quality_score,<br/>last_commit, last_modified, change_count and more (tableSchema.ts:62-82)"]
    SCH --> W["wire analytics joined by render index: cluster, community with nullId 0,<br/>centrality, anomaly (tableSchema.ts:85-88)"]
    W --> AL["offsets imported from the one layout declaration, stride 5 with SSSP at 4<br/>tableSchema.ts:21-27, analyticsLayout.ts:9-14"]
    AL -.-> STORE["the store builds its buffer from the same constants<br/>nodeAnalyticsStore.ts:4-11"]
    AL -.-> PIN["INVARIANT: the layout imports nothing, and the store writes each field at its offset<br/>analyticsLayout.test.ts:15-18, :20-33"]
    SCH --> BNC["buildNodeColumns: rows = slotCountFor(indexMap), present = delivered<br/>tableSchema.ts:165-175"]
    BNC --> TYP["the typed arrays are the table: Float64 NaN-null, Int8 boolean,<br/>Int32 dictionary codes, string arrays (tableSchema.ts:97-104, :17-18)"]
    TYP --> PINT["test pins render-index rows, an empty slot, typed columns and nulls<br/>nodeTable.test.ts:8-28"]
    SCH -.-> T1["TENSION: ADR-2184 says the schema is generated from Rust where the type exists,<br/>it is hand-declared TypeScript (ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:58)"]
```

**What it shows.** Where each column comes from and why row `i` of the table is slot `i` of the mask and the encoding buffers. **Why it is this way.** ADR-2184 D2 types metadata strings by a declared schema (`ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:41`); the layout is declared once in an import-free module because `nodeAnalyticsStore`'s module graph creates the settings store and cannot load in a worker (`analyticsLayout.ts:5-7`).

**Debt (parallel):** the worker still keeps its own analytics buffer at its own stride constant, `WORKER_ANALYTICS_STRIDE = 5` (`binary-processor.ts:22`), filled per frame (`binary-processor.ts:80-86`, `graph.worker.ts:171`), and answers `getAnalyticsBuffer` (`graph.worker.ts:499-500`), while the proxy's `getAnalyticsBuffer` is a deprecation stub that throws (`graphWorkerProxy.ts:493-497`): a second layout nobody on the main thread can read.

**Debt (parallel):** the maturity tiers are declared twice, as `MATURITY_RANK` (`nodeFilter.ts:26-28`) and `MATURITY_ORDER` (`tableSchema.ts:55`), and the latter's comment still points at `useGraphFiltering`'s copy (`tableSchema.ts:54`), which moved to `nodeFilter.ts`.

## VC-56.4 Crossfilter: one clause per widget, counts exclude their own clause

```mermaid
flowchart TB
    SC["setClause(widget, clause): subtract the old fail vector, add the new<br/>crossfilter.ts:97-107"]
    SC --> EV["evaluate: 1 where the row fails, kinds range, values, search, ids<br/>crossfilter.ts:23-36, :267-304"]
    EV --> FC["failCount per row, a Uint8Array<br/>crossfilter.ts:72, :80"]
    FC --> PS["passes(row, except W) = eligible and failCount minus own fails is 0<br/>crossfilter.ts:110-114"]
    PS --> HG["histogram: bins over min to p99, last bin open, fixed brush edges kept<br/>crossfilter.ts:158-186, :242"]
    PS --> CT["categories: declared order, else by total<br/>crossfilter.ts:189-213"]
    PS --> CN["count except W, the N of M figure<br/>crossfilter.ts:117-121"]
    FC --> IP["idsPassingClause(W): ids passing that one clause, over present rows<br/>crossfilter.ts:124-131"]
    IP --> MASK["becomes widget W's include source on the mask (VC-56.5)"]
    EL["eligible = delivered and in population scope<br/>crossfilter.ts:15-16, :79"] --> PS
    SR["search haystack: label, tags, definition, slug, lower-cased once<br/>crossfilter.ts:39, :249-264"] --> EV
    DK["DuckDB-WASM declined: typed-array passes at about 9.7k rows<br/>crossfilter.ts:10-13"] -.-> SC
    UN["DEBT: selectedIds, the combined selection, has no production caller<br/>crossfilter.ts:134-140"] -.-> IP
```

**What it shows.** The standard crossfilter rule in O(1) per row: a widget's own bars keep the full distribution while the others narrow. **Why it is this way.** ADR-2185 D2 (`ADR-2185-linked-timeline-histogram-and-search-widgets-write-the-highlight-mask.md:39`); D5 sanctions the typed-array fallback if DuckDB is not used.

**Tension (ADR-2184 D5 vs code):** the record names DuckDB-WASM with Mosaic as the widgets' query engine (`ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:50-51`) and is marked implementation complete, while the code declines DuckDB in a comment (`crossfilter.ts:10-13`) on the strength of ADR-2185 D5's fallback (`ADR-2185-linked-timeline-histogram-and-search-widgets-write-the-highlight-mask.md:42`).

## VC-56.5 exploreStore clauses: widget sources and context clauses

```mermaid
flowchart TB
    subgraph WID["Widget clauses: timeline, histogram, categories, search"]
        WSC["setClause: empty search clears, crossfilter updated, mask written, version++<br/>exploreStore.ts:134-140"]
        WSC --> WM["writeMask: ids passing the clause, as filter:explore:widget include<br/>exploreStore.ts:164-169, :31"]
        WM --> WG["no table yet: the mask is left alone<br/>exploreStore.ts:166-167"]
    end
    subgraph CTX["Context clauses: counted, never written to the mask"]
        GS["graph:selection, ids include<br/>exploreStore.ts:34, :80-90"]
        EXT["ext:source mirrors every non-explore filter source, e.g. filter:settings<br/>exploreStore.ts:35, :113-131"]
    end
    WID --> CTX
    REG["filterRegistry listener calls syncExternalFilters<br/>exploreStore.ts:48, filterRegistry.ts:155-159"] --> EXT
    CTX --> LIT["litCount = count except graph:selection, so settings grey-out is counted<br/>exploreStore.ts:102-104"]
    WID --> CLR["clearAll: drop widget clauses, clear every filter:explore source<br/>exploreStore.ts:152-162"]
    CLR -.-> D1["DEBT: one mask recomposition per cleared source, HighlightMask.batch has no caller<br/>exploreStore.ts:158-160, highlightMask.ts:161-167"]
    WM --> INC["INVARIANT: intersection, any include source missing a node greys it<br/>highlightMask.ts:212-216"]
```

**What it shows.** Each widget is its own mask source, so the graph shows the intersection; selection and the settings grey-out enter the crossfilter only as counting context. **Why it is this way.** ADR-2185 D3 makes the graph a consumer of the combined selection (`ADR-2185-linked-timeline-histogram-and-search-widgets-write-the-highlight-mask.md:40`), realised as one source per widget so a single widget can be cleared alone.

**Debt (loose end):** `SOURCE.EXPLORE`, the single combined `filter:explore:crossfilter` source, is declared (`filterRegistry.ts:38-39`) and re-exported (`explore/index.ts:12`) but never written: widgets write per-widget sources (`exploreStore.ts:31`).

**Drift:** the explore public-API header still says mounting ExplorePanel or GraphLegend "feeds the node table" (`explore/index.ts:4`); the Arrow node table was deleted in 29cc1cd8d and nothing is fed.

**Open (risk):** on a new graph load the mask adopts the new index at once (`highlightMask.ts:183-190`), but each active widget's include set is from the previous table until the worker build lands (`useExplore.ts:62-68`, `exploreStore.ts:72`), and an include source greys every mapped node missing from its set (`highlightMask.ts:213-216`), so new nodes are greyed until then.

## VC-56.6 Selection-to-widgets clause

```mermaid
sequenceDiagram
    autonumber
    participant SEL as useGraphSelection<br/>useGraphSelection.ts:174
    participant FRG as writeEmphasis<br/>filterRegistry.ts:117
    participant XS as setGraphSelection<br/>exploreStore.ts:80
    participant PNL as ExplorePanel<br/>ExplorePanel.tsx:314
    SEL->>SEL: neighbours from graphData.edges (:181-187)
    SEL->>FRG: emph:selection and emph:neighbours (useGraphSelection.ts:188-189)
    SEL->>XS: selected id plus neighbours (useGraphSelection.ts:191)
    XS->>XS: same set as before, return (exploreStore.ts:88)
    XS->>XS: graph:selection ids include, crossfilter only, bump (exploreStore.ts:89, :106-110)
    PNL->>XS: graphSelectionSize for the banner (ExplorePanel.tsx:321)
    PNL->>PNL: Widgets show the graph selection, N nodes (ExplorePanel.tsx:338-340)
    PNL->>SEL: Clear selection dispatches visionclaw:node-deselect (ExplorePanel.tsx:341-342)
    SEL->>XS: no selection, setGraphSelection null (useGraphSelection.ts:175-178)
    Note over XS: INVARIANT: the selection never writes the mask, the graph already emphasises it<br/>(exploreStore.ts:9-11, :77-78)
```

**What it shows.** Clicking a node narrows every widget to it and its neighbours, while the graph shows the same set as emphasis. **Why it is this way.** ADR-2185 D3: "selecting nodes in the graph adds a clause that the widgets reflect" (`ADR-2185-linked-timeline-histogram-and-search-widgets-write-the-highlight-mask.md:40`); the effect re-runs on every edges change, and the set comparison keeps that churn from re-aggregating (`exploreStore.ts:88`).

**Debt (parallel):** two searches with different reach: the panel's search widget is a grey-out clause over label, tags, definition and slug (`crossfilter.ts:39`, `ExplorePanel.tsx:291-299`), and the `visionclaw:search` event is a label-only emphasis (`useGraphSelection.ts:33-43`, `useGraphSelection.ts:214-215`).

## VC-56.7 filterRegistry: the write path, de-duplication and hideFiltered

```mermaid
flowchart TB
    WF["writeFilter: source must start filter:, null clears<br/>filterRegistry.ts:92-106"]
    WF --> DD["same mode, same ids, still live on the mask: skip, no recomposition<br/>filterRegistry.ts:107-109"]
    DD --> SET["record, mask.setFilter, registry version++<br/>filterRegistry.ts:110-113"]
    SET --> RC["HighlightMask.recompose: fill 0, OR emphasis, FILTERED_OUT for any filter, version++<br/>highlightMask.ts:193-223"]
    RC --> LV["highlightLevel: selected or hovered beat greyed, which beats search and neighbour<br/>highlightMask.ts:54-58"]
    WE["writeEmphasis, emph: sources, empty set clears<br/>filterRegistry.ts:117-137"] --> RC
    MSUB["mask subscriber prunes records of sources someone else cleared<br/>filterRegistry.ts:58-75"] --> SET
    SET --> IFO["isFilteredOut(id): by id from the registry, not the mask bytes, index-free<br/>filterRegistry.ts:143-148"]
    IFO --> HF["hideFiltered opt-in removes what any filter greys<br/>useGraphFiltering.ts:135-149"]
    HF --> FIRST["checks greyedNodeIds directly so the first render already hides<br/>useGraphFiltering.ts:142-146"]
    WF -.-> DRIFT["DRIFT: header names useExploreFilters.ts for the hooks, no such file exists<br/>filterRegistry.ts:19"]
```

**What it shows.** Every grey and emphasis write, and why graph-data churn costs no recomposition. **Why it is this way.** Each mask write is O(n) plus a re-render of every subscriber (`filterRegistry.ts:11-13`); `hideFiltered` reads the registry because the mask bytes are indexed by a map that is one render stale on a new load (`filterRegistry.ts:14-17`), the opt-in ADR-2183 foresaw (`ADR-2183-interactive-filters-grey-out-through-a-highlight-mask.md:49`).

## VC-56.8 Node-filter settings: removal, grey-out, and hide

```mermaid
flowchart TB
    S["settings nodeFilter, resolved with client fallbacks, off by default at 0.4<br/>nodeFilter.ts:57-68, :75-89"]
    S --> R1["REMOVE, population scope: linked_page while excluded<br/>nodeFilter.ts:101-103, useGraphFiltering.ts:106"]
    S --> R2["REMOVE, hierarchy collapse: expansion state held in the hook<br/>useGraphFiltering.ts:94, :107-108"]
    S --> G["GREY: minConnections, minMaturity, then quality and authority, only enabled<br/>predicates combine, unscored is neutral (nodeFilter.ts:109-139)"]
    R1 --> SN["scopedNodes, the render set<br/>useGraphFiltering.ts:103-112"]
    R2 --> SN
    SN --> G
    G --> GS["greyedNodeIds written as filter:settings exclude, empty clears<br/>useGraphFiltering.ts:115-132"]
    GS --> HF["hideFiltered on: greyed also removed (VC-56.7)<br/>useGraphFiltering.ts:140-147"]
    R1 --> TB["the table's eligible mask applies the scope only<br/>buildTable.ts:32-36"]
    R2 -.-> GAP["OPEN: collapse is not in the table's eligibility, so widget N of M counts<br/>hidden nodes (ExplorePanel.tsx:336)"]
```

**What it shows.** The ADR-2183 D4 split: scope and collapse remove, the interactive predicates grey, and only the opt-in hides. **Why it is this way.** ADR-2183 D4 keeps hierarchy collapse a removal because it "changes what a node is" (`ADR-2183-interactive-filters-grey-out-through-a-highlight-mask.md:42`); the predicate semantics and the 0.4 default come from the 2026-11-06 implementation notes (`nodeFilter.ts:13-20`, `nodeFilter.ts:51-57`).

**Tension (crossfilter vs render set):** the crossfilter says only rows in the render set count "so counts match the screen" (`crossfilter.ts:15-16`), but hierarchy collapse lives in per-instance hook state inside GraphManager (`GraphManager.tsx:87`, `useGraphFiltering.ts:94`) that the table builder cannot see (`buildTable.ts:32-36`), so the panel has to disclaim it (`ExplorePanel.tsx:336`).

## VC-56.9 Column encoder and legend

```mermaid
flowchart TB
    SET["settings explore colorBy, sizeBy, and nodes.colorScheme<br/>useExplore.ts:157-159"]
    SET --> CE["createColumnEncoding(table, colorBy, sizeBy)<br/>useExplore.ts:161, columnEncoder.ts:277-296"]
    CE --> CAT["categorical: domain palette, wire community and cluster hue formulas,<br/>else frequency palette then other (columnEncoder.ts:151-224)"]
    CE --> SEQ["sequential viridis ramp over p2 to p98, live max for centrality<br/>columnEncoder.ts:125-143, :226-241"]
    CE --> SZ["size: default size x 0.5 to 2.5<br/>columnEncoder.ts:57-58, :247-259"]
    CAT --> HUE["community and cluster hues from the one formula module<br/>columnEncoder.ts:197, :199, analyticColours.ts:12-19"]
    CAT --> ACC["values read at encode time from metadata or the live analytics buffer,<br/>never from a table row (columnEncoder.ts:23-26, :107-116)"]
    SEQ --> ACC
    CE --> INST["setEncoder, restored to default on unmount<br/>useExplore.ts:166-169, nodeEncoding.ts:188-195"]
    INST --> EV["encoderVersion is part of the encode snapshot, so the one encodeAll caller re-runs<br/>useNodeEncodingSync.ts:82, :100"]
    EV --> EA["encodeAll by render index, version++<br/>nodeEncoding.ts:206-228"]
    SET --> SL["colorBy scheme: legend for today's look per colorScheme<br/>useExplore.ts:163-165, columnEncoder.ts:344-368"]
    CE --> LG["GraphLegend renders entries unless showLegend is off<br/>GraphLegend.tsx:126-132"]
    SL --> LG
    CAT -.-> CNT["OPEN: legend counts use present rows, the crossfilter eligible rows<br/>columnEncoder.ts:159-160, crossfilter.ts:79"]
```

**What it shows.** How a colour-by or size-by choice becomes a new encoder and a legend, and why live wire columns recolour without a table rebuild. **Why it is this way.** ADR-2184 D4: any categorical or numeric column, a clamp on size, a legend always shown, defaults reproduce today's look (`ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:43-48`).

The community, cluster and centrality hues are written once, in `analyticColours.ts` (`analyticColours.ts:12-25`), and the encoder, its scheme legend and the CSS ramp call them (`columnEncoder.ts:197`, `columnEncoder.ts:199`, `columnEncoder.ts:332`, `columnEncoder.ts:355`), as does the default encoder (`nodeEncoding.ts:95`, `nodeEncoding.ts:97`, `nodeEncoding.ts:100`).

## VC-56.10 ExplorePanel widgets and brushing

```mermaid
flowchart TB
    P["ExplorePanel: N lit of M eligible, Clear filters with the active count<br/>ExplorePanel.tsx:318-332"]
    P --> SW["Search: 200 ms debounce, a search clause<br/>ExplorePanel.tsx:35, :287-299"]
    P --> TW["Timeline: last_commit else last_modified, 36 bins, empty state without dates<br/>ExplorePanel.tsx:31-32, :175-189"]
    P --> HW["Histogram: any numeric column with values, 20 bins<br/>ExplorePanel.tsx:33, :209-216"]
    P --> CW["Categories: domain first, at most 12 bars then a plus N more line<br/>ExplorePanel.tsx:34, :241-256, :281"]
    TW --> BB["BinBars: click, shift-click, drag, arrow keys, Escape clears<br/>ExplorePanel.tsx:88-104"]
    HW --> BB
    BB --> RCL["rangeClause keeps the brush-time edges, open ends at both extremes<br/>ExplorePanel.tsx:159-164, :167-169"]
    RCL --> ST["store.setClause(widget)<br/>ExplorePanel.tsx:200, :233"]
    CW --> ST
    SW --> ST
    SW -.-> T1["TENSION: nodes match shows litCount, every clause and the settings grey-out,<br/>not the search alone (ExplorePanel.tsx:300, :307)"]
    CW -.-> D1["DEBT: categories past the 12th cannot be picked, only counted<br/>ExplorePanel.tsx:256, :281"]
```

**What it shows.** The four ADR-2185 D1 widgets and how a brush becomes a clause that survives a live domain shift. **Why it is this way.** ADR-2185 D1 lists timeline, numeric histograms, category bars and search (`ADR-2185-linked-timeline-histogram-and-search-widgets-write-the-highlight-mask.md:38`); the panel is a scene occluder so the camera fit keeps the graph out from under it (`ExplorePanel.tsx:13-14`).

## VC-56.11 Refresh cadence: what a 5 s analytics change costs

```mermaid
flowchart TB
    T["timer every ANALYTICS_REFRESH_MS = 5000 while any mount is live<br/>useExplore.ts:24, :100-104"]
    T --> CMP["compare analytics contents, not identity<br/>useExplore.ts:41-46, :102-103"]
    CMP -->|"changed"| RB["full table rebuild in the worker, all nodes cloned again<br/>useExplore.ts:55-69, tableBuilder.ts:64-74"]
    RB --> XF["new Crossfilter, clauses re-evaluated, widget sources re-written<br/>exploreStore.ts:68-74"]
    XF --> DEDUP["writeFilter skips an unchanged id set<br/>filterRegistry.ts:109"]
    RB --> NEWT["new table identity re-runs createColumnEncoding<br/>useExplore.ts:161"]
    NEWT --> SE["setEncoder bumps encoderVersion, a whole-graph encodeAll<br/>nodeEncoding.ts:188-195, useNodeEncodingSync.ts:82"]
    RA["readAnalytics uses the shared index map when sizes agree, to keep the store's<br/>one-slot cache warm (useExplore.ts:72-83, nodeAnalyticsStore.ts:97)"] --> CMP
    SE -.-> OPEN1["OPEN: with live GPU analytics every tick differs, so the whole chain runs<br/>every 5 s even when no encoding reads a wire column (useExplore.ts:161, :166-169)"]
```

**What it shows.** The one polling loop in the lane and the work a changed analytics buffer triggers. **Why it is this way.** ADR-2184 D1 refreshes analytics "as frames arrive" (`ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:37`); the code polls instead, and the encoder reads wire values at encode time (`columnEncoder.ts:23-26`), so the rebuild serves the widgets' domains and counts more than the colours.
