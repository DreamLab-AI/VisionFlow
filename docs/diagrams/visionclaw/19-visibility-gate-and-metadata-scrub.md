---
id: VC-19
title: The visibility gate shared by REST and /wss, the page-name scrub, gated analytics and the ontology deny-list
area: visionclaw
governing:
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/SECURITY-profiles.md
adrs: [ADR-2003, ADR-2012, ADR-2184]
sources:
  - ../project/crates/visionclaw-domain/src/utils/visibility_filter.rs
  - ../project/src/handlers/graph_access.rs
  - ../project/crates/visionclaw-domain/src/models/metadata.rs
  - ../project/src/settings/auth_extractor.rs
  - ../project/src/utils/auth.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/src/handlers/api_handler/graph/mod.rs
  - ../project/src/handlers/api_handler/graph/fold.rs
  - ../project/src/handlers/api_handler/files/mod.rs
  - ../project/src/handlers/graph_export_handler.rs
  - ../project/src/adapters/sqlite_share_repository.rs
  - ../project/src/handlers/api_handler/analytics/mod.rs
  - ../project/src/handlers/api_handler/analytics/gpu_routes.rs
  - ../project/src/handlers/api_handler/analytics/community.rs
  - ../project/src/handlers/api_handler/analytics/pathfinding.rs
  - ../project/src/handlers/api_handler/analytics/visible_analytics.rs
  - ../project/src/handlers/api_handler/analytics/visible_results.rs
  - ../project/src/handlers/api_handler/analytics/pagerank_handlers.rs
  - ../project/src/handlers/api_handler/analytics/sssp_handlers.rs
  - ../project/src/handlers/api_handler/analytics/clustering_handlers.rs
  - ../project/src/handlers/api_handler/analytics/cycles_handlers.rs
  - ../project/src/handlers/api_handler/analytics/insights_handlers.rs
  - ../project/src/handlers/api_handler/semantic_forces.rs
  - ../project/src/handlers/api_handler/ontology/mod.rs
  - ../project/src/handlers/constraints_handler.rs
  - ../project/src/handlers/pages_handler.rs
  - ../project/src/handlers/semantic_handler.rs
  - ../project/src/handlers/semantic_pathfinding_handler.rs
  - ../project/src/handlers/schema_handler.rs
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/handlers/bots_visualization_handler.rs
  - ../project/src/handlers/ontology_handler.rs
  - ../project/src/handlers/ontology_agent_handler.rs
  - ../project/src/services/ontology_visibility.rs
  - ../project/src/services/pathfinding.rs
  - ../project/src/adapters/actix_semantic_adapter.rs
  - ../project/src/actors/gpu/shortest_path_actor.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/main.rs
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/SECURITY-profiles.md
  - ../project/docs/adr/ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

Every read that returns node identities applies one visibility policy, whether it is REST, the `/wss` initial load, an analytics answer or an ontology read. `NodeVisibility::of_node` is the only reader of the `public` and `owner_pubkey` metadata keys. `node_is_visible` applies the rule: keep a node that is public or owned by the caller, drop everything else. `visible_subgraph` builds the caller's graph from that rule: visible nodes, edges whose two endpoints are both visible, metadata rows a visible node names, and visible metadata with hidden page names scrubbed out, wikilinks inside a `definition` included (VC-19.1, VC-19.2). The `/wss` `initialGraphLoad` and `GET /api/graph/data` are pinned to the same output for three callers (VC-19.3). Handlers reach the policy through `graph_access` (VC-19.4). Since the 2026-11-06 sweep there is one decision type, `CallerGate`, which carries the caller's view and two distinct sets: visible node ids and visible GPU positions. A gated caller's analytics are computed on the CPU over the view (SSSP, point-to-point, components, APSP landmarks, communities with modularity, PageRank). Full-access callers go to the GPU through `gpu_routes`. Only structural anomalies and semantic centrality are still filtered on the way out (VC-19.5). Hidden reads answer exactly like missing ones, and share deletion is checked against the recorded creator (VC-19.6). The Oxigraph ontology store still holds formal classes whose pages are private, so every ontology read now takes a per-caller deny-list of hidden class IRIs (VC-19.7). The live position stream applies the same rule through a drop set (VC-03.9).

## For the business

A visitor who is not signed in sees only public pages. A signed-in user also sees their own private pages, and never anyone else's. The same rule applies to the graph view, to every analysis the server runs, to file contents, to shared snapshots and now to the ontology browser and the agent ontology tools. A private page someone else owns looks exactly like a page that does not exist. Analyses for a user who cannot see everything are computed over what that user can see, so a private page no longer shapes their community scores or rankings. Two analyses, anomaly detection and semantic centrality, still use scores that private pages helped shape. Some ontology queries (yes/no questions, counts, text matches) can still hint at a private class.

## VC-19.1 The policy core: one reader of the visibility keys, one subgraph
```mermaid
flowchart TB
    ENV["PUBKEY_VISIBILITY_FILTER<br/>visibility_filter.rs:31"] --> PARSE["parse_visibility_flag: absent or unrecognised means ON<br/>only 0, false, off or no disables<br/>visibility_filter.rs:34-42"]
    PARSE --> FE["filter_enabled, read once through OnceLock<br/>visibility_filter.rs:46-51"]
    ON["NodeVisibility::of_node: public iff metadata public equals true,<br/>owner_pubkey read only for a private node<br/>visibility_filter.rs:101-115"] --> DROP["is_dropped_for: public keeps,<br/>caller equal to owner keeps, anything else drops<br/>visibility_filter.rs:127-137"]
    DROP --> NIV["node_is_visible<br/>visibility_filter.rs:160-162"]
    DROP --> CPO["compute_private_opaque_ids, the position-stream drop set<br/>visibility_filter.rs:144-153, private_nodes from of_node at<br/>src/actors/graph_state_actor.rs:173-179 (VC-03.9)"]
    FE --> VS
    NIV --> VS["visible_subgraph(graph, caller, filter_on)<br/>visibility_filter.rs:171-232"]
    VS --> OFF["filter off: the same Arc, unchanged<br/>visibility_filter.rs:176-178"]
    VS --> SAME["nothing to remove, every node visible and every metadata row named:<br/>the same Arc, no copy<br/>visibility_filter.rs:185-196"]
    VS --> NEW["otherwise a new GraphData<br/>visibility_filter.rs:198-231"]
    NEW --> N1["nodes: visible only, page-reference keys scrubbed (VC-19.2)<br/>visibility_filter.rs:201-208"]
    NEW --> N2["edges: both endpoints visible<br/>visibility_filter.rs:209-214"]
    NEW --> N3["metadata and id_to_metadata: rows a visible node names, scrubbed<br/>an orphan row fails closed<br/>visibility_filter.rs:215-230"]
    INV["INVARIANT: of_node is the only reader of the public and owner_pubkey keys,<br/>so REST, initialGraphLoad and the stream cannot disagree (visibility_filter.rs:97-100)"]
    ON --- INV
    DR["DRIFT: the module doc still says the handler layer position_updates owns the env gate<br/>(visibility_filter.rs:4-5), but filter_enabled lives here (visibility_filter.rs:46)"]
    FE --- DR
    OA["OptionalAuth yields None for a bad or absent credential, so that caller is anonymous<br/>src/settings/auth_extractor.rs:87, graph_access.rs:18-22"]
    OA --> VS
    DEV["dev builds only: with DEV_AUTH_LOOPBACK=1 behind a same-host proxy any client is loopback<br/>and may name any pubkey, so the caller identity itself is forgeable<br/>src/utils/auth.rs:147, operator-accepted in ADR-2012, see VC-03.11"]
    DEV -.-> OA
```

**What it shows.** The single rule and the one function that applies it to a whole graph. **Why it is this way.** ADR-2003 makes the filter on by default and fail-closed. A shared `Arc` is returned whenever the caller loses nothing, so `gate_for_graph` uses `Arc::ptr_eq` to tell an ungated caller from a gated one (`../project/src/handlers/graph_access.rs:136`).

## VC-19.2 The PageNames scrub inside visible rows and nodes
```mermaid
flowchart TB
    PN["PageNames::new(graph, kept, kept_metadata_ids)<br/>visibility_filter.rs:246-279"] --> VIS["visible names: metadata_id and label of kept nodes,<br/>plus kept metadata keys<br/>visibility_filter.rs:247-256"]
    PN --> HID["hidden names: metadata_id, label, owl_class_iri, file_name, page_iri, class_iri<br/>of removed nodes, plus removed row keys and their file_name<br/>visibility_filter.rs:257-276"]
    VIS --> NORM["norm: trim, strip .md, lowercase<br/>visibility_filter.rs:242-244"]
    HID --> NORM
    NORM --> ISH["is_hidden: hidden and NOT also a visible name<br/>visibility_filter.rs:282-285"]
    ISH --> ROW["scrub_row on each kept Metadata row<br/>visibility_filter.rs:287-296"]
    ROW --> R1["topic_counts: keep only keys that are visible page names<br/>visibility_filter.rs:289-290"]
    ROW --> R2["is_subclass_of and belongs_to_domain: drop hidden names<br/>visibility_filter.rs:291-292"]
    ROW --> R3["definition: scrub_links<br/>visibility_filter.rs:293-295"]
    ISH --> SL["scrub_links: cut each wikilink whose page, before a hash, caret or pipe, is hidden,<br/>alias included, and an embed with its bang<br/>visibility_filter.rs:302-360, target at :311, embed at :316-320"]
    R3 --> SL
    ISH --> NODE["scrub_node on each kept node<br/>visibility_filter.rs:367-388"]
    NODE --> K0["node definition key: scrub_links<br/>visibility_filter.rs:368-371"]
    K0 --> SL
    NODE --> K1["node keys is_subclass_of and tags: split, drop hidden names, rejoin,<br/>remove the key when nothing is left<br/>visibility_filter.rs:372-387"]
    NODE --> K2["domain and source_domain are left alone: removing a value that shares<br/>a hidden page's name would itself reveal that page<br/>visibility_filter.rs:362-366"]
    T["tests hidden_page_names_are_scrubbed_from_visible_rows_and_nodes<br/>and a_definition_never_names_a_hidden_page<br/>visibility_filter.rs:806, :916"]
    K1 --- T
    SL --- T
```

**What it shows.** How a visible page stops naming a hidden one. **Why it is this way.** Security review M1 found visible metadata naming hidden pages (commit 395b62924). Tags were added and plain domain text removed in efec9b27d, because a value that disappears is itself a signal. The 2026-11-06 sweep added the `definition` wikilinks (SECURITY-profiles "Metadata scrub", `../project/docs/SECURITY-profiles.md:110`).

**Open (risk):** `scrub_links` only looks for `[[` (`../project/crates/visionclaw-domain/src/utils/visibility_filter.rs:305`), so a Markdown link such as `[text](Hidden.md)` in an authored definition still names a hidden page to a caller who cannot see it. Whether the corpus writes such links was not checked.

## VC-19.3 One projection for /wss initialGraphLoad and GET /api/graph/data
```mermaid
sequenceDiagram
    autonumber
    participant WS as send_full_state_sync<br/>src/handlers/socket_flow_handler/types.rs:177
    participant IGL as initial_graph_load_projection<br/>src/handlers/socket_flow_handler/types.rs:492
    participant REST as get_graph_data<br/>src/handlers/api_handler/graph/mod.rs:224
    participant GDP as graph_data_projection<br/>src/handlers/api_handler/graph/mod.rs:341
    participant VS as visible_subgraph<br/>crates/visionclaw-domain/src/utils/visibility_filter.rs:171
    participant T as visibility_parity_tests<br/>src/handlers/api_handler/graph/mod.rs:2403

    WS->>WS: caller_pubkey = session pubkey, visibility_filter_on (types.rs:183, :185)
    WS->>WS: state_sync counts are the session's view, visible_counts (types.rs:210-221)
    WS->>IGL: projection(graph, caller, filter_on, initial_node_limit) (types.rs:271-276)
    IGL->>VS: visible_subgraph(graph, caller, filter_on), scrub included, review N1 (types.rs:505-513)
    VS-->>IGL: the caller's view, scrubbed
    IGL->>IGL: sort by quality_score, take initial_node_limit, keep edges inside the set (types.rs:516-540)
    REST->>GDP: graph_data_projection(graph, query, caller, filter_enabled()) (graph/mod.rs:276-280)
    GDP->>VS: visible_subgraph first (graph/mod.rs:352-356)
    VS-->>GDP: the same view
    GDP->>GDP: then the population and linked-page filters, edges with both ends kept
    T->>IGL: wss_view (graph/mod.rs:2477)
    T->>GDP: rest_view (graph/mod.rs:2466)
    Note over T: anonymous, non-owner and owner callers must see the same node ids on both paths<br/>(graph/mod.rs:2601, :2606, :2611), and wss node metadata is scrubbed like REST (graph/mod.rs:2495)
    Note over IGL,GDP: INVARIANT: neither path reads the visibility keys itself, both call visible_subgraph<br/>(types.rs:509, graph/mod.rs:352). This is the ADR-2184 D1 precondition (graph/mod.rs:2403)
    Note over IGL,VS: DRIFT: PROTOCOL-registry.md:405 places visible_subgraph at visibility_filter.rs line 184 and,<br/>with ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:40,<br/>initial_graph_load_projection at types.rs line 488. They are at visibility_filter.rs:171 and types.rs:492
    Note over VS: TENSION: ids keep their gaps (visibility_filter.rs:201-208), so a gated load shows how many<br/>nodes are hidden. PROTOCOL records it as documented, not renumbered (PROTOCOL-registry.md:405)
```

**What it shows.** The two first-load paths, the session counts sent before them, and the test that holds them equal. **Why it is this way.** ADR-2184 builds the desktop node table from the REST load and refuses a second read path; its precondition now reads "met 2026-11-06" (`../project/docs/adr/ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:40`).

**Tension (parallel):** the ontology graph scrub renumbers node ids densely "so no gap marks where a node was" (`../project/src/services/ontology_visibility.rs:137-170`), while `visible_subgraph` keeps ids and their gaps. Two hidden-node policies give opposite answers on whether an id gap is a leak.

## VC-19.4 graph_access: one CallerGate, and every route that uses the module
```mermaid
flowchart TB
    subgraph GA["src/handlers/graph_access.rs"]
        CV["caller_view(graph, OptionalAuth), caller_view_for(graph, pubkey)<br/>graph_access.rs:23-28, :36-42"]
        CG["CallerGate: graph, view, VisibleIds, VisibleSlots<br/>graph_access.rs:82-87, newtypes :47, :53"]
        GFG["gate_for_graph: None when the view is the same Arc,<br/>else ids of the view, and slots whose node id equals the slot<br/>graph_access.rs:120-153, slot rule :140-146"]
        CGA["caller_gate: filter off is Ok None without fetching the graph,<br/>an unreadable graph is Err<br/>graph_access.rs:159-168"]
        IAS["id_at_slot: a position names a node only when visible, fail closed<br/>graph_access.rs:109-114"]
        VC["visible_counts: nodes, edges, metadata of the view<br/>graph_access.rs:224-236"]
        HC["caller_hidden_classes, the ontology deny-list, VC-19.7<br/>graph_access.rs:200-208"]
        GFG --> CG
        CGA --> GFG
        CG --> IAS
        CGA --> HC
    end
    subgraph RG["/api/graph, graph/mod.rs"]
        G1["/data/paginated and /refresh<br/>graph/mod.rs:464, :554"]
        G3["fetch_graph_view, view and a whole-graph flag<br/>graph/mod.rs:1155-1175"]
        G4["/positions ids, node relations, expand, query pattern<br/>graph/mod.rs:814, :1195, :1230, :1666"]
        G5["/fold: a gated plan never enters the shared memo<br/>fold.rs:580, :597"]
        G3 --> G4
        G3 --> G5
    end
    subgraph SW["gated by the sweep"]
        S1["GET /api/pages<br/>pages_handler.rs:84"]
        S2["insights and insights/realtime<br/>insights_handlers.rs:44, :183"]
        S3["hierarchy-levels<br/>semantic_forces.rs:68"]
        S4["GET /api/constraints/list<br/>constraints_handler.rs:269"]
        S5["refresh_graph and state_sync counts<br/>files/mod.rs:261, types.rs:210"]
    end
    subgraph OTH["other users"]
        O1["export and share, the caller's view<br/>graph_export_handler.rs:111"]
        O2["semantic-path, query-traversal, chunk-traversal<br/>semantic_pathfinding_handler.rs:46, :76, :108"]
        O3["cycles, schema, layout mode, mock agents<br/>cycles_handlers.rs:66, schema_handler.rs:116-117,<br/>layout_handler.rs:141, bots_visualization_handler.rs:360"]
        O4["analytics: community, anomaly, PageRank, SSSP, DBSCAN, VC-19.5<br/>analytics/mod.rs:78, :115, pagerank_handlers.rs:138,<br/>sssp_handlers.rs:229, clustering_handlers.rs:390"]
    end
    CV --> G1
    CV --> G3
    CV --> O1
    CV --> O2
    CV --> O3
    CGA --> S1
    CGA --> S3
    CGA --> S4
    GFG --> S2
    VC --> S5
    CGA --> O4
    DRS["DRIFT: SECURITY-profiles.md:105 places caller_gate at graph_access line 175 and<br/>SECURITY-profiles.md:107 places visible_counts at line 200, which is caller_hidden_classes (graph_access.rs:200).<br/>They are at graph_access.rs:159 and graph_access.rs:224"]
    CGA -.- DRS
```

**What it shows.** The module's one decision type and every route that runs a graph through it. **Why it is this way.** The sweep replaced four per-route "is anything hidden" helpers with `CallerGate`; ids and positions are separate types because they agree only while the graph is compact (`../project/src/handlers/graph_access.rs:49-51`), pinned by `ids_and_positions_are_distinct_sets_on_a_non_compact_graph` (`graph_access.rs:269`).

**Debt (parallel):** three thin wrappers sit beside `caller_gate`: `gated_view`, which fetches the graph even with the filter off (`../project/src/handlers/api_handler/analytics/pathfinding.rs:456-478`), a private `caller_gate` with another signature (`../project/src/handlers/semantic_handler.rs:252-260`) and the `/positions` id set (`../project/src/handlers/api_handler/graph/mod.rs:812-815`). The graph itself is read three ways: through the supervisor (`../project/src/handlers/graph_access.rs:239-246`), straight from the graph state actor (`../project/src/handlers/api_handler/graph/mod.rs:1161-1171`) and from `graph_repository` in point-to-point (`../project/src/handlers/api_handler/analytics/pathfinding.rs:580`).

## VC-19.5 Gated analytics: computed over the view on the CPU, full access on the GPU
```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant R as analytics routes<br/>src/handlers/api_handler/analytics/mod.rs:71, :108
    participant GA as caller_gate<br/>src/handlers/graph_access.rs:159
    participant VA as visible_analytics<br/>src/handlers/api_handler/analytics/visible_analytics.rs:60
    participant DJ as single_source Dijkstra<br/>src/services/pathfinding.rs:180
    participant GR as GpuAnalyticsRoutes<br/>src/handlers/api_handler/analytics/gpu_routes.rs:24
    participant SPA as ShortestPathActor APSP<br/>src/actors/gpu/shortest_path_actor.rs:284

    C->>R: POST community/detect, pagerank/compute, pathfinding sssp, apsp, components, point-to-point
    R->>GA: caller_gate or gate_for_graph (analytics/mod.rs:78-82, pagerank_handlers.rs:138-142, analytics/pathfinding.rs:456-478)
    alt the view hides nodes
        R->>VA: visible_communities, Louvain local moving plus modularity over the view (community.rs:154-163, visible_analytics.rs:331-337)
        R->>VA: visible_pagerank over the view, keyed by visible id (pagerank_handlers.rs:144-157, visible_analytics.rs:408-416)
        R->>VA: visible_sssp, visible_components, visible_point_to_point (analytics/pathfinding.rs:195, :388, :703)
        VA->>DJ: shortest paths over the view only, an edge leaving the view is skipped (services/pathfinding.rs:176-180)
        R->>VA: visible_apsp, landmark rows k x n, at most 256 landmarks (analytics/pathfinding.rs:301, visible_analytics.rs:511-530)
        VA-->>C: visible ids only, a hidden SSSP source reads as past the end (visible_analytics.rs:66, analytics/pathfinding.rs:446-451)
    else full access, or the filter off
        R->>GR: community and PageRank through the GPU manager (community.rs:54-61, pagerank_handlers.rs:176, gpu_routes.rs:33-45)
        GR-->>C: no GPU manager means an error, never an empty answer (gpu_routes.rs:43-45, pagerank_handlers.rs:275-277)
        R->>SPA: APSP for a full-access caller
        SPA-->>C: refused for every caller, NFR-7 (shortest_path_actor.rs:287-299)
    end
    C->>R: POST anomaly/detect
    R->>R: GPU run, then gate_anomalies by VisibleSlots (analytics/mod.rs:121-133, visible_results.rs:14)
    Note over R: DEBT: anomalies are still filtered on the way out, LOF and Z-score over positions<br/>a hidden node helped lay out (visible_results.rs:1-7)
    Note over GR: INVARIANT: a gated caller never reaches the GPU actors or their PageRank cache,<br/>a cached result reads as nothing cached for them (pagerank_handlers.rs:197-214)
    Note over R,VA: DEBT: DBSCAN and semantic centrality are still whole-graph runs cut afterwards. gate_clusters<br/>recounts, but ids and membership were formed with hidden nodes (visible_analytics.rs:177-219)
    Note over R: OPEN RESOLVED at visionclaw 46edc196: pagerank/clear now takes the caller and only a power user may evict the whole-graph cache<br/>every full-access caller shares, anyone else gets a 404 (pagerank_handlers.rs:240-249, :260-262)
    Note over C,R: DEBT: the point-to-point comment still says a GPU run would be published to the shared stream<br/>(analytics/pathfinding.rs:697-698), but ComputeSSP.publish and node_sssp are gone
    Note over C,VA: INVARIANT: semantic routes filter analyser answers, sound only while every analysis returns empty,<br/>pinned by semantic_analyses_still_return_nothing (actix_semantic_adapter.rs:453-498)
```

**What it shows.** Which strategy each analytics route takes now, and where full-access requests go. **Why it is this way.** The sweep moved every per-node answer onto the view because filtering a whole-graph answer left modularity, community numbering and PageRank values shaped by hidden nodes (PROTOCOL 0.1.20). `gpu_routes` exists because the community route used to reach `ForceComputeActor`, which refused it, and PageRank read an address nothing set (`../project/src/handlers/api_handler/analytics/gpu_routes.rs:9-12`).

**Tension (parallel):** APSP gives a gated caller a CPU landmark answer (`../project/src/handlers/api_handler/analytics/pathfinding.rs:296-311`) while a full-access caller is refused by NFR-7 (`../project/src/actors/gpu/shortest_path_actor.rs:287-299`), so seeing less of the graph gets a caller more of the API.

**Open (risk):** every gated CPU answer (Louvain up to 10,000 iterations, PageRank up to 1,000, APSP up to 256 Dijkstra runs) runs inline on the actix worker, with no `spawn_blocking` or `web::block` anywhere in the analytics handlers (`../project/src/handlers/api_handler/analytics/community.rs:159-163`, `../project/src/handlers/api_handler/analytics/pagerank_handlers.rs:148-154`). On a large view one request can stall one of the four workers.

## VC-19.6 Not-found equals hidden: node reads, get_content and shares
```mermaid
flowchart TB
    subgraph NODE["node-scoped reads"]
        NR["GET /api/graph/node/id/relations and POST /node/id/expand<br/>graph/mod.rs:1188, :1220"] --> NV["the caller's view has no such id<br/>graph/mod.rs:1206, :1241"] --> NF["404 Node id not found, the same text for hidden and absent<br/>graph/mod.rs:1207-1208, :1242-1243"]
        SSP["POST /analytics/sssp/compute, a source outside gate.ids<br/>sssp_handlers.rs:229-239"] --> SNF["the same reply as a missing source, source_not_found<br/>sssp_handlers.rs:237"]
    end
    subgraph FILE["GET /api/files/get_content/filename, files/mod.rs:366"]
        F0["get_file_content, filter_enabled, caller pubkey<br/>files/mod.rs:141-152"] --> F1["path traversal: 400<br/>files/mod.rs:175-184, :204-213"]
        F1 --> F2["one path for absent and present: canonicalize, then read,<br/>None when either fails<br/>files/mod.rs:197-222"]
        F2 --> F3["page_is_visible runs on whatever was read, empty for an absent file<br/>files/mod.rs:223-224"]
        F3 --> F4["only a .md that vault_core parses with public: true, then node_is_visible<br/>files/mod.rs:233-246"]
        F4 -->|"absent or not visible"| F5["file_not_found, the one no-such-file reply<br/>files/mod.rs:156-161, :225-228"]
    end
    subgraph SHARE["/api/graph-export, RequireAuth on the scope, graph_export_handler.rs:357"]
        S1["POST /share stores the caller's view and the caller as creator<br/>graph_export_handler.rs:188-204"]
        S1A["the store refuses a share with no creator<br/>sqlite_share_repository.rs:110-115"]
        S2["DELETE /shared/id: delete_owned, recorded creator equals caller,<br/>a power user may delete_unowned a legacy share, else 404<br/>graph_export_handler.rs:313-325, sqlite_share_repository.rs:193-236"]
        S3["GET /shared/id: no creator check, the link, expiry, access claim and password gate it<br/>graph_export_handler.rs:217-260"]
        S1 --> S1A
        S1 --> S2
        S1 --> S3
    end
    VC4["the share store itself, expiry and backup are drawn in VC-04"]
    S1A -.- VC4
```

**What it shows.** Where a hidden thing answers exactly like a missing one, and where the caller is checked against a stored owner. **Why it is this way.** ADR-060's rule is that a hidden name and an unknown one must be indistinguishable (`../project/src/handlers/api_handler/files/mod.rs:154-155`). The sweep made a missing page take the same steps as a hidden one, because it used to answer before any read (`files/mod.rs:197-200`), and moved shares into one SQLite store so the creator check holds across workers.

**Open (risk):** GET of a share returns the creator's view, their private nodes included, to anyone authenticated who holds the link (`../project/src/handlers/graph_export_handler.rs:217-260`, view taken at `:111`). The link is the only key.

**Drift (SECURITY-profiles vs code):** `SECURITY-profiles.md:109` names the share deletion rule `graph_export_handler.rs` `may_delete_share_as`. No such function exists at this commit; the rule is `delete_owned` and `delete_unowned` (`../project/src/handlers/graph_export_handler.rs:315-325`).

## VC-19.7 The ontology deny-list: a hidden class reads as one never stored
```mermaid
flowchart TB
    REQ["an ontology read: /api/ontology GETs, query and sparql,<br/>or /api/ontology-agent discover, read, query, traverse, validate, check<br/>ontology/mod.rs:1846-1869, ontology_agent_handler.rs:470-476"] --> CHC
    CHC["caller_hidden_classes(supervisor, caller)<br/>graph_access.rs:200-208"] --> CHI["caller_hidden_class_iris: caller_gate, then hidden_class_iris,<br/>empty when nothing is hidden<br/>graph_access.rs:186-194"]
    CHI --> HCI["hidden_class_iris: owl_class_iri of every graph node outside the caller's ids<br/>graph_access.rs:174-181"]
    CHC --> F503["graph unreadable: 503, never the unfiltered store<br/>graph_access.rs:203-207"]
    HCI --> HC["HiddenClasses, an empty set is a pass-through<br/>ontology_visibility.rs:20, :35-37"]
    HC --> SC["class and classes: a hidden class is None or dropped,<br/>survivors lose parents, ADR-014 lists, other relationships,<br/>IRI-valued properties and domains naming one<br/>ontology_visibility.rs:60-108"]
    HC --> AX["axioms, inference results, Whelk pairs: drop any naming a hidden class<br/>ontology_visibility.rs:126-135"]
    HC --> GR["graph: drop hidden class nodes and their axiom records, renumber ids densely<br/>ontology_visibility.rs:141-193"]
    HC --> SP["sparql_json and query_rows: drop a row with a binding equal to a hidden IRI<br/>ontology_visibility.rs:196-224"]
    HC --> TR["triples_json, validation_report, provenance_chain<br/>ontology_visibility.rs:228-301"]
    SC --> USE["call sites: ontology_handler.rs:92, :173, :220, :1025, :1106,<br/>ontology/mod.rs:1088, :1741, ontology_agent_handler.rs:135, :200, :240"]
    T["tests: ontology_handler hidden_class_tests, ontology_agent_handler hidden_class_tests,<br/>graph_access the_deny_list_is_the_class_iris_of_nodes_outside_the_view<br/>ontology_handler.rs:1421, ontology_agent_handler.rs:488, graph_access.rs:383"]
    USE --- T
    RES["OPEN: SPARQL ASK, aggregates, queries that do not project the class and literal forms<br/>pass the row filter, which compares binding values to IRIs only (ontology_visibility.rs:204-211)"]
    SP -.- RES
    COST["DEBT: each read builds the caller's whole scrubbed view only to collect IRIs<br/>(graph_access.rs:190-193, :166-167), and a gated /graph re-reads every axiom (ontology_handler.rs:114-115)"]
    CHI -.- COST
```

**What it shows.** How a class whose page is private disappears from every ontology read for a caller who cannot see the page. **Why it is this way.** The vault admits a formal class page with `public: false`, so the store holds classes the graph hides; the operator chose "hidden reads as unknown" and accepted hierarchy gaps (SECURITY-profiles 0.1.6, `../project/docs/SECURITY-profiles.md:111`; PROTOCOL 0.1.22, `../project/docs/PROTOCOL-registry.md:9`).

**Open (risk):** SECURITY-profiles lists the residuals the deny-list does not catch (`../project/docs/SECURITY-profiles.md:111`): inferences computed over the whole store on routes outside per-view reasoning (the SPARQL routes over the inferred graph and the validator's inferred triples), store-wide totals in validation reports (left as computed, `../project/src/services/ontology_visibility.rs:266-268`), ontology nodes stored under a slug IRI with no `owl_class_iri` (`hidden_class_iris` reads only that field, `../project/src/handlers/graph_access.rs:179`), and `jev_nlq` and `natural_language_query_service`, which query the store outside these routes. The per-view reasoning of 0.1.8 has since closed the stored-run reads and the `/inferred` run id (`../project/docs/SECURITY-profiles.md:112`).

**Open (risk):** the deny-list is keyed by IRI, not by node. If a visible page and a hidden page carry the same `owl_class_iri`, the class is denied to a caller who can see one of them (`../project/src/handlers/graph_access.rs:174-181`). This fails closed, but nothing tests it.
