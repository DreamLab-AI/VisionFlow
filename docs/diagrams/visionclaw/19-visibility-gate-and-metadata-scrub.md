---
id: VC-19
title: The visibility gate shared by REST and /wss, the page-name scrub and gated analytics
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
  - ../project/src/handlers/api_handler/analytics/mod.rs
  - ../project/src/handlers/api_handler/analytics/community.rs
  - ../project/src/handlers/api_handler/analytics/pathfinding.rs
  - ../project/src/handlers/api_handler/analytics/visible_analytics.rs
  - ../project/src/handlers/api_handler/analytics/visible_results.rs
  - ../project/src/handlers/api_handler/analytics/pagerank_handlers.rs
  - ../project/src/handlers/api_handler/analytics/sssp_handlers.rs
  - ../project/src/handlers/api_handler/analytics/clustering_handlers.rs
  - ../project/src/handlers/api_handler/analytics/cycles_handlers.rs
  - ../project/src/handlers/semantic_handler.rs
  - ../project/src/handlers/semantic_pathfinding_handler.rs
  - ../project/src/handlers/schema_handler.rs
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/handlers/bots_visualization_handler.rs
  - ../project/src/adapters/actix_semantic_adapter.rs
  - ../project/src/actors/gpu/shortest_path_actor.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/main.rs
  - ../project/docs/PROTOCOL-registry.md
  - ../project/docs/SECURITY-profiles.md
  - ../project/docs/adr/ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md
verified_commit: {visionclaw: ec785afa9331be1fa049cee6282a4dcda16ee86d}
---

## For developers

Every read that returns node identities applies one visibility policy, whether it is REST, the `/wss` initial load or an analytics answer. `NodeVisibility::of_node` is the only reader of the `public` and `owner_pubkey` metadata keys. `node_is_visible` applies the rule: keep a node that is public or owned by the caller, drop everything else. `visible_subgraph` builds the caller's graph from that rule: visible nodes, edges whose two endpoints are both visible, metadata rows a visible node names, and visible metadata with hidden page names scrubbed out (VC-19.1, VC-19.2). `graph_access` wraps the policy for handlers (VC-19.4). The `/wss` `initialGraphLoad` and `GET /api/graph/data` are pinned to the same output for three callers (VC-19.3). The policy's default-on, fail-closed parse comes from ADR-2003. Closing the REST gap was the ADR-2184 precondition, recorded in PROTOCOL 0.1.17 and 0.1.18. Analytics actors compute over the whole graph and hold no per-caller state. A gated caller either gets a CPU answer computed over their own view (SSSP, components, point-to-point) or a whole-graph answer filtered on the way out (communities, anomalies, PageRank, DBSCAN) (VC-19.5). What filtering on the way out still leaks, and the other gates (identical not-found replies, `get_content`, share deletion), are in VC-19.6. The live position stream applies the same rule through a drop set (VC-03.9).

## For the business

A visitor who is not signed in sees only public pages. A signed-in user also sees their own private pages, and never anyone else's. The same rule applies to the graph view, to every analysis the server runs, to file contents and to shared snapshots. A private page someone else owns looks exactly like a page that does not exist. Some numbers that describe the whole graph, such as overall community scores, still include private pages. The security review accepted that.

## VC-19.1 The policy core: one reader of the visibility keys, one subgraph
```mermaid
flowchart TB
    ENV["PUBKEY_VISIBILITY_FILTER<br/>visibility_filter.rs:31"] --> PARSE["parse_visibility_flag: absent or unrecognised means ON<br/>only 0, false, off or no disables<br/>visibility_filter.rs:34-42"]
    PARSE --> FE["filter_enabled, read once through OnceLock<br/>visibility_filter.rs:46-51"]
    ON["NodeVisibility::of_node: public iff metadata public equals true,<br/>owner_pubkey read only for a private node<br/>visibility_filter.rs:101-115"] --> DROP["is_dropped_for: public keeps,<br/>caller equal to owner keeps, anything else drops<br/>visibility_filter.rs:127-137"]
    DROP --> NIV["node_is_visible<br/>visibility_filter.rs:173-175"]
    DROP --> CPO["compute_private_opaque_ids, the position-stream drop set<br/>visibility_filter.rs:144-153, private_nodes from of_node at<br/>src/actors/graph_state_actor.rs:173-179 (VC-03.9)"]
    FE --> VS
    NIV --> VS["visible_subgraph(graph, caller, filter_on)<br/>visibility_filter.rs:184-245"]
    VS --> OFF["filter off: the same Arc, unchanged<br/>visibility_filter.rs:189-191"]
    VS --> SAME["nothing to remove, every node visible and every metadata row named:<br/>the same Arc, no copy<br/>visibility_filter.rs:198-209"]
    VS --> NEW["otherwise a new GraphData<br/>visibility_filter.rs:211-244"]
    NEW --> N1["nodes: visible only, page-reference keys scrubbed (VC-19.2)<br/>visibility_filter.rs:214-221"]
    NEW --> N2["edges: both endpoints visible<br/>visibility_filter.rs:222-227"]
    NEW --> N3["metadata and id_to_metadata: rows a visible node names, scrubbed<br/>an orphan row fails closed<br/>visibility_filter.rs:228-243"]
    INV["INVARIANT: of_node is the only reader of the public and owner_pubkey keys,<br/>so REST, initialGraphLoad and the stream cannot disagree (visibility_filter.rs:97-100)"]
    ON --- INV
    DR["DRIFT: the module doc still says the handler layer position_updates owns the env gate<br/>(visibility_filter.rs:4-5), but filter_enabled lives here (visibility_filter.rs:46)"]
    FE --- DR
    OA["OptionalAuth yields None for a bad or absent credential, so that caller is anonymous<br/>src/settings/auth_extractor.rs:87, graph_access.rs:18-22"]
    OA --> VS
    DEV["dev builds only: with DEV_AUTH_LOOPBACK=1 behind a same-host proxy any client is loopback<br/>and may name any pubkey, so the caller identity itself is forgeable<br/>src/utils/auth.rs:147, operator-accepted in ADR-2012, see VC-03.11"]
    DEV -.-> OA
```

**What it shows.** The single rule and the one function that applies it to a whole graph. **Why it is this way.** ADR-2003 makes the filter on by default and fail-closed. A shared `Arc` is returned whenever the caller loses nothing, so callers can use `Arc::ptr_eq` to tell an ungated caller from a gated one (`../project/src/handlers/graph_access.rs:86`).

**Drift (SECURITY-profiles vs code):** `SECURITY-profiles.md:41` and `:96` place `parse_visibility_flag` at `visibility_filter.rs:26` and `filter_enabled` at `:38`, within `:23-43`. At this commit they are at `:34` and `:46`, and the env constant is at `:31`.

## VC-19.2 The PageNames scrub inside visible rows and nodes
```mermaid
flowchart TB
    PN["PageNames::new(graph, kept, kept_metadata_ids)<br/>visibility_filter.rs:259-292"] --> VIS["visible names: metadata_id and label of kept nodes,<br/>plus kept metadata keys<br/>visibility_filter.rs:260-269"]
    PN --> HID["hidden names: metadata_id, label, owl_class_iri, file_name, page_iri, class_iri<br/>of removed nodes, plus removed row keys and their file_name<br/>visibility_filter.rs:270-289"]
    VIS --> NORM["norm: trim, strip .md, lowercase<br/>visibility_filter.rs:255-257"]
    HID --> NORM
    NORM --> ISH["is_hidden: hidden and NOT also a visible name<br/>visibility_filter.rs:295-298"]
    ISH --> ROW["scrub_row on each kept Metadata row<br/>visibility_filter.rs:300-306"]
    ROW --> R1["topic_counts: keep only keys that are visible page names<br/>visibility_filter.rs:302-303"]
    ROW --> R2["is_subclass_of and belongs_to_domain: drop hidden names<br/>visibility_filter.rs:304-305"]
    ISH --> NODE["scrub_node on each kept node<br/>visibility_filter.rs:312-329"]
    NODE --> K1["node keys is_subclass_of and tags only: split, drop hidden names, rejoin,<br/>remove the key when nothing is left<br/>visibility_filter.rs:313-327"]
    NODE --> K2["domain and source_domain are left alone: removing a value that shares<br/>a hidden page's name would itself reveal that page<br/>visibility_filter.rs:308-311"]
    T["test hidden_page_names_are_scrubbed_from_visible_rows_and_nodes<br/>visibility_filter.rs:739"]
    K1 --- T
```

**What it shows.** How a visible page stops naming a hidden one. **Why it is this way.** Security review M1 found visible metadata naming hidden pages (commit 395b62924). Tags were added and plain domain text removed in efec9b27d, because a value that disappears is itself a signal. PROTOCOL 0.1.18 records the scrubbed fields (`../project/docs/PROTOCOL-registry.md:8`).

**Open (risk):** `scrub_row` leaves the rest of a visible row as it is, including the free-text `definition`, `owl_class` and `hyperlink_count` (`../project/crates/visionclaw-domain/src/models/metadata.rs:62`, `:46`, `:15`). An authored definition that links a hidden page, or a link count that includes one, still reaches a caller who cannot see that page (`visibility_filter.rs:300-306`).

## VC-19.3 One projection for /wss initialGraphLoad and GET /api/graph/data
```mermaid
sequenceDiagram
    autonumber
    participant WS as send_full_state_sync<br/>src/handlers/socket_flow_handler/types.rs:177
    participant IGL as initial_graph_load_projection<br/>src/handlers/socket_flow_handler/types.rs:488
    participant REST as get_graph_data<br/>src/handlers/api_handler/graph/mod.rs:218
    participant GDP as graph_data_projection<br/>src/handlers/api_handler/graph/mod.rs:306
    participant VS as visible_subgraph<br/>crates/visionclaw-domain/src/utils/visibility_filter.rs:184
    participant T as visibility_parity_tests<br/>src/handlers/api_handler/graph/mod.rs:2312-2316

    WS->>WS: caller_pubkey = session pubkey, visibility_filter_on (types.rs:183, :185)
    WS->>IGL: projection(graph, caller, filter_on, initial_node_limit) (types.rs:267-272)
    IGL->>VS: visible_subgraph(graph, caller, filter_on) (types.rs:505-509)
    VS-->>IGL: the caller's view, scrubbed
    IGL->>IGL: sort by quality_score, take initial_node_limit, keep edges inside the set (types.rs:516-535)
    REST->>GDP: graph_data_projection(graph, query, caller, filter_enabled()) (graph/mod.rs:259-263)
    GDP->>VS: visible_subgraph first (graph/mod.rs:316-320)
    VS-->>GDP: the same view
    GDP->>GDP: then the population and linked-page filters, edges with both ends kept
    T->>IGL: wss_view (graph/mod.rs:2387)
    T->>GDP: rest_view (graph/mod.rs:2377)
    Note over T: anonymous, non-owner and owner callers must see the same node ids on both paths<br/>(graph/mod.rs:2448, :2453, :2458), and wss node metadata is scrubbed like REST (graph/mod.rs:2405)
    Note over IGL,GDP: INVARIANT: neither path reads the visibility keys itself, both call visible_subgraph<br/>(types.rs:505, graph/mod.rs:316). This is the ADR-2184 D1 precondition (graph/mod.rs:2312)
```

**What it shows.** The two first-load paths and the test that holds them equal. **Why it is this way.** ADR-2184 builds the desktop node table from the REST load and refuses a second read path. PROTOCOL 0.1.17 records that `/api/graph` had no gate until this lane (`../project/docs/PROTOCOL-registry.md:9`).

**Drift (ADR-2184 vs code):** `ADR-2184-a-client-side-columnar-node-table-drives-visual-encoding.md:40` still says no visibility gate exists in REST `/api/graph/data`, and cites the `/wss` gate at `position_updates.rs:64-72`. At this commit the REST read is gated (`../project/src/handlers/api_handler/graph/mod.rs:316`), and the `/wss` gate is `types.rs:505`.

**Tension (risk):** node ids stay compact `0..N-1` with gaps where nodes are hidden, so the gaps in a gated `/api/graph/data` or `initialGraphLoad` reveal how many nodes are hidden. PROTOCOL records this residual as documented, not renumbered (`../project/docs/PROTOCOL-registry.md:401`), and `visible_subgraph` keeps each node's own id (`visibility_filter.rs:214-221`).

## VC-19.4 graph_access: the handler-facing helpers and every route that uses them
```mermaid
flowchart TB
    subgraph GA["src/handlers/graph_access.rs"]
        CV["caller_view(graph, OptionalAuth)<br/>graph_access.rs:23-28"] --> CVF["caller_view_for(graph, pubkey)<br/>graph_access.rs:31-37"]
        VP["visible_positions: a position counts only when its node is in the view<br/>AND its id equals its position, so drift fails closed<br/>graph_access.rs:43-52"]
        SS["stream_safe: true only when the anonymous view hides nothing<br/>graph_access.rs:58-70"]
        CG["caller_gate: filter off is Ok None without fetching the graph,<br/>ptr_eq is Ok None, else Some of view and visible positions,<br/>an unreadable graph is Err<br/>graph_access.rs:77-91"]
    end
    subgraph RG["/api/graph, graph/mod.rs"]
        G1["/data/paginated, counts after the gate<br/>graph/mod.rs:403-405"]
        G2["/refresh<br/>graph/mod.rs:495"]
        G3["fetch_graph_view, returns view and a whole-graph flag<br/>graph/mod.rs:1057-1077"]
        G4["/positions, /node/id/relations, /node/id/expand, /query/pattern<br/>graph/mod.rs:678, :1107, :1142, :1578"]
        G5["/fold: a gated plan never enters the shared memo<br/>fold.rs:580, :603-605"]
    end
    subgraph OTH["outside /api/graph"]
        O1["graph export and share<br/>graph_export_handler.rs:109"]
        O2["semantic-path, query-traversal, chunk-traversal<br/>semantic_pathfinding_handler.rs:46, :76, :108"]
        O3["analytics /cycles<br/>cycles_handlers.rs:66"]
        O4["schema: the caller's view, and the anonymous view for the shared cache<br/>schema_handler.rs:111-118"]
        O5["POST /api/layout/mode placement<br/>layout_handler.rs:141"]
        O6["mock agent injection<br/>bots_visualization_handler.rs:360"]
    end
    CVF --> G1
    CVF --> G2
    CVF --> G3
    G3 --> G4
    G3 --> G5
    CV --> O1
    CV --> O2
    CV --> O3
    CVF --> O4
    CVF --> O5
    CVF --> O6
    CG --> AN["analytics gates, VC-19.5"]
    VP --> AN
    SS --> AN
```

**What it shows.** The four helpers, and every route that runs a graph through them before answering. **Why it is this way.** One wrapper over `visible_subgraph` means no handler reads the visibility keys itself. `caller_gate` skips the graph fetch when the filter is off, so ungated deployments pay nothing (`graph_access.rs:81-83`).

**Debt (parallel):** the "is anything hidden from this caller" decision is written four times. It is `caller_gate` (`../project/src/handlers/graph_access.rs:77-91`), `gated_view` (`../project/src/handlers/api_handler/analytics/pathfinding.rs:408-436`), `visible_ids` (`../project/src/handlers/semantic_handler.rs:245-257`) and `caller_clusters` (`../project/src/handlers/api_handler/analytics/clustering_handlers.rs:381-403`). The last one keys its visible set by node id (`:399`), while the other three use `visible_positions`.

**Drift (PROTOCOL vs code):** `PROTOCOL-registry.md:400` places `caller_view` at `graph/mod.rs:1076`. The wrapper is at `graph/mod.rs:1087`, and the line before it is a call (`:1074`).

## VC-19.5 Gated analytics shapes: compute over the view, or filter on the way out
```mermaid
sequenceDiagram
    autonumber
    participant C as gated caller
    participant PF as compute_sssp and compute_connected_components<br/>src/handlers/api_handler/analytics/pathfinding.rs:157, :324
    participant VA as visible_sssp and visible_components<br/>src/handlers/api_handler/analytics/visible_analytics.rs:78, :159
    participant GPU as ShortestPathActor<br/>src/actors/gpu/shortest_path_actor.rs:356
    participant AM as community, anomaly and PageRank routes<br/>src/handlers/api_handler/analytics/mod.rs:70, :112
    participant VR as gate_communities, gate_pagerank, gate_anomalies<br/>src/handlers/api_handler/analytics/visible_results.rs:20, :43, :81

    C->>PF: POST /analytics/pathfinding/sssp
    PF->>PF: gated_view (pathfinding.rs:167)
    alt source not visible
        PF-->>C: the same error as a source past the end of the graph (pathfinding.rs:169-174, :396-400)
    else
        PF->>VA: CPU Dijkstra over visible adjacency only (pathfinding.rs:179-184)
        VA-->>C: SsspPayload Visible, distances keyed by visible id, unreachable absent (pathfinding.rs:68, visible_analytics.rs:65)
    end
    C->>PF: POST /analytics/pathfinding/connected-components
    PF->>VA: visible_components (pathfinding.rs:333-338)
    VA-->>C: labels keyed by visible id, numbered densely by smallest member (visible_analytics.rs:181-188)
    Note over PF,GPU: an ungated caller reaches the GPU actor, which writes node_sssp only when publish is set,<br/>and publish = stream_safe(full) (pathfinding.rs:202, shortest_path_actor.rs:317)
    Note over GPU: APSP is refused for every caller, NFR-7 (shortest_path_actor.rs:359-371), pinned by apsp_verdict_tests (:389)
    C->>AM: POST /analytics/community/detect
    AM->>AM: caller_gate (analytics/mod.rs:77-81), then the whole-graph GPU run (analytics/mod.rs:83)
    AM->>VR: gate_communities drops hidden members, recounts size, recentres (analytics/mod.rs:86-89, visible_results.rs:27-33)
    Note over AM,VR: DEBT: modularity and each modularity_contribution stay whole-graph values<br/>(community.rs:169-173, analytics/mod.rs:89 replaces only communities and total), and labels keep<br/>the whole-graph community numbers, Community N (community.rs:179)
    C->>AM: POST /analytics/pagerank/compute
    AM->>VR: gate_pagerank: hidden positions become NaN, serialised as null, stats and top-k over visible values (visible_results.rs:45-75)
    Note over VR: DEBT: the visible values themselves were computed with hidden nodes in the graph, so<br/>a hidden node's influence shows in its visible neighbours' scores (visible_results.rs:1-8)
```

**What it shows.** The two gating strategies and which routes use which. **Why it is this way.** Security review item 5 (805dc6c8f) and 84ccd9aaa found GPU answers that gave hidden ids or joined visible nodes through hidden ones. Path-like answers are recomputed over the view. Community, anomaly and PageRank actors are shared and keep no per-caller state, so their answers are filtered on the way out (`../project/src/handlers/api_handler/analytics/pagerank_handlers.rs:93-105`).

**Tension (risk):** filtering on the way out leaves structure computed over the whole graph in a gated answer: global modularity, community numbering and PageRank values for visible nodes (`../project/src/handlers/api_handler/analytics/community.rs:169-179`, `../project/src/handlers/api_handler/analytics/visible_results.rs:43-48`). The commits record these as review follow-ups, but no record found at this commit marks them as accepted.

**Invariant:** the semantic routes filter the analyser's answers per caller, which is sound only while every analysis returns empty. `semantic_analyses_still_return_nothing_so_response_filtering_is_sound` fails as soon as one returns data (`../project/src/adapters/actix_semantic_adapter.rs:453-498`).

## VC-19.6 Not-found equals hidden: node reads, get_content and shares
```mermaid
flowchart TB
    subgraph NODE["node-scoped reads"]
        NR["GET /api/graph/node/id/relations and POST /node/id/expand<br/>graph/mod.rs:1100, :1132"] --> NV["the caller's view has no such id<br/>graph/mod.rs:1118, :1153"] --> NF["404 Node id not found, the same text for hidden and absent<br/>graph/mod.rs:1119-1120, :1154-1155"]
        SSP["POST /analytics/sssp/compute, hidden source<br/>sssp_handlers.rs:226-237"] --> SNF["the same reply as a missing source, source_not_found"]
    end
    subgraph FILE["GET /api/files/get_content/filename, files/mod.rs:358"]
        F0["get_file_content, filter_enabled, caller pubkey<br/>files/mod.rs:141-152"] --> F1["path traversal: 400<br/>files/mod.rs:175-184, :206-215"]
        F1 --> F2["canonicalize fails, file absent: file_not_found<br/>files/mod.rs:197-202"]
        F1 --> F3["read_to_string, then page_is_visible<br/>files/mod.rs:217-221"]
        F3 --> F4["only a .md that vault_core parses with public: true, then node_is_visible<br/>files/mod.rs:233-246"]
        F4 -->|"not visible"| F5["file_not_found, the one no-such-file reply<br/>files/mod.rs:154-161"]
    end
    subgraph SHARE["/api/graph-export, RequireAuth on the scope, graph_export_handler.rs:350"]
        S1["POST /share stores the caller's view and records creator_id<br/>graph_export_handler.rs:180-193"]
        S2["DELETE /shared/id: may_delete_share, recorded creator equals caller,<br/>else 404 Shared graph not found<br/>graph_export_handler.rs:294-299, :316-319, :333"]
        S3["GET /shared/id: no creator check, the link, expiry, access limit and password gate it<br/>graph_export_handler.rs:220-246"]
        S1 --> S2
        S1 --> S3
    end
    TIMING["OPEN: an absent file returns at canonicalize (files/mod.rs:197-202),<br/>a hidden one only after a read and a parse (files/mod.rs:217-220),<br/>so response time can tell hidden from absent"]
    F2 -.- TIMING
    F5 -.- TIMING
    VC4["whole-graph counts that bypass the view, POST /api/files/refresh_graph<br/>and the /positions bounds, are drawn in VC-04"]
```

**What it shows.** Where a hidden thing answers exactly like a missing one, and where the caller is checked against a stored owner. **Why it is this way.** ADR-060's rule is that a hidden name and an unknown one must be indistinguishable (`../project/src/handlers/api_handler/files/mod.rs:154-155`). Review L1 added the share creator (95ca5cb80), and `get_content` was gated in 0489303c7.

**Open (risk):** `GraphExportHandler::new` runs inside `configure_routes` (`../project/src/handlers/graph_export_handler.rs:343-344`), which runs once per actix worker (`../project/src/main.rs:1204`, four workers at `:1276`). Each worker therefore has its own in-memory share map (`graph_export_handler.rs:41`). On source evidence, a share created on one worker answers 404 to GET and DELETE served by another worker, and every share is lost on restart.

**Open (risk):** a share with no recorded `creator_id` can never be deleted (`../project/src/handlers/graph_export_handler.rs:298`). The scope requires authentication (`:350`), so only shares created before the creator was recorded have no creator. GET of a share returns the creator's view, their private nodes included, to anyone authenticated who holds the link (`:220-246`).
