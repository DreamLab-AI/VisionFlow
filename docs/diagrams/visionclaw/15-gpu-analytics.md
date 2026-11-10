---
id: VC-15
title: GPU analytics kernels and pathfinding
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2007, ADR-2053, ADR-2054, ADR-2061, ADR-2003, ADR-2184]
sources:
  - ../project/src/actors/gpu/clustering_actor.rs
  - ../project/src/actors/gpu/analytics_supervisor.rs
  - ../project/src/actors/gpu/anomaly_detection_actor.rs
  - ../project/src/actors/gpu/pagerank_actor.rs
  - ../project/src/actors/gpu/shortest_path_actor.rs
  - ../project/src/actors/gpu/connected_components_actor.rs
  - ../project/src/actors/gpu/graph_analytics_supervisor.rs
  - ../project/src/actors/gpu/gpu_manager_actor.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/handlers/api_handler/analytics/mod.rs
  - ../project/src/handlers/api_handler/analytics/clustering_handlers.rs
  - ../project/src/handlers/api_handler/analytics/clustering.rs
  - ../project/src/handlers/api_handler/analytics/real_gpu_functions.rs
  - ../project/src/handlers/api_handler/analytics/community.rs
  - ../project/src/handlers/api_handler/analytics/anomaly.rs
  - ../project/src/handlers/api_handler/analytics/pagerank_handlers.rs
  - ../project/src/handlers/api_handler/analytics/sssp_handlers.rs
  - ../project/src/handlers/api_handler/analytics/pathfinding.rs
  - ../project/src/handlers/api_handler/analytics/gpu_routes.rs
  - ../project/src/services/pathfinding.rs
  - ../project/src/handlers/semantic_pathfinding_handler.rs
  - ../project/src/services/semantic_pathfinding_service.rs
  - ../project/src/gpu/visual_analytics.rs
  - ../project/src/gpu/mod.rs
  - ../project/src/utils/gpu_safety.rs
  - ../project/crates/visionclaw-analytics-oracle/src/lib.rs
  - ../project/crates/visionclaw-gpu/src/cuda_sources/gpu_clustering_kernels.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/pagerank.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/gpu_landmark_apsp.cu
  - ../project/docs/GPU-wire-abi.md
  - ../project/src/main.rs
  - ../project/src/handlers/api_handler/analytics/params_handlers.rs
  - ../project/src/handlers/api_handler/analytics/types.rs
  - ../project/src/utils/unified_gpu_compute/sssp.rs
  - ../project/src/handlers/api_handler/analytics/visible_analytics.rs
  - ../project/src/handlers/api_handler/analytics/visible_results.rs
  - ../project/src/handlers/graph_access.rs
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

The analytics routes: clustering, community, anomaly, PageRank, shortest paths and components. At 809a859aa every gated analytics route takes its decision from one `CallerGate`; point-to-point A*, bidirectional and semantic paths run over `caller_view` instead (`analytics/pathfinding.rs:557`). A caller whose view hides nodes is answered on the CPU over that view for communities, PageRank, SSSP, APSP and components; full-access callers reach the GPU actors through `GPUManagerActor`. Anomalies and DBSCAN clusters are still filtered after a whole-graph run, and no SSSP run is written to the wire any more.

## For the business

These are the graph insights a user can ask for: clusters, important nodes, paths between ideas. A user who cannot see a private node does not learn of it through an analytics answer, and community and importance scores are now worked out from what that user can see. Some routes still answer with stubs, and the register lists them.

## VC-15.1 POST /analytics/clustering/run — spectral/kmeans/louvain/default dispatch with CPU fallback

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:203
    participant RC as run_clustering<br/>clustering_handlers.rs:20
    participant PC as perform_clustering<br/>clustering_handlers.rs:407
    participant RG as real_gpu_functions<br/>real_gpu_functions.rs:100-269
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:444
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:680
    participant CA as ClusteringActor<br/>clustering_actor.rs:1288

    C->>RT: POST /analytics/clustering/run
    RT->>RC: run_clustering(ClusteringRequest)
    RC->>RC: CLUSTERING_TASKS.insert(task_id, status=running) :33-46
    RC->>PC: tokio::spawn perform_clustering :52-53
    RC-->>C: 200 ClusteringResponse task_id, clusters none (clustering_handlers.rs:91-98)
    PC->>PC: graph_service_addr.send(GetGraphData) :415
    PC->>PC: mcp_client.query_agent_list() :429, an MCP failure yields an empty agent list :437-443
    alt request.method == spectral
        PC->>RG: perform_gpu_spectral_clustering :448
    else request.method == kmeans
        PC->>RG: perform_gpu_kmeans_clustering :451
    else request.method == louvain
        PC->>RG: perform_gpu_louvain_clustering :454
    else any other method, dbscan included
        PC->>RG: perform_gpu_default_clustering :456
        Note right of RG: node_count under 100 kmeans<br/>100 to 999 spectral<br/>else louvain (real_gpu_functions.rs:260-268)
    end
    RG->>GM: PerformGPUClustering{method,params,task_id}<br/>via gpu_manager_addr (real_gpu_functions.rs:111-122)
    GM->>AS: PerformGPUClustering forward (gpu_manager_actor.rs:444-465)
    AS->>CA: PerformGPUClustering forward (analytics_supervisor.rs:686-704)
    alt method == dbscan
        CA->>CA: perform_dbscan_clustering :515<br/>unreachable from this route, real_gpu_functions sends<br/>spectral, kmeans or louvain only
    else method in [louvain,leiden,communities]
        CA->>CA: perform_community_detection :332<br/>louvain un-refined, leiden and communities refined<br/>(clustering_actor.rs:1317-1326)
    else
        CA->>CA: perform_kmeans_clustering :175<br/>kmeans and spectral share the k-means path<br/>(clustering_actor.rs:1359-1368)
    end
    alt gpu_manager_addr present and GPU call Ok
        CA-->>AS: Ok(cluster list)
        AS-->>GM: Ok(cluster list)
        GM-->>RG: Ok(cluster list)
        RG-->>PC: clusters
    else gpu_manager_addr absent OR GPU call Err
        RG->>RG: generate_cpu_fallback_clustering :271-293
        alt agents non-empty
            RG->>RG: generate_agent_based_clusters (clustering_handlers.rs:468)
        else
            RG->>RG: generate_label_propagation_clusters (real topology CPU fallback, real_gpu_functions.rs:290)
        end
    end
    PC->>GM: WriteClusterAnalytics{clusters} do_send (clustering_handlers.rs:69)
    GM->>AS: WriteClusterAnalytics forward (gpu_manager_actor.rs:984)
    AS->>CA: WriteClusterAnalytics (clustering_actor.rs:1222)
    CA->>CA: write_cluster_id_from_assignments (ADR-031 D3 single writer)
    PC->>PC: CLUSTERING_TASKS[task_id].status = completed :79-81
    Note over PC,RG: DEAD CODE - clustering.rs:6 perform_clustering, a second PerformGPUClustering-only<br/>dispatcher with validate_clustering_params, has no caller anywhere in the tree<br/>clustering_handlers.rs:407 is the live perform_clustering
```

## VC-15.2 POST /analytics/clustering/dbscan — direct RunDBSCAN, GPU-only, precondition refusals

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:206
    participant RD as run_dbscan_clustering<br/>clustering_handlers.rs:226
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:398
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:544
    participant CA as ClusteringActor<br/>clustering_actor.rs:1193

    C->>RT: POST /analytics/clustering/dbscan {epsilon,minPoints}
    RT->>RD: run_dbscan_clustering(body)
    alt epsilon <= 0.0
        RD-->>C: 200 success=false epsilon must be positive :246-251
    else minPoints == 0
        RD-->>C: 200 success=false minPoints must be at least 1 :252-257
    else gpu_manager_addr is None
        RD-->>C: 200 success=false GPU compute not available :371-374
    else
        RD->>GM: RunDBSCAN{epsilon,min_points} (clustering_handlers.rs:266)
        GM->>AS: RunDBSCAN forward (gpu_manager_actor.rs:398-418)
        AS->>CA: RunDBSCAN forward (analytics_supervisor.rs:544-572)
        CA->>CA: perform_dbscan_clustering :515
        alt epsilon not finite or <= 0.0
            CA-->>AS: Err epsilon must be finite and > 0 :536-541
        else min_points == 0
            CA-->>AS: Err min_points must be >= 1 :542-544
        else position_clustering_refusal (MIN_NODES_FOR_CLUSTERING=3, MIN_POSITION_SPREAD=1e-3)
            CA-->>AS: Err DBSCAN clustering refused :560-564
        else
            CA->>CA: unified_compute.run_dbscan_clustering(epsilon,min_points) :568
            CA->>CA: record_execution(Dbscan, Gpu) :576
            CA->>CA: write_cluster_id_from_assignments (node_analytics single writer) :638-655
            CA-->>AS: Ok(DBSCANResult)
        end
        AS-->>GM: propagate result
        GM-->>RD: propagate result
        alt Ok
            RD->>RD: CLUSTERING_TASKS.insert(task_id, completed) :279-291
            RD->>RD: caller_clusters narrows the stored full result to the caller's view and recounts the stats, no graph means success=false (clustering_handlers.rs:293-302, :384-403)
            RD-->>C: 200 success=true clusters, stats, gpuAccelerated=true
        else Err(e)
            RD-->>C: 200 success=false error (clustering_handlers.rs:354-361)
        end
    end
```

## VC-15.3 POST /analytics/community/detect — full access reaches ClusteringActor, a gated caller is answered on the CPU over its view

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:207
    participant RCD as run_community_detection<br/>analytics/mod.rs:71
    participant VC as visible_community_response<br/>community.rs:154
    participant GC as run_community_detection_via<br/>community.rs:66
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:375
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:520
    participant CA as ClusteringActor<br/>clustering_actor.rs:1176

    C->>RT: POST /analytics/community/detect {algorithm}
    RT->>RCD: run_community_detection(request), AuthenticatedUser extractor analytics/mod.rs:72
    RCD->>RCD: caller_gate for the caller's pubkey (analytics/mod.rs:78-82)
    alt Err, the graph cannot be read
        RCD-->>C: 500 success=false communities=[] modularity=0.0 (analytics/mod.rs:93-104)
    else Ok(Some(gate)), the view hides nodes
        RCD->>VC: visible_community_response(gate, request) analytics/mod.rs:86
        VC->>VC: visible_communities, Louvain local moving over the view, iterations clamped 1 to 10000 (community.rs:159-163)
        Note over VC: INVARIANT a gated caller never reaches the whole-graph GPU run, so no grouping, label or modularity is shaped by a hidden node<br/>analytics/mod.rs:83-86, visible_analytics.rs:331
        VC-->>RCD: algorithm louvain_local_moving, executionPath CpuVisibleSubgraph (community.rs:171-176)
    else Ok(None), nothing hidden from this caller
        RCD->>GC: run_gpu_community_detection, GpuAnalyticsRoutes.of(app_state) (community.rs:54-60)
        alt gpu_manager_addr is None
            GC-->>RCD: Err GPU analytics not available (community.rs:58-59)
        else cfg not(feature = gpu)
            GC-->>RCD: Err GPU features not enabled in this build (community.rs:301-307)
        else unsupported algorithm string
            GC-->>RCD: Err Unsupported community detection algorithm (community.rs:80-85)
        else
            GC->>GC: validate_community_params (community.rs:97)
            GC->>GM: RunCommunityDetection via routes.community (community.rs:101, gpu_routes.rs:33-38)
            GM->>AS: forward (gpu_manager_actor.rs:375)
            AS->>CA: forward (analytics_supervisor.rs:520)
            CA-->>GC: Ok(result) or Err
            alt Ok(Ok(result))
                GC->>GC: convert_gpu_result_to_communities, graph node ids not buffer slots (community.rs:107, :194)
                GC-->>RCD: success=true (community.rs:116-127)
            else Ok(Err(e))
                GC-->>RCD: success=false with the actor's error, HTTP 200 (community.rs:129-143)
            else mailbox error
                GC-->>RCD: Err Failed to communicate with GPU actor (community.rs:144-147)
            end
        end
    end
    RCD-->>C: 200 response, or 500 on Err (analytics/mod.rs:93-104)
    Note over CA: perform_community_detection clustering_actor.rs:332<br/>refuses below MIN_NODES_FOR_CLUSTERING :368-377, runs label propagation :384<br/>Louvain :395 or Leiden :409, records the kernel per branch :390
    Note over CA: modularity gate - apply_modularity_gate writes node_analytics.community_id<br/>only when modularity reaches 0.3 (clustering_actor.rs:472)<br/>else community_id is reset (clustering_actor.rs:477-483), single writer per ADR-031 D3
```

**What it shows:** the route splits on the caller gate before any GPU work. A caller whose view hides nodes is answered by CPU local moving over that view; a full-access caller goes through `GPUManagerActor` to `ClusteringActor`, which the old `ForceComputeActor` refusal stub never reached.

**Why it is this way:** commits ab05ee9d6 and 405edcfe1 (2026-11) replaced filter-after with compute-over-the-view and routed full access through `gpu_routes.rs` (ADR-2184).

**Tension (parallel):** a gated caller always gets Louvain local moving whatever `algorithm` it asked for (`src/handlers/api_handler/analytics/community.rs:171`), while a full-access caller gets label propagation, Louvain or Leiden (`src/handlers/api_handler/analytics/community.rs:76-79`), so two callers asking the same question get different algorithms.

## VC-15.4 POST /analytics/anomaly/detect — LOF / Z-score structural anomaly

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:209
    participant RAD as run_anomaly_detection<br/>analytics/mod.rs:108
    participant GA as run_gpu_anomaly_detection<br/>anomaly.rs:25
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:421
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:574
    participant ADA as AnomalyDetectionActor<br/>anomaly_detection_actor.rs:118

    C->>RT: POST /analytics/anomaly/detect {method,kNeighbors,threshold}
    RT->>RAD: run_anomaly_detection(request)
    RAD->>RAD: caller_gate for the caller's pubkey (analytics/mod.rs:115-120)
    RAD->>GA: run_gpu_anomaly_detection(method,k_neighbors,radius,threshold) analytics/mod.rs:121
    alt app_state.gpu_manager_addr is None
        GA-->>RAD: Err GPU manager actor not available anomaly.rs:40
    else method not lof, local_outlier_factor, zscore or z_score
        GA-->>RAD: Err Unsupported anomaly detection method anomaly.rs:42-46
    else
        GA->>GA: validate_anomaly_params anomaly.rs:59
        GA->>GM: RunAnomalyDetection{params} anomaly.rs:63
        GM->>AS: RunAnomalyDetection forward (gpu_manager_actor.rs:421)
        AS->>ADA: RunAnomalyDetection forward (analytics_supervisor.rs:574-597)
        alt ADA.shared_context is None
            ADA-->>AS: Err GPU not initialized (anomaly_detection_actor.rs:127-132)
        else ADA.gpu_state.num_nodes == 0
            ADA-->>AS: Err No nodes available for anomaly detection (anomaly_detection_actor.rs:137-140)
        else k_neighbors >= num_nodes
            ADA-->>AS: Err k_neighbors must be less than total nodes (anomaly_detection_actor.rs:147-152)
        else
            alt method == LocalOutlierFactor
                ADA->>ADA: run_lof_anomaly_detection(k,threshold) (anomaly_detection_actor.rs:207)
                ADA->>ADA: record_execution(Lof, Gpu) (anomaly_detection_actor.rs:212)
            else method == ZScore
                ADA->>ADA: get_node_positions then run_zscore_anomaly_detection (anomaly_detection_actor.rs:237-252)
            end
            ADA->>ADA: node_analytics.anomaly = anomaly_score (single writer) (anomaly_detection_actor.rs:387-418)
            ADA-->>AS: Ok(AnomalyResult)
        end
        AS-->>GM: propagate
        GM-->>GA: propagate
        alt Ok(result)
            GA->>GA: convert_gpu_anomaly_result_to_anomalies anomaly.rs:74
            GA-->>RAD: Ok(Vec anomalies)
        else Err(e)
            GA-->>RAD: Err(e) anomaly.rs:81
        end
    end
    alt Ok
        RAD->>RAD: gate_anomalies keeps only visible positions when the caller's view hides nodes (analytics/mod.rs:129-132)
        RAD-->>C: 200 success=true anomalies, total, method
    else Err(e)
        RAD-->>C: 500 success=false anomalies=[] total=0 (analytics/mod.rs:148-155)
    end
    Note over ADA: the actor also has an internal DBSCAN noise arm (anomaly_detection_actor.rs:277-295), unreachable here because the public AnomalyMethod maps only to LOF and ZScore (anomaly_detection_actor.rs:155-160)
```

## VC-15.5 PageRank compute / cached result / cache clear — gated callers on the CPU, full access through the GPU manager

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant CP as compute_pagerank<br/>pagerank_handlers.rs:126
    participant GP as get_pagerank_result<br/>pagerank_handlers.rs:191
    participant CL as clear_pagerank_cache<br/>pagerank_handlers.rs:240
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:819
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:631
    participant PRA as PageRankActor<br/>pagerank_actor.rs:374

    C->>CP: POST /analytics/pagerank/compute (analytics/mod.rs:238), OptionalAuth caller
    CP->>CP: caller_gate over the current graph (pagerank_handlers.rs:138-142)
    alt Err, graph unreadable
        CP-->>C: 500 success=false (pagerank_handlers.rs:160-166)
    else Ok(Some(gate)), the view hides nodes
        CP->>CP: visible_pagerank over the view, keyed by visible node id, top 10 (pagerank_handlers.rs:144-157, visible_analytics.rs:408-415)
        CP-->>C: 200 PageRankPayload Visible, the shared actor and its cache are never touched
    else Ok(None), full access
        CP->>GM: ComputePageRank via whole_graph_pagerank (pagerank_handlers.rs:176, :278-283)
        alt gpu_manager_addr is None
            CP-->>C: 500 GPU analytics not available (pagerank_handlers.rs:275-277)
        else
            GM->>AS: forward (gpu_manager_actor.rs:819)
            AS->>PRA: forward to the PageRankActor it spawned (analytics_supervisor.rs:132-134)
            PRA->>PRA: run_pagerank_centrality :419, record_execution(Pagerank, Gpu) :427
            PRA->>PRA: last_result = Some(result) :487, publish_centrality single writer :491
            PRA-->>CP: Ok(PageRankResult), 200 PageRankPayload Whole (pagerank_handlers.rs:285-294)
        end
    end
    C->>GP: GET /analytics/pagerank/result (analytics/mod.rs:299)
    GP->>GP: caller_gate (pagerank_handlers.rs:200-204)
    alt Ok(Some), the view hides nodes
        GP-->>C: 200 cached=false, the same reply as an empty cache (pagerank_handlers.rs:207-214)
    else Ok(None)
        GP->>GM: GetPageRankResult (pagerank_handlers.rs:320, gpu_manager_actor.rs:843)
        GM->>AS: forward, None when PageRankActor is not running (analytics_supervisor.rs:601-610)
        AS->>PRA: GetPageRankResult, get_cached_result pagerank_actor.rs:345
        PRA-->>C: 200 cached and the whole-graph result
    end
    C->>CL: POST /analytics/pagerank/clear (analytics/mod.rs:242)
    CL->>GM: ClearPageRankCache, only for a power user, anyone else gets the reply of a missing route (pagerank_handlers.rs:240-256)
    GM->>AS: forward (gpu_manager_actor.rs:858, analytics_supervisor.rs:614)
    AS->>PRA: clear_cache sets last_result None (pagerank_actor.rs:350-351)
    Note over CP,PRA: INVARIANT the shared PageRank cache only ever holds and serves a whole-graph run, a gated caller is answered over the view and told nothing is cached<br/>pagerank_handlers.rs:136-137, :197-199
```

**What it shows:** the three PageRank routes after the 2026-11 visibility sweep. The `AppState.analytics.pagerank` address that nothing ever set is gone; full access reaches `PageRankActor` through `GPUManagerActor` and `AnalyticsSupervisor`.

**Why it is this way:** commit 405edcfe1 added the `GetPageRankResult` and `ClearPageRankCache` forwards and `gpu_routes.rs`, so the full-access path stops failing (ADR-2184).

**Open (risk):** RESOLVED at visionclaw 46edc196 — `POST /analytics/pagerank/clear` now refuses everyone but a power user, with the reply of a missing route (`src/handlers/api_handler/analytics/pagerank_handlers.rs:240-249`, `:260-262`), so only an operator can evict the whole-graph cache that full-access callers read.

## VC-15.6 POST /analytics/pathfinding/sssp and /pathfinding/apsp — supervised ShortestPathActor, gated callers over the view

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant CS as compute_sssp<br/>analytics/pathfinding.rs:167
    participant CAp as compute_apsp<br/>analytics/pathfinding.rs:262
    participant VA as visible_sssp and visible_apsp<br/>visible_analytics.rs:60
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:732
    participant GAS as GraphAnalyticsSupervisor<br/>graph_analytics_supervisor.rs:392
    participant SPA as ShortestPathActor (supervised)<br/>shortest_path_actor.rs:198

    C->>CS: POST /analytics/pathfinding/sssp {sourceIdx,maxDistance,delta} (analytics/mod.rs:246)
    CS->>CS: gated_view, current graph then gate_for_graph,<br/>an unreadable graph answers 503 (analytics/pathfinding.rs:177, :431-452)
    alt Ok(Some(gate)), the view hides nodes
        CS->>VA: visible_sssp(gate, source_idx, max_distance) analytics/pathfinding.rs:179-183
        alt source hidden or past the end
            VA-->>C: 500 with the exact out-of-range text (analytics/pathfinding.rs:189-193, :421-426)
        else
            VA-->>C: 200 SsspPayload Visible, one CPU Dijkstra over the view,<br/>never the GPU (analytics/pathfinding.rs:184-188)
        end
    else data.gpu_manager_addr is None
        CS-->>C: 500 Shortest path actor not available analytics/pathfinding.rs:239-245
    else full access
        CS->>GM: ComputeSSP{source_idx,max_distance,delta} analytics/pathfinding.rs:204-208
        GM->>GAS: forward_to_graph_analytics ComputeSSP (gpu_manager_actor.rs:755-758)
        alt child absent or not running
            GAS-->>GM: Err ShortestPathActor not available (graph_analytics_supervisor.rs:396-402)
        else
            GAS->>SPA: ComputeSSP (graph_analytics_supervisor.rs:404-410)
            alt SPA.shared_context is None
                SPA-->>GAS: Err GPU context not initialized (shortest_path_actor.rs:214-216)
            else
                SPA->>SPA: unified_compute.run_sssp(source_idx,delta) (shortest_path_actor.rs:223),<br/>bounds-checked, an index past num_nodes errors<br/>instead of panicking (src/utils/unified_gpu_compute/sssp.rs:21-25)
                SPA->>SPA: record_execution(Sssp, Gpu) (shortest_path_actor.rs:231)
                SPA->>SPA: optional max_distance filter (shortest_path_actor.rs:247-254)
                SPA-->>GAS: Ok(SSSPResult{distances,nodes_reached,max_distance}),<br/>returned only, written nowhere (shortest_path_actor.rs:274-280)
            end
        end
        CS-->>C: 200 or 500 SSSPResponse (analytics/pathfinding.rs:210-238)
    end
    C->>CAp: POST /analytics/pathfinding/apsp {numLandmarks,seed} (analytics/mod.rs:250)
    CAp->>CAp: gated_view (analytics/pathfinding.rs:271)
    alt Ok(Some(gate))
        CAp->>VA: visible_apsp, landmark rows k x n keyed by visible id,<br/>k defaults to ceil sqrt n, capped at 256 (visible_analytics.rs:511-515)
        VA-->>C: 200 ApspPayload Visible (analytics/pathfinding.rs:272-283)
    else full access
        CAp->>GM: ComputeAPSP{num_landmarks,seed} analytics/pathfinding.rs:293-298
        GM->>GAS: forward ComputeAPSP (gpu_manager_actor.rs:759-762, graph_analytics_supervisor.rs:491)
        GAS->>SPA: ComputeAPSP
        SPA-->>CAp: Err APSP disabled by NFR-7, O(n^2) memory forbidden,<br/>use SSSP per source (shortest_path_actor.rs:294-299)
        CAp-->>C: 500 APSPResponse success=false (analytics/pathfinding.rs:310-316)
    end
    Note over SPA: RESOLVED ADR-2054 - the APSP kernel is removed (crates/visionclaw-gpu/src/cuda_sources/gpu_landmark_apsp.cu:13-25), ComputeAPSP is retained and refuses explicitly under NFR-7
    Note over CS,GAS: RESOLVED 2026-11 - the publish flag, stream_safe and the node_sssp map are deleted,<br/>no SSSP run reaches the wire, ComputeSSP carries no publish field (src/actors/gpu/shortest_path_actor.rs:26-36)
```

**What it shows:** SSSP and APSP split on the caller gate. The GPU actor now only returns its distances; the run-to-wire publish path and its `stream_safe` precondition were removed in ca9113b2f and 7a0674e79.

**Tension (parallel):** a gated caller is answered on `/pathfinding/apsp` with landmark rows (`src/handlers/api_handler/analytics/pathfinding.rs:271-283`), while a full-access caller always gets the NFR-7 refusal (`src/actors/gpu/shortest_path_actor.rs:294-299`), so seeing less of the graph yields more of the answer.

## VC-15.7 POST /analytics/sssp/compute — the one CPU Dijkstra via GraphServiceSupervisor, result discarded

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:212
    participant CS as compute_sssp<br/>sssp_handlers.rs:213
    participant GSS as GraphServiceSupervisor (graph_service_addr)<br/>graph_service_supervisor.rs:1689
    participant GSA as GraphStateActor<br/>graph_state_actor.rs:1100

    C->>RT: POST /analytics/sssp/compute {sourceNode}
    RT->>CS: compute_sssp(request), sourceNode defaults to 0 (sssp_handlers.rs:220-224)
    alt caller_gate says the caller's view hides nodes and the source is not in it (sssp_handlers.rs:228-238)
        CS-->>C: 500 Failed to compute SSSP Source node N not found, the same text and status as a missing source (sssp_handlers.rs:206-211, graph_state_actor.rs:1096-1098)
    else unreadable graph
        CS-->>C: 500 Failed to communicate with graph service, fail closed (sssp_handlers.rs:240-243)
    end
    CS->>GSS: ComputeShortestPaths{source_node_id} (sssp_handlers.rs:247-251)
    GSS->>GSA: forward to graph_state (graph_service_supervisor.rs:1698-1708)
    GSA->>GSA: compute_shortest_paths delegates to shortest_paths_over (graph_state_actor.rs:749-754, :1071)
    alt source_node_id not in node_map
        GSA-->>GSS: Err source_not_found (graph_state_actor.rs:1076-1077, :1096-1098)
    else
        GSA->>GSA: services::pathfinding::single_source, binary-heap Dijkstra, undirected weighted edges,<br/>an edge to an id outside the graph is skipped (src/services/pathfinding.rs:180-194)
        GSA->>GSA: path_to rebuilds each hop path from the predecessors (graph_state_actor.rs:1079-1085, src/services/pathfinding.rs:168-173)
        GSA-->>GSS: Ok(PathfindingResult{distances,paths,computation_time_ms}) (graph_state_actor.rs:1125-1130)
    end
    GSS-->>CS: propagate
    alt Ok
        CS-->>C: 200 success=true SSSP computation started (sssp_handlers.rs:254-260)
        Note over CS: DEBT the computed distances and paths are dropped<br/>Ok(Ok(_)) at sssp_handlers.rs:254 discards them<br/>and the reply says started for a finished computation
    else Err(e)
        CS-->>C: 500 success=false error through the shared sssp_compute_failed (sssp_handlers.rs:262-265)
    end
    Note over GSA: RESOLVED 2026-11 - the BTreeSet Dijkstra is deleted, this route and every gated route share single_source (src/services/pathfinding.rs:150-155)<br/>it never touches ShortestPathActor, and unlike GPU ComputeSSP it reconstructs hop paths
    Note over GSS,GSA: DEAD CODE - GPUManagerActor Handler-ComputeShortestPaths (gpu_manager_actor.rs:773)<br/>and GraphAnalyticsSupervisor Handler-ComputeShortestPaths (graph_analytics_supervisor.rs:514)<br/>are unreachable, the only sender is sssp_handlers.rs:249 and it targets graph_service_addr
```

## VC-15.8 POST /analytics/pathfinding/connected-components — supervised GPU actor, no CPU fallback

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:258
    participant CC as compute_connected_components<br/>analytics/pathfinding.rs:349
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:796
    participant GAS as GraphAnalyticsSupervisor<br/>graph_analytics_supervisor.rs:574
    participant CCA as ConnectedComponentsActor (supervised)<br/>connected_components_actor.rs:179

    C->>RT: POST /analytics/pathfinding/connected-components {maxIterations}
    RT->>CC: compute_connected_components(payload)
    CC->>CC: the same gated_view first, a caller whose view hides nodes gets visible_components on the CPU, so no label joins two visible nodes through a hidden one (analytics/pathfinding.rs:356-370)
    alt data.gpu_manager_addr is None
        CC-->>C: 500 Connected components actor not available analytics/pathfinding.rs:411-417
    else
            CC->>GM: ComputeConnectedComponents{max_iterations} analytics/pathfinding.rs:373-379
        GM->>GAS: forward (gpu_manager_actor.rs:805-814)
        alt child absent or not running
            GAS-->>GM: Err ConnectedComponentsActor not available (graph_analytics_supervisor.rs:582-590)
        else
            GAS->>CCA: ComputeConnectedComponents (graph_analytics_supervisor.rs:592-596)
            alt CCA.shared_context is None
                CCA-->>GAS: Err GPU context not initialized (connected_components_actor.rs:228-229)
            else
                CCA->>CCA: unified_compute.run_connected_components_gpu(max_iterations) (connected_components_actor.rs:202)
                alt GPU call Ok
                    CCA->>CCA: record_execution(ConnectedComponents, Gpu) (connected_components_actor.rs:205-208)
                    CCA->>CCA: analyze_components(labels) (connected_components_actor.rs:233-234)
                    CCA-->>GAS: Ok(ConnectedComponentsResult{execution_path}) (connected_components_actor.rs:246-255)
                else GPU call Err(e)
                    CCA-->>GAS: Err GPU path failed and no CPU fallback is available (connected_components_actor.rs:221-224)
                end
            end
        end
        CC-->>C: 200 or 500 ConnectedComponentsResponse (analytics/pathfinding.rs:380-409)
    end
    Note over GM,CCA: RESOLVED ADR-2053 - stats reads use GetSupervisedComponentsStats (analytics/pathfinding.rs:480), a Result-returning wrapper, because the bare stats message has no Default and absence must stay explicit
    Note over CCA: RESOLVED ADR-2054 - the CPU fallback that returned every node as a singleton component is removed, a GPU Err propagates as a hard error (connected_components_actor.rs:212-224)
```

## VC-15.9 POST /analytics/pathfinding/path — point-to-point A* / Bidirectional Dijkstra / Semantic / SSSP dispatch

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:254
    participant PTP as compute_point_to_point<br/>analytics/pathfinding.rs:543-544
    participant GR as graph_repository<br/>analytics/pathfinding.rs:555
    participant AS as AStarPathfinder<br/>services/pathfinding.rs
    participant BD as BidirectionalDijkstra<br/>services/pathfinding.rs
    participant SEM as SemanticPathfinder (word-level Jaccard similarity)<br/>services/pathfinding.rs
    participant GM as GPUManagerActor to supervised ShortestPathActor

    C->>RT: POST /analytics/pathfinding/path {sourceId,targetId,algorithm,query}
    RT->>PTP: compute_point_to_point(payload)
    PTP->>GR: graph_repository.get_graph() analytics/pathfinding.rs:555
    PTP->>PTP: keep the full graph and the caller's view, caller_view, every algorithm below runs on the view (analytics/pathfinding.rs:556-559)
    alt graph fetch Err
        PTP-->>C: 500 Failed to retrieve graph data analytics/pathfinding.rs:562-566
    else
        alt algorithm == Astar
            PTP->>AS: AStarPathfinder::find_path(graph,source,target) analytics/pathfinding.rs:572
        else algorithm == Bidirectional
            PTP->>BD: BidirectionalDijkstra::find_path(graph,source,target) analytics/pathfinding.rs:598
        else algorithm == Semantic
            alt query is empty or missing
                PTP-->>C: 500 Query string is required for semantic pathfinding analytics/pathfinding.rs:631-638
            else
                PTP->>SEM: SemanticPathfinder(alpha).find_path(graph,source,target,query) analytics/pathfinding.rs:641-644
            end
        else algorithm == Sssp
            opt gate_for_graph over the full graph says the view hides nodes (analytics/pathfinding.rs:674-677)
                PTP-->>C: CPU visible_point_to_point over the view, never the GPU, whose indices are full-graph positions (analytics/pathfinding.rs:678-703)
            end
            PTP->>PTP: locate source_idx by node id in graph_data.nodes analytics/pathfinding.rs:710-714
            alt source not found
                PTP-->>C: 500 Source node not found in graph analytics/pathfinding.rs:716-725
            else
                PTP->>GM: ComputeSSP{source_idx} via gpu_manager_addr analytics/pathfinding.rs:708, :728-732
                GM-->>PTP: SSSPResult (same supervised path as VC-15.6)
            end
        end
        PTP-->>C: 200 PointToPointResponse result (json_val per algorithm)
    end
```

## VC-15.10 POST /pathfinding/semantic-path, /query-traversal, /chunk-traversal — CPU semantic A*

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant CFG as configure_pathfinding_routes<br/>semantic_pathfinding_handler.rs:151
    participant FSP as find_semantic_path<br/>semantic_pathfinding_handler.rs:29
    participant QT as query_traversal<br/>semantic_pathfinding_handler.rs:62
    participant CT as chunk_traversal<br/>semantic_pathfinding_handler.rs:94
    participant GSA as GraphServiceSupervisor (web::Data Addr)<br/>graph_service_supervisor.rs:1490
    participant SVC as SemanticPathfindingService<br/>semantic_pathfinding_service.rs:86

    Note over CFG: mounted at web-scope pathfinding by main.rs:1210, distinct from the analytics-scope path family in VC-15.6 to VC-15.9
    Note over CFG: the scope is wrapped in one process-wide rate limiter, 30 requests per minute keyed on the verified pubkey<br/>with an IP fallback for unauthenticated reads (semantic_pathfinding_handler.rs:127-145, :154, PATHFINDING_REQUESTS_PER_MINUTE :125)
    Note over QT,CT: traversal_budget defaults max_nodes to 50 and caps it at MAX_TRAVERSAL_NODES = 1000 (semantic_pathfinding_handler.rs:123, :147-148)

    C->>CFG: POST /pathfinding/semantic-path {startId,endId,query}
    CFG->>FSP: find_semantic_path(request)
    FSP->>GSA: GetGraphData (semantic_pathfinding_handler.rs:40-42)
    alt Ok(graph_data)
        FSP->>FSP: caller_view, the search runs on the caller's visible subgraph,<br/>as do the two traversals (semantic_pathfinding_handler.rs:46, :76, :108)
        FSP->>SVC: find_semantic_path(graph,start_id,end_id,query) (semantic_pathfinding_handler.rs:47, semantic_pathfinding_service.rs:98)
        SVC->>SVC: A* over build_adjacency_list with calculate_semantic_cost per edge (semantic_pathfinding_service.rs:108-193)
        alt max_explored (1000) reached before goal
            SVC-->>FSP: None (semantic_pathfinding_service.rs:136-138)
        else path found
            SVC-->>FSP: Some(PathResult{path,cost,relevance,explanation}) (semantic_pathfinding_service.rs:128-133)
        end
        alt Some(path)
            FSP-->>C: 200 path
        else None
            FSP-->>C: error_json No path found (semantic_pathfinding_handler.rs:54)
        end
    else Err(e)
        FSP-->>C: error_json Graph error (semantic_pathfinding_handler.rs:57)
    end

    C->>CFG: POST /pathfinding/query-traversal {startId,query,maxNodes}
    CFG->>QT: query_traversal(request)
    QT->>GSA: GetGraphData (semantic_pathfinding_handler.rs:70-72)
    alt query is None
        QT-->>C: error_json Query parameter required (semantic_pathfinding_handler.rs:86)
    else
        QT->>SVC: query_traversal(graph,start_id,query,traversal_budget(max_nodes)) (semantic_pathfinding_handler.rs:78-83, semantic_pathfinding_service.rs:202)
        SVC-->>QT: results
        QT-->>C: 200 results
    end

    C->>CFG: POST /pathfinding/chunk-traversal {startId,maxNodes}
    CFG->>CT: chunk_traversal(request)
    CT->>GSA: GetGraphData (semantic_pathfinding_handler.rs:102-104)
    CT->>SVC: chunk_traversal(graph,start_id,traversal_budget(max_nodes)) (semantic_pathfinding_service.rs:299)
    SVC-->>CT: results
    CT-->>C: 200 results
```

## VC-15.12 Visual analytics GPU types (src/gpu/visual_analytics.rs) with validate() failure modes

```mermaid
classDiagram
    class Vec4 {
        x: f32
        y: f32
        z: f32
        t: f32
        new(x,y,z,t) Result~Vec4,GPUSafetyError~
        validate() Result~(),GPUSafetyError~
    }
    note for Vec4 "visual_analytics.rs:13, new at :21 rejects non-finite components and abs greater than 1e6 (MAX_VAL :28), validate at :50"

    class IsolationLayer {
        layer_id: i32
        opacity: f32
        z_offset: f32
        focus_center: Vec4
        focus_radius: f32
        temporal_range: f32_2
        edge_opacity: f32
        new(layer_id) IsolationLayer
        validate() Result~(),GPUSafetyError~
    }
    note for IsolationLayer "visual_analytics.rs:75, new() at :95, validate at :113 rejects opacity/edge_opacity outside 0..1 and a negative layer_id"

    class VisualAnalyticsParams {
        total_nodes: i32
        total_edges: i32
        active_layers: i32
        force_scale: f32_4
        rest_length: f32
        primary_focus_node: i32
        embedding_dims: i32
        camera_position: Vec4
        viewport_bounds: Vec4
        validate() Result~(),GPUSafetyError~
    }
    note for VisualAnalyticsParams "visual_analytics.rs:207 (38 fields), Default impl at :257, validate() at :312 covers node/edge counts and force scales"

    class VisualAnalyticsBuilder {
        <<builder, live external consumer>>
    }
    note for VisualAnalyticsBuilder "visual_analytics.rs:537, used by src/handlers/api_handler/analytics/params_handlers.rs:347"

    class PerformanceMetrics {
        <<live external consumer>>
    }
    note for PerformanceMetrics "visual_analytics.rs:500, used by src/handlers/api_handler/analytics/types.rs:4"

    class GPUSafetyError {
        <<enum, subset shown>>
        InvalidKernelParams
        BufferBoundsExceeded
        ResourceExhaustion
        InvalidBufferSize
        DeviceError
    }
    note for GPUSafetyError "src/utils/gpu_safety.rs:176, twelve variants in all, the five above are the ones these validators and allocations name"

    IsolationLayer --> Vec4
    VisualAnalyticsParams --> Vec4
    Vec4 ..> GPUSafetyError
    IsolationLayer ..> GPUSafetyError
    VisualAnalyticsParams ..> GPUSafetyError

    note "RESOLVED ADR-2054 (src/gpu/mod.rs:32-37): TSNode, TSEdge, VisualAnalyticsGPU and<br/>VisualAnalyticsEngine are REMOVED - dead code, never instantiated outside their<br/>own module. Vec4, IsolationLayer, VisualAnalyticsParams, VisualAnalyticsBuilder<br/>and PerformanceMetrics remain, each with a live external consumer<br/>(physics_messages.rs, params_handlers.rs, types.rs)."
```

## VC-15.13 Analytics kernel trust status and the CPU reference oracle

```mermaid
flowchart TD
    subgraph LIVE["Live GPU kernels (crates/visionclaw-gpu/src/cuda_sources)"]
        LOUV["Louvain community detection TRUSTED<br/>gpu_clustering_kernels.cu:606 D1 fix marker, output-verified by ADR-2061"]
        PR["PageRank TRUSTED<br/>pagerank.cu:263 D8 fix marker, global two-kernel dangling-mass path, output-verified by ADR-2061"]
        DBS["DBSCAN TRUSTED<br/>gpu_clustering_kernels.cu:1104 border handling in propagate/finalise, output-verified by ADR-2061"]
        LOF["LOF local outlier factor TRUSTED on the fixture<br/>lof_gather_neighbors gpu_clustering_kernels.cu:326 includes kth-distance ties<br/>lof_lrd_from_neighbors gpu_clustering_kernels.cu:418 uses neighbour k-distance"]
        ONT["Ontology constraints<br/>fixed and live, keystone wiring"]
    end

    subgraph QUARANTINED["Removed / absent"]
        APSP["Landmark APSP kernel REMOVED<br/>gpu_landmark_apsp.cu:13-25 records the removal, the file keeps only landmark selection and stress kernels"]
        EMB["Node embeddings<br/>no embedding kernels exist in the tree"]
    end

    subgraph ORACLE["Test-time reference oracle, crates/visionclaw-analytics-oracle/src/lib.rs"]
        ENC["encode_record_52 lib.rs:84 / decode_record_52 lib.rs:104"]
        FIX["GraphFixture lib.rs:139 - two_clique lib.rs:190, triangle lib.rs:204, star lib.rs:213, linear_chain lib.rs:223, canonical_live_scale lib.rs:244"]
        CPUMOD["CPU reference modularity lib.rs:301"]
        CPUPR["CPU reference pagerank lib.rs:340"]
        CPUDB["CPU reference dbscan lib.rs:390"]
        CPULOF["CPU reference lof lib.rs:441"]
    end

    LOUV -.->|"conformance test recorded PASS"| CPUMOD
    PR -.->|"conformance test recorded PASS"| CPUPR
    DBS -.->|"conformance test recorded PASS"| CPUDB
    LOF -.->|"recorded PASS: max delta 4.759e-7 below 1e-3"| CPULOF

    NOTE["Trust table docs/GPU-wire-abi.md:196-209 (ADR-2061, 2026-09-07): PageRank max delta 3.4e-11,<br/>DBSCAN exact, Louvain 16 of 16 communities, LOF corrected from 0.702 to 4.759e-7 on the 1e-3 bar<br/>Test crates/visionclaw-gpu/tests/analytics_oracle_conformance.rs - these are recorded results, not re-run here"]
    LIVE -.-> NOTE
    ORACLE -.->|"fixture crate only, not wired into any runtime request path"| LIVE
    TEL["Runtime reach: full-access PageRank and community detection reach these kernels through GPUManagerActor<br/>gpu_routes.rs:33-44, gated callers are answered on the CPU over the view, see VC-15.3 and VC-15.5"]
    LIVE -.-> TEL
    PAR["GPU vs CPU parity for a full-access caller<br/>visible_analytics.rs:922 communities, :951 PageRank<br/>both ignored without a CUDA device, visible_analytics.rs:921"]
    TEL -.-> PAR
```

**Debt:** `POST /analytics/sssp/compute` computes a full CPU Dijkstra and then discards it, answering `SSSP computation started` (`src/handlers/api_handler/analytics/sssp_handlers.rs:254-260`).

**Drift:** before this restamp the topic placed the Louvain D1 marker at `gpu_clustering_kernels.cu:581` (a loop line; the marker is at `:606`) and described the APSP kernel as `#if 0`-guarded, but ADR-2054 had already removed it (`crates/visionclaw-gpu/src/cuda_sources/gpu_landmark_apsp.cu:13-25`).

**Invariant:** `CallerGate` is the one ADR-060 "is anything hidden from this caller" decision; `gate_for_graph` and `caller_gate` are its only constructors (`src/handlers/graph_access.rs:77-82`, `:120`, `:159`), and ids and GPU positions are distinct types so they cannot be mixed (`src/handlers/graph_access.rs:44-53`).

**Debt (parallel):** the gate still reads the graph two ways. Point-to-point reads `graph_repository` (`src/handlers/api_handler/analytics/pathfinding.rs:555`) while every other gate reads `GraphServiceSupervisor` (`src/handlers/graph_access.rs:159-168`), and `gated_view` fetches the graph even when the filter is off (`src/handlers/api_handler/analytics/pathfinding.rs:431-451`) where `caller_gate` skips the fetch (`src/handlers/graph_access.rs:163-165`).

**Invariant:** there is one CPU single-source shortest path, `single_source` (`src/services/pathfinding.rs:180`), used by `/sssp/compute` (`src/actors/graph_state_actor.rs:1079`) and the gated routes; the GPU `run_sssp` (`src/utils/unified_gpu_compute/sssp.rs:20`) is its accelerator for full-access callers.

**Tension (risk):** anomalies are still scored over the whole graph and only filtered on the way out (`src/handlers/api_handler/analytics/visible_results.rs:1-7`), because LOF scores layout positions every node shaped; and DBSCAN clusters keep their whole-graph cluster ids after hidden members are dropped (`src/handlers/api_handler/analytics/visible_analytics.rs:173-177`). A visible node's anomaly score or cluster number can still reflect hidden nodes.

**Open (risk):** the GPU-vs-CPU parity tests for communities and PageRank are `#[ignore]` without a CUDA device (`src/handlers/api_handler/analytics/visible_analytics.rs:921-922`), so nothing in a GPU-less run checks that a gated and a full-access caller see the same algorithm's answer.

**Drift (loose end):** the point-to-point SSSP comment still says a GPU run would "be published to the shared stream" (`src/handlers/api_handler/analytics/pathfinding.rs:671-673`); the publish path was deleted in ca9113b2f.
