---
id: VC-15
title: GPU analytics kernels and pathfinding
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2007, ADR-2053, ADR-2054, ADR-2061]
sources:
  - ../project/src/actors/gpu/clustering_actor.rs
  - ../project/src/actors/gpu/analytics_supervisor.rs
  - ../project/src/actors/gpu/anomaly_detection_actor.rs
  - ../project/src/actors/gpu/pagerank_actor.rs
  - ../project/src/actors/gpu/shortest_path_actor.rs
  - ../project/src/actors/gpu/connected_components_actor.rs
  - ../project/src/actors/gpu/graph_analytics_supervisor.rs
  - ../project/src/actors/gpu/gpu_manager_actor.rs
  - ../project/src/actors/gpu/resource_supervisor.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/app_state.rs
  - ../project/src/handlers/api_handler/analytics/mod.rs
  - ../project/src/handlers/api_handler/analytics/clustering_handlers.rs
  - ../project/src/handlers/api_handler/analytics/clustering.rs
  - ../project/src/handlers/api_handler/analytics/real_gpu_functions.rs
  - ../project/src/handlers/api_handler/analytics/community.rs
  - ../project/src/handlers/api_handler/analytics/anomaly.rs
  - ../project/src/handlers/api_handler/analytics/pagerank_handlers.rs
  - ../project/src/handlers/api_handler/analytics/sssp_handlers.rs
  - ../project/src/handlers/api_handler/analytics/pathfinding.rs
  - ../project/src/handlers/semantic_pathfinding_handler.rs
  - ../project/src/handlers/mod.rs
  - ../project/src/services/semantic_pathfinding_service.rs
  - ../project/src/gpu/visual_analytics.rs
  - ../project/src/gpu/mod.rs
  - ../project/src/utils/gpu_safety.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/crates/visionclaw-analytics-oracle/src/lib.rs
  - ../project/crates/visionclaw-gpu/src/cuda_sources/gpu_clustering_kernels.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/pagerank.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/gpu_landmark_apsp.cu
  - ../project/docs/GPU-wire-abi.md
  - ../project/crates/visionclaw-gpu/tests/analytics_oracle_conformance.rs
  - ../project/src/main.rs
  - ../project/src/handlers/api_handler/analytics/params_handlers.rs
  - ../project/src/handlers/api_handler/analytics/types.rs
verified_commit: 86cdd76355823a51bb64d503c005fe277726e724
---

## VC-15.1 POST /analytics/clustering/run — spectral/kmeans/louvain/default dispatch with CPU fallback

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:175
    participant RC as run_clustering<br/>clustering_handlers.rs:20
    participant PC as perform_clustering<br/>clustering_handlers.rs:332
    participant RG as real_gpu_functions<br/>real_gpu_functions.rs:100-269
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:444
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:648
    participant CA as ClusteringActor<br/>clustering_actor.rs:1288

    C->>RT: POST /analytics/clustering/run
    RT->>RC: run_clustering(ClusteringRequest)
    RC->>RC: CLUSTERING_TASKS.insert(task_id, status=running) :33-46
    RC->>PC: tokio::spawn perform_clustering :52-53
    RC-->>C: 200 ClusteringResponse task_id, clusters none (clustering_handlers.rs:91-98)
    PC->>PC: graph_service_addr.send(GetGraphData) :340
    PC->>PC: mcp_client.query_agent_list() :354, an MCP failure yields an empty agent list :362-368
    alt request.method == spectral
        PC->>RG: perform_gpu_spectral_clustering :373
    else request.method == kmeans
        PC->>RG: perform_gpu_kmeans_clustering :376
    else request.method == louvain
        PC->>RG: perform_gpu_louvain_clustering :379
    else any other method, dbscan included
        PC->>RG: perform_gpu_default_clustering :381
        Note right of RG: node_count under 100 kmeans<br/>100 to 999 spectral<br/>else louvain (real_gpu_functions.rs:260-268)
    end
    RG->>GM: PerformGPUClustering{method,params,task_id}<br/>via gpu_manager_addr (real_gpu_functions.rs:111-122)
    GM->>AS: PerformGPUClustering forward (gpu_manager_actor.rs:444-465)
    AS->>CA: PerformGPUClustering forward (analytics_supervisor.rs:654-672)
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
            RG->>RG: generate_agent_based_clusters (clustering_handlers.rs:393)
        else
            RG->>RG: generate_label_propagation_clusters (real topology CPU fallback, real_gpu_functions.rs:290)
        end
    end
    PC->>GM: WriteClusterAnalytics{clusters} do_send (clustering_handlers.rs:69)
    GM->>AS: WriteClusterAnalytics forward (gpu_manager_actor.rs:981)
    AS->>CA: WriteClusterAnalytics (clustering_actor.rs:1222)
    CA->>CA: write_cluster_id_from_assignments (ADR-031 D3 single writer)
    PC->>PC: CLUSTERING_TASKS[task_id].status = completed :79-81
    Note over PC,RG: DEAD CODE - clustering.rs:6 perform_clustering, a second PerformGPUClustering-only<br/>dispatcher with validate_clustering_params, has no caller anywhere in the tree<br/>clustering_handlers.rs:332 is the live perform_clustering
```

## VC-15.2 POST /analytics/clustering/dbscan — direct RunDBSCAN, GPU-only, precondition refusals

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:178
    participant RD as run_dbscan_clustering<br/>clustering_handlers.rs:211
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:398
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:544
    participant CA as ClusteringActor<br/>clustering_actor.rs:1193

    C->>RT: POST /analytics/clustering/dbscan {epsilon,minPoints}
    RT->>RD: run_dbscan_clustering(body)
    alt epsilon <= 0.0
        RD-->>C: 200 success=false epsilon must be positive :231-236
    else minPoints == 0
        RD-->>C: 200 success=false minPoints must be at least 1 :237-242
    else gpu_manager_addr is None
        RD-->>C: 200 success=false GPU compute not available :325-328
    else
        RD->>GM: RunDBSCAN{epsilon,min_points} (clustering_handlers.rs:251)
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
            RD->>RD: CLUSTERING_TASKS.insert(task_id, completed) :264-276
            RD-->>C: 200 success=true clusters, stats, gpuAccelerated=true
        else Err(e)
            RD-->>C: 200 success=false error (clustering_handlers.rs:308-315)
        end
    end
```

## VC-15.3 POST /analytics/community/detect — sent to ForceComputeActor, which refuses

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:179
    participant RCD as run_community_detection<br/>analytics/mod.rs:68
    participant GC as run_gpu_community_detection<br/>community.rs:51
    participant FCA as ForceComputeActor<br/>get_gpu_compute_addr app_state.rs:1672
    participant CA as ClusteringActor<br/>clustering_actor.rs:1176

    C->>RT: POST /analytics/community/detect {algorithm}
    RT->>RCD: run_community_detection(request), AuthenticatedUser extractor analytics/mod.rs:69
    RCD->>GC: run_gpu_community_detection(app_state, request) analytics/mod.rs:75
    alt cfg not(feature = gpu)
        GC-->>RCD: Err GPU features not enabled in this build (community.rs:263-268)
    else app_state.get_gpu_compute_addr() is None
        GC-->>RCD: Err GPU compute actor not available (community.rs:61-64)
    else unsupported algorithm string
        GC-->>RCD: Err Unsupported community detection algorithm (community.rs:70-75)
    else
        GC->>GC: map algorithm string to CommunityDetectionAlgorithm (community.rs:66-69), validate_community_params (community.rs:87)
        GC->>FCA: RunCommunityDetection{params} (community.rs:91)
        FCA-->>GC: Err Community detection should be handled by ClusteringActor (force_compute_actor.rs:3347-3352)
        GC-->>RCD: Err
    end
    RCD-->>C: 500 success=false communities=[] modularity=0.0 (analytics/mod.rs:77-88)
    Note over FCA,CA: DEBT the working chain GPUManagerActor (gpu_manager_actor.rs:375)<br/>to AnalyticsSupervisor (analytics_supervisor.rs:520) to ClusteringActor exists<br/>but this route never uses it - anomaly.rs:35-38 routes through gpu_manager_addr for this reason
    Note over CA: if reached, perform_community_detection clustering_actor.rs:332<br/>refuses below MIN_NODES_FOR_CLUSTERING :368-377, runs label propagation :384<br/>Louvain :395 or Leiden :409, records the kernel per branch :390
    Note over CA: modularity gate - apply_modularity_gate writes node_analytics.community_id<br/>only when modularity reaches 0.3 (clustering_actor.rs:472)<br/>else community_id is reset (clustering_actor.rs:477-483), single writer per ADR-031 D3
    Note over GC: the response converter convert_gpu_result_to_communities (community.rs:97)<br/>is reachable only on an Ok that this route cannot receive
```

## VC-15.4 POST /analytics/anomaly/detect — LOF / Z-score structural anomaly

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:181
    participant RAD as run_anomaly_detection<br/>analytics/mod.rs:92
    participant GA as run_gpu_anomaly_detection<br/>anomaly.rs:25
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:421
    participant AS as AnalyticsSupervisor<br/>analytics_supervisor.rs:574
    participant ADA as AnomalyDetectionActor<br/>anomaly_detection_actor.rs:118

    C->>RT: POST /analytics/anomaly/detect {method,kNeighbors,threshold}
    RT->>RAD: run_anomaly_detection(request)
    RAD->>GA: run_gpu_anomaly_detection(method,k_neighbors,radius,threshold) analytics/mod.rs:99
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
        RAD-->>C: 200 success=true anomalies, total, method
    else Err(e)
        RAD-->>C: 500 success=false anomalies=[] total=0 (analytics/mod.rs:120-127)
    end
    Note over ADA: the actor also has an internal DBSCAN noise arm (anomaly_detection_actor.rs:277-295), unreachable here because the public AnomalyMethod maps only to LOF and ZScore (anomaly_detection_actor.rs:155-160)
```

## VC-15.5 PageRank compute / cached result / cache clear — the AppState address is never set

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RTc as Route POST /pagerank/compute<br/>analytics/mod.rs:210
    participant RTr as Route GET /pagerank/result<br/>analytics/mod.rs:271
    participant RTx as Route POST /pagerank/clear<br/>analytics/mod.rs:214
    participant CP as compute_pagerank<br/>pagerank_handlers.rs:112
    participant GP as get_pagerank_result<br/>pagerank_handlers.rs:181
    participant CL as clear_pagerank_cache<br/>pagerank_handlers.rs:236
    participant PRA as PageRankActor (data.analytics.pagerank)<br/>pagerank_actor.rs:361

    Note over CP,PRA: AppState builds AnalyticsSubsystem with pagerank None (app_state.rs:1296-1300) and nothing assigns it afterwards, so every branch below that needs PRA is unreachable at this pin
    C->>RTc: POST /analytics/pagerank/compute {dampingFactor,maxIterations}
    RTc->>CP: compute_pagerank(payload)
    alt data.analytics.pagerank is None (pagerank_handlers.rs:121)
        CP-->>C: 500 success=false PageRank actor not available pagerank_handlers.rs:163-169
    else unreachable here
        CP->>PRA: ComputePageRank{params}
        PRA->>PRA: run_pagerank_centrality(damping,max_iter,epsilon,normalize,use_optimized) :406
        PRA->>PRA: record_execution(Pagerank, Gpu) :414
        PRA->>PRA: calculate_statistics, extract_top_nodes(10) :454-462
        PRA->>PRA: last_result cache = Some(result) :474
        PRA->>PRA: publish_centrality (ADR-031 D3 single writer of centrality) :478
        PRA-->>CP: Ok(PageRankResult)
    end
    C->>RTr: GET /analytics/pagerank/result
    RTr->>GP: get_pagerank_result
    alt data.analytics.pagerank is None (pagerank_handlers.rs:184)
        GP-->>C: error, PageRank actor not available
    else unreachable here
        GP->>PRA: GetPageRankResult, cached Some (get_cached_result pagerank_actor.rs:332) or None (pagerank_handlers.rs:198-206)
    end
    C->>RTx: POST /analytics/pagerank/clear
    RTx->>CL: clear_pagerank_cache
    alt data.analytics.pagerank is None (pagerank_handlers.rs:239)
        CL-->>C: error, PageRank actor not available
    else unreachable here
        CL->>PRA: ClearPageRankCache, clear_cache sets last_result=None pagerank_actor.rs:337
    end
    Note over CP,PRA: DEBT the supervised path exists - GPUManagerActor Handler-ComputePageRank gpu_manager_actor.rs:850 forwards to AnalyticsSupervisor analytics_supervisor.rs:599, which holds the running PageRankActor it spawned (analytics_supervisor.rs:132) - but no route sends ComputePageRank there
```

## VC-15.6 POST /analytics/pathfinding/sssp and /pathfinding/apsp — supervised ShortestPathActor

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RTs as Route POST /pathfinding/sssp<br/>analytics/mod.rs:218
    participant RTa as Route POST /pathfinding/apsp<br/>analytics/mod.rs:222
    participant CS as compute_sssp<br/>pathfinding.rs:135
    participant CAp as compute_apsp<br/>pathfinding.rs:203
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:763
    participant GAS as GraphAnalyticsSupervisor<br/>graph_analytics_supervisor.rs:430
    participant SPA as ShortestPathActor (supervised)<br/>shortest_path_actor.rs:213

    C->>RTs: POST /analytics/pathfinding/sssp {sourceIdx,maxDistance,delta}
    RTs->>CS: compute_sssp(payload)
    alt data.gpu_manager_addr is None
        CS-->>C: 500 Shortest path actor not available pathfinding.rs:180-186
    else
        CS->>GM: ComputeSSP{source_idx,max_distance,delta} pathfinding.rs:144-151
        GM->>GAS: forward_to_graph_analytics ComputeSSP (gpu_manager_actor.rs:786-789)
        alt child absent or not running
            GAS-->>GM: Err ShortestPathActor not available (graph_analytics_supervisor.rs:434-440)
        else
            GAS->>SPA: ComputeSSP (graph_analytics_supervisor.rs:442-448)
            alt SPA.shared_context is None
                SPA-->>GAS: Err GPU context not initialized (shortest_path_actor.rs:229-231)
            else
                SPA->>SPA: unified_compute.run_sssp(source_idx,delta) (shortest_path_actor.rs:238)
                SPA->>SPA: record_execution(Sssp, Gpu) (shortest_path_actor.rs:246)
                SPA->>SPA: optional max_distance filter (shortest_path_actor.rs:283-290)
                SPA->>SPA: publish distances to node_sssp map, parent=-1 always (no predecessor array) (shortest_path_actor.rs:309-329)
                SPA-->>GAS: Ok(SSSPResult{distances,nodes_reached,max_distance})
            end
        end
        CS-->>C: 200 or 500 SSSPResponse (pathfinding.rs:152-178)
    end
    C->>RTa: POST /analytics/pathfinding/apsp {numLandmarks,seed}
    RTa->>CAp: compute_apsp(payload)
    CAp->>GM: ComputeAPSP{num_landmarks,seed} pathfinding.rs:210-219
    GM->>GAS: forward ComputeAPSP (gpu_manager_actor.rs:790-793, graph_analytics_supervisor.rs:529)
    GAS->>SPA: ComputeAPSP
    SPA-->>CAp: Err APSP disabled by NFR-7, O(n^2) memory forbidden, use SSSP per source (shortest_path_actor.rs:358-363)
    CAp-->>C: 500 APSPResponse success=false (pathfinding.rs:231-237)
    Note over SPA: RESOLVED ADR-2054 - the APSP kernel is removed (crates/visionclaw-gpu/src/cuda_sources/gpu_landmark_apsp.cu:13-25), ComputeAPSP is retained and refuses explicitly under NFR-7
    Note over CS,GAS: RESOLVED ADR-2053 - the routes address data.gpu_manager_addr, and the supervisor's children receive SetSharedGPUContext (graph_analytics_supervisor.rs:342)
```

## VC-15.7 POST /analytics/sssp/compute — CPU Dijkstra via GraphServiceSupervisor, result discarded

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:184
    participant CS as compute_sssp<br/>sssp_handlers.rs:205
    participant GSS as GraphServiceSupervisor (graph_service_addr)<br/>graph_service_supervisor.rs:1689
    participant GSA as GraphStateActor<br/>graph_state_actor.rs:1160

    C->>RT: POST /analytics/sssp/compute {sourceNode}
    RT->>CS: compute_sssp(request), sourceNode defaults to 0 (sssp_handlers.rs:212-216)
    CS->>GSS: ComputeShortestPaths{source_node_id} (sssp_handlers.rs:219-223)
    GSS->>GSA: forward to graph_state (graph_service_supervisor.rs:1698-1708)
    GSA->>GSA: compute_shortest_paths(source_node_id) (graph_state_actor.rs:764)<br/>Dijkstra with a BTreeSet frontier that rescans every edge<br/>in both directions per pop (graph_state_actor.rs:787-814)
    alt source_node_id not in node_map
        GSA-->>GSS: Err Source node not found (graph_state_actor.rs:768-770)
    else
        GSA->>GSA: reconstruct predecessor chain into path (graph_state_actor.rs:819-836)
        GSA-->>GSS: Ok(PathfindingResult{distances,paths,computation_time_ms}) (graph_state_actor.rs:1185-1190)
    end
    GSS-->>CS: propagate
    alt Ok
        CS-->>C: 200 success=true SSSP computation started (sssp_handlers.rs:226-232)
        Note over CS: DEBT the computed distances and paths are dropped<br/>Ok(Ok(_)) at sssp_handlers.rs:226 discards them<br/>and the reply says started for a finished computation
    else Err(e)
        CS-->>C: 500 success=false error (sssp_handlers.rs:234-240)
    end
    Note over GSA: DIVERGENCE - a second, independent CPU-only SSSP implementation<br/>it never touches ShortestPathActor or the GPU kernel used by VC-15.6<br/>and unlike GPU ComputeSSP it does reconstruct hop paths
    Note over GSS,GSA: DEAD CODE - GPUManagerActor Handler-ComputeShortestPaths (gpu_manager_actor.rs:804)<br/>and GraphAnalyticsSupervisor Handler-ComputeShortestPaths (graph_analytics_supervisor.rs:552)<br/>are unreachable, the only sender is sssp_handlers.rs:221 and it targets graph_service_addr
```

## VC-15.8 POST /analytics/pathfinding/connected-components — supervised GPU actor, no CPU fallback

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:230
    participant CC as compute_connected_components<br/>pathfinding.rs:270
    participant GM as GPUManagerActor<br/>gpu_manager_actor.rs:827
    participant GAS as GraphAnalyticsSupervisor<br/>graph_analytics_supervisor.rs:611
    participant CCA as ConnectedComponentsActor (supervised)<br/>connected_components_actor.rs:179

    C->>RT: POST /analytics/pathfinding/connected-components {maxIterations}
    RT->>CC: compute_connected_components(payload)
    alt data.gpu_manager_addr is None
        CC-->>C: 500 Connected components actor not available pathfinding.rs:315-321
    else
        CC->>GM: ComputeConnectedComponents{max_iterations} pathfinding.rs:277-283
        GM->>GAS: forward (gpu_manager_actor.rs:836-845)
        alt child absent or not running
            GAS-->>GM: Err ConnectedComponentsActor not available (graph_analytics_supervisor.rs:619-627)
        else
            GAS->>CCA: ComputeConnectedComponents (graph_analytics_supervisor.rs:629-633)
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
        CC-->>C: 200 or 500 ConnectedComponentsResponse (pathfinding.rs:284-313)
    end
    Note over GM,CCA: RESOLVED ADR-2053 - stats reads use GetSupervisedComponentsStats (pathfinding.rs:350), a Result-returning wrapper, because the bare stats message has no Default and absence must stay explicit
    Note over CCA: RESOLVED ADR-2054 - the CPU fallback that returned every node as a singleton component is removed, a GPU Err propagates as a hard error (connected_components_actor.rs:212-224)
```

## VC-15.9 POST /analytics/pathfinding/path — point-to-point A* / Bidirectional Dijkstra / Semantic / SSSP dispatch

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant RT as Route<br/>analytics/mod.rs:226
    participant PTP as compute_point_to_point<br/>pathfinding.rs:413-414
    participant GR as graph_repository<br/>pathfinding.rs:424
    participant AS as AStarPathfinder<br/>services/pathfinding.rs
    participant BD as BidirectionalDijkstra<br/>services/pathfinding.rs
    participant SEM as SemanticPathfinder (word-level Jaccard similarity)<br/>services/pathfinding.rs
    participant GM as GPUManagerActor to supervised ShortestPathActor

    C->>RT: POST /analytics/pathfinding/path {sourceId,targetId,algorithm,query}
    RT->>PTP: compute_point_to_point(payload)
    PTP->>GR: graph_repository.get_graph() pathfinding.rs:424
    alt graph fetch Err
        PTP-->>C: 500 Failed to retrieve graph data pathfinding.rs:428-432
    else
        alt algorithm == Astar
            PTP->>AS: AStarPathfinder::find_path(graph,source,target) pathfinding.rs:438
        else algorithm == Bidirectional
            PTP->>BD: BidirectionalDijkstra::find_path(graph,source,target) pathfinding.rs:464
        else algorithm == Semantic
            alt query is empty or missing
                PTP-->>C: 500 Query string is required for semantic pathfinding pathfinding.rs:497-504
            else
                PTP->>SEM: SemanticPathfinder(alpha).find_path(graph,source,target,query) pathfinding.rs:507-510
            end
        else algorithm == Sssp
            PTP->>PTP: locate source_idx by node id in graph_data.nodes pathfinding.rs:541-545
            alt source not found
                PTP-->>C: 500 Source node not found in graph pathfinding.rs:547-556
            else
                PTP->>GM: ComputeSSP{source_idx} via gpu_manager_addr pathfinding.rs:539, message pathfinding.rs:559-565
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
    participant CFG as configure_pathfinding_routes<br/>semantic_pathfinding_handler.rs:145
    participant FSP as find_semantic_path<br/>semantic_pathfinding_handler.rs:29
    participant QT as query_traversal<br/>semantic_pathfinding_handler.rs:60
    participant CT as chunk_traversal<br/>semantic_pathfinding_handler.rs:90
    participant GSA as GraphServiceSupervisor (web::Data Addr)<br/>graph_service_supervisor.rs:1490
    participant SVC as SemanticPathfindingService<br/>semantic_pathfinding_service.rs:86

    Note over CFG: mounted at web-scope pathfinding by main.rs:1176, distinct from the analytics-scope path family in VC-15.6 to VC-15.9
    Note over CFG: the scope is wrapped in one process-wide rate limiter, 30 requests per minute keyed on the verified pubkey with an IP fallback for unauthenticated reads (semantic_pathfinding_handler.rs:121-139, :148, PATHFINDING_REQUESTS_PER_MINUTE :119)
    Note over QT,CT: traversal_budget defaults max_nodes to 50 and caps it at MAX_TRAVERSAL_NODES = 1000 (semantic_pathfinding_handler.rs:117, :141-142)

    C->>CFG: POST /pathfinding/semantic-path {startId,endId,query}
    CFG->>FSP: find_semantic_path(request)
    FSP->>GSA: GetGraphData (semantic_pathfinding_handler.rs:39-41)
    alt Ok(graph_data)
        FSP->>SVC: find_semantic_path(graph,start_id,end_id,query) (semantic_pathfinding_handler.rs:45, semantic_pathfinding_service.rs:98)
        SVC->>SVC: A* over build_adjacency_list with calculate_semantic_cost per edge (semantic_pathfinding_service.rs:108-193)
        alt max_explored (1000) reached before goal
            SVC-->>FSP: None (semantic_pathfinding_service.rs:136-138)
        else path found
            SVC-->>FSP: Some(PathResult{path,cost,relevance,explanation}) (semantic_pathfinding_service.rs:128-133)
        end
        alt Some(path)
            FSP-->>C: 200 path
        else None
            FSP-->>C: error_json No path found (semantic_pathfinding_handler.rs:52)
        end
    else Err(e)
        FSP-->>C: error_json Graph error (semantic_pathfinding_handler.rs:55)
    end

    C->>CFG: POST /pathfinding/query-traversal {startId,query,maxNodes}
    CFG->>QT: query_traversal(request)
    QT->>GSA: GetGraphData (semantic_pathfinding_handler.rs:67-69)
    alt query is None
        QT-->>C: error_json Query parameter required (semantic_pathfinding_handler.rs:82)
    else
        QT->>SVC: query_traversal(graph,start_id,query,traversal_budget(max_nodes)) (semantic_pathfinding_handler.rs:74-79, semantic_pathfinding_service.rs:202)
        SVC-->>QT: results
        QT-->>C: 200 results
    end

    C->>CFG: POST /pathfinding/chunk-traversal {startId,maxNodes}
    CFG->>CT: chunk_traversal(request)
    CT->>GSA: GetGraphData (semantic_pathfinding_handler.rs:97-99)
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

    NOTE["Trust table docs/GPU-wire-abi.md:189-202 (ADR-2061, 2026-09-07): PageRank max delta 3.4e-11,<br/>DBSCAN exact, Louvain 16 of 16 communities, LOF corrected from 0.702 to 4.759e-7 on the 1e-3 bar<br/>Test crates/visionclaw-gpu/tests/analytics_oracle_conformance.rs - these are recorded results, not re-run here"]
    LIVE -.-> NOTE
    ORACLE -.->|"fixture crate only, not wired into any runtime request path"| LIVE
    TEL["Runtime reach: PageRank and community detection have no working HTTP route at this pin, see VC-15.3 and VC-15.5"]
    LIVE -.-> TEL
```

**Debt:** two analytics routes cannot reach a GPU kernel at the pin. `POST /analytics/community/detect` sends `RunCommunityDetection` to `ForceComputeActor` (`src/handlers/api_handler/analytics/community.rs:61-64`, `:91`), whose handler only returns an error (`src/actors/gpu/force_compute_actor.rs:3347-3352`). The PageRank routes read `AppState.analytics.pagerank` (`src/handlers/api_handler/analytics/pagerank_handlers.rs:121`), which is built as `None` (`src/app_state.rs:1296-1300`) and never assigned. Both have working supervised paths through `GPUManagerActor` (`src/actors/gpu/gpu_manager_actor.rs:375`, `:850`), the pattern ADR-2053 already applied to the pathfinding routes. Closing this needs both routes moved to `gpu_manager_addr`, plus route-level tests that fail on the stub replies.

**Debt:** `POST /analytics/sssp/compute` computes a full CPU Dijkstra and then discards it, answering `SSSP computation started` (`src/handlers/api_handler/analytics/sssp_handlers.rs:226-232`).

**Drift:** before this restamp the topic placed the Louvain D1 marker at `gpu_clustering_kernels.cu:581` (a loop line; the marker is at `:606`) and described the APSP kernel as `#if 0`-guarded, but ADR-2054 had already removed it (`crates/visionclaw-gpu/src/cuda_sources/gpu_landmark_apsp.cu:13-25`).
