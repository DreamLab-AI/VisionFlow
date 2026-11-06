---
id: VC-25
title: Insight loop, KPI, briefing, NLQ and semantic classification
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2014, ADR-2040, ADR-2004, ADR-2063, ADR-2065]
sources:
  - ../project/src/services/insight_loop.rs
  - ../project/src/handlers/insight_loop_handler.rs
  - ../project/src/adapters/sqlite_enrichment_repository.rs
  - ../project/src/services/kpi_compute.rs
  - ../project/src/adapters/sqlite_kpi_repository.rs
  - ../project/src/handlers/kpi_handler.rs
  - ../project/src/services/briefing_service.rs
  - ../project/src/handlers/briefing_handler.rs
  - ../project/src/services/natural_language_query_service.rs
  - ../project/src/services/jev_nlq.rs
  - ../project/src/handlers/natural_language_query_handler.rs
  - ../project/src/services/perplexity_service.rs
  - ../project/src/services/semantic_analyzer.rs
  - ../project/src/services/semantic_type_registry.rs
  - ../project/src/services/edge_classifier.rs
  - ../project/src/actors/semantic_processor_actor.rs
  - ../project/src/actors/messages/physics_messages.rs
  - ../project/src/actors/messages/analytics_messages.rs
  - ../project/crates/visionclaw-actors/src/messages/analytics_messages.rs
  - ../project/src/actors/metadata_actor.rs
  - ../project/crates/visionclaw-actors/src/messages/graph_messages.rs
  - ../project/src/services/file_service.rs
  - ../project/src/services/graph_serialization.rs
  - ../project/src/app_state.rs
  - ../project/data/metadata/metadata.json
  - ../project/src/handlers/ontology_handler.rs
  - ../project/src/ports/knowledge_graph_repository.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/src/services/liveness_harness.rs
  - ../project/src/services/management_api_client.rs
  - ../project/src/services/nostr_bead_publisher.rs
  - ../project/src/services/ontology_enrichment_service.rs
  - ../project/src/main.rs
  - ../project/src/test_helpers.rs
  - ../project/src/adapters/actix_semantic_adapter.rs
  - ../project/src/handlers/semantic_handler.rs
  - ../project/src/handlers/graph_access.rs
  - ../project/src/services/schema_service.rs
  - ../project/src/handlers/api_handler/analytics/visible_analytics.rs
  - ../project/src/services/pathfinding.rs
  - ../project/src/handlers/graph_export_handler.rs
  - ../project/docs/SECURITY-profiles.md
  - ../project/src/utils/response_macros.rs
verified_commit: {visionclaw: 809a859aae9e3d96d4d6b99690378018e48e3ad8}
---

## For developers

This topic covers the read-side analytics: the insight loop trace, KPI compute and lineage, the briefing service, natural-language queries to SPARQL, and the semantic processor and edge classifier. At ec785afa9 the `/api/semantic` routes filter their answers per caller. At 809a859aa a gated caller's communities and shortest paths are computed over that caller's view instead; centrality is still filtered after a whole-graph run, which is sound only while the analyser returns empty results, as a tripwire test enforces (VC-25.14).

## For the business

These features turn the graph into answers: health indicators, briefings and plain-English questions. The community and importance views on the semantic routes return nothing yet, so they are safe to expose but not yet useful.

## VC-25.1 Insight loop trace assembly (REC-10, compute-on-read)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as insight_loop_handler<br/>src/handlers/insight_loop_handler.rs:53
    participant R as SqliteEnrichmentRepository<br/>src/adapters/sqlite_enrichment_repository.rs:853
    participant L as insight_loop::summarise<br/>src/services/insight_loop.rs:209
    participant B as insight_loop::build_trace<br/>src/services/insight_loop.rs:103
    participant LH as LivenessHarness<br/>src/services/liveness_harness.rs

    Note over H,B: DIVERGENCE: no tokio::time::interval scheduler exists for this loop -<br/>every stage is computed fresh on each GET (compute-on-read), not a periodic job
    C->>H: GET /api/insight-loop/trace?limit=
    H->>R: loop_traces(limit) - SELECT p JOIN d ON MAX(decided_at_ms)<br/>src/adapters/sqlite_enrichment_repository.rs:853-870
    R-->>H: Vec~LoopTraceRow~<br/>src/adapters/sqlite_enrichment_repository.rs:288
    H->>L: summarise(rows)
    loop for each LoopTraceRow in rows
        L->>B: build_trace(row)
        B->>B: stage_instants(row) - propose/queued from body or created_at*1000<br/>src/services/insight_loop.rs:88
        Note over B: INVARIANT: stage order propose(35)->queued(36)->broker_decision(37)->merged_enrichment(38)->amplification(39)
        alt writeback_committed_at_ms Some
            B->>B: merged status = complete, loop_closed = true
        else decided_at_ms Some and writeback_triggered Some(true)
            B->>B: merged status = pending
        else decided_at_ms Some
            B->>B: merged status = not_applicable (rejection/unattributed - never fabricated pending)
        else decided_at_ms None
            B->>B: merged status = pending (still queued)
        end
        B->>B: amplification stage = planned, at_ms=None always<br/>src/services/insight_loop.rs:159-165
        B->>B: monotonic = windows(2).all(w[0]<=w[1])<br/>src/services/insight_loop.rs:177
        B-->>L: InsightLoopTrace
    end
    L->>L: mesh_velocity_mean_ms = mean over closed loops only, None if none closed<br/>src/services/insight_loop.rs:213-217
    L-->>H: InsightLoopSummary
    opt a closed and monotonic trace is present
        H->>LH: observe(CANARY_REC10_LOOP, evidence)<br/>src/handlers/insight_loop_handler.rs:35-50
    end
    H-->>C: 200 InsightLoopSummary
```

## VC-25.2 insight_loop_handler routes

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant Cfg as configure_routes<br/>src/handlers/insight_loop_handler.rs:89
    participant T as traces<br/>src/handlers/insight_loop_handler.rs:53
    participant TC as trace_by_case<br/>src/handlers/insight_loop_handler.rs:68
    participant R as SqliteEnrichmentRepository<br/>src/adapters/sqlite_enrichment_repository.rs:888

    Note over Cfg: scope /insight-loop mounted under /api - src/handlers/insight_loop_handler.rs:89-94
    C->>T: GET /insight-loop/trace?limit=N
    Note right of T: limit.clamp(1, MAX_LIMIT=1000), default DEFAULT_LIMIT=100<br/>src/handlers/insight_loop_handler.rs:25-26,54
    T-->>C: 200 InsightLoopSummary
    C->>TC: GET /insight-loop/trace/{case_id}
    TC->>R: loop_trace_for(case_id)<br/>src/adapters/sqlite_enrichment_repository.rs:888
    alt Ok(Some(row))
        R-->>TC: LoopTraceRow
        TC-->>C: 200 InsightLoopTrace
    else Ok(None)
        TC-->>C: 404 not-found
    else Err(e)
        TC-->>C: 500 loop-trace read failed
    end
```

## VC-25.3 KPI compute and persist (kpi_snapshots / kpi_lineage / kpi_agent_events)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant K as compute_and_persist<br/>src/services/kpi_compute.rs:387
    participant KR as SqliteKpiRepository<br/>src/adapters/sqlite_kpi_repository.rs:274
    participant ER as SqliteEnrichmentRepository<br/>src/adapters/sqlite_enrichment_repository.rs
    participant LH as LivenessHarness<br/>src/services/liveness_harness.rs

    Note over K: KPI_WINDOW_MS = 30 days rolling window (ADR-043 legacy ref, not in docs/adr/)<br/>src/services/kpi_compute.rs:45
    C->>K: compute_and_persist()
    K->>KR: count_agent_events_since(window_start) - kpi_agent_events<br/>src/adapters/sqlite_kpi_repository.rs:404
    KR-->>K: agent_volume
    K->>ER: decisions_since(window_start) - enrichment_decisions<br/>src/adapters/sqlite_enrichment_repository.rs:728
    ER-->>K: decisions (outcome, activity_urn)
    K->>ER: decided_cases_since(window_start)<br/>src/adapters/sqlite_enrichment_repository.rs:761
    ER-->>K: decided-case rows (case, outcome, decided_by)
    K->>K: augmentation_ratio(agent_volume, escalation_volume)<br/>src/services/kpi_compute.rs:82, called at :423
    alt escalation_volume == 0
        K->>K: (value=0.0, confidence=0.0) - undefined ratio, never Inf/NaN
    else escalation_volume > 0
        K->>K: value = agent_volume/escalation_volume, confidence = sample_confidence(sum)
    end
    critical persist augmentation_ratio snapshot + lineage in one transaction
        K->>KR: insert_snapshot_with_lineage(ar_snapshot, ar_lineage)<br/>src/adapters/sqlite_kpi_repository.rs:427
        KR->>KR: tx = c.transaction() then INSERT kpi_snapshots then INSERT kpi_lineage per row then tx.commit()<br/>src/adapters/sqlite_kpi_repository.rs:436, :440, :461
        KR-->>K: ar_id
    end
    K->>K: trust_variance(outcomes) - Gini-Simpson 1-sum(p^2) normalised<br/>src/services/kpi_compute.rs:98, called at :458
    Note over K: ADR-2110 INVARIANT: an outcome decided by the reserved non-DID actor<br/>system:whelk-gate is excluded from the Trust Variance human-outcome<br/>series and from BOTH terms of HITL Precision, so a reasoner verdict<br/>cannot read as a human override.<br/>src/services/kpi_compute.rs:140, :143
    critical persist trust_variance snapshot + lineage in one transaction
        K->>KR: insert_snapshot_with_lineage(tv_snapshot, tv_lineage)
        KR-->>K: tv_id
    end
    K->>K: decided_cases_with_intent then hitl_precision<br/>src/services/kpi_compute.rs:270, :213, called at :419, :420
    K->>LH: observe(CANARY_REC4_KPI, evidence)<br/>src/services/kpi_compute.rs:510
    K-->>C: KpiSummary with ar_tile, tv_tile, mesh_tile, hitl_tile<br/>src/services/kpi_compute.rs:622
    Note over K: ADR-2110 FR5.3: hitl_tile is no longer a stub. It is computed as<br/>warranted over decided across human-decided cases, and at a zero<br/>denominator it reports status no_decided_cases with value None and<br/>the denominator still stated - never a number.<br/>src/services/kpi_compute.rs:554, :556, :559
    Note over K: mesh_tile alone still reports status awaiting_data_source with no value -<br/>never fabricated. src/services/kpi_compute.rs:330, :545

    Note over KR: separate passive tap task (not this request path):
    loop rx.recv().await on /wss/agent-events hub - run_agent_event_tap src/services/kpi_compute.rs:646, :650
        KR->>KR: record_agent_trajectory - INSERT kpi_agent_events with event_id,<br/>source_agent_id, action_type, agent_did, handoff_id and the<br/>ADR-2110 intent column<br/>src/adapters/sqlite_kpi_repository.rs:334, :245
    end
```

## VC-25.4 kpi_handler read routes

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant Cfg as configure_routes<br/>src/handlers/kpi_handler.rs:47
    participant S as summary handler<br/>src/handlers/kpi_handler.rs:25
    participant Lin as lineage handler<br/>src/handlers/kpi_handler.rs:33
    participant K as KpiComputeService<br/>src/services/kpi_compute.rs:359

    Note over Cfg: scope /kpi mounted under /api - src/handlers/kpi_handler.rs:47-52
    C->>S: GET /kpi/summary
    S->>K: compute_and_persist() (see VC-25.3)
    alt Ok(summary)
        S-->>C: 200 KpiSummary
    else Err(e)
        S-->>C: 500 {error: e}
    end
    C->>Lin: GET /kpi/lineage/{snapshot_id}
    Lin->>K: lineage_for(snapshot_id)<br/>src/services/kpi_compute.rs:630
    K->>K: kpi_repo.lineage_for(snapshot_id) - SELECT kpi_lineage WHERE snapshot_id<br/>src/adapters/sqlite_kpi_repository.rs:505
    alt Ok(rows)
        Lin-->>C: 200 {snapshot_id, lineage: Vec~KpiLineageRow~}
    else Err(e)
        Lin-->>C: 500 {error: e}
    end
```

## VC-25.6 Briefing service and handler (submit + debrief)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as briefing_handler<br/>src/handlers/briefing_handler.rs:22
    participant BS as BriefingService<br/>src/services/briefing_service.rs:11
    participant MA as ManagementApiClient<br/>src/services/management_api_client.rs
    participant NP as NostrBeadPublisher<br/>src/services/nostr_bead_publisher.rs

    Note over H: routes: POST /briefs (submit_brief), POST /briefs/{brief_id}/debrief (request_debrief)<br/>src/handlers/briefing_handler.rs:116-122
    C->>H: POST /api/briefs {briefing, user_context}
    H->>BS: submit_brief(request, user_context)<br/>src/services/briefing_service.rs:24
    BS->>MA: create_brief(content, roles, user_context, version, brief_type, slug)
    MA-->>BS: brief_result{brief_id, brief_path, bead_id}
    BS->>MA: execute_brief(brief_id, brief_path, roles, user_context, bead_id)
    MA-->>BS: role_tasks: Vec~RoleTask~
    alt Ok(response)
        BS-->>H: BriefingResponse
        H-->>C: 201 Created {brief_id, brief_path, bead_id, role_tasks}
    else Err(BriefingError::ApiError(msg))
        BS-->>H: BriefingError::ApiError<br/>src/services/briefing_service.rs:104
        H-->>C: 502 BadGateway {error, message}
    end
    C->>H: POST /api/briefs/{brief_id}/debrief {role_tasks, user_context}
    H->>H: bead_id = first role_task.bead_id or brief_id<br/>src/handlers/briefing_handler.rs:69-74
    H->>BS: request_debrief(brief_id, role_tasks, user_context)<br/>src/services/briefing_service.rs:80
    BS->>MA: create_debrief(brief_id, role_tasks, user_context)
    alt Ok(debrief_path)
        opt nostr_publisher configured
            H->>NP: tokio::spawn publish_bead_complete(bead_id, brief_id, user_pubkey, debrief_path)<br/>src/handlers/briefing_handler.rs:88-97
            Note right of NP: fire-and-forget - does not affect the HTTP response
        end
        H-->>C: 201 Created {brief_id, debrief_path}
    else Err(BriefingError::ApiError(msg))
        H-->>C: 502 BadGateway {error, message}
    end
```

## VC-25.7 Natural language query: parse -> plan -> execute

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as translate_query<br/>src/handlers/natural_language_query_handler.rs:92
    participant S as NaturalLanguageQueryService<br/>src/services/natural_language_query_service.rs:36
    participant SC as SchemaService<br/>src/services/schema_service.rs
    participant P as PerplexityService::chat_completion<br/>src/services/perplexity_service.rs:77

    Note over S,P: config-driven, not an env flag - api_url/api_key/model read from<br/>AppFullSettings.perplexity at call time (src/services/perplexity_service.rs:103-113)
    C->>H: POST /nl-query/translate {query, suggestAlternatives}
    H->>S: translate_to_sparql(query) or suggest_queries(query)<br/>src/services/natural_language_query_service.rs:72,101
    Note over S: Jev option, opt-in and off unless VISIONCLAW_JEV is set (src/main.rs:517-522): translate_to_sparql first calls try_jev_plan,<br/>an enumerated read-only template chosen by Jev with a code-extracted entity, accepted only above<br/>VISIONCLAW_JEV_AUTO_MIN_CONFIDENCE and re-checked by validate_read_only_sparql, else the generative route below runs unchanged<br/>src/services/natural_language_query_service.rs:65,76-78 src/services/jev_nlq.rs:148, :209 - suggest_queries has no Jev path (src/services/natural_language_query_service.rs:101)
    S->>SC: get_llm_context() - schema for the prompt
    SC-->>S: schema_context
    S->>S: get_system_prompt() then build_translation_prompt(query, schema_context)<br/>src/services/natural_language_query_service.rs:163,225
    Note over S: RESOLVED ADR-2063 - the prompt now describes the real Oxigraph named graphs and vocabulary<br/>and asks for read-only SPARQL. It previously claimed to be a SPARQL generator while instructing and<br/>validating for Cypher GraphNode/EDGE labels, so every generated query was unrunnable against the store.
    S->>P: chat_completion([(system,...), (user, prompt)])
    alt perplexity_config missing (api_url/api_key/model None)
        P-->>S: Err("Perplexity API URL/Key/model not configured")
        S-->>H: Err("LLM service error: ...")
        H-->>C: 500 {error: "Translation failed", message} via error_json<br/>src/handlers/natural_language_query_handler.rs:119, src/utils/response_macros.rs:82
    else config present
        P-->>S: response text
        S->>S: extract_sparql_block(response) - find fenced sparql block or fallback fence<br/>src/services/natural_language_query_service.rs:300
        alt no fenced block found
            S-->>H: Err - no SPARQL query found in response
            H-->>C: 500 {error: "Translation failed", message}
        else block found
            S->>S: validate_sparql(sparql) - delegates to the shared read-only validator<br/>src/services/natural_language_query_service.rs:129
            S->>S: validate_read_only_sparql - SELECT/ASK/CONSTRUCT/DESCRIBE only<br/>src/handlers/ontology_handler.rs:812
            alt mutating form (INSERT/DELETE/DROP/CLEAR/LOAD)
                S-->>H: Err - rejected, not a read-only query
                H-->>C: 500 {error: "Translation failed", message}
            else valid read-only SPARQL
                S-->>H: QueryTranslation{sparql_query, explanation, confidence, warnings}
                H-->>C: 200 {translations: [...]}
            end
        end
    end
    Note over S,H: INVARIANT one validator - the NLQ path reuses ontology_handler::validate_read_only_sparql<br/>rather than defining a second read-only policy (ADR-2063)
    C->>H: POST /nl-query/validate {sparql}
    H->>S: validate_sparql(sparql)<br/>src/handlers/natural_language_query_handler.rs:212
    H-->>C: 200 {valid, errors}
    C->>H: POST /nl-query/explain {sparql}
    H->>S: validate_sparql then explain_sparql(sparql)<br/>src/handlers/natural_language_query_handler.rs:170
    alt rejected by the read-only validator
        H-->>C: 500 {error: "Invalid SPARQL syntax", message}<br/>src/handlers/natural_language_query_handler.rs:178
    else valid
        S->>P: chat_completion - explain prompt<br/>src/services/natural_language_query_service.rs:134
        H-->>C: 200 {sparql, explanation}
    end
    C->>H: GET /nl-query/examples
    H-->>C: 200 {examples: QueryPatterns::examples()}
    Note over H: routes registered by configure_nl_query_routes - translate, examples, explain, validate<br/>src/handlers/natural_language_query_handler.rs:236-244
```

**Tension:** every refusal on the NLQ routes is reported as a server fault. `translate_query` and `explain_sparql` answer a failed translation, an LLM reply with no SPARQL block, a mutating query and an invalid query through `error_json!` (`../project/src/handlers/natural_language_query_handler.rs:119`, `:178`), which builds `HttpResponse::InternalServerError` (`../project/src/utils/response_macros.rs:82`). A caller cannot tell "your query was refused" from "the service broke" by status; only `/nl-query/validate` answers 200 with `{valid, errors}`. Earlier stamps of this panel showed these as 200.

## VC-25.8 SemanticProcessorActor message handling

```mermaid
sequenceDiagram
    autonumber
    participant Caller as GraphServiceSupervisor / handler
    participant A as SemanticProcessorActor<br/>src/actors/semantic_processor_actor.rs:171

    Note over A: since af3dff3f2 the actor has a production owner: main registers a SemanticService over ActixSemanticAdapter for /api/semantic/*, and the adapter starts this actor on its first initialize (estate register ER-25)<br/>src/main.rs:860-870 src/adapters/actix_semantic_adapter.rs:95-103
    Note over A: Actor started/stopped log config.enable_ai_features and final stats<br/>src/actors/semantic_processor_actor.rs:1361-1376

    Caller->>A: UpdateConstraints{constraint_data}<br/>src/actors/messages/physics_messages.rs:292
    A-->>Caller: Result<(),String> via handle_constraint_update<br/>src/actors/semantic_processor_actor.rs:1381-1388

    Caller->>A: GetConstraints<br/>src/actors/messages/physics_messages.rs:276
    A-->>Caller: ConstraintSet clone<br/>src/actors/semantic_processor_actor.rs:1390-1399

    Caller->>A: TriggerStressMajorization<br/>src/actors/messages/physics_messages.rs:321
    A->>A: web::block(stress optimisation) on thread pool<br/>src/actors/semantic_processor_actor.rs:1402
    alt thread pool Ok(Ok(()))
        A-->>Caller: Ok(())
    else Ok(Err(e)) or thread pool error
        A-->>Caller: Err(e) / "Thread pool error"
    end

    Caller->>A: RegenerateSemanticConstraints<br/>src/actors/messages/physics_messages.rs:341
    A->>A: deactivate semantic_similarity/semantic_clustering/importance_based/topic_based groups<br/>src/actors/semantic_processor_actor.rs:1438-1446
    A->>A: web::block(generate_semantic_constraints_blocking)
    A-->>Caller: ResponseFuture<Result<(),String>>

    Note over A: REMOVED since af3dff3f2: no Handler for UpdateAdvancedParams remains - the message left physics_messages.rs and the actor<br/>dropped its advanced_params, relationship_threshold and constraint_cache fields, so the solver is no longer rebuilt from AdvancedParams<br/>src/actors/semantic_processor_actor.rs:31-34 (import list)

    Caller->>A: SetGraphData{graph_data}<br/>src/actors/semantic_processor_actor.rs:1490
    Caller->>A: ProcessMetadata{metadata_id, metadata}<br/>src/actors/semantic_processor_actor.rs:1505
    A->>A: web::block(process_metadata_blocking)<br/>src/actors/semantic_processor_actor.rs:1510-1530

    Caller->>A: GetSemanticStats<br/>src/actors/semantic_processor_actor.rs:1539
    A-->>Caller: SemanticStats clone

    Caller->>A: UpdateSemanticConfig{config}<br/>src/actors/semantic_processor_actor.rs:1551
    A-->>Caller: () via update_config

    Caller->>A: ComputeShortestPaths{source_node_id}<br/>src/actors/messages/analytics_messages.rs:147
    alt gpu_analyzer is None
        A-->>Caller: Err("GPU analyzer not available")<br/>src/actors/semantic_processor_actor.rs:1574-1577
    else gpu_analyzer present
        A->>A: gpu_analyzer.initialize(graph) then compute_shortest_paths(source_node_id)<br/>src/actors/semantic_processor_actor.rs:1583-1591
        A-->>Caller: Ok(PathfindingResult) / Err("Pathfinding failed")
    end

    Caller->>A: ComputeAllPairsShortestPaths<br/>crates/visionclaw-actors/src/messages/analytics_messages.rs:166
    alt gpu_analyzer is None
        A-->>Caller: Err("GPU analyzer not available")<br/>src/actors/semantic_processor_actor.rs:1622-1625
    else present
        A->>A: compute_all_pairs_shortest_paths() - GPU landmark approximation<br/>src/actors/semantic_processor_actor.rs:1631-1638
        A-->>Caller: Ok(HashMap<(u32,u32),Vec~u32~>) / Err("APSP failed")
    end
```

## VC-25.9 semantic_analyzer feature extraction + semantic_type_registry lookup/register

ADR-2114 unified corpus reads (local vault + GitHub via one `CorpusSource`) so `relation_edges` now
takes a `ParsedPage` and vault-core `Vocabulary` rather than JSON-LD; the registry lookup pattern is
otherwise unchanged. `SemanticAnalyzer` is a separate, unrelated caller shown for the same actor.

```mermaid
sequenceDiagram
    autonumber
    participant SPA as SemanticProcessorActor::process_metadata_blocking<br/>src/actors/semantic_processor_actor.rs:246
    participant SA as SemanticAnalyzer::analyze_metadata<br/>src/services/semantic_analyzer.rs:261
    participant GS as GitHubSyncService::relation_edges<br/>src/services/github_sync_service.rs:1878
    participant REG as SemanticTypeRegistry<br/>src/services/semantic_type_registry.rs:90

    SPA->>SA: analyze_metadata(metadata)
    alt config.enable_caching and id cached
        SA-->>SPA: cached SemanticFeatures clone<br/>src/services/semantic_analyzer.rs:264-267
    else not cached
        SA->>SA: extract_topics, classify_domains, extract_temporal/structural/content_features
        SA->>SA: calculate_importance_score(topics, temporal, structural)<br/>src/services/semantic_analyzer.rs:270-286
        SA->>SA: feature_cache.insert(id, features.clone())<br/>src/services/semantic_analyzer.rs:288-290
        SA-->>SPA: SemanticFeatures
    end
    Note over SA: compute_similarity weights: topic 0.4 + domain 0.2 + file_type 0.1 + depth 0.1 + temporal 0.1 + importance 0.1<br/>src/services/semantic_analyzer.rs:545-580

    Note over GS: one typed edge per vocab.relations key whose ParsedPage frontmatter has<br/>a wikilink target — an unmapped edge-type predicate is skipped, not defaulted<br/>src/services/github_sync_service.rs:1888-1898
    GS->>REG: get_or_register_id(edge_type)<br/>src/services/semantic_type_registry.rs:652,src/services/github_sync_service.rs:1899
    REG->>REG: get_id(uri) - read_uri_map lookup<br/>src/services/semantic_type_registry.rs:646-648
    alt uri already registered
        REG-->>GS: existing id
    else uri unknown
        REG->>REG: register(uri, RelationshipForceConfig::default())<br/>src/services/semantic_type_registry.rs:628
        REG->>REG: register_internal(uri, config) - assign next_id, push uri/config<br/>src/services/semantic_type_registry.rs:612
        REG-->>GS: new id
    end
    GS->>REG: get_config(reg_id) - strength*2.0 normalised to spring weight<br/>src/services/semantic_type_registry.rs:662,src/services/github_sync_service.rs:1901-1902
    Note over REG: version() = next_id atomic counter, used for hot-reload detection<br/>src/services/semantic_type_registry.rs:703
```

## VC-25.10 edge_classifier classification rules

```mermaid
flowchart TD
    Start(["classify_edge(source_label, target_label, source_class, target_class, context)<br/>src/services/edge_classifier.rs:189"]) --> Lower["context_lower = context.to_lowercase()"]
    Lower --> Loop["for each Pattern in patterns.values()<br/>src/services/edge_classifier.rs:202"]
    Loop --> Match{"keyword found in context_lower?"}
    Match -->|"yes, matches>0"| Score["avg_score = sum(pattern.confidence)/matches<br/>track best_match by highest avg_score"]
    Match -->|no| Loop
    Score --> Loop
    Loop --> Done{"best_match found after all patterns?"}
    Done -->|yes| ReturnMatch["return Some(property_iri) - e.g. mv:hasCEO(0.95), mv:worksAt(0.9), mv:hasFounder(0.9)<br/>src/services/edge_classifier.rs:38-152"]
    Done -->|no| Fallback{"source_class and target_class both Some?"}
    Fallback -->|yes| ClassPair["classify_by_class_pair(src,tgt)<br/>src/services/edge_classifier.rs:256"]
    ClassPair --> Pairs{"Person->Company / Person->Project /<br/>Company->Project / Project->Technology / Concept->Concept?"}
    Pairs -->|match| ReturnFallback["return Some(mapped property_iri)"]
    Pairs -->|no match| ReturnNone["return None - no classification"]
    Fallback -->|no| ReturnNone
    ReturnMatch --> Caller["ontology_enrichment_service::enrich_edges<br/>src/services/ontology_enrichment_service.rs:138"]
    ReturnFallback --> Caller
    ReturnNone --> Jev["opt-in Jev gap-fill only for a None result: accepted above VISIONCLAW_JEV_AUTO_MIN_CONFIDENCE and the domain/range gate,<br/>else property_iri stays None - src/services/ontology_enrichment_service.rs:146-163"]
    Jev --> Caller
    Debt["DEBT: OntologyEnrichmentService has no production constructor - only test_helpers builds it and with_jev is called by tests alone,<br/>so enrich_edges and the Jev gap-fill do not run in the server - src/test_helpers.rs:266-270, src/services/ontology_enrichment_service.rs:36"]
    Caller -.- Debt
```

## VC-25.12 MetadataActor message handling

```mermaid
sequenceDiagram
    autonumber
    participant Caller as AppState / handler
    participant MA as MetadataActor<br/>src/actors/metadata_actor.rs:23

    Note over MA: replaces Arc<RwLock<MetadataStore>> - started at src/app_state.rs:763 (BASELINE-architecture.md "Actor system topology")
    Caller->>MA: GetMetadata<br/>crates/visionclaw-actors/src/messages/graph_messages.rs:221
    MA-->>Caller: Ok(MetadataStore clone)<br/>src/actors/metadata_actor.rs:58-64
    Caller->>MA: UpdateMetadata{metadata}<br/>crates/visionclaw-actors/src/messages/graph_messages.rs:225
    MA->>MA: update_metadata(new_metadata) - self.metadata = new_metadata<br/>src/actors/metadata_actor.rs:36-39,66-73
    MA-->>Caller: Ok(())
    Note right of MA: RESOLVED ADR-2097 (2026-09-05): RefreshMetadata and refresh_metadata are deleted - the message had no senders<br/>and the actor holds no source. metadata.json is owned by FileService, which pushes rebuilt stores in via UpdateMetadata
```

**Drift:** the governing baseline places MetadataActor's start at `app_state.rs:774` (`../project/docs/BASELINE-architecture.md:170`); at this revision the actor starts at `../project/src/app_state.rs:763`, with its log line at `:762`.

## VC-25.13 file_service + graph_serialization + empty-graph guard

```mermaid
sequenceDiagram
    autonumber
    participant Boot as AppState::new (startup)
    participant FS as FileService::load_graph_from_files<br/>src/services/file_service.rs:1089
    participant Repo as KnowledgeGraphRepository (Oxigraph)<br/>src/ports/knowledge_graph_repository.rs
    participant GSS as GraphSerializationService::export_graph<br/>src/services/graph_serialization.rs:29

    Boot->>FS: load_graph_from_files(graph_repo)
    FS->>Repo: load_graph() - idempotency guard (ADR-2004, not in docs/adr/ ledger)<br/>src/services/file_service.rs:1095
    alt existing.nodes not empty
        Repo-->>FS: existing graph populated
        FS-->>Boot: Ok(()) - skip local-file seed, GitHub sync is authoritative<br/>src/services/file_service.rs:1096-1102
    else store empty or query failed
        FS->>FS: load_or_create_metadata()<br/>src/services/file_service.rs:1115
        alt metadata.is_empty()
            FS-->>Boot: Ok(()) - warn "no data to load", nothing seeded<br/>src/services/file_service.rs:1116-1118
        else metadata present
            FS->>FS: Phase 1 - build AppNode per file, classify ontology_node vs page via owl_class_iri<br/>src/services/file_service.rs:1123-1178
            FS->>FS: Phase 2 - wikilink regex extracts AppEdge set, dedup via seen_edges<br/>src/services/file_service.rs:1183-1194
            FS->>Repo: save_graph(&graph_data)<br/>src/services/file_service.rs:1216
            FS-->>Boot: Ok(())
        end
    end

    Note over FS: the legacy GitHub-to-markdown mirror initialize_local_storage now returns LocalStorageSync and is skipped outright for a local corpus (ADR-2114)<br/>src/services/file_service.rs:301-316
    Note over FS: RESOLVED ADR-2065: src/services/empty_graph_check.rs (check_empty_graph) had zero call sites<br/>and has been deleted - the only live empty-graph guard is the idempotency check at file_service.rs:1095

    Note over GSS: distinct empty-graph handling path - export_graph has no explicit empty check,<br/>writes whatever GraphData it is given, through write_export (src/services/graph_serialization.rs:29-36, :41)
    Boot->>GSS: export_graph(graph, request)
    GSS->>GSS: serialize_to_json/gexf/graphml/csv/dot per ExportFormat<br/>src/services/graph_serialization.rs:59-65
    opt request.compress
        GSS->>GSS: compress_data via GzEncoder (compression_level=6)<br/>src/services/graph_serialization.rs:67-70
    end
    alt file_size > max_file_size (100 MiB)
        GSS-->>Boot: Err("Export file size exceeds limit")<br/>src/services/graph_serialization.rs:76-81
    else within limit
        GSS->>GSS: fs::write(file_path, final_data)
        GSS-->>Boot: ExportResponse{export_id, download_url, expires_at: now+24h}<br/>src/services/graph_serialization.rs:83-88
    end
    Note over GSS: DEBT: download_url is /api/graph/download/id (graph_serialization.rs:85), a route nothing mounts -<br/>the export scope has no download route (src/handlers/graph_export_handler.rs:351-363)
```

## VC-25.14 /api/semantic routes: gated callers computed over their own view

<!-- 2026-11-06 restamp ec785afa9: new. 84ccd9aaa gates the semantic routes per caller, 14c429baf adds the M2 tripwire test -->
<!-- 2026-11-06 restamp 809a859aa: redrawn. 405edcfe1 - communities and shortest paths for a gated caller are computed over the caller's view on the CPU, never a filtered whole-graph run -->
```mermaid
flowchart TB
    R["routes communities, centrality, shortest-path, constraints/generate, statistics, cache/invalidate<br/>src/handlers/semantic_handler.rs:401-413"]
    R --> CG{"caller_gate: None when the view hides nothing<br/>src/handlers/semantic_handler.rs:252-260, graph_access.rs:120"}
    CG -->|"gated, communities"| VC["visible_communities over the view: numbering, sizes, modularity<br/>src/handlers/semantic_handler.rs:89-90, :269-274, visible_analytics.rs:331"]
    CG -->|"gated, shortest-path"| VS["Dijkstra over the view from the source slot<br/>src/handlers/semantic_handler.rs:191-192, :314-321, pathfinding.rs:180"]
    CG -->|"gated, centrality"| CF["full-graph analysis, no top-k cut, then gate_centrality filters<br/>src/handlers/semantic_handler.rs:132, :158, :163, :294"]
    CG -->|"ungated"| SS["SemanticService.initialize(full graph) then the adapter<br/>src/handlers/semantic_handler.rs:92, :200, src/main.rs:864-869"]
    CF --> SS
    SS --> AD["ActixSemanticAdapter port handlers return empty data, modularity 0.0<br/>src/adapters/actix_semantic_adapter.rs:264, :267-273"]
    AD --> EMPTY["shortest paths, SSSP and importance also empty<br/>src/adapters/actix_semantic_adapter.rs:290-297, :309, :371"]
    GEN["constraints/generate takes no caller and runs on the full graph<br/>src/handlers/semantic_handler.rs:220-229"] --> SS
    AD -.-> INV["INVARIANT: centrality filtering is sound only while the analyser is empty,<br/>tripwire actix_semantic_adapter.rs:460, :482-497"]
```

**Invariant:** filtering per caller after the analysis is sound only while the analyser returns nothing. `semantic_analyses_still_return_nothing_so_response_filtering_is_sound` (`src/adapters/actix_semantic_adapter.rs:460`) fails as soon as communities, centrality, shortest paths or SSSP distances return data (`src/adapters/actix_semantic_adapter.rs:482-497`). Communities and shortest paths no longer depend on it for gated callers (`src/handlers/semantic_handler.rs:89-90`, `:191-192`); centrality still does (`src/handlers/semantic_handler.rs:163`).

**Debt (loose end):** the port handlers return empty results for communities, shortest paths, SSSP and importance (`src/adapters/actix_semantic_adapter.rs:267-273`, `:290-297`, `:309`, `:371`). The shortest-path stub builds a `ComputeShortestPaths` message and drops it (`src/adapters/actix_semantic_adapter.rs:285-288`). The GPU-backed `ComputeShortestPaths` handler (`src/actors/semantic_processor_actor.rs:1564`) is never reached from `/api/semantic`, so an ungated caller gets 200 with empty data while a gated caller gets real CPU results.

**Tension (parallel):** centrality is the one gated semantic route still filtered after a whole-graph run (`src/handlers/semantic_handler.rs:132`, `:163`), while the analytics PageRank already has a view-local `visible_pagerank` (`src/handlers/api_handler/analytics/visible_analytics.rs:408`). The day the adapter returns scores, a gated caller's visible scores will have been shaped by hidden nodes.

**Open (risk):** the NLQ prompt context lists up to ten `owl_class_iri` values and a total drawn from the whole graph (`src/services/schema_service.rs:145-146`, `:83-89`), sent to the LLM for every translation (`src/services/natural_language_query_service.rs:81`), and Jev templates query the ontology graph directly (`src/services/jev_nlq.rs:56`). SECURITY-profiles 0.1.6 names `jev_nlq` and `natural_language_query_service` as outside the ontology deny-list (`docs/SECURITY-profiles.md:111`).
