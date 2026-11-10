---
id: VC-61
title: Reasoner, sync-pipeline and node-thinning switches — five stored keys and where each acts
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2192-the-reasoner-the-sync-pipeline-and-node-thinning-are-stored-operator-switches.md
adrs: [ADR-2192, ADR-2140, ADR-2183, ADR-2189, ADR-2173, ADR-2071]
sources:
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
  - ../project/src/settings/models.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/src/actors/messages/physics_messages.rs
  - ../project/src/actors/gpu/ontology_constraint_actor.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/physics_supervisor.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/src/handlers/api_handler/ontology_physics/mod.rs
  - ../project/src/actors/client_filter.rs
  - ../project/src/actors/client_coordinator_actor.rs
  - ../project/src/handlers/api_handler/graph/mod.rs
  - ../project/docs/adr/ADR-2192-the-reasoner-the-sync-pipeline-and-node-thinning-are-stored-operator-switches.md
verified_commit: {visionclaw: cae4e7844ddeb907caee726d570babe94fefa3d7}
---

## For developers

ADR-2192 (VisionClaw PR 38) turns the reasoner's layout constraints, the sync pipeline's reasoning and materialisation, and node thinning into stored switches, retires three dead copies of the constraint switch, and deletes `OntologyPipelineService`. Read with VC-20 (ontology pipeline), VC-58 (edge producers) and VC-19 (the node filter).

## For the business

An operator can now turn off the reasoner's pull on the layout, stop the reasoner or the derived links from running at the next sync, and cap how many nodes a viewer is sent, without code changes. The cap thins what each viewer sees, not the shared simulation.

## VC-61.1 Constraint switch and strength: stored physics keys, sent as a policy

```mermaid
sequenceDiagram
    autonumber
    participant PATCH as apply_physics_patch<br/>settings_routes.rs:679
    participant PUSH as push_ontology_constraint_policy_if_changed<br/>settings_routes.rs:620
    participant SUP as restore_saved_physics<br/>physics_supervisor.rs:204
    participant OCA as OntologyConstraintActor<br/>ontology_constraint_actor.rs:676
    participant FCA as constraint_sync_action<br/>force_compute_actor.rs:1776
    Note over PATCH: physics.ontologyConstraintsEnabled default true, ontologyConstraintStrength default 0.6<br/>physics_config.rs:122, :126, :423, :431
    PATCH->>PUSH: previous and new policy (settings_routes.rs:793)
    PUSH->>OCA: SetOntologyConstraintPolicy only when it changed (settings_routes.rs:625)
    SUP->>OCA: the same policy on every supervisor start (physics_supervisor.rs:216)
    OCA->>OCA: live buffer is empty when off, else the unscaled base at strength (ontology_constraint_actor.rs:96)
    OCA->>FCA: replace the cached set
    FCA->>FCA: an empty pending cache over a non-empty engine is Clear (force_compute_actor.rs:1783, :1695)
```

**What it shows:** the two physics keys travel as one policy message to the constraint actor, which keeps the unscaled set so switching back on needs no sync. **Why it is this way:** ADR-2140 decision 1, one stored definition per value; the earlier switch lived in a service nothing called.

## VC-61.2 Three sync-time switches read from quality gates

```mermaid
flowchart TB
    QG["qualityGates document, global layer"] --> SW["SyncSwitches.from_quality_gates, a missing key reads as on<br/>github_sync_service.rs:233-245"]
    SW --> R["ontologyReasoning<br/>github_sync_service.rs:241"]
    SW --> D["materialiseDomainRoots<br/>github_sync_service.rs:242"]
    SW --> I["materialiseInferredEdges<br/>github_sync_service.rs:243"]
    R -->|"off"| NOWH["start-up Whelk load skipped<br/>github_sync_service.rs:1444-1446"]
    R -->|"off"| ASSERT["post-sync: stored inferred edges removed,<br/>asserted axioms only as constraints<br/>github_sync_service.rs:1466-1469"]
    D --> ROOTS["materialise_domain_roots with the switch<br/>github_sync_service.rs:714, github_sync_service.rs:991"]
    I --> WIE["writes_inferred_edges needs reasoning and the edge switch<br/>github_sync_service.rs:249-250"]
    WIE -->|"false"| PURGE["purge_inferred_edges<br/>github_sync_service.rs:1623"]
    SW --> WHEN["read each sync, applies at the next sync or restart<br/>github_sync_service.rs:334"]
```

**What it shows:** reasoning, domain roots and inferred edges are each a stored quality gate defaulting to on, read key by key at every sync. **Why it is this way:** ADR-2192 decision 2; on the dev stack the operator could not switch off reasoning whose output buried the graph.

## VC-61.3 Node thinning: the filter's maxNodes cap and its counts

```mermaid
flowchart TB
    NF["nodeFilter.maxNodes, default 0 no cap<br/>models.rs:172, models.rs:209"] --> CF["ClientFilter.max_nodes, 0 means none<br/>client_filter.rs:88"]
    CF --> REC["recompute_filtered_nodes, only when enabled and serverRemove<br/>client_filter.rs:25-26"]
    REC --> CAP{"more nodes pass than the cap<br/>client_filter.rs:56-57"}
    CAP -->|"yes"| TOP["top_nodes: quality, then degree, then id<br/>client_filter.rs:115-118, client_filter.rs:58"]
    TOP --> COUNT["filter_update_success with visible and total<br/>client_coordinator_actor.rs:111-116"]
    REC --> REST["GET /api/graph/data?apply_node_filter=true<br/>graph/mod.rs:110, graph/mod.rs:265"]
    TOP -.-> SCOPE["thins what a client receives and draws,<br/>not the shared GPU simulation"]
```

**What it shows:** the cap ranks the nodes that already pass the filter and keeps the best; the WebSocket reply now reports the counts and the REST reload applies the same removal. **Why it is this way:** ADR-2192 decision 6 reuses the ADR-2183 filter rather than a second population scope.

**Open:** the cap thins delivery only; every viewer and the headset still share one simulation over every node, so a capped viewer's layout is shaped by nodes it does not receive (`../project/docs/adr/ADR-2192-the-reasoner-the-sync-pipeline-and-node-thinning-are-stored-operator-switches.md:58`).
