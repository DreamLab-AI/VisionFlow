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
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

ADR-2192 (VisionClaw PR 38, merged as 010e1109b; `activation_status: inactive` until deployed) turns the reasoner's layout constraints, the sync pipeline's reasoning and materialisation, and node thinning into stored switches, retires three dead copies of the constraint switch, and deletes `OntologyPipelineService`. Read with VC-20 (ontology pipeline), VC-58 (edge producers) and VC-19 (the node filter).

## For the business

An operator can now turn off the reasoner's pull on the layout, stop the reasoner or the derived links from running at the next sync, and cap how many nodes a viewer is sent, without code changes. The cap thins what each viewer sees, not the shared simulation.

## VC-61.1 Constraint switch and strength: stored physics keys, sent as a policy on every write

```mermaid
sequenceDiagram
    autonumber
    participant PATCH as apply_physics_patch<br/>settings_routes.rs:661
    participant APPLY as apply_ontology_constraint_policy<br/>settings_routes.rs:639
    participant SUP as restore_saved_physics<br/>physics_supervisor.rs:204
    participant OCA as OntologyConstraintActor<br/>ontology_constraint_actor.rs:650
    participant FCA as constraint_sync_action<br/>force_compute_actor.rs:1827
    Note over PATCH: physics.ontologyConstraintsEnabled default true, ontologyConstraintStrength default 0.6<br/>physics_config.rs:123, :128, :431, :439
    PATCH->>APPLY: on every physics write, before SimParams (settings_routes.rs:741)
    APPLY->>OCA: SetOntologyConstraintPolicy from the stored physics (settings_routes.rs:643)
    SUP->>OCA: the same policy on every supervisor start (physics_supervisor.rs:216)
    OCA->>OCA: an identical applied policy answers unchanged (ontology_constraint_actor.rs:668)
    OCA->>OCA: live buffer rebuilt from the unscaled base (ontology_constraint_actor.rs:679, :96)
    OCA->>FCA: notify, or clear applied_policy so the next write retries (ontology_constraint_actor.rs:689-690)
    FCA->>FCA: an empty pending cache over a non-empty engine is Clear (force_compute_actor.rs:1834)
```

**What it shows:** the two physics keys travel as one policy message on every physics write; the actor skips an unchanged policy and keeps the unscaled set so switching back on needs no sync. **Why it is this way:** ADR-2140 decision 1, one stored definition per value; the store is written first, so only the actor can tell whether a policy was applied (comment at `settings_routes.rs:634-638`).

## VC-61.2 Three sync-time switches read from quality gates

```mermaid
flowchart TB
    QG["qualityGates document, global layer<br/>sync_switches github_sync_service.rs:334-338"] --> SW["SyncSwitches.from_quality_gates, a missing key reads as on<br/>github_sync_service.rs:233"]
    SW --> R["ontologyReasoning<br/>github_sync_service.rs:241"]
    SW --> D["materialiseDomainRoots<br/>github_sync_service.rs:242"]
    SW --> I["materialiseInferredEdges<br/>github_sync_service.rs:243"]
    R -->|"off"| NOWH["start-up Whelk load skipped<br/>github_sync_service.rs:1444-1448"]
    R -->|"off"| ASSERT["post-sync: inference forgotten, inferred edges purged,<br/>asserted axioms only as constraints<br/>github_sync_service.rs:1476-1480"]
    D --> ROOTS["materialise_domain_roots with the switch<br/>github_sync_service.rs:714, github_sync_service.rs:991"]
    I --> WIE["writes_inferred_edges needs reasoning and the edge switch<br/>github_sync_service.rs:249"]
    WIE -->|"false"| PURGE["purge_inferred_edges<br/>github_sync_service.rs:1547-1552, github_sync_service.rs:1646"]
```

**What it shows:** reasoning, domain roots and inferred edges are each a stored quality gate defaulting to on, read at every sync and at start-up. **Why it is this way:** ADR-2192 decision 2; on the dev stack the operator could not switch off reasoning whose output buried the graph.

## VC-61.3 Node thinning: the filter's maxNodes cap and its counts

```mermaid
flowchart TB
    NF["nodeFilter.maxNodes, default 0 no cap<br/>models.rs:172, models.rs:209"] --> CF["filter_update sets max_nodes, 0 means none<br/>client_filter.rs:110-113"]
    CF --> REC["recompute_filtered_nodes, removal only when enabled and serverRemove<br/>client_filter.rs:26, client_filter.rs:42"]
    REC --> CAP{"more nodes pass than the cap<br/>client_filter.rs:72-73"}
    CAP -->|"yes"| TOP["top_nodes: quality, then degree, then id<br/>client_filter.rs:140, client_filter.rs:160"]
    REC --> COUNT["filter_update_success, visible of considered<br/>client_coordinator_actor.rs:113, client_coordinator_actor.rs:150"]
    REC --> REST["GET graph data with the caller's node_filter snapshot<br/>same function, bad JSON gives 400<br/>graph/mod.rs:110, graph/mod.rs:265-272"]
    TOP -.-> SCOPE["thins what a client receives and draws,<br/>not the shared GPU simulation"]
```

**What it shows:** the cap ranks the nodes that already pass the filter and keeps the best; the WebSocket reply reports the counts and a REST reload given the same filter snapshot applies the same removal. **Why it is this way:** ADR-2192 decision 6 reuses the ADR-2183 filter rather than a second population scope.

**Open:** the cap thins delivery only; every viewer and the headset still share one simulation over every node, so a capped viewer's layout is shaped by nodes it does not receive (`../project/docs/adr/ADR-2192-the-reasoner-the-sync-pipeline-and-node-thinning-are-stored-operator-switches.md:65`).
