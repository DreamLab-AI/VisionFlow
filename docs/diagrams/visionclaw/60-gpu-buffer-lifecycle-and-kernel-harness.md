---
id: VC-60
title: GPU buffer lifecycle and the force-kernel harness — one buffer table, graph-change invalidation, SSSP bounds, golden dumps
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2191-graph-sized-gpu-buffers-are-declared-once-and-sssp-spring-adjust-is-off-by-default.md
  - ../project/docs/GPU-wire-abi.md
adrs: [ADR-2191, ADR-2140, ADR-2143, ADR-2169, ADR-2173]
sources:
  - ../project/src/utils/unified_gpu_compute/buffers.rs
  - ../project/src/utils/unified_gpu_compute/memory.rs
  - ../project/src/utils/unified_gpu_compute/sssp.rs
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/handlers/api_handler/analytics/pathfinding.rs
  - ../project/src/handlers/api_handler/analytics/sssp_handlers.rs
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
  - ../project/src/adapters/sqlite_settings_repository.rs
  - ../project/scripts/gpu-physics-gate.sh
  - ../project/crates/visionclaw-gpu/tests/force_kernel_baseline/main.rs
  - ../project/docs/adr/ADR-2191-graph-sized-gpu-buffers-are-declared-once-and-sssp-spring-adjust-is-off-by-default.md
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

ADR-2191 (VisionClaw PR 37, merged as bc6c16aef) puts every graph-sized device buffer in one table, clears node-indexed state on every graph change, bounds SSSP against the table, switches SSSP spring adjustment off by default with a one-time migration of stored settings, and adds a golden force-kernel harness and a GPU gate script. Read with VC-12 (SimParams and buffers), VC-15 (analytics) and VC-59 (force terms).

## For the business

The layout engine's memory now grows and shrinks as one piece, so a graph that changes size can no longer leave stale shortest-path or community data pushing the wrong nodes. Before the planned physics rework, every force was recorded as it behaves today, so the rework can prove what it changed.

## VC-60.1 One table generates allocation, resize and the length check

```mermaid
flowchart TB
    TABLE["graph_sized_buffers! table<br/>buffers.rs:122<br/>positions, CSR, edge rest snapshot, partials, node_degrees"] --> EXT["Extent: Nodes, NodesPlusOne, Edges, NodeBlocks<br/>buffers.rs:32-53"]
    TABLE --> ALLOC["allocate the whole set<br/>buffers.rs:90"]
    TABLE --> LEN["graph_sized_lengths for checks<br/>buffers.rs:115"]
    FIXED["FIXED_EXTENT: buffers that do not follow the graph, with a reason<br/>buffers.rs:208"] --> REGTEST["CPU registry test: a DeviceBuffer field in neither list fails"]
    TABLE --> REGTEST
    RESIZE["resize_buffers<br/>memory.rs:472"] --> INV["invalidate_node_indexed_state first<br/>memory.rs:473"]
    INV --> FRESH["GraphSizedBuffers.allocate in full<br/>memory.rs:501"]
    FRESH --> CARRY["positions and velocities carried across<br/>memory.rs:537"]
    CARRY --> INSTALL["install_graph_sized, only after everything succeeded<br/>memory.rs:544"]
```

**What it shows:** the macro table is the single declaration of every node- or edge-sized buffer; resize allocates a complete fresh set and installs it only after the carry succeeds. **Why it is this way:** ADR-2191 decisions 1 and 8: buffers resized one by one had drifted apart, and a failed growth must leave the engine intact.

**Invariant:** a growth replaces the whole buffer set at once: allocation (`../project/src/utils/unified_gpu_compute/memory.rs:501`) and the carry (`:537`) complete before `install_graph_sized` (`:544`).

## VC-60.2 A graph change discards everything indexed by node

```mermaid
flowchart TB
    INIT["initialize_graph<br/>memory.rs:844"] --> CNT{"node or edge count changed<br/>memory.rs:853"}
    CNT -->|"yes"| RES["resize_buffers<br/>memory.rs:854"]
    CNT -->|"no"| FP{"CSR fingerprint differs<br/>memory.rs:855"}
    FP -->|"yes"| INVN["invalidate_node_indexed_state<br/>memory.rs:857"]
    FP -->|"no, identical re-upload"| KEEP["state kept"]
    RES --> INVN
    INVN --> LIST["SSSP distances, compute mask, community labels,<br/>degree weights, hold overlay rest lengths, SemanticGpu<br/>buffers.rs:292-300"]
    LIST -.-> NOTC["OPEN: the constraint index is not covered,<br/>constraint actors re-send on their own schedule<br/>buffers.rs:290-291"]
    UP["upload_edges_csr<br/>memory.rs:860"] --> SSSPI["invalidate_graph_derived_state, SSSP only<br/>memory.rs:222, buffers.rs:272-275"]
```

**What it shows:** a resize, or the same counts with a different topology, clears every piece of node-indexed host state, so a term is off until the actor re-uploads it rather than acting on whichever nodes now hold old indices. **Why it is this way:** ADR-2191 decision 9 (review round 1).

**Open:** constraints still name nodes by index and are not cleared on a graph change (`../project/src/utils/unified_gpu_compute/buffers.rs:290-291`); between a topology change and the constraint actors' next send, a resident constraint can act on the wrong node.

## VC-60.3 SSSP: bounded runs, results that die with the graph, off for springs by default

```mermaid
sequenceDiagram
    autonumber
    participant API as compute_sssp handler<br/>pathfinding.rs
    participant RUN as run_sssp<br/>sssp.rs:72
    participant MEM as sssp_distances_for_force_pass<br/>memory.rs:900
    participant EX as force pass setup<br/>execution.rs:675
    API->>API: delta not finite or not above 0 gives HTTP 400 before any GPU work (sssp.rs:40-45)
    API->>RUN: run with delta
    RUN->>RUN: require_graph_capacity on every indexed buffer (sssp.rs:96)
    Note over RUN: at most SSSP_MAX_DELTA_BUCKETS 64 buckets plus 2 passes, then an error (sssp.rs:25, :163)
    EX->>MEM: d_sssp only if current, available and num_nodes long, else null
    MEM-->>EX: pointer or null (execution.rs:684)
    Note over EX: SSSP_SPRING_ADJUST_DEFAULT false (physics_config.rs:107)<br/>and in LinLog mode the term has no rest length to adjust
```

**What it shows:** SSSP checks its buffers against the table, cannot spin forever on a tiny `delta`, and the force pass reads distances only when they belong to the current graph. **Why it is this way:** ADR-2191 decisions 2–4 and 7; spring adjustment is off because SSSP reads spring stiffness as path length, so a stiffer relation got a longer rest length (decision 6).

## VC-60.4 Stored ssspEnabled is migrated off once

```mermaid
flowchart TB
    OPEN["SqliteSettingsRepository open<br/>sqlite_settings_repository.rs:333"] --> MIG["migrate_sssp_spring_adjust_off<br/>sqlite_settings_repository.rs:130"]
    MIG --> DONE{"0003_sssp_spring_adjust_off in schema_migrations<br/>sqlite_settings_repository.rs:119, sqlite_settings_repository.rs:133"}
    DONE -->|"already run"| SKIP["nothing changes, a later true sticks"]
    DONE -->|"first run"| ARCH["copy each physics row with ssspEnabled true<br/>into settings_archive<br/>sqlite_settings_repository.rs:141, sqlite_settings_repository.rs:179"]
    ARCH --> SET["set that key alone to false<br/>sqlite_settings_repository.rs:188"]
    SET --> WARN["warning logged<br/>sqlite_settings_repository.rs:339"]
    DEF["serde default and /sssp/status fallback<br/>physics_config.rs:110, sssp_handlers.rs:179"] --> SKIP
```

**What it shows:** on first open after the upgrade, every stored physics row that had SSSP spring adjustment on is archived verbatim and switched off; the migration marker makes it run once. **Why it is this way:** operator decision of 10 November 2026, recorded as ADR-2191 decision 10, which supersedes decision 6's "a stored true is honoured" for rows stored before the migration.

**Tension (ADR-2140 vs ADR-2191):** ADR-2140 says a value is applied exactly as stored (`../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:28`); this migration rewrites a stored value once, which ADR-2191 records as an explicit exception amending ADR-2140 (`../project/docs/adr/ADR-2191-graph-sized-gpu-buffers-are-declared-once-and-sssp-spring-adjust-is-off-by-default.md:65`).

## VC-60.5 The gate: golden force dumps and lifecycle tests on a named GPU

```mermaid
flowchart TB
    GATE["gpu-physics-gate.sh"] --> PRE["preflight: GPU on the expected PCI bus<br/>gpu-physics-gate.sh:66"]
    PRE --> CPU["CPU crates build, clippy, test<br/>gpu-physics-gate.sh:92"]
    CPU --> LIFE["buffer-lifecycle and SSSP tests under compute-sanitizer<br/>gpu-physics-gate.sh:103, at least 12 gpu-physics-gate.sh:44"]
    LIFE --> HARN["force-kernel golden harness<br/>gpu-physics-gate.sh:114, at least 40 gpu-physics-gate.sh:45"]
    HARN --> REQ["require_executed: no SKIP line, enough tests<br/>and device lines on the bus gpu-physics-gate.sh:53-63"]
    HARN --> GOLD["each kernel vs a CPU reference of the law as implemented<br/>and vs a committed golden dump, main.rs:1-10"]
    HARN --> INT["five intended_law tests, expected failures until the rework<br/>main.rs:928, main.rs:958, main.rs:990, main.rs:1051, main.rs:1067"]
```

**What it shows:** the gate refuses to pass without the GPU, pins every force kernel to today's behaviour, and keeps five tests that assert the stated law and are expected to fail. **Why it is this way:** the rework must show every force change against a recorded baseline; a skip on a host without a device would otherwise read as a pass (ADR-2191 decision 12).

**Debt:** the five expected failures are the departures VC-59 records: the relation spring is not Hookean (`../project/crates/visionclaw-gpu/tests/force_kernel_baseline/main.rs:928`), the repulsion relation type attracts (`:958`), the attribute-spring rest length leaves its bounds (`:990`), and the stability kernel applies neither gravity (`:1051`) nor class repulsion (`:1067`).
