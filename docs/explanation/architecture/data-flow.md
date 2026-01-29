---
title: Complete Data Flow Architecture
description: **VisionFlow End-to-End Pipeline**
category: explanation
tags:
  - architecture
  - api
  - backend
updated-date: 2025-12-18
difficulty-level: advanced
---


# Complete Data Flow Architecture
**VisionFlow End-to-End Pipeline**

**Purpose**: Document the complete data flow from GitHub to GPU to Client

---

## Table of Contents

1. [System Overview](#system-overview)
2. [GitHub to Database Pipeline](#github-to-database-pipeline)
3. [Ontology Reasoning Pipeline](#ontology-reasoning-pipeline)
4. [GPU Semantic Physics Pipeline](#gpu-semantic-physics-pipeline)
5. [Client Visualization Pipeline](#client-visualization-pipeline)
6. [Performance Metrics](#performance-metrics)

---

## System Overview

### Complete Architecture Diagram

```mermaid
graph TB
    subgraph GitHub["🌐 GitHub Repository (jjohare/logseq)"]
        MD1["📄 Knowledge Graph<br>.md files (public:: true)"]
        MD2["📄 Ontology<br>.md files (OntologyBlock)"]
    end

    subgraph Sync["⬇️ GitHub Sync Service"]
        DIFF["🔍 Differential Sync<br>(SHA1 comparison)"]
        KGP["📝 KnowledgeGraphParser"]
        ONTOP["🧬 OntologyParser"]
    end

    subgraph Database["💾 Unified Database (unified.db)"]
        GRAPH-TABLES["graph-nodes<br>graph-edges"]
        OWL-TABLES["owl-classes<br>owl-properties<br>owl-axioms<br>owl-hierarchy"]
        META["file-metadata"]
    end

    subgraph Reasoning["🧠 Ontology Reasoning"]
        WHELK["Whelk-rs Reasoner<br>(OWL 2 EL)"]
        INFER["Inferred Axioms<br>(is-inferred=1)"]
        CACHE["LRU Cache<br>(90x speedup)"]
    end

    subgraph Physics["⚡ GPU Semantic Physics"]
        CONSTRAINTS["Semantic Constraints<br>(8 types)"]
        CUDA["CUDA Physics Engine<br>(39 kernels)"]
        FORCES["Force Calculations<br>(Ontology-driven)"]
    end

    subgraph Client["🖥️ Client Visualization"]
        WS["Binary WebSocket<br>(36 bytes/node)"]
        RENDER["3D Rendering<br>(Three.js/Babylon.js)"]
        GRAPH["Self-Organizing Graph"]
    end

    MD1 --> DIFF
    MD2 --> DIFF
    DIFF --> KGP
    DIFF --> ONTOP
    KGP --> GRAPH-TABLES
    ONTOP --> OWL-TABLES
    DIFF --> META

    OWL-TABLES --> WHELK
    WHELK --> INFER
    INFER --> OWL-TABLES
    WHELK --> CACHE

    OWL-TABLES --> CONSTRAINTS
    CONSTRAINTS --> CUDA
    GRAPH-TABLES --> CUDA
    CUDA --> FORCES

    FORCES --> WS
    WS --> RENDER
    RENDER --> GRAPH

    style GitHub fill:#e1f5ff
    style Sync fill:#fff3e0
    style Database fill:#f0e1ff
    style Reasoning fill:#e8f5e9
    style Physics fill:#ffe1e1
    style Client fill:#fff9c4
```

---

## GitHub to Database Pipeline

### 1. Initialization Flow

```mermaid
sequenceDiagram
    participant App as AppState::new()
    participant Sync as GitHubSyncService
    participant GH as GitHub API
    participant Parser as Content Parsers
    participant Repo as UnifiedGraphRepository
    participant DB as unified.db

    App->>Sync: Initialize sync service
    App->>Sync: sync-graphs()

    activate Sync
    Sync->>DB: Query file-metadata for SHA1 hashes
    DB-->>Sync: Previous file states

    Sync->>GH: Fetch file list (jjohare/logseq)
    GH-->>Sync: Markdown files metadata

    loop For each file
        Sync->>Sync: Compute SHA1 hash
        alt File changed or FORCE-FULL-SYNC
            Sync->>GH: Fetch file content
            GH-->>Sync: Raw markdown
            Sync->>Parser: Route to appropriate parser
            Parser-->>Sync: Parsed data
            Sync->>Repo: Store nodes/edges/classes
            Repo->>DB: INSERT/UPDATE
            Sync->>DB: Update file-metadata
        else File unchanged
            Sync->>Sync: Skip (no processing)
        end
    end

    Sync-->>App: SyncStatistics (316 nodes, timing)
    deactivate Sync
```

### 2. File Type Detection

```rust
// File routing based on content markers
fn detect-file-type(content: &str) -> FileType {
    if content.starts-with("public:: true") {
        FileType::KnowledgeGraph
    } else if content.contains("- ### OntologyBlock") {
        FileType::Ontology
    } else {
        FileType::Skip
    }
}
```

### 3. Knowledge Graph Parsing

**Input Format**:
```markdown
public:: true
---
# Artificial Intelligence
- [[Machine Learning]] is a subset
- tag:: #ai #technology
- property:: active
```

**Output** (to unified.db):
```sql
-- graph-nodes table
INSERT INTO graph-nodes (metadata-id, label, metadata)
VALUES ('artificial-intelligence', 'Artificial Intelligence',
        '{"tags": ["ai", "technology"], "property": "active"}');

-- graph-edges table
INSERT INTO graph-edges (source, target, weight)
VALUES (1, 2, 1.0); -- AI → Machine Learning
```

### 4. Ontology Parsing

**Input Format**:
```markdown
- ### OntologyBlock
  - owl-class:: Agent
    - label:: Intelligent Agent
    - subClassOf:: Entity
  - objectProperty:: hasCapability
    - domain:: Agent
    - range:: Capability
```

**Output** (to unified.db):
```sql
-- owl-classes table
INSERT INTO owl-classes (iri, label, description)
VALUES ('Agent', 'Intelligent Agent', NULL);

-- owl-class-hierarchy table
INSERT INTO owl-class-hierarchy (class-iri, parent-iri)
VALUES ('Agent', 'Entity');

-- owl-properties table
INSERT INTO owl-properties (iri, label, property-type, domain, range)
VALUES ('hasCapability', 'hasCapability', 'ObjectProperty', 'Agent', 'Capability');

-- owl-axioms table (asserted)
INSERT INTO owl-axioms (axiom-type, subject, predicate, object, is-inferred)
VALUES ('SubClassOf', 'Agent', 'rdfs:subClassOf', 'Entity', 0);
```

---

## Ontology Reasoning Pipeline

### 1. Reasoning Workflow

```mermaid
graph TB
    START["🔄 Sync Complete"]

    subgraph Load["1️⃣ Load Ontology"]
        LOAD-CLASSES["Load owl-classes"]
        LOAD-AXIOMS["Load owl-axioms<br>(is-inferred=0)"]
        LOAD-PROPS["Load owl-properties"]
    end

    subgraph Reason["2️⃣ Whelk-rs Reasoning"]
        BUILD["Build OWL graph"]
        COMPUTE["Compute inferences<br>(10-100x faster)"]
        CHECK["Consistency check"]
    end

    subgraph Store["3️⃣ Store Results"]
        INFER-AX["Insert inferred axioms<br>(is-inferred=1)"]
        UPDATE-META["Update reasoning-metadata"]
        CACHE-WARM["Warm LRU cache"]
    end

    subgraph Generate["4️⃣ Generate Constraints"]
        SUBCLASS["SubClassOf → Attraction"]
        DISJOINT["DisjointWith → Repulsion"]
        EQUIV["EquivalentClasses → Strong Attraction"]
        PROP["ObjectProperty → Alignment"]
        WEAKEN["Inferred axioms → 0.3x force"]
    end

    START --> LOAD-CLASSES
    LOAD-CLASSES --> LOAD-AXIOMS
    LOAD-AXIOMS --> LOAD-PROPS

    LOAD-PROPS --> BUILD
    BUILD --> COMPUTE
    COMPUTE --> CHECK

    CHECK --> INFER-AX
    INFER-AX --> UPDATE-META
    UPDATE-META --> CACHE-WARM

    CACHE-WARM --> SUBCLASS
    SUBCLASS --> DISJOINT
    DISJOINT --> EQUIV
    EQUIV --> PROP
    PROP --> WEAKEN

    WEAKEN --> GPU["⚡ Upload to GPU"]

    style START fill:#c8e6c9
    style GPU fill:#ffe1e1
```

### 2. Inference Examples

**Asserted Axiom**:
```sql
-- User defines: "Cat SubClassOf Animal"
INSERT INTO owl-axioms (axiom-type, subject, predicate, object, is-inferred)
VALUES ('SubClassOf', 'Cat', 'rdfs:subClassOf', 'Animal', 0);
```

**Inferred Axiom** (by Whelk-rs):
```sql
-- System infers: "Cat SubClassOf LivingThing" (via Animal → LivingThing)
INSERT INTO owl-axioms (axiom-type, subject, predicate, object, is-inferred)
VALUES ('SubClassOf', 'Cat', 'rdfs:subClassOf', 'LivingThing', 1);
```

### 3. Performance Metrics

| Metric | Value | Details |
|--------|-------|---------|
| **Reasoning Speed** | 10-100x | vs. Java-based reasoners |
| **LRU Cache Speedup** | 90x | For repeated queries |
| **Ontology Size** | 900+ classes | Current jjohare/logseq ontology |
| **Inference Time** | <2s | Complete reasoning pass |
| **Memory Usage** | ~50MB | In-memory graph representation |

---

## GPU Semantic Physics Pipeline

### 1. Constraint Generation

**Semantic Constraint Types**:

| Axiom Type | Physics Force | Visual Effect |
|------------|---------------|---------------|
| **SubClassOf** | Spring attraction (k=0.5) | Child classes cluster near parents |
| **DisjointWith** | Coulomb repulsion (k=-0.8) | Disjoint classes pushed apart |
| **EquivalentClasses** | Strong spring (k=1.0) | Synonyms rendered together |
| **ObjectProperty** | Directional alignment | Property domains/ranges aligned |
| **Inferred axioms** | Weaker forces (0.3x) | Subtle influence vs. asserted |

**Constraint Structure**:
```rust
pub struct SemanticConstraint {
    pub constraint-type: ConstraintType, // Spring, Repulsion, Alignment, etc.
    pub node-a: u32,
    pub node-b: u32,
    pub strength: f32,      // Force multiplier
    pub is-inferred: bool,  // Apply 0.3x reduction if true
}
```

### 2. CUDA Physics Pipeline

```mermaid
graph LR
    subgraph CPU["CPU (Rust)"]
        CONS["Generate<br>Constraints"]
        UPLOAD["Upload to GPU"]
    end

    subgraph GPU["GPU (CUDA)"]
        K1["Kernel 1:<br>Spring Forces"]
        K2["Kernel 2:<br>Repulsion Forces"]
        K3["Kernel 3:<br>Alignment Forces"]
        K-INFER["Apply 0.3x<br>to inferred"]
        INTEGRATE["Integrate<br>Velocities"]
        UPDATE["Update<br>Positions"]
    end

    subgraph Output["Output"]
        POSITIONS["New Node<br>Positions"]
        DOWNLOAD["Download to CPU"]
    end

    CONS --> UPLOAD
    UPLOAD --> K1
    K1 --> K2
    K2 --> K3
    K3 --> K-INFER
    K-INFER --> INTEGRATE
    INTEGRATE --> UPDATE
    UPDATE --> POSITIONS
    POSITIONS --> DOWNLOAD

    style CPU fill:#fff3e0
    style GPU fill:#ffe1e1
    style Output fill:#e8f5e9
```

### 3. Force Calculation Example

**Asserted SubClassOf** (full strength):
```rust
// Cat SubClassOf Animal (is-inferred=0)
let force = spring-force(cat-pos, animal-pos, k=0.5);
// Result: cat-pos moves toward animal-pos with full force
```

**Inferred SubClassOf** (reduced strength):
```rust
// Cat SubClassOf LivingThing (is-inferred=1)
let force = spring-force(cat-pos, living-pos, k=0.5 * 0.3); // 70% weaker
// Result: cat-pos gently influenced by living-pos
```

### 4. GPU Performance

| Metric | Value | Hardware |
|--------|-------|----------|
| **Nodes** | 10,000+ | RTX 3080 |
| **Constraints** | 50,000+ | RTX 3080 |
| **FPS** | 60 sustained | RTX 3080 |
| **Latency** | <16ms per frame | RTX 3080 |
| **Kernels** | 39 CUDA kernels | Custom physics engine |

---

## Client Visualization Pipeline

### 1. Binary WebSocket Protocol

**Message Format** (36 bytes per node):
```rust
pub struct NodeUpdate {
    pub id: u32,           // 4 bytes
    pub x: f32,            // 4 bytes
    pub y: f32,            // 4 bytes
    pub z: f32,            // 4 bytes
    pub vx: f32,           // 4 bytes (velocity)
    pub vy: f32,           // 4 bytes
    pub vz: f32,           // 4 bytes
    pub color: u32,        // 4 bytes (RGBA)
    pub size: f32,         // 4 bytes
}
```

**Bandwidth Calculation**:
- 316 nodes × 36 bytes = 11.4 KB per frame
- 60 FPS = 684 KB/s = 0.68 MB/s
- **Efficient**: 10x smaller than JSON protocol

### 2. Client Rendering Flow

```mermaid
sequenceDiagram
    participant WS as WebSocket Client
    participant Parser as Binary Parser
    participant Scene as 3D Scene
    participant Render as Renderer
    participant GPU as Client GPU

    WS->>Parser: Binary node updates (11.4 KB)
    Parser->>Scene: Parse 316 NodeUpdate structs
    Scene->>Scene: Update node positions
    Scene->>Render: Request frame render
    Render->>GPU: Upload geometry
    GPU->>GPU: Render 3D scene (60 FPS)
    GPU-->>WS: Display to user
```

### 3. Self-Organizing Graph

**Visual Representation of Ontology**:

```
     LivingThing
         │
    ┌────┴────┐
    │         │
  Animal    Plant
    │         │
  ┌─┴─┐     ┌─┴─┐
  │   │     │   │
 Cat Dog  Tree Flower

Legend:
• Vertical lines = SubClassOf relationships
• Close proximity = Asserted axioms (strong forces)
• Loose proximity = Inferred axioms (weak forces)
• Repelled nodes = DisjointWith axioms
```

**Real-Time Interaction**:
1. User drags "Cat" node
2. Spring forces pull it back toward "Animal"
3. Repulsion forces push it away from "Plant"
4. Inferred relationships (Cat → LivingThing) provide subtle guidance
5. Graph self-organizes into ontologically meaningful clusters

---

## Performance Metrics

### End-to-End Pipeline Timing

```mermaid
gantt
    title Complete Data Flow Timing (from GitHub to Client)
    dateFormat X
    axisFormat %L ms

    section GitHub Sync
    Fetch files          :0, 2000
    Parse content        :2000, 1000
    Store to DB          :3000, 500

    section Reasoning
    Load ontology        :3500, 200
    Whelk-rs inference   :3700, 1500
    Store inferred       :5200, 300

    section GPU Physics
    Generate constraints :5500, 100
    Upload to GPU        :5600, 50
    Compute forces       :5650, 16
    Download positions   :5666, 34

    section Client
    WebSocket transmit   :5700, 50
    Render frame         :5750, 16
```

**Total Latency Breakdown**:
1. **GitHub Sync**: ~3.5s (one-time on startup, then differential)
2. **Ontology Reasoning**: ~2s (one-time after sync)
3. **GPU Physics**: ~16ms per frame (60 FPS sustained)
4. **Client Rendering**: ~16ms per frame (60 FPS)

**Key Optimizations**:
- ✅ Differential sync: Only process changed files (90%+ skip rate)
- ✅ LRU caching: 90x speedup for repeated reasoning queries
- ✅ Binary WebSocket: 10x bandwidth reduction vs. JSON
- ✅ GPU parallelism: 100x faster than CPU physics

---

## Data Lineage

### Complete Traceability

```mermaid
graph TB
    GH["📁 GitHub File:<br>artificial-intelligence.md"]

    META["📋 file-metadata:<br>SHA1: abc123...<br>last-modified: 2025-11-03"]

    NODE["🔵 graph-nodes:<br>id: 1<br>metadata-id: 'artificial-intelligence'<br>label: 'Artificial Intelligence'"]

    CLASS["🧬 owl-classes:<br>iri: 'AI'<br>label: 'AI System'"]

    AXIOM-A["📐 owl-axioms:<br>subject: 'AI'<br>predicate: 'subClassOf'<br>object: 'ComputationalSystem'<br>is-inferred: 0"]

    AXIOM-I["📐 owl-axioms:<br>subject: 'AI'<br>predicate: 'subClassOf'<br>object: 'InformationProcessor'<br>is-inferred: 1<br>(inferred by Whelk-rs)"]

    CONS1["⚙️ Semantic Constraint:<br>type: Spring<br>node-a: 1<br>node-b: 2<br>strength: 0.5<br>is-inferred: false"]

    CONS2["⚙️ Semantic Constraint:<br>type: Spring<br>node-a: 1<br>node-b: 3<br>strength: 0.15<br>is-inferred: true (0.3x)"]

    FORCE["⚡ GPU Force:<br>node 1 attracted to 2 (strong)<br>node 1 attracted to 3 (weak)"]

    POS["📍 Node Position:<br>x: 42.3, y: 15.7, z: -8.2"]

    CLIENT["🖥️ Client Display:<br>3D rendered at (42.3, 15.7, -8.2)"]

    GH --> META
    GH --> NODE
    GH --> CLASS
    CLASS --> AXIOM-A
    AXIOM-A --> AXIOM-I
    AXIOM-A --> CONS1
    AXIOM-I --> CONS2
    CONS1 --> FORCE
    CONS2 --> FORCE
    FORCE --> POS
    POS --> CLIENT

    style GH fill:#e1f5ff
    style META fill:#fff3e0
    style NODE fill:#f0e1ff
    style CLASS fill:#e8f5e9
    style AXIOM-A fill:#fff9c4
    style AXIOM-I fill:#ffecb3
    style CONS1 fill:#ffe1e1
    style CONS2 fill:#ffcdd2
    style FORCE fill:#ff8a80
    style POS fill:#c8e6c9
    style CLIENT fill:#a5d6a7
```

---

## Monitoring & Observability

### Key Metrics Dashboard

```
┌─────────────────────────────────────────────────────────┐
│  VisionFlow Data Flow Metrics (Real-Time)              │
├─────────────────────────────────────────────────────────┤
│  GitHub Sync:                                           │
│    Last sync:          2025-11-03 12:45:32              │
│    Files scanned:      189                              │
│    Files processed:    12 (6% changed)                  │
│    Nodes loaded:       316                              │
│    Duration:           3.2s                             │
│                                                         │
│  Ontology Reasoning:                                    │
│    Classes:            247                              │
│    Asserted axioms:    1,834                            │
│    Inferred axioms:    4,217                            │
│    Reasoning time:     1.8s                             │
│    Cache hit rate:     94%                              │
│                                                         │
│  GPU Physics:                                           │
│    Active nodes:       316                              │
│    Active constraints: 2,145                            │
│    FPS:                60                               │
│    Frame time:         14.2ms                           │
│    GPU utilization:    42%                              │
│                                                         │
│  WebSocket Clients:                                     │
│    Connected:          3                                │
│    Bandwidth:          2.1 MB/s total                   │
│    Latency (p99):      18ms                             │
└─────────────────────────────────────────────────────────┘
```

---

## Conclusion

**System Characteristics**:
- ✅ **Complete**: GitHub → Database → Reasoning → GPU → Client
- ✅ **Efficient**: Differential sync, LRU caching, binary protocol
- ✅ **Intelligent**: Ontology reasoning drives visualization
- ✅ **Scalable**: Handles 10,000+ nodes at 60 FPS
- ✅ **Traceable**: Complete data lineage from source to display

**Architecture Benefits**:
1. **Unified Database**: Single source of truth (unified.db)
2. **Ontology-Driven**: Semantic relationships control physics
3. **GPU-Accelerated**: Real-time 3D graph simulation
4. **Binary Efficient**: 10x bandwidth reduction vs. JSON
5. **Self-Organizing**: Graph naturally clusters by ontological structure

---

**Documentation Version**: 1.0
**Last Updated**: November 3, 2025
**Maintained By**: VisionFlow Architecture Team
