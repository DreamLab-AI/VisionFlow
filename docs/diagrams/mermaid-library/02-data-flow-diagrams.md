---
title: Data Flow Diagrams - End-to-End Processes
description: User->>REST: POST /api/admin/sync/streaming     REST->>GitHub: trigger_sync(repo_url, token)
category: reference
tags:
  - api
  - api
  - api
  - database
  - database
related-docs:
  - diagrams/mermaid-library/01-system-architecture-overview.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Neo4j database
---

# Data Flow Diagrams - End-to-End Processes

## 1. GitHub Sync - Complete Flow

```mermaid
sequenceDiagram
    participant User
    participant REST as REST API
    participant GitHub as GitHub Service
    participant GH as GitHub API
    participant Parser as Markdown Parser
    participant Neo4j as Neo4j Database
    participant GSA as GraphStateActor
    participant PO as PhysicsOrchestrator
    participant WS as WebSocket Broadcaster
    participant Client as Web Clients

    User->>REST: POST /api/admin/sync/streaming
    REST->>GitHub: trigger_sync(repo_url, token)

    GitHub->>GH: GET /repos/:owner/:repo/git/trees/:sha
    GH-->>GitHub: File tree (paginated, 100 files/request)

    loop For each markdown file
        GitHub->>GH: GET /repos/:owner/:repo/contents/:path
        GH-->>GitHub: File content (base64 encoded)

        GitHub->>Parser: parse_frontmatter(content)
        Parser->>Parser: Extract metadata (id, label, properties)

        alt Contains OWL Ontology Block
            Parser->>Neo4j: CREATE (:OwlClass {iri, label, subclass_of})
            Parser->>Neo4j: CREATE (:OwlProperty {iri, domain, range})
        end

        alt public: true in frontmatter
            Parser->>Neo4j: CREATE (:Node {id, label, metadata})
            Note over Neo4j: Store in knowledge graph
        end

        alt Contains edge definitions
            Parser->>Neo4j: CREATE (:Node)-[:EDGE {type}]->(:Node)
        end

        GitHub-->>User: Stream progress (JSON-ND)
        Note over User: Real-time progress updates
    end

    GitHub->>GSA: ReloadGraphFromDatabase
    GSA->>Neo4j: MATCH (n:Node)-[e:EDGE]->(m:Node) RETURN *
    Neo4j-->>GSA: Complete graph data (nodes + edges)

    GSA->>GSA: Build in-memory graph state
    GSA->>PO: InitializePhysics {nodes, edges}

    PO->>PO: Transfer to GPU memory
    PO->>PO: Initialize force-directed layout

    loop Physics Simulation (60 Hz)
        PO->>PO: Compute forces (CUDA)
        PO->>PO: Update positions (Verlet integration)
        PO->>WS: BroadcastPositions (binary)
        WS->>Client: WebSocket frame (36 bytes/node)
        Client->>Client: Update Three.js scene
    end

    REST-->>User: 200 OK {synced_nodes: N, synced_ontologies: M}
```

## 2. Real-time Graph Update Flow

```mermaid
sequenceDiagram
    participant Client as Web Client
    participant API as REST API
    participant Validator as Command Validator
    participant GSA as GraphStateActor
    participant Neo4j as Neo4j Database
    participant PO as PhysicsOrchestrator
    participant GPU as GPU Compute
    participant WS as WebSocket Service
    participant Clients as All Clients

    Client->>API: POST /api/graph/nodes<br/>{label: "New Node", properties: {...}}

    API->>Validator: Validate AddNodeCommand
    Validator->>Validator: Check business rules<br/>- Unique ID<br/>- Valid properties<br/>- Schema compliance

    alt Validation Failed
        Validator-->>Client: 400 Bad Request<br/>{error: "Validation failed"}
    else Validation Passed
        Validator->>GSA: AddNode Message
        GSA->>GSA: Update in-memory state
        GSA->>Neo4j: CREATE (:Node {id, label, ...})

        par Parallel Operations
            Neo4j-->>GSA: Node created (id: 12345)

            and

            GSA->>PO: AddNodeToPhysics {id: 12345, position: (0,0,0)}
            PO->>GPU: Transfer node data to GPU
            GPU->>GPU: Initialize physics properties<br/>- Position<br/>- Velocity<br/>- Mass<br/>- Forces
            GPU-->>PO: GPU buffer updated

            and

            GSA->>WS: BroadcastNodeAdded {id, label, position}
            WS->>Clients: Binary WebSocket frame<br/>36 bytes per node
        end

        Clients->>Clients: Add node to Three.js scene

        GSA-->>API: Success {node_id: 12345}
        API-->>Client: 201 Created<br/>{id: 12345, label: "New Node"}
    end

    Note over PO,GPU: Physics simulation continues<br/>at 60 FPS
```

## 3. Settings Update Flow (Debounced)

```mermaid
sequenceDiagram
    participant UI as Settings UI
    participant Store as Settings Store<br/>(Zustand)
    participant Debounce as Debounce Queue<br/>(500ms)
    participant AutoSave as AutoSave Manager
    participant API as Settings API
    participant Backend as Rust Backend
    participant Neo4j as Neo4j Database
    participant WS as WebSocket Broadcaster
    participant Clients as Other Clients

    UI->>Store: setSetting("physics.enabled", true)
    Store->>Store: Update partial state
    Store->>Debounce: Queue change {key, value, timestamp}

    Note over Debounce: Wait 500ms for more changes

    UI->>Store: setSetting("physics.gravity", 9.8)
    Store->>Store: Update partial state
    Store->>Debounce: Queue change {key, value, timestamp}

    UI->>Store: setSetting("physics.damping", 0.95)
    Store->>Store: Update partial state
    Store->>Debounce: Queue change {key, value, timestamp}

    Note over Debounce: 500ms elapsed, batch ready

    Debounce->>AutoSave: Flush batch [3 changes]
    AutoSave->>API: POST /api/settings/batch<br/>{changes: [{key, value}, ...]}

    API->>Backend: Actix Web Handler
    Backend->>Backend: Validate settings schema
    Backend->>Neo4j: MATCH (u:UserSettings {pubkey})<br/>SET u.physics_enabled = true<br/>SET u.physics_gravity = 9.8<br/>SET u.physics_damping = 0.95

    Neo4j-->>Backend: Settings updated
    Backend->>WS: Broadcast settings change
    WS->>Clients: Binary message (settings updated)

    Backend-->>API: 200 OK {updated: 3}
    API-->>AutoSave: Success
    AutoSave-->>Store: Confirm persistence
    Store-->>UI: Update save indicator (green checkmark)

    Clients->>Clients: Apply new settings to local state
```

## 4. Agent Coordination Flow

```mermaid
sequenceDiagram
    participant User
    participant API as Management API
    participant Swarm as Swarm Controller
    participant Agent1 as Agent 1 (Coder)
    participant Agent2 as Agent 2 (Tester)
    participant Agent3 as Agent 3 (Reviewer)
    participant Memory as Shared Memory
    participant WS as WebSocket Service
    participant Client as Client UI

    User->>API: POST /api/agents/spawn<br/>{type: "swarm", topology: "hierarchical"}

    API->>Swarm: Initialize swarm
    Swarm->>Agent1: Spawn coder agent
    Swarm->>Agent2: Spawn tester agent
    Swarm->>Agent3: Spawn reviewer agent

    Agent1-->>WS: Agent spawned (id: agent-1)
    Agent2-->>WS: Agent spawned (id: agent-2)
    Agent3-->>WS: Agent spawned (id: agent-3)
    WS-->>Client: Update agent visualization

    User->>API: POST /api/agents/tasks<br/>{task: "Implement feature X"}

    API->>Swarm: Orchestrate task
    Swarm->>Agent1: Task: Write code for feature X

    Agent1->>Agent1: Generate code
    Agent1->>Memory: Store code<br/>key: "feature-x-code"
    Agent1->>Agent2: Request: Run tests

    Agent2->>Memory: Retrieve code<br/>key: "feature-x-code"
    Agent2->>Agent2: Execute tests
    Agent2->>Memory: Store results<br/>key: "feature-x-tests"

    alt Tests Failed
        Agent2->>Agent1: Feedback: Fix test failures
        Agent1->>Agent1: Revise code
        Agent1->>Memory: Update code
        Agent1->>Agent2: Request: Re-run tests
    else Tests Passed
        Agent2->>Agent3: Request: Code review
        Agent3->>Memory: Retrieve code + tests
        Agent3->>Agent3: Perform review
        Agent3->>Memory: Store review<br/>key: "feature-x-review"

        Agent3->>Swarm: Task complete
        Swarm->>API: Success {result: {...}}
        API-->>User: 200 OK {status: "complete"}
    end

    loop Position Updates (60 Hz)
        Agent1->>WS: Binary position update
        Agent2->>WS: Binary position update
        Agent3->>WS: Binary position update
        WS->>Client: Batch WebSocket frames
        Client->>Client: Render agent nodes
    end
```

## 5. Voice Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Voice UI
    participant Audio as AudioInputService
    participant WS as Voice WebSocket
    participant Backend as Rust Backend
    participant Whisper as Whisper STT
    participant LLM as Claude/Perplexity
    participant GSA as GraphStateActor
    participant TTS as Kokoro TTS
    participant Speaker as Audio Output

    User->>UI: Press voice button
    UI->>Audio: startRecording()
    Audio->>Audio: Capture microphone stream
    Audio->>WS: Stream binary audio chunks

    Note over WS: Streaming audio data<br/>(PCM 16kHz mono)

    WS->>Backend: Binary audio frames
    Backend->>Whisper: Process speech-to-text
    Whisper-->>Backend: Transcribed text:<br/>"Show me nodes related to AI"

    Backend->>LLM: Process natural language query
    LLM->>LLM: Generate Cypher query:<br/>MATCH (n:Node) WHERE n.label CONTAINS 'AI'<br/>RETURN n

    Backend->>GSA: SearchNodes {query: "AI"}
    GSA->>GSA: Execute search
    GSA-->>Backend: Found 23 nodes

    Backend->>LLM: Generate response:<br/>"I found 23 nodes related to AI"
    LLM-->>Backend: Natural language response

    Backend->>TTS: Generate speech
    TTS-->>Backend: Audio data (WAV)

    Backend->>WS: Binary audio response
    WS->>Speaker: Play audio

    Speaker-->>User: Voice output:<br/>"I found 23 nodes related to AI"

    Backend->>WS: Highlight nodes (binary update)
    WS->>UI: Update visualization
    UI->>UI: Highlight 23 AI nodes in 3D scene
```

## 6. WebSocket Binary Protocol Flow

```mermaid
sequenceDiagram
    participant Client as Web Client
    participant WS as WebSocket Connection
    participant Decoder as Binary Protocol Decoder
    participant Handler as Message Handler
    participant GSA as GraphStateActor
    participant Encoder as Binary Protocol Encoder
    participant Broadcast as Broadcast Queue

    Note over Client,Broadcast: Connection Established (WSS)

    GSA->>Handler: PositionUpdate {nodes: [100k nodes]}
    Handler->>Encoder: Encode batch (binary protocol V4)

    Encoder->>Encoder: For each node:<br/>- id: u16 (2 bytes)<br/>- x,y,z: f32 (12 bytes)<br/>- vx,vy,vz: f32 (12 bytes)<br/>- sssp_distance: f32 (4 bytes)<br/>- sssp_parent: i32 (4 bytes)<br/>Total: 34 bytes/node

    Encoder->>Broadcast: Binary frame (3.4 MB for 100k nodes)

    Broadcast->>Broadcast: Split into chunks (64 KB max)
    Note over Broadcast: Chunking for TCP efficiency

    loop For each client (50+ concurrent)
        Broadcast->>WS: Send chunk 1/53
        Broadcast->>WS: Send chunk 2/53
        Broadcast->>WS: ...
        Broadcast->>WS: Send chunk 53/53
    end

    WS->>Client: Binary WebSocket frames

    Client->>Decoder: Parse binary data
    Decoder->>Decoder: Extract node data (34 bytes each)

    loop For each node
        Decoder->>Client: {id: 1, x: 10.5, y: 20.3, z: 5.1, ...}
        Client->>Client: Update Three.js mesh position
    end

    Note over Client: Frame rendered at 60 FPS

    Client->>WS: Acknowledge receipt (control message)
    WS->>Handler: ACK received
```

## 7. GPU Physics Simulation Flow

```mermaid
sequenceDiagram
    participant PO as PhysicsOrchestrator
    participant FC as ForceComputeActor
    participant GPU as CUDA Kernels
    participant Memory as GPU Memory
    participant SM as StressMajorizationActor
    participant WS as WebSocket Broadcaster
    participant Clients as Web Clients

    Note over PO: Simulation step starts (60 Hz)

    PO->>FC: ComputeForces {iteration: 1234}

    FC->>Memory: Read node positions (GPU buffer)
    FC->>GPU: Launch barnes_hut_force_kernel<br/>- Grid: 256 blocks<br/>- Threads: 256/block<br/>- 100k nodes

    GPU->>GPU: Build octree (spatial acceleration)
    GPU->>GPU: Compute forces (O(n log n))
    GPU->>GPU: Store forces (GPU buffer)

    GPU-->>FC: Forces computed (16ms)

    par Parallel Physics Steps
        FC->>GPU: Launch velocity_integration_kernel<br/>- Verlet integration<br/>- Adaptive timestep
        GPU->>GPU: Update velocities
        GPU->>GPU: Update positions
        GPU-->>FC: Positions updated

        and

        PO->>SM: OptimizeLayout {convergence: 0.01}
        SM->>GPU: Launch stress_majorization_kernel
        GPU->>GPU: Minimize graph stress
        GPU-->>SM: Layout optimized
    end

    FC->>Memory: Read updated positions (GPU→CPU transfer)
    Memory-->>FC: Position data (3.2 MB for 100k nodes)

    FC->>PO: PhysicsStepComplete {positions: [...]}
    PO->>WS: BroadcastPositions (binary protocol)

    WS->>Clients: WebSocket frame (3.4 MB, 34 bytes/node)
    Clients->>Clients: Update Three.js scene (60 FPS)

    Note over PO: Step complete in 16ms (60 FPS maintained)
```

## 8. Multi-Workspace Flow

```mermaid
sequenceDiagram
    participant User1 as User 1
    participant User2 as User 2
    participant API as REST API
    participant WA as WorkspaceActor
    participant Neo4j as Neo4j Database
    participant GSA as GraphStateActor
    participant WS1 as WebSocket (User 1)
    participant WS2 as WebSocket (User 2)

    User1->>API: POST /api/workspaces<br/>{name: "Project Alpha"}
    API->>WA: CreateWorkspace {name, owner: user1}
    WA->>Neo4j: CREATE (:Workspace {id, name, owner})<br/>CREATE (:Workspace)-[:CONTAINS]->(:Node)

    Neo4j-->>WA: Workspace created (id: ws-1)
    WA-->>API: Success {workspace_id: "ws-1"}
    API-->>User1: 201 Created {id: "ws-1"}

    User1->>API: POST /api/graph/nodes?workspace=ws-1<br/>{label: "Task 1"}
    API->>GSA: AddNode {workspace_id: "ws-1", node: {...}}
    GSA->>Neo4j: MATCH (ws:Workspace {id: "ws-1"})<br/>CREATE (ws)-[:CONTAINS]->(:Node {label: "Task 1"})

    GSA->>WS1: NodeAdded (binary, workspace-filtered)
    WS1->>User1: Update visualization (workspace ws-1)

    Note over WS2: User 2 not subscribed to ws-1<br/>No update sent

    User2->>API: POST /api/workspaces<br/>{name: "Project Beta"}
    API->>WA: CreateWorkspace {name, owner: user2}
    WA->>Neo4j: CREATE (:Workspace {id: "ws-2"})

    WA-->>API: Success {workspace_id: "ws-2"}
    API-->>User2: 201 Created {id: "ws-2"}

    User2->>API: GET /api/graph?workspace=ws-2
    API->>GSA: GetGraph {workspace_id: "ws-2"}
    GSA->>Neo4j: MATCH (:Workspace {id: "ws-2"})-[:CONTAINS]->(n:Node)<br/>RETURN n

    Neo4j-->>GSA: Empty graph (no nodes)
    GSA-->>API: {nodes: [], edges: []}
    API-->>User2: 200 OK (empty graph)

    Note over User1,User2: Workspaces are completely isolated
```
