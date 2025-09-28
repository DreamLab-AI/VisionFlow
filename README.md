# 🌌 VisionFlow

[![License](https://img.shields.io/badge/License-Mozilla%202.0-blue.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/Performance-60FPS%20@%20100k%20nodes-red.svg)](docs/)
[![Agents](https://img.shields.io/badge/AI%20Agents-50%2B%20Concurrent-orange.svg)](docs/)
[![CUDA](https://img.shields.io/badge/CUDA-40%20Kernels-green.svg)](docs/)

### **Immersive Multi-User Multi-Agent Knowledge Graphing**
**VisionFlow deploys self-sovereign AI agent teams that continuously research, analyse, and surface insights from your entire data corpus—visualised for collaborative teams in a stunning, real-time 3D interface.**



<div align="center">
  <table>
    <tr>
      <td><img src="./visionflow.gif" alt="VisionFlow Visualisation" style="width:100%; border-radius:10px;"></td>
      <td><img src="./jarvisSept.gif" alt="Runtime Screenshot" style="width:100%; border-radius:10px;"></td>
    </tr>
  </table>
</div>

---

## ✨ Core Features

*   **🧠 Continuous AI Analysis**: Deploy swarms of specialist AI agents (Researcher, Analyst, Coder) that work 24/7 in the background, using advanced GraphRAG to uncover deep semantic connections within your private data.
*   **🤝 Real-Time Collaborative 3D Space**: Invite your team into a shared virtual environment. Watch agents work, explore the knowledge graph together, and maintain independent specialist views while staying perfectly in sync.
*   **🎙️ Voice-First Interaction**: Converse naturally with your AI agents. Guide research, ask questions, and receive insights through seamless, real-time voice-to-voice communication with spatial audio.
*   **🔐 Enterprise-Grade & Self-Sovereign**: Your data remains yours. Built on a thin-client, secure-server architecture with Git-based version control for all knowledge updates, ensuring a complete audit trail and human-in-the-loop oversight.
*   **🔌 Seamless Data Integration**: Connect to your existing knowledge sources with our powerful Markdown-based data management system, built on [Logseq](https://logseq.com/). Enjoy block-based organisation, bidirectional linking, and local-first privacy.

| VisionFlow | ChatGPT Pulse |
| :--- | :--- |
| ✅ **Continuous**, real-time agent research | ❌ Asynchronous daily research |
| ✅ Discovers patterns in **your private knowledge corpus** | ❌ Surfaces insights from past chats |
| ✅ **Interactive 3D visualisation** you explore | ❌ Static visual summaries |
| ✅ **Human-in-the-loop** collaboration | ❌ Passive insight delivery |
| ✅ **Self-sovereign** and enterprise-secure | ❌ Hosted on third-party infrastructure |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/VisionFlow
cd VisionFlow

# 2. Configure your environment
cp .env.example .env
# Edit .env to add your data sources and API keys

# 3. Deploy with Docker
docker-compose up -d

# 4. Access your AI research universe
echo "VisionFlow is running at http://localhost:3001"
```

**[📚 Full Documentation](docs/)** | **[🎯 Getting Started Guide](docs/getting-started/02-quick-start.md)** | **[🔧 Installation Details](docs/getting-started/01-installation.md)**

---

## 🛠️ Technical Deep Dive

> 💡 **Click the sections below to explore detailed architecture diagrams and technical specifications**

<details>
<summary><strong>🧠 Advanced Knowledge Architecture & Agent Orchestration</strong></summary>

### Advanced Knowledge Graph Architecture

```mermaid
graph TB
    subgraph "Transitional Architecture - Bridge Pattern"
        TransitionalSupervisor["TransitionalGraphSupervisor<br/>Bridge Pattern Wrapper"]
        GraphActor["GraphServiceActor<br/>35,193 lines, Being Refactored"]

        subgraph "Extracted Actor Services"
            GraphStateActor["GraphStateActor<br/>State Management"]
            PhysicsOrchestrator["PhysicsOrchestratorActor<br/>GPU Physics"]
            SemanticProcessor["SemanticProcessorActor<br/>AI Analysis"]
            ClientCoordinator["ClientCoordinatorActor<br/>WebSocket Management"]
        end

        TransitionalSupervisor -->|Manages| GraphActor
        TransitionalSupervisor -->|Supervises| GraphStateActor
        TransitionalSupervisor -->|Orchestrates| PhysicsOrchestrator
        TransitionalSupervisor -->|Coordinates| SemanticProcessor
        TransitionalSupervisor -->|Routes| ClientCoordinator
    end

    style TransitionalSupervisor fill:#ff9800
    style GraphActor fill:#ffd54f
```

- **Microsoft GraphRAG Integration**: We build hierarchical knowledge structures with subject-object-predicate relationships, capturing deep semantic meaning beyond simple vector similarity.
- **Leiden Clustering Algorithm**: Automatically organises your knowledge into well-connected communities, revealing hidden relationships and structuring information from high-level domains down to specific details.
- **Cutting-Edge Shortest Path Analysis**: Utilises [new research](https://arxiv.org/abs/2504.17033) for multi-hop reasoning, enabling you to connect distant concepts and trace the flow of information.

### Intelligent Agent Orchestration
VisionFlow deploys specialised AI agents that work continuously in the background:
- **Researcher Agents**: Deep-dive into topics using GraphRAG's local search.
- **Analyst Agents**: Identify patterns and correlations using clustering algorithms.
- **Coder Agents**: Parse and understand codebases, documentation, and dependencies.
- **Planner & Reviewer Agents**: Coordinate research strategies and validate findings.

### Living Knowledge Graph with Git Integration
- Your data evolves in real-time as agents discover relationships.
- All changes are submitted as **merge requests** for human oversight, tracked with a complete Git version history.
- **Time Travel**: Visually rewind and fast-forward through the history of your data in the immersive graph.

</details>

<details>
<summary><strong>⚡ High-Performance Technology Stack</strong></summary>

### GPU Computation Layer

```mermaid
graph LR
    subgraph GCL["GPU Computation Layer - 40 Production CUDA Kernels"]
        subgraph VU["visionflow_unified.cu (28 kernels)"]
            Physics["Force-Directed Layout<br/>Spring-Mass Physics"]
            Clustering1["K-means++ Clustering<br/>Spectral Analysis"]
            Anomaly1["Local Outlier Factor<br/>Statistical Z-score"]
        end

        subgraph GCK["gpu_clustering_kernels.cu (8 kernels)"]
            Louvain["Louvain Modularity<br/>Community Detection"]
            LabelProp["Label Propagation<br/>Graph Partitioning"]
        end

        subgraph SK["Specialised Kernels (4)"]
            Stability["Stability Gates<br/>2 kernels"]
            SSSP["Shortest Path<br/>2 kernels"]
        end
    end

    style Physics fill:#4caf50
    style Clustering1 fill:#2196f3
    style Anomaly1 fill:#ff5722
```

### Binary Protocol Architecture

```mermaid
graph TD
    subgraph "34-Byte Wire Protocol (Actual Implementation)"
        WireFormat["Wire Packet Structure<br/>34 bytes total"]

        subgraph "Packet Layout"
            NodeID["node_id: u16 (2 bytes)"]
            Position["position: [f32; 3] (12 bytes)"]
            Velocity["velocity: [f32; 3] (12 bytes)"]
            Distance["sssp_distance: f32 (4 bytes)"]
            Parent["sssp_parent: i32 (4 bytes)"]
        end

        WireFormat --> NodeID
        WireFormat --> Position
        WireFormat --> Velocity
        WireFormat --> Distance
        WireFormat --> Parent
    end

    Comparison["JSON: 680 bytes → Binary: 34 bytes<br/>95% reduction"]

    style WireFormat fill:#673ab7
    style Comparison fill:#4caf50
```

| Layer | Component | Specification | Performance |
| :--- | :--- | :--- | :--- |
| **GPU Acceleration** | 40 CUDA Kernels | Physics, clustering, pathfinding | 100x CPU speedup |
| **Networking** | Binary WebSocket | 34-byte custom protocol | <10ms latency, 95% bandwidth saving |
| **Visualisation** | React Three Fiber | WebGL 3D Rendering Pipeline | 60 FPS @ 100k+ nodes |
| **Backend** | Rust + Actix | Supervised actor system | 1,000+ requests/min |
| **AI Orchestration**| MCP Protocol | ClaudeFlowActor & Agent Swarms | 50+ concurrent agents |

</details>

<details>
<summary><strong>👥 Multi-User Voice-Enabled Collaboration Architecture</strong></summary>

```mermaid
graph TB
    subgraph "Immersive Voice-Enabled Collaboration"
        subgraph "Human Experts with Independent Views"
            Expert1[Research Lead<br/>🎙️ Voice + Custom View]
            Expert2[Data Scientist<br/>🎙️ Voice + Analytics View]
            Expert3[Developer<br/>🎙️ Voice + Code View]
        end

        subgraph "Voice-Responsive AI Swarms"
            ResearchSwarm[Research Agents<br/>🔊 Voice Response]
            CoderSwarm[Coder Agents<br/>🔊 Code Narration]
            AnalystSwarm[Analyst Agents<br/>🔊 Data Insights]
        end

        subgraph "Shared Knowledge + Individual Lenses"
            KnowledgeGraph[Living Knowledge Graph<br/>Common Data Layer]

            subgraph "Personalised Views"
                View1[Research View<br/>Publications Focus]
                View2[Analytics View<br/>Metrics Focus]
                View3[Code View<br/>Architecture Focus]
            end

            HolographicVis[200x Holographic Sphere<br/>Synchronised Perspectives]
        end

        Expert1 <-->|🎙️ Voice Commands| ResearchSwarm
        Expert2 <-->|🎙️ Voice Queries| AnalystSwarm
        Expert3 <-->|🎙️ Voice Reviews| CoderSwarm

        KnowledgeGraph --> View1 & View2 & View3
        View1 --> Expert1
        View2 --> Expert2
        View3 --> Expert3
    end

    subgraph "Voice & Sync Infrastructure"
        VoiceSystem[Dual Voice System<br/>Legacy + Centralised]
        SpatialAudio[3D Spatial Audio<br/>Positioned Voices]
        WebSocketSync[Binary WebSocket<br/>34-byte packets]
        IndependentState[Per-User State<br/>Custom Settings]
    end

    VoiceSystem --> Expert1 & Expert2 & Expert3
    SpatialAudio --> HolographicVis
    WebSocketSync --> KnowledgeGraph

    style Expert1 fill:#4caf50
    style VoiceSystem fill:#ff5722
    style View1 fill:#e3f2fd
```

### Key Scenarios:
- **Voice-First Collaborative Research**: Teams guide agent swarms and discuss findings using natural voice commands in a shared 3D space with spatial audio.
- **Independent Specialist Views**: A data scientist can view statistical overlays while a developer sees code dependency graphs—both looking at the same core data, at the same time, without interrupting each other.
- **Team-Based Knowledge Discovery**: AI agents route findings to the most relevant human expert, who can then validate the insight and guide the next phase of research via merge request approval.
- **Collaborative Code Intelligence**: Use coder agents for live pair-programming sessions, architectural discussions, and automated knowledge capture from senior developers.

</details>

<details>
<summary><strong>🔬 Client Architecture & 3D Rendering Pipeline</strong></summary>

```mermaid
graph TB
    subgraph R3F["React Three Fiber Visualisation Pipeline"]
        subgraph CR["Core Rendering (60 FPS @ 100k nodes)"]
            GraphCanvas["GraphCanvas.tsx<br/>R3F Main Canvas"]
            GraphManager["GraphManager<br/>Scene Orchestration"]
            HolographicDataSphere["HolographicDataSphere<br/>200x Scale Hologram System"]
        end

        subgraph BW["Binary WebSocket (34-byte protocol)"]
            UnifiedApiClient["UnifiedApiClient<br/>31 References Across Codebase"]
            BinaryProtocol["Binary Protocol<br/>85% Bandwidth Reduction"]
            WebSocketService["WebSocket Service<br/><10ms Latency"]
        end

        subgraph MU["Multi-User Synchronisation"]
            ClientCoordinator["ClientCoordinatorActor<br/>User Session Management"]
            PresenceSystem["Presence Tracking<br/>Real-time Locations"]
            CollaborationLayer["Collaboration Layer<br/>Shared State Sync"]
        end
    end

    style GraphCanvas fill:#e1f5fe
    style HolographicDataSphere fill:#fff3e0
    style CollaborationLayer fill:#c8e6c9
```

### Data Flow Architecture

```mermaid
sequenceDiagram
    participant Backend as Rust Backend
    participant WS as WebSocket Service
    participant Binary as Binary Protocol
    participant GraphData as Graph Data Manager
    participant Canvas as Graph Canvas
    participant Agents as Agent Visualisation

    Note over Backend,Agents: Real-time Position Updates

    Backend->>WS: Binary frame (34 bytes/node)
    WS->>Binary: Parse binary data
    Binary->>Binary: Validate node format
    Binary->>GraphData: Update positions

    par Graph Updates
        GraphData->>Canvas: Node positions
        and
        GraphData->>Agents: Agent positions
    end

    Canvas->>Canvas: Update Three.js scene
    Agents->>Agents: Update agent meshes

    Note over Backend,Agents: Agent Metadata via REST

    loop Every 10 seconds
        Agents->>Backend: GET /api/bots/data
        Backend-->>Agents: Agent metadata (JSON)
        Agents->>Agents: Update agent details
    end
```

<details>
<summary><strong>🎙️ Voice-to-Voice Architecture</strong></summary>

```mermaid
graph LR
    subgraph VIO["Voice Input/Output Pipeline"]
        subgraph HVI["Human Voice Input"]
            Mic["Microphone<br/>Voice Capture"]
            STT["Speech-to-Text<br/>OpenAI Whisper"]
            Intent["Intent Recognition<br/>Context Analysis"]
        end

        subgraph AVR["AI Voice Response"]
            TTS["Text-to-Speech<br/>OpenAI/Kokoro"]
            Spatial["3D Spatial Audio<br/>Positioned Output"]
            Speaker["Voice Output<br/>Natural Response"]
        end

        subgraph DVS["Dual Voice System"]
            Legacy["useVoiceInteraction<br/>196 lines - Active"]
            Central["useVoiceInteractionCentralised<br/>856 lines - Available"]
            Hooks["9 Specialised Hooks<br/>Domain-Specific"]
        end
    end

    Mic --> STT --> Intent
    Intent --> Legacy
    Intent --> Central
    Legacy --> TTS
    Central --> TTS
    TTS --> Spatial --> Speaker

    style Mic fill:#4caf50
    style TTS fill:#2196f3
    style Central fill:#ff9800
```

### Voice Data Flow

```mermaid
sequenceDiagram
    participant User as User
    participant LegacyHook as useVoiceInteraction
    participant AudioService as Audio Input Service
    participant WS as WebSocket Service
    participant Backend as Rust Backend
    participant Whisper as Whisper STT
    participant Kokoro as Kokoro TTS

    User->>LegacyHook: Press voice button
    LegacyHook->>AudioService: Start recording
    AudioService->>AudioService: Capture audio stream
    AudioService->>WS: Send binary audio
    WS->>Backend: Forward audio data
    Backend->>Whisper: Process STT
    Whisper-->>Backend: Transcribed text
    Backend->>Backend: Process command
    Backend->>Kokoro: Generate TTS
    Kokoro-->>Backend: Audio response
    Backend->>WS: Send binary audio response
    WS->>LegacyHook: Audio response received
    LegacyHook-->>User: Voice feedback
```

<details>
<summary><strong>🏗️ System Integration Overview</strong></summary>

```mermaid
graph TB
    Client["Web Clients<br/>Unity/Browser"]
    
    subgraph "Actor System (Transitional Architecture)"
        TransitionalSupervisor["TransitionalGraphSupervisor<br/>Bridge Pattern"]
        GraphActor["GraphServiceActor<br/>Being Refactored"]
        GPUManager["GPUManagerActor"]
        ClientCoordinator["ClientCoordinatorActor"]
        ClaudeFlowActor["ClaudeFlowActor<br/>MCP Integration"]
    end
    
    subgraph "External Integrations"
        GitHub["GitHub API"]
        Docker["Docker Services<br/>multi-agent-container"]
        MCP["MCP Servers<br/>TCP :9500"]
        Speech["Speech Services"]
    end
    
    Client --> ClientCoordinator
    TransitionalSupervisor --> GraphActor
    ClaudeFlowActor --> MCP
    GPUManager --> GraphActor
    
    style TransitionalSupervisor fill:#ff9800
    style GPUManager fill:#4caf50
```

### Performance Metrics

| Component | Specification | Performance |
|-----------|--------------|-------------|
| **GPU Kernels** | 40 CUDA kernels | 100x CPU speedup |
| **Binary Protocol** | 34-byte packets | 95% bandwidth saving |
| **Actor System** | 20 Actix actors | 1000+ req/min |
| **WebSocket Latency** | Binary streaming | <10ms updates |
| **3D Rendering** | Three.js + WebGL | 60 FPS @ 100k nodes |
| **Agent Swarms** | MCP orchestration | 50+ concurrent agents |
| **Memory Efficiency** | Per-node overhead | 34 bytes only |
| **Hologram Scale** | HolographicDataSphere | 200x visual scale |

</details>

---

## 🔮 Roadmap

- ✅ **Current**: Real-time multi-user collaboration, voice-to-voice AI, 50+ concurrent agents, GPU acceleration.
- 🔄 **Coming Soon**: AR/VR (Quest 3) interface, multi-language voice support, email integration, mobile companion app.
- 🎯 **Future Vision**: Predictive intelligence, autonomous workflows, and a community plugin marketplace.

---

## 🤝 Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-org/VisionFlow/issues)
- **Discord**: [Join our community](https://discord.gg/ar-ai-kg)
- **Documentation**: [Full Documentation Hub](docs/)

#### 🙏 Acknowledgements
Inspired by the innovative work of **Prof. Rob Aspin** and powered by the tools and concepts from **Anthropic**, **OpenAI**, and the incredible open-source community.

---

## 📄 Licence

This project is licensed under the Mozilla Public License 2.0. See the [LICENSE](LICENSE) file for details.