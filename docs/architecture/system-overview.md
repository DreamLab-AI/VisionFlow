# VisionFlow System Architecture

## Overview

VisionFlow is built on a decoupled, actor-based architecture that enables real-time 3D visualization of knowledge graphs and AI agent swarms. The system combines a high-performance Rust backend with a modern React/TypeScript frontend, leveraging GPU acceleration and WebXR for immersive experiences.

## Core Architecture Diagram

```mermaid
graph TB
    subgraph "Frontend Layer"
        subgraph "React Application"
            UI[UI Components]
            Store[State Management]
            Three[Three.js Renderer]
            XR[WebXR Manager]
        end
        
        subgraph "WebSocket Clients"
            WSFlow[Socket Flow Client]
            WSSpeech[Speech Client]
            WSMCP[MCP Relay Client]
            WSBots[Bots Viz Client]
        end
    end
    
    subgraph "Backend Layer"
        subgraph "HTTP Server"
            REST[REST API<br/>Actix-Web]
            Static[Static Files]
            Auth[Auth Handler]
        end
        
        subgraph "WebSocket Server"
            WSHandler[WS Handler]
            Binary[Binary Protocol]
            Stream[Stream Manager]
        end
        
        subgraph "Actor System"
            CFActor[Claude Flow Actor<br/>Enhanced MCP]
            GraphActor[Graph Service Actor]
            GPUActor[GPU Compute Actor]
            ClientMgr[Client Manager Actor]
            SettingsActor[Settings Actor]
            MetaActor[Metadata Actor]
            ProtectedActor[Protected Settings]
        end
        
        subgraph "Services Layer"
            MCPRelay[MCP Relay Manager]
            GitHubSvc[GitHub Service]
            NostrSvc[Nostr Service]
            SpeechSvc[Speech Service]
            BotsClient[Bots Client]
            AgentViz[Agent Viz Processor]
        end
    end
    
    subgraph "GPU Layer"
        CUDA[CUDA Kernels]
        Physics[Physics Engine]
        DualGraph[Dual Graph Processor]
        Analytics[Visual Analytics]
    end
    
    subgraph "External Services"
        ClaudeFlow[Claude Flow<br/>Port 3002]
        GitHub[GitHub API]
        RAGFlow[RAGFlow Service]
        Perplexity[Perplexity API]
        Nostr[Nostr Network]
    end
    
    UI --> Store
    Store --> Three
    Three --> XR
    UI --> WSFlow
    UI --> WSSpeech
    UI --> WSMCP
    UI --> WSBots
    
    WSFlow --> WSHandler
    WSSpeech --> WSHandler
    WSMCP --> WSHandler
    WSBots --> WSHandler
    
    WSHandler --> Binary
    Binary --> Stream
    Stream --> ClientMgr
    
    REST --> Auth
    Auth --> NostrSvc
    REST --> GraphActor
    REST --> SettingsActor
    REST --> BotsClient
    
    ClientMgr --> GraphActor
    GraphActor --> GPUActor
    GPUActor --> CUDA
    CUDA --> Physics
    CUDA --> DualGraph
    CUDA --> Analytics
    
    CFActor --> MCPRelay
    MCPRelay --> ClaudeFlow
    GraphActor --> GitHubSvc
    GitHubSvc --> GitHub
    NostrSvc --> Nostr
    
    BotsClient --> AgentViz
    AgentViz --> GraphActor
```

## Component Architecture

### Frontend Components

```mermaid
graph LR
    subgraph "Component Hierarchy"
        App[App.tsx]
        App --> MainLayout[MainLayout]
        App --> Quest3Layout[Quest3 AR Layout]
        
        MainLayout --> GraphCanvas[Graph Canvas]
        MainLayout --> ControlPanel[Control Panel]
        MainLayout --> BotsPanel[Bots Panel]
        
        GraphCanvas --> Viewport[3D Viewport]
        Viewport --> Renderer[WebGL Renderer]
        Viewport --> Camera[Camera Controller]
        Viewport --> Effects[Post-Processing]
        
        ControlPanel --> Settings[Settings Panel]
        ControlPanel --> Commands[Command Palette]
        ControlPanel --> Voice[Voice Controls]
        
        BotsPanel --> AgentList[Agent List]
        BotsPanel --> SwarmViz[Swarm Visualization]
        BotsPanel --> Metrics[Performance Metrics]
    end
```

### Actor Communication Flow

```mermaid
sequenceDiagram
    participant Client
    participant WebSocket
    participant ClientManager
    participant GraphActor
    participant GPUActor
    participant CUDA
    
    Client->>WebSocket: Connect
    WebSocket->>ClientManager: Register Client
    ClientManager->>Client: Send Initial State
    
    Client->>WebSocket: Update Request
    WebSocket->>ClientManager: Forward Message
    ClientManager->>GraphActor: Process Update
    GraphActor->>GPUActor: Compute Physics
    GPUActor->>CUDA: Execute Kernel
    CUDA-->>GPUActor: Return Results
    GPUActor-->>GraphActor: Physics Results
    GraphActor-->>ClientManager: Graph Update
    ClientManager-->>Client: Binary Update Stream
```

## Data Flow Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        MD[Markdown Files]
        JSON[JSON Metadata]
        API[External APIs]
        Agents[AI Agents]
    end
    
    subgraph "Processing Pipeline"
        Parser[Data Parser]
        Semantic[Semantic Analyzer]
        Edge[Edge Generator]
        Layout[Layout Engine]
    end
    
    subgraph "Storage"
        GraphState[Graph State]
        MetaCache[Metadata Cache]
        Settings[Settings Store]
    end
    
    subgraph "Distribution"
        Binary[Binary Protocol]
        Diff[Differential Updates]
        Stream[Stream Manager]
    end
    
    MD --> Parser
    JSON --> Parser
    API --> Parser
    Agents --> Parser
    
    Parser --> Semantic
    Semantic --> Edge
    Edge --> Layout
    Layout --> GraphState
    
    GraphState --> MetaCache
    GraphState --> Binary
    Binary --> Diff
    Diff --> Stream
    Stream --> Clients[Connected Clients]
```

## GPU Processing Pipeline

```mermaid
graph LR
    subgraph "Input"
        Nodes[Node Data]
        Edges[Edge Data]
        Params[Physics Params]
    end
    
    subgraph "GPU Memory"
        DevNodes[Device Nodes]
        DevEdges[Device Edges]
        DevForces[Force Buffer]
    end
    
    subgraph "CUDA Kernels"
        Force[Force Calculation]
        Stress[Stress Majorization]
        Integration[Velocity Integration]
        Position[Position Update]
    end
    
    subgraph "Output"
        Updated[Updated Positions]
        Metrics[Performance Metrics]
    end
    
    Nodes --> DevNodes
    Edges --> DevEdges
    Params --> Force
    
    DevNodes --> Force
    DevEdges --> Force
    Force --> DevForces
    DevForces --> Stress
    Stress --> Integration
    Integration --> Position
    Position --> Updated
    Position --> Metrics
```

## MCP Integration Architecture

```mermaid
graph TB
    subgraph "VisionFlow Backend"
        CFActor[Claude Flow Actor]
        MCPRelay[MCP Relay Manager]
        WSRelay[WebSocket Relay]
    end
    
    subgraph "Claude Flow Service"
        MCPServer[MCP Server<br/>Port 3002]
        Tools[50+ MCP Tools]
        Swarm[Swarm Manager]
        Memory[Memory Service]
    end
    
    subgraph "Agent Types"
        Coord[Coordinator]
        Research[Researcher]
        Coder[Coder]
        Analyst[Analyst]
        Architect[Architect]
        Others[15+ Types]
    end
    
    CFActor <--> MCPRelay
    MCPRelay <--> WSRelay
    WSRelay <--> MCPServer
    
    MCPServer --> Tools
    MCPServer --> Swarm
    MCPServer --> Memory
    
    Swarm --> Coord
    Swarm --> Research
    Swarm --> Coder
    Swarm --> Analyst
    Swarm --> Architect
    Swarm --> Others
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Containers"
        subgraph "Main Container"
            Nginx[NGINX<br/>Port 80/443]
            Rust[Rust Backend<br/>Port 3001]
            Vite[Vite Dev<br/>Port 5173]
        end
        
        subgraph "Services Container"
            Claude[Claude Flow<br/>Port 3002]
            RAG[RAGFlow<br/>Port 80]
        end
    end
    
    subgraph "Host System"
        GPU[NVIDIA GPU]
        CUDA_Host[CUDA Driver]
        Docker[Docker Engine]
    end
    
    subgraph "Volumes"
        Data[Data Volume]
        Logs[Logs Volume]
        Config[Config Volume]
    end
    
    Internet[Internet] --> Nginx
    Nginx --> Rust
    Nginx --> Vite
    Rust --> Claude
    Rust --> RAG
    
    Rust --> GPU
    GPU --> CUDA_Host
    
    Rust --> Data
    Rust --> Logs
    Rust --> Config
```

## Security Architecture

```mermaid
graph TB
    subgraph "Authentication Layer"
        Nostr[Nostr Auth]
        NIP07[NIP-07 Extension]
        Signer[Event Signer]
    end
    
    subgraph "Authorization"
        RBAC[Role-Based Access]
        Features[Feature Flags]
        Protected[Protected Settings]
    end
    
    subgraph "Network Security"
        TLS[TLS/SSL]
        CORS[CORS Policy]
        CSP[Content Security Policy]
    end
    
    subgraph "Data Security"
        Validation[Input Validation]
        Sanitization[Data Sanitization]
        Encryption[At-Rest Encryption]
    end
    
    Client[Client] --> TLS
    TLS --> Nostr
    Nostr --> NIP07
    NIP07 --> Signer
    Signer --> RBAC
    RBAC --> Features
    Features --> Protected
    
    TLS --> CORS
    CORS --> CSP
    CSP --> Validation
    Validation --> Sanitization
    Sanitization --> Encryption
```

## Performance Optimization

### Caching Strategy
- **Metadata Cache**: In-memory caching of graph metadata
- **Settings Cache**: Client-side settings persistence
- **GPU Buffer Cache**: Reusable CUDA memory allocations
- **WebSocket Message Cache**: Differential update tracking

### Scalability Features
- **Actor Supervision**: Automatic actor restart on failure
- **Connection Pooling**: Efficient database connections
- **Load Balancing**: NGINX reverse proxy distribution
- **Horizontal Scaling**: Stateless backend design

### Performance Metrics
| Component | Target | Actual |
|-----------|--------|--------|
| REST API Latency | <100ms | 50ms |
| WebSocket Latency | <10ms | 5ms |
| GPU Kernel Time | <16ms | 10ms |
| Frame Rate | 60 FPS | 60 FPS |
| Memory Usage | <4GB | 2.5GB |

## Technology Stack

### Backend Technologies
- **Language**: Rust 1.75+
- **Web Framework**: Actix-Web 4.4
- **Async Runtime**: Tokio
- **GPU**: CUDA 11.8+
- **Serialization**: Serde, Bincode
- **WebSocket**: Actix-WS

### Frontend Technologies
- **Framework**: React 18
- **Language**: TypeScript 5
- **3D Graphics**: Three.js, React Three Fiber
- **XR**: @react-three/xr
- **State Management**: Zustand
- **Build Tool**: Vite

### Infrastructure
- **Containerization**: Docker
- **Proxy**: NGINX
- **Process Manager**: Supervisord
- **Logging**: Custom structured logging
- **Monitoring**: Built-in metrics collection

## Key Design Decisions

1. **Actor Model**: Provides fault tolerance and concurrent state management
2. **Binary Protocol**: Minimizes bandwidth for real-time updates
3. **GPU Acceleration**: Enables massive graph visualization
4. **Dual Graph**: Supports both knowledge and agent graphs simultaneously
5. **WebXR Integration**: Future-proofs for AR/VR interfaces
6. **MCP Protocol**: Standardized AI tool communication
7. **Differential Updates**: Optimizes network traffic
8. **Modular Architecture**: Allows independent component scaling