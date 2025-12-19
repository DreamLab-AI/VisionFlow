---
title: Deployment & Infrastructure Diagrams
description: subgraph "Application Cluster (3 Nodes)"             App1[App Server 1<br/>Rust Backend<br/>8 CPU / 16GB RAM]             App2[App Server 2<br/>Rust Backend<br/>8 CPU / 16GB RAM]             App3[A...
category: explanation
tags:
  - architecture
  - structure
  - api
  - api
  - api
related-docs:
  - diagrams/mermaid-library/01-system-architecture-overview.md
  - diagrams/mermaid-library/02-data-flow-diagrams.md
updated-date: 2025-12-18
difficulty-level: intermediate
dependencies:
  - Docker installation
  - Rust toolchain
  - Node.js runtime
  - Neo4j database
---

# Deployment & Infrastructure Diagrams

## 1. Deployment Topology

```mermaid
graph TB
    subgraph "Production Environment - Cloud Infrastructure"
        subgraph "Load Balancer Layer"
            LB[NGINX Load Balancer<br/>TLS Termination<br/>Rate Limiting: 1000 req/s]
            LB2[WebSocket LB<br/>Sticky Sessions<br/>50k concurrent connections]
        end

        subgraph "Application Cluster (3 Nodes)"
            App1[App Server 1<br/>Rust Backend<br/>8 CPU / 16GB RAM]
            App2[App Server 2<br/>Rust Backend<br/>8 CPU / 16GB RAM]
            App3[App Server 3<br/>Rust Backend<br/>8 CPU / 16GB RAM]
        end

        subgraph "GPU Cluster (2 Nodes)"
            GPU1[GPU Node 1<br/>NVIDIA RTX 4090<br/>24GB VRAM<br/>Physics + ML]
            GPU2[GPU Node 2<br/>NVIDIA RTX 4090<br/>24GB VRAM<br/>Physics + ML]
        end

        subgraph "Database Cluster"
            Neo4jMaster[(Neo4j Master<br/>Write Operations<br/>16 CPU / 32GB RAM)]
            Neo4jReplica1[(Neo4j Replica 1<br/>Read Operations<br/>8 CPU / 16GB RAM)]
            Neo4jReplica2[(Neo4j Replica 2<br/>Read Operations<br/>8 CPU / 16GB RAM)]
        end

        subgraph "Cache Layer"
            Redis1[Redis Master<br/>Session Cache<br/>Settings Cache]
            Redis2[Redis Replica<br/>Read Replica<br/>Failover]
        end

        subgraph "Monitoring & Observability"
            Prometheus[Prometheus<br/>Metrics Collection<br/>30s scrape interval]
            Grafana[Grafana<br/>Dashboards<br/>Alerting]
            Loki[Loki<br/>Log Aggregation<br/>7-day retention]
            Jaeger[Jaeger<br/>Distributed Tracing<br/>OpenTelemetry]
        end

        subgraph "Storage"
            S3[S3 / MinIO<br/>File Storage<br/>Backups<br/>Exports]
        end
    end

    subgraph "Clients"
        Browsers[Web Browsers<br/>Chrome/Edge/Firefox]
        XRDevices[XR Devices<br/>Meta Quest 3]
        Mobile[Mobile Apps<br/>iOS/Android]
    end

    Browsers --> LB
    XRDevices --> LB2
    Mobile --> LB

    LB --> App1
    LB --> App2
    LB --> App3

    LB2 --> App1
    LB2 --> App2
    LB2 --> App3

    App1 --> Neo4jMaster
    App2 --> Neo4jMaster
    App3 --> Neo4jMaster

    App1 --> Neo4jReplica1
    App2 --> Neo4jReplica2
    App3 --> Neo4jReplica1

    App1 --> GPU1
    App2 --> GPU2
    App3 --> GPU1

    App1 --> Redis1
    App2 --> Redis1
    App3 --> Redis1

    Neo4jMaster -.->|Replication| Neo4jReplica1
    Neo4jMaster -.->|Replication| Neo4jReplica2

    Redis1 -.->|Replication| Redis2

    App1 --> Prometheus
    App2 --> Prometheus
    App3 --> Prometheus

    Prometheus --> Grafana
    App1 --> Loki
    App2 --> Loki
    App3 --> Loki

    App1 --> Jaeger
    App2 --> Jaeger
    App3 --> Jaeger

    Neo4jMaster --> S3
    App1 --> S3

    style LB fill:#ff6b6b,color:#fff
    style GPU1 fill:#e1ffe1
    style GPU2 fill:#e1ffe1
    style Neo4jMaster fill:#f0e1ff
```

## 2. Docker Container Architecture

```mermaid
graph TB
    subgraph "Docker Host - Multi-User Development Environment"
        subgraph "Primary Container: turbo-flow-unified"
            direction TB

            subgraph "User Isolation"
                DevUser[devuser UID:1000<br/>Primary Development<br/>Claude Code]
                GeminiUser[gemini-user UID:1001<br/>Gemini Flow<br/>66 Agents]
                OpenAIUser[openai-user UID:1002<br/>OpenAI Codex<br/>API Integration]
                ZaiUser[zai-user UID:1003<br/>Z.AI Service<br/>Port 9600]
            end

            subgraph "Services (Supervisord)"
                MgmtAPI[Management API<br/>Port 9090<br/>FastAPI]
                ZaiService[Z.AI Service<br/>Port 9600<br/>4 Workers]
                VNC[VNC Server<br/>Port 5901<br/>Desktop Access]
                SSH[SSH Server<br/>Port 22<br/>Mapped to 2222]
                CodeServer[code-server<br/>Port 8080<br/>VS Code Web]
            end

            subgraph "Development Tools"
                Rust[Rust 1.75+<br/>Cargo + Clippy]
                Python[Python 3.11<br/>Poetry + pip]
                Node[Node.js 20<br/>npm + pnpm]
                CUDA[CUDA 12.4<br/>Toolkit + Drivers]
            end

            subgraph "MCP Servers (5)"
                ClaudeFlow[claude-flow@alpha<br/>Swarm Orchestration]
                RuvSwarm[ruv-swarm<br/>Advanced Coordination]
                FlowNexus[flow-nexus@latest<br/>Cloud Features]
                GeminiFlow[gemini-flow<br/>66 Agents]
                Custom[Custom MCP Servers<br/>Project-specific]
            end

            subgraph "tmux Workspace (8 Windows)"
                W0[Win 0: Claude-Main<br/>Primary Shell]
                W1[Win 1: Claude-Agent<br/>Agent Execution]
                W2[Win 2: Services<br/>Supervisord Monitor]
                W3[Win 3: Development<br/>Python/Rust/CUDA]
                W4[Win 4: Logs<br/>Service Logs]
                W5[Win 5: System<br/>htop Monitoring]
                W6[Win 6: VNC-Status<br/>VNC Info]
                W7[Win 7: SSH-Shell<br/>General Shell]
            end
        end

        subgraph "Volume Mounts"
            WorkspaceVol[/workspace<br/>Project Files<br/>Persistent]
            CacheVol[/cache<br/>Build Cache<br/>Ephemeral]
            LogsVol[/logs<br/>Service Logs<br/>Persistent]
        end

        subgraph "Network"
            Bridge[Docker Bridge<br/>turbo-net]
        end
    end

    subgraph "Host Machine"
        Ports[Exposed Ports:<br/>2222 → SSH<br/>5901 → VNC<br/>8080 → code-server<br/>9090 → Management API]
        HostGPU[NVIDIA GPU<br/>Passthrough]
    end

    DevUser --> MgmtAPI
    DevUser --> ZaiService
    GeminiUser --> GeminiFlow
    ZaiUser --> ZaiService

    MgmtAPI --> ClaudeFlow
    MgmtAPI --> RuvSwarm
    MgmtAPI --> FlowNexus

    DevUser --> W0
    DevUser --> W1
    DevUser --> W3

    Rust --> WorkspaceVol
    Python --> WorkspaceVol
    Node --> WorkspaceVol
    CUDA --> HostGPU

    VNC --> Ports
    SSH --> Ports
    CodeServer --> Ports
    MgmtAPI --> Ports

    WorkspaceVol --> Bridge
    CacheVol --> Bridge
    LogsVol --> Bridge

    style DevUser fill:#4ecdc4
    style MgmtAPI fill:#ffe66d
    style ZaiService fill:#ff8b94
    style ClaudeFlow fill:#a8e6cf
```

## 3. Database Schema ER Diagram

```mermaid
erDiagram
    Node ||--o{ Edge : "connects"
    Node {
        u32 id PK
        string label
        string metadata_id
        string public
        string owl_class_iri FK
        jsonb properties
    }

    Edge {
        u32 id PK
        u32 source_id FK
        u32 target_id FK
        string edge_type
        jsonb properties
    }

    OwlClass ||--o{ Node : "classifies"
    OwlClass ||--o{ OwlProperty : "has"
    OwlClass {
        string iri PK
        string label
        string[] subclass_of
        string[] disjoint_with
        string[] equivalent_classes
        jsonb annotations
    }

    OwlProperty {
        string iri PK
        string label
        string[] domain
        string[] range
        string property_type
        jsonb restrictions
    }

    UserSettings ||--o{ VisualizationSettings : "has"
    UserSettings ||--o{ PhysicsSettings : "configures"
    UserSettings {
        string pubkey PK
        boolean is_power_user
        datetime created_at
        datetime updated_at
        string default_workspace_id FK
    }

    VisualizationSettings {
        string pubkey PK_FK
        boolean enable_bloom
        boolean physics_enabled
        float node_size
        float edge_width
        string color_scheme
        jsonb advanced_config
    }

    PhysicsSettings {
        string pubkey PK_FK
        float gravity
        float damping
        float spring_strength
        float repulsion
        boolean use_gpu
        jsonb constraints
    }

    Workspace ||--o{ Node : "contains"
    Workspace ||--o{ UserSettings : "owned_by"
    Workspace {
        string id PK
        string name
        string owner_pubkey FK
        datetime created_at
        datetime updated_at
        boolean is_public
        jsonb metadata
    }

    Agent ||--o{ AgentTask : "executes"
    Agent ||--o{ AgentMetrics : "reports"
    Agent {
        string id PK
        string type
        string status
        string swarm_id FK
        jsonb capabilities
        datetime spawned_at
    }

    AgentTask {
        string id PK
        string agent_id FK
        string task_type
        string status
        jsonb input_data
        jsonb output_data
        datetime created_at
        datetime completed_at
    }

    AgentMetrics {
        string agent_id PK_FK
        float cpu_usage
        float memory_usage
        int tasks_completed
        int tasks_failed
        datetime timestamp
    }

    Swarm ||--o{ Agent : "coordinates"
    Swarm {
        string id PK
        string topology
        int max_agents
        string strategy
        datetime created_at
        jsonb configuration
    }
```

## 4. CI/CD Pipeline

```mermaid
graph LR
    subgraph "Source Control"
        GitHub[GitHub Repository<br/>main branch]
        PR[Pull Request<br/>Feature Branch]
    end

    subgraph "CI Pipeline (GitHub Actions)"
        Checkout[Checkout Code<br/>actions/checkout@v4]
        RustTest[Rust Tests<br/>cargo test --all]
        ClientTest[Client Tests<br/>npm test]
        Lint[Linting<br/>clippy + eslint]
        Build[Build<br/>cargo build --release<br/>npm run build]
        Docker[Docker Build<br/>Multi-stage]
    end

    subgraph "Quality Gates"
        Coverage[Code Coverage<br/>Requirement: 70%]
        Security[Security Scan<br/>cargo audit<br/>npm audit]
        Performance[Performance Tests<br/>Benchmark Suite]
    end

    subgraph "Artifact Registry"
        DockerHub[Docker Hub<br/>visionflow/backend]
        NPM[NPM Registry<br/>@visionflow/client]
        GHPackages[GitHub Packages<br/>Release Artifacts]
    end

    subgraph "Deployment Environments"
        Dev[Development<br/>Auto-deploy<br/>dev.visionflow.io]
        Staging[Staging<br/>Manual Approval<br/>staging.visionflow.io]
        Prod[Production<br/>Manual Approval<br/>visionflow.io]
    end

    subgraph "Monitoring"
        Datadog[Datadog<br/>APM + Logs]
        Sentry[Sentry<br/>Error Tracking]
        StatusPage[Status Page<br/>Public Uptime]
    end

    PR --> Checkout
    GitHub --> Checkout

    Checkout --> RustTest
    Checkout --> ClientTest
    Checkout --> Lint

    RustTest --> Coverage
    ClientTest --> Coverage
    Lint --> Security

    Coverage --> Build
    Security --> Build

    Build --> Docker

    Docker --> Performance

    Performance --> DockerHub
    Performance --> NPM
    Performance --> GHPackages

    DockerHub --> Dev
    Dev --> Staging
    Staging --> Prod

    Prod --> Datadog
    Prod --> Sentry
    Prod --> StatusPage

    style GitHub fill:#333,color:#fff
    style Build fill:#4ecdc4
    style Docker fill:#ffe66d
    style Prod fill:#ff6b6b,color:#fff
```

## 5. Network Architecture

```mermaid
graph TB
    subgraph "External Network"
        Internet[Internet]
        CDN[CloudFlare CDN<br/>Static Assets<br/>DDoS Protection]
    end

    subgraph "DMZ - Perimeter Security"
        WAF[Web Application Firewall<br/>OWASP Rules<br/>Rate Limiting]
        LB[Load Balancer<br/>TLS 1.3<br/>HTTPS Only]
    end

    subgraph "Public Subnet - 10.0.1.0/24"
        NGINX1[NGINX 1<br/>10.0.1.10]
        NGINX2[NGINX 2<br/>10.0.1.11]
    end

    subgraph "Application Subnet - 10.0.2.0/24"
        App1[Backend 1<br/>10.0.2.10<br/>Port 4000]
        App2[Backend 2<br/>10.0.2.11<br/>Port 4000]
        App3[Backend 3<br/>10.0.2.12<br/>Port 4000]
    end

    subgraph "GPU Subnet - 10.0.3.0/24"
        GPU1[GPU Node 1<br/>10.0.3.10<br/>CUDA Service]
        GPU2[GPU Node 2<br/>10.0.3.11<br/>CUDA Service]
    end

    subgraph "Data Subnet - 10.0.4.0/24"
        Neo4jM[(Neo4j Master<br/>10.0.4.10<br/>Port 7687)]
        Neo4jR1[(Neo4j Replica 1<br/>10.0.4.11<br/>Port 7687)]
        Neo4jR2[(Neo4j Replica 2<br/>10.0.4.12<br/>Port 7687)]
        RedisM[Redis Master<br/>10.0.4.20<br/>Port 6379]
        RedisR[Redis Replica<br/>10.0.4.21<br/>Port 6379]
    end

    subgraph "Management Subnet - 10.0.5.0/24"
        Bastion[Bastion Host<br/>10.0.5.10<br/>SSH Only]
        Prometheus[Prometheus<br/>10.0.5.20]
        Grafana[Grafana<br/>10.0.5.21]
    end

    subgraph "Firewall Rules"
        FW1[Internet → WAF: 443]
        FW2[WAF → NGINX: 443]
        FW3[NGINX → App: 4000]
        FW4[App → Neo4j: 7687]
        FW5[App → Redis: 6379]
        FW6[App → GPU: 9000]
        FW7[Bastion → All: 22]
    end

    Internet --> CDN
    CDN --> WAF
    WAF --> LB

    LB --> NGINX1
    LB --> NGINX2

    NGINX1 --> App1
    NGINX1 --> App2
    NGINX2 --> App2
    NGINX2 --> App3

    App1 --> Neo4jM
    App2 --> Neo4jM
    App3 --> Neo4jM

    App1 --> Neo4jR1
    App2 --> Neo4jR2
    App3 --> Neo4jR1

    App1 --> RedisM
    App2 --> RedisM
    App3 --> RedisM

    App1 --> GPU1
    App2 --> GPU2
    App3 --> GPU1

    Neo4jM -.->|Replication| Neo4jR1
    Neo4jM -.->|Replication| Neo4jR2
    RedisM -.->|Replication| RedisR

    Bastion --> App1
    Bastion --> App2
    Bastion --> Neo4jM

    Prometheus --> App1
    Prometheus --> App2
    Prometheus --> App3
    Prometheus --> Grafana

    style WAF fill:#ff6b6b,color:#fff
    style Neo4jM fill:#f0e1ff
    style GPU1 fill:#e1ffe1
    style Bastion fill:#ffe66d
```

---

---

## Related Documentation

- [System Architecture Overview - Complete Mermaid Diagrams](01-system-architecture-overview.md)
- [Complete System Data Flow Documentation](../data-flow/complete-data-flows.md)
- [ASCII Diagram Deprecation - Complete Report](../../ASCII_DEPRECATION_COMPLETE.md)
- [Agent Orchestration & Multi-Agent Systems](04-agent-orchestration.md)
- [Server Architecture](../../concepts/architecture/core/server.md)

## 6. Backup & Disaster Recovery

```mermaid
graph TB
    subgraph "Production Systems"
        Neo4jProd[(Neo4j Production<br/>Primary Database)]
        AppProd[Application Servers<br/>State + Sessions]
        FilesProd[File Storage<br/>User Uploads]
    end

    subgraph "Backup Strategy"
        direction TB

        subgraph "Daily Backups (7-day retention)"
            Daily1[(Day 1 Backup<br/>Full)]
            Daily2[(Day 2 Backup<br/>Incremental)]
            Daily3[(Day 3 Backup<br/>Incremental)]
            Daily4[(Day 4 Backup<br/>Incremental)]
            Daily5[(Day 5 Backup<br/>Incremental)]
            Daily6[(Day 6 Backup<br/>Incremental)]
            Daily7[(Day 7 Backup<br/>Full)]
        end

        subgraph "Weekly Backups (4-week retention)"
            Weekly1[(Week 1 Backup)]
            Weekly2[(Week 2 Backup)]
            Weekly3[(Week 3 Backup)]
            Weekly4[(Week 4 Backup)]
        end

        subgraph "Monthly Backups (12-month retention)"
            Monthly1[(Jan Backup)]
            Monthly2[(Feb Backup)]
            Month12[(Dec Backup)]
        end
    end

    subgraph "Backup Storage"
        S3Primary[S3 Primary<br/>us-east-1]
        S3Secondary[S3 Secondary<br/>eu-west-1]
        Glacier[Glacier Deep Archive<br/>Long-term Storage]
    end

    subgraph "Disaster Recovery"
        DRSite[DR Site<br/>Different Region]
        StandbyDB[(Standby Database<br/>Async Replication)]
        DRApp[DR Application<br/>Warm Standby]
    end

    subgraph "Recovery Procedures"
        RTO[RTO: 4 hours<br/>Recovery Time Objective]
        RPO[RPO: 1 hour<br/>Recovery Point Objective]
        Testing[DR Testing<br/>Quarterly]
    end

    Neo4jProd --> Daily1
    Neo4jProd --> Daily2
    AppProd --> Daily1
    FilesProd --> Daily1

    Daily7 --> Weekly1
    Weekly4 --> Monthly1

    Daily1 --> S3Primary
    Daily2 --> S3Primary
    Daily7 --> S3Primary

    S3Primary -.->|Cross-region replication| S3Secondary
    Monthly1 --> Glacier

    Neo4jProd -.->|Async replication| StandbyDB
    StandbyDB --> DRSite
    DRApp --> DRSite

    S3Primary --> RTO
    StandbyDB --> RPO
    DRSite --> Testing

    style Neo4jProd fill:#f0e1ff
    style S3Primary fill:#ffe66d
    style DRSite fill:#ff6b6b,color:#fff
```
