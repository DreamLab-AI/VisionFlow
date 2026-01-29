---
title: C4 Context Diagram - VisionFlow System
description: C4 Level 1 diagram showing VisionFlow's external system boundaries and user interactions
category: explanation
tags:
  - architecture
  - c4
  - diagrams
  - context
updated-date: 2026-01-29
difficulty-level: intermediate
---

# C4 Context Diagram - VisionFlow System

This diagram shows VisionFlow at the system context level (C4 Level 1), illustrating how external users and systems interact with the platform.

## System Context

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
graph TB
    subgraph "External Users"
        Developer[Developer<br/>Uses web interface for<br/>knowledge exploration]
        DataScientist[Data Scientist<br/>Analyzes graph patterns<br/>and clusters]
        XRUser[XR/VR User<br/>Immersive 3D<br/>visualization]
    end

    subgraph "External Systems"
        GitHubAPI[GitHub API<br/>Source of markdown<br/>documentation]
        AIServices[AI Services<br/>Claude + Perplexity<br/>Semantic analysis]
        NostrNetwork[Nostr Network<br/>Decentralized identity<br/>and authentication]
    end

    subgraph "VisionFlow Platform"
        VF[VisionFlow<br/>Knowledge Graph Visualization<br/>with GPU-Accelerated Physics<br/>and AI-Powered Analysis]
    end

    subgraph "Infrastructure"
        Neo4j[(Neo4j 5.13<br/>Graph Database<br/>Cypher queries)]
        GPU[GPU Compute<br/>CUDA 12.4<br/>87 physics kernels]
    end

    Developer -->|HTTPS/WSS| VF
    DataScientist -->|HTTPS/WSS| VF
    XRUser -->|WebXR| VF

    GitHubAPI -->|REST API| VF
    AIServices -->|API calls| VF
    NostrNetwork -->|NIP-07| VF

    VF -->|Bolt protocol| Neo4j
    VF -->|CUDA FFI| GPU

    style VF fill:#4A90D9,color:#fff,stroke:#333,stroke-width:3px
    style Neo4j fill:#f0e1ff,stroke:#333
    style GPU fill:#e1ffe1,stroke:#333
    style GitHubAPI fill:#e3f2fd,stroke:#333
    style AIServices fill:#fff3e0,stroke:#333
    style NostrNetwork fill:#fce4ec,stroke:#333
```

## User Personas

| Persona | Primary Use Case | Key Features Used |
|---------|------------------|-------------------|
| Developer | Navigate codebase knowledge graph | Graph exploration, search, GitHub sync |
| Data Scientist | Analyze patterns and clusters | Clustering algorithms, PageRank, community detection |
| XR/VR User | Immersive visualization | WebXR, spatial interaction, voice commands |

## External System Integrations

| System | Protocol | Data Flow | Purpose |
|--------|----------|-----------|---------|
| GitHub API | REST + Webhooks | Bidirectional | Sync markdown documentation |
| AI Services | HTTPS REST | Request-Response | Semantic analysis, embeddings |
| Nostr Network | NIP-07 WebSocket | Bidirectional | Decentralized authentication |
| Neo4j | Bolt (TCP) | Bidirectional | Graph persistence |
| GPU (CUDA) | FFI | Local | Physics simulation |

## System Boundaries

**Inside VisionFlow:**
- Web application (React + Three.js)
- REST and WebSocket APIs
- Actor-based backend (Actix)
- GPU physics engine (CUDA)
- OWL/RDF ontology processing

**Outside VisionFlow:**
- User browsers and XR devices
- GitHub repositories
- AI/ML service providers
- Nostr relay network
- External databases

## Related Documentation

- [C4 Container Diagram](c4-container.md)
- [System Architecture Overview](../overview.md)
- [Data Flow Documentation](../../diagrams/data-flow/complete-data-flows.md)
