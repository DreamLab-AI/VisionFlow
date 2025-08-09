# VisionFlow Documentation

Welcome to the VisionFlow documentation hub. VisionFlow is a high-performance, GPU-accelerated knowledge graph and AI agent swarm visualization system built with Rust, WebGL, and modern AI integration.

## Getting Started

- [**Quick Start Guide**](quick-start.md) - Get VisionFlow running in minutes
- [**Docker Deployment**](deployment/docker-guide.md) - Production deployment with Docker
- [**Development Setup**](development/setup.md) - Set up your development environment

## Core Concepts

- [**System Overview**](architecture/system-overview.md) - High-level architecture and design principles
- [**Dual Graph Architecture**](architecture/dual-graph.md) - Parallel visualization of Knowledge and Agent graphs
- [**GPU Compute**](architecture/gpu-compute.md) - CUDA-accelerated physics and visualization
- [**MCP Integration**](architecture/mcp-integration.md) - Claude Flow agent orchestration

## Documentation Sections

### Architecture
Detailed technical documentation of the system architecture, actor model, and core components.
- [System Overview](architecture/system-overview.md)
- [Dual Graph System](architecture/dual-graph.md)
- [GPU Compute Pipeline](architecture/gpu-compute.md)
- [MCP/Claude Flow Integration](architecture/mcp-integration.md)

### API Reference
Complete API documentation for REST endpoints, WebSocket protocols, and binary data formats.
- [REST API](api/rest.md)
- [WebSocket Protocols](api/websocket-protocols.md)
- [Binary Protocol](api/binary-protocol.md)

### Server Documentation
Backend implementation details, actor system, and server features.
- [Actor System](server/actors.md)
- [Agent Swarm Feature](server/features/agent-swarm.md)
- [Physics Engine](server/features/physics-engine.md)

### Client Documentation
Frontend implementation with React, Three.js, and WebXR.
- [Overview](client/index.md)
- [3D Visualization](client/3d-visualization.md)
- [WebXR/AR Support](client/webxr.md)

### Configuration
System configuration, settings management, and environment variables.
- [Configuration Guide](configuration/index.md)
- [Settings System](configuration/settings.md)
- [Feature Access Control](configuration/feature-access.md)

### Deployment
Production deployment guides and Docker configuration.
- [Docker Guide](deployment/docker-guide.md)
- [Environment Variables](deployment/environment.md)
- [Performance Tuning](deployment/performance.md)

### Development
Developer guides, build instructions, and contribution guidelines.
- [Development Setup](development/setup.md)
- [Building from Source](development/building.md)
- [Testing](development/testing.md)

### Technical Deep Dives
Advanced technical documentation and implementation details.
- [Decoupled Graph Architecture](technical/decoupled-graph-architecture.md)
- [MCP Tool Usage](technical/mcp_tool_usage.md)
- [Binary Protocol Specification](technical/binary-protocol-spec.md)

## Key Features

### AI Agent Swarm Visualization
Real-time visualization of AI agent swarms orchestrated through Claude Flow/MCP, showing agent relationships, task distribution, and communication patterns.

### Dual Graph Architecture
Simultaneous visualization of:
- **Knowledge Graph**: Logseq markdown-based knowledge structures
- **Agent Graph**: Live AI agent swarm activity and interactions

### High-Performance Backend
- **Rust/Actix**: Type-safe, memory-efficient backend
- **Actor Model**: Concurrent, fault-tolerant architecture
- **Binary Protocol**: Efficient 28-byte per node updates
- **GPU Acceleration**: CUDA kernels for physics simulation

### Immersive Frontend
- **React/TypeScript**: Modern, type-safe frontend
- **Three.js**: Hardware-accelerated 3D rendering
- **WebXR Support**: VR/AR capabilities for Quest 3 and other devices
- **Real-time Updates**: 60 FPS visualization with WebSocket streaming

### Integrated AI Services
- **RAGFlow**: Document processing and retrieval
- **Perplexity**: AI-powered search integration
- **Speech Services**: Voice interaction capabilities
- **Claude Flow**: Agent orchestration and management

## System Requirements

- **Docker**: 20.10+ with Docker Compose
- **GPU**: NVIDIA GPU with CUDA 11.8+ support (for GPU features)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB available space
- **Network**: Stable internet for AI service connections

## Performance Benchmarks

- **Node Capacity**: 100,000+ nodes
- **Edge Capacity**: 1,000,000+ edges
- **Update Rate**: 60 FPS sustained
- **Network Efficiency**: 28 bytes per node update
- **GPU Utilization**: 80%+ efficiency on force calculations

## Community

- [GitHub Repository](https://github.com/yourusername/visionflow)
- [Issue Tracker](https://github.com/yourusername/visionflow/issues)
- [Discussions](https://github.com/yourusername/visionflow/discussions)

## License

VisionFlow is licensed under the MIT License. See [LICENSE](../LICENSE) for details.