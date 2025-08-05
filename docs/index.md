# LogseqXR Agent Visualization Documentation

Welcome to the comprehensive documentation for the LogseqXR agent visualization system. This system provides real-time 3D visualization of AI agent swarms, knowledge graphs, and collaborative workflows.

## Quick Start

- **[Getting Started Guide](quick-start-swarm.md)** - Spawn your first agent swarm in 5 minutes
- **[Docker Deployment](deployment/docker-mcp-integration.md)** - Deploy the full system with Docker
- **[API Reference](api/index.md)** - REST and WebSocket API documentation

## Architecture Overview

### System Architecture
- **[System Overview](architecture/system-overview.md)** - High-level architecture and components
- **[Parallel Graphs Architecture](architecture/parallel-graphs.md)** - Multi-graph visualization system
- **[Bots/VisionFlow System](architecture/bots-visionflow-system.md)** - AI agent visualization architecture
- **[GPU Migration Architecture](architecture/visionflow-gpu-migration.md)** - GPU-accelerated physics system

### Key Features
- **Dual-Graph System**: Simultaneous visualization of Logseq knowledge graphs and AI agent swarms
- **Real-time Physics**: GPU-accelerated spring physics for smooth 60 FPS visualization
- **Binary Protocol**: Efficient data streaming for 200+ agents
- **MCP Integration**: Full Claude Flow hive-mind AI agent support

## API Documentation

### Core APIs
- **[REST API Reference](api/rest.md)** - HTTP endpoints for agent management
- **[WebSocket Protocols](api/websocket-protocols.md)** - Real-time communication protocols
- **[Binary Protocol Specification](api/binary-protocol.md)** - High-performance position streaming

### Integration APIs
- **[MCP Tool Reference](server/features/claude-flow-mcp-integration.md)** - Claude Flow MCP tools
- **[Agent Control API](deployment/docker-mcp-integration.md#agent-control-system-setup)** - TCP control interface

## Server Documentation

### Core Server
- **[Server Architecture](server/architecture.md)** - Rust backend architecture
- **[Actor System](server/actors.md)** - Actix actor-based design
- **[GPU Compute](server/gpu-compute.md)** - GPU physics integration
- **[Handlers](server/handlers.md)** - HTTP and WebSocket handlers

### Features
- **[Claude Flow Integration](server/features/claude-flow-mcp-integration.md)** - MCP agent integration
- **[AI Services](server/ai-services.md)** - AI-powered features
- **[Feature Access Control](server/feature-access.md)** - Feature flag management

## Client Documentation

### Frontend Architecture
- **[Client Architecture](client/architecture.md)** - React/Three.js architecture
- **[Component Library](client/components.md)** - Reusable UI components
- **[3D Visualization](client/visualization.md)** - Three.js graph rendering
- **[State Management](client/state.md)** - Redux and context patterns

### User Interface
- **[UI Components](client/ui-components.md)** - Material-UI component system
- **[Settings Panel](client/settings-panel-redesign.md)** - Advanced configuration UI
- **[Command Palette](client/command-palette.md)** - Keyboard-driven interface
- **[Help System](client/help-system.md)** - Interactive user guidance

### Advanced Features
- **[Parallel Graphs](client/parallel-graphs.md)** - Multi-graph visualization
- **[WebSocket Integration](client/websocket.md)** - Real-time data handling
- **[XR Support](client/xr.md)** - Virtual reality features

## Deployment & Operations

### Deployment Guides
- **[Docker & MCP Integration](deployment/docker-mcp-integration.md)** - Complete deployment guide
- **[Docker Configuration](deployment/docker.md)** - Container setup
- **[Production Deployment](deployment/docker-mcp-integration.md#production-deployment)** - Security and scaling

### Development
- **[Development Setup](development/setup.md)** - Local development environment
- **[Testing Guide](development/testing.md)** - Unit and integration testing
- **[Debugging Tools](development/debugging.md)** - Troubleshooting guide

### Configuration
- **[Configuration Reference](configuration/index.md)** - All configuration options
- **[Environment Variables](deployment/docker-mcp-integration.md#environment-variables)** - Runtime configuration

## Advanced Topics

### Technical Deep Dives
- **[Decoupled Graph Architecture](technical/decoupled-graph-architecture.md)** - Graph system design
- **[MCP Tool Usage](technical/mcp_tool_usage.md)** - Advanced MCP patterns
- **[Voice System](voice-system.md)** - Voice control integration

### Performance & Optimization
- **[GPU Physics Pipeline](architecture/visionflow-gpu-migration.md#gpu-processing-pipeline)** - GPU compute details
- **[Binary Protocol Optimization](api/websocket-protocols.md#performance-metrics)** - Bandwidth optimization
- **[Scaling Strategies](deployment/docker-mcp-integration.md#scaling-considerations)** - Horizontal scaling

## Contributing

- **[Contributing Guide](contributing.md)** - How to contribute to the project
- **[Code of Conduct](contributing.md#code-of-conduct)** - Community guidelines
- **[Development Workflow](contributing.md#development-workflow)** - PR process

## Reference

- **[Glossary](glossary.md)** - Terminology and definitions
- **[Migration Guides](architecture/migration-guide.md)** - Upgrading from older versions
- **[Troubleshooting](deployment/docker-mcp-integration.md#troubleshooting)** - Common issues and solutions

## External Resources

- [Claude Flow Documentation](https://github.com/Agentic-Insights/claude-flow)
- [MCP Specification](https://github.com/anthropics/mcp)
- [Logseq API Reference](https://docs.logseq.com/)
- [Three.js Documentation](https://threejs.org/docs/)