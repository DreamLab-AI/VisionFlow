# VisionFlow Documentation

Welcome to the official documentation for VisionFlow, a high-performance, GPU-accelerated platform for real-time 3D visualisation of AI multi-agent systems and knowledge graphs.

This documentation provides a comprehensive guide for users, developers, and operators. Whether you are getting started, developing new features, or deploying to production, you will find the necessary information here.

## Documentation Structure

Our documentation is organised into several key areas to help you find information quickly.

```mermaid
graph TD
    A[VisionFlow Docs] --> B[Getting Started];
    A --> C[Guides];
    A --> D[Concepts];
    A --> E[Reference];
    A --> F[Development];

    subgraph B[01. Getting Started]
        B1[Installation]
        B2[Quickstart]
    end

    subgraph C[02. Guides]
        C1[Agent Orchestration]
        C2[Settings & Tuning]
        C3[Deployment]
        C4[Troubleshooting]
    end

    subgraph D[03. Concepts]
        D1[System Architecture]
        D2[Actor Model]
        D3[GPU Acceleration]
        D4[Dual-Graph System]
    end

    subgraph E[04. Reference]
        E1[API Reference]
        E2[Binary Protocol]
        E3[Configuration]
        E4[Glossary]
    end

    subgraph F[05. Development]
        F1[Contributing]
        F2[Environment Setup]
        F3[Testing & Debugging]
    end
```

## Quick Links

*   **New to VisionFlow?** Start with the **[Getting Started](./getting-started/index.md)** guide.
*   **Want to perform a specific task?** Check our **[How-to Guides](./guides/index.md)**.
*   **Need to understand the architecture?** Read the **[Concepts](./architecture/index.md)** documentation.
*   **Looking for technical details?** Dive into the **[Reference](./reference/index.md)** section.
*   **Want to contribute?** See the **[Development](./development/index.md)** guides.

## Navigation by Role

### For Users & Analysts
- **Installation**: [getting-started/installation.md](./getting-started/installation.md)
- **First Visualisation**: [getting-started/quickstart.md](./getting-started/quickstart.md)
- **Tuning and Settings**: [guides/settings-guide.md](./guides/settings-guide.md)

### For DevOps & System Administrators
- **Deployment with Docker**: [deployment/docker.md](./deployment/docker.md)
- **Configuration Reference**: [configuration/index.md](./configuration/index.md)
- **Troubleshooting**: [guides/troubleshooting.md](./guides/troubleshooting.md)

### For Developers & Contributors
- **System Architecture**: [architecture/system-overview.md](./architecture/system-overview.md)
- **API Reference**: [api/index.md](./api/index.md)
- **Development Workflow**: [development/index.md](./development/index.md)
- **Contributing Guide**: [contributing.md](./contributing.md)

## Documentation Sections

### 📚 [Getting Started](./getting-started/index.md)
Learn how to install VisionFlow and create your first visualisation. Perfect for new users who want to get up and running quickly.

- Installation guide with prerequisites
- Quick start tutorial
- First visualisation walkthrough

### 🔧 [Guides](./guides/index.md)
Practical how-to guides for accomplishing specific tasks with VisionFlow.

- Agent orchestration and multi-agent configuration
- Settings and performance tuning
- Docker deployment and scaling
- Troubleshooting common issues
- Development workflow best practices

### 🧠 [Architecture](./architecture/index.md)
Understand the core concepts and architecture behind VisionFlow.

- System architecture and design principles
- Actor model for concurrent processing
- GPU acceleration and CUDA integration
- Dual-graph system for knowledge and agents

### 📖 [Reference](./reference/index.md)
Complete technical reference documentation.

- REST and WebSocket API specifications
- Binary protocol format
- Configuration options and schemas
- Glossary of terms

### 🛠️ [Development](./development/index.md)
Everything you need to contribute to VisionFlow development.

- Contributing guidelines
- Development environment setup
- Testing strategies
- Debugging techniques

## Key Features Documentation

### 🤖 AI Agent Visualisation
- [Agent Orchestration Guide](./features/agent-orchestration.md) - Configure and manage AI agent swarms
- [MCP Integration](./architecture/mcp-integration.md) - Claude Flow Model Context Protocol setup

### 📊 Knowledge Graph Integration
- [Parallel Graphs](./client/parallel-graphs.md) - Understanding parallel visualisation
- [Graph Configuration](./configuration/index.md) - Customising graph behaviour

### 🚀 Performance & Optimisation
- [GPU Acceleration](./server/gpu-compute.md) - CUDA-powered physics engine
- [Performance Tuning](./guides/settings-guide.md) - Optimising for your hardware

### 🔌 Integration & APIs
- [API Reference](./api/index.md) - Complete API documentation
- [Binary Protocol](./api/websocket-protocols.md) - High-performance streaming protocol

## Production Deployment

For production deployments, follow our comprehensive deployment guide:

1. [Docker Deployment](./deployment/docker.md) - Container-based deployment
2. [Configuration Reference](./configuration/index.md) - Production configuration
3. [Troubleshooting](./guides/troubleshooting.md) - Common issues and solutions

## Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/visionflow/visionflow/issues)
- **Discord**: [Join our community](https://discord.gg/visionflow)
- **Contributing**: [How to contribute](./contributing.md)

## Documentation Standards

This documentation follows the [Diátaxis framework](https://diataxis.fr/), organising content into:

- **Tutorials** (Getting Started) - Learning-oriented guides
- **How-to Guides** (Guides) - Task-oriented instructions
- **Explanations** (Concepts) - Understanding-oriented discussion
- **Reference** - Information-oriented technical descriptions

## Version Information

- **Documentation Version**: 2.0.0
- **VisionFlow Version**: 1.0.0
- **Last Updated**: January 2025

## License

VisionFlow is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

---

**Need help?** Start with our [Getting Started](./getting-started/index.md) guide or check the [Troubleshooting](./guides/troubleshooting.md) section.