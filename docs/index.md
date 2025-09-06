# VisionFlow Documentation

## Overview

VisionFlow is a sophisticated real-time 3D visualisation platform that combines AI agent orchestration, GPU-accelerated physics, and cutting-edge XR capabilities. Built with a decoupled actor-based architecture using Rust backend and React/TypeScript frontend, it provides a powerful environment for visualising and interacting with complex knowledge graphs and AI agents.

## Documentation Structure

### Getting Started
- [Quick Start Guide](guides/quick-start.md) - Get up and running quickly
- [System Requirements](README.md) - Hardware and software requirements

### Core Features
- [GPU Compute System](server/gpu-compute.md) - CUDA-accelerated processing
- [Physics Engine](server/physics-engine.md) - Force-directed layout calculations
- [Agent Orchestration](features/agent-orchestration.md) - AI agent management

### Development
- [Development Setup](development/setup.md) - Local development environment
- [Debug System](development/debugging.md) - Debugging tools and techniques
- [API Reference](api/index.md) - Complete API documentation

### Configuration
- [Settings Guide](guides/settings-guide.md) - User interface settings
- [Configuration Reference](configuration/index.md) - System configuration options

### Deployment
- [Docker MCP Integration](deployment/docker-mcp-integration.md) - Containerised MCP deployment
- [Deployment Guide](deployment/index.md) - Production deployment strategies

### Architecture
- [System Overview](architecture/system-overview.md) - High-level system design
- [Technical Documentation](technical/) - Detailed technical specifications

### Client Documentation
- [Client Architecture](client/) - Frontend architecture and components

### Server Documentation
- [Server Components](server/) - Backend services and APIs

### Additional Resources
- [Glossary](glossary.md) - Technical terms and definitions
- [Contributing](contributing.md) - Development guidelines and contribution process
- [Security](security/) - Security policies and best practices

## Quick Start

### Prerequisites
- Docker 20.10+ with Docker Compose
- NVIDIA GPU with CUDA 11.8+ (for GPU features)
- Node.js 22+ and Rust 1.75+ (for development)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ext

# Production deployment
docker-compose up -d

# Development environment
./scripts/dev.sh

# Access application
open http://localhost:3001
```

## Key Features

- **AI Agent Orchestration**: 15+ specialised agent types with hierarchical, mesh, ring, and star topologies
- **GPU-Accelerated Physics**: CUDA implementation optimised for NVIDIA hardware with dual-graph support
- **XR/AR Capabilities**: Meta Quest 3 support with hand tracking and WebXR integration
- **Real-time Communication**: Binary protocol with WebSocket endpoints for minimal bandwidth usage

## Technology Stack

### Backend
- Rust 1.75+ with Actix-Web framework
- CUDA 11.8+ for GPU acceleration
- WebSocket and MCP protocol support

### Frontend
- React 18 with TypeScript 5
- Three.js and React Three Fiber for 3D graphics
- WebXR integration for AR/VR support

## Contributing

See [Contributing Guide](contributing.md) for development workflow, code standards, and submission process.

## Support

- **Documentation**: Complete guides and API references
- **Issues**: GitHub issue tracker
- **Discussions**: Community support channels

Based on the provided repository files, here is a detailed explanation of the GPU force-directed graph feature set, how to interpret the client and agent visualizations, and the underlying mechanics of the agent telemetry graph.

### 1. GPU Force-Directed Graph: Feature Set

The system utilizes a high-performance, GPU-accelerated physics engine to render two simultaneous force-directed graphs: a primary knowledge graph (from Logseq) and a secondary agent telemetry graph. The entire physics simulation is handled by a unified CUDA kernel (`visionflow_unified.cu`) for maximum efficiency.

The core feature set of the physics engine includes several tunable forces and behaviors that dictate the graph's layout and dynamics:

*   **Repulsive Force (`repel_k`)**: This is a fundamental force where every node pushes every other node away, preventing clumps and ensuring labels are readable. The strength of this force is a key parameter for adjusting the overall spread of the graph.
*   **Attractive Spring Force (`spring_k`)**: This force only applies between connected nodes (nodes with an edge). It acts like a spring, pulling connected nodes together. The interplay between repulsion and attraction is what forms the characteristic clusters and structures in the graph.
*   **Damping**: This force acts like friction, gradually slowing down the movement of all nodes. High damping leads to a stable, static layout more quickly, while low damping results in a more dynamic, constantly shifting graph.
*   **Velocity & Force Clamping**: To prevent the simulation from "exploding" (where nodes fly off to infinity due to extreme forces), there are built-in limits on the maximum velocity a node can achieve and the maximum force that can be applied in a single step.
*   **Boundary Constraints**: The simulation space has defined boundaries. If a node travels beyond these bounds, a "soft" force gently pushes it back towards the centre, keeping the graph contained.
*   **Progressive Warmup**: When the graph first loads, forces are introduced gradually over the first 200 iterations. This prevents the initial, randomly-placed nodes from violently flying apart, leading to a much smoother and more stable initialization.
*   **Node Collapse Prevention**: A hard-coded minimum distance (`MIN_DISTANCE`) is enforced between nodes. If two nodes get closer than this, a strong, localized repulsive force pushes them apart to prevent them from overlapping and collapsing into the same point.

These parameters are managed through a complete pipeline, starting from UI controls in `PhysicsEngineControls.tsx`, flowing through the Rust backend via the `/api/settings` and `/api/physics/update` endpoints, and ultimately updating the `SimParams` structure used by the CUDA kernel.

### 2. Client Visualization: Interpreting the Knowledge Graph Nodes

The primary graph visualization renders your knowledge base (e.g., Logseq notes). Each node's appearance is not random; it's a rich visual representation of its underlying metadata, primarily defined in `MetadataShapes.tsx`.

Here is how to interpret the visual properties of a node:

*   **Shape**: A node's geometry is determined by its connectivity (`hyperlinkCount`).
    *   **Sphere**: A node with few or no connections.
    *   **Cube (Box)**: A node with some connections.
    *   **Octahedron**: A well-connected node.
    *   **Icosahedron**: A highly-connected, important hub node.

*   **Size (Scale)**: A node's size is a function of both its content size and its connectivity.
    *   **File Size**: Larger files result in larger nodes (on a logarithmic scale).
    *   **Connectivity**: Nodes with more connections are rendered larger, making hubs naturally stand out.

*   **Color & "Heat"**: The color is based on the node's base color setting but is modulated by its recency.
    *   **Recency**: More recently modified files (`lastModified`) have a "heat" effect applied. They appear brighter, more saturated, and their hue is shifted slightly towards warmer colors like yellow and orange. This allows you to see recent activity in your knowledge graph at a glance.
    *   **Node Type**: If recency data is unavailable, the color is tinted based on the node's `metadata.type` (e.g., folder, file, function).

*   **Glow & AI Processing**: Nodes that have been processed by an AI service (indicated by a `perplexityLink` in their metadata) have a distinct gold-tinted emissive glow. This visually flags all AI-enriched content in your graph.

*   **Pulse Animation**: The speed of the subtle hologram pulse effect is determined by the `fileSize`, adding another layer of data visualization to the node's animation.

*   **Labels**: Node labels display the file or concept name, and can be configured to show additional metadata like file size or type, providing further context directly in the visualization.

### 3. Agent Telemetry: Interpreting the Force Graph

The application also visualizes a real-time force graph of a multi-agent AI system, driven by telemetry data from the "Claude Flow MCP" (Multi-Agent Control Plane). This visualization provides insights into the health, status, and activity of the AI agent swarm.

The feature set and interpretation are as follows:

*   **Data Source**: The backend connects to the Claude Flow MCP via a direct TCP connection (`claude_flow_actor_tcp.rs`). The `bots_handler.rs` processes this telemetry and streams it to the client for visualization.
*   **Force-Directed Layout**: The agent graph uses the same GPU-accelerated physics engine as the knowledge graph. The `weight` of the edges between agents is determined by their **communication intensity**, meaning agents that communicate more frequently are pulled closer together.

The visual representation of each agent node in `BotsVisualizationFixed.tsx` conveys its current state:

*   **Glow Color (Health)**: The color of a node's glow is a direct indicator of its health.
    *   **Green**: Healthy (> 80%).
    *   **Gold/Yellow**: Warning (> 50%).
    *   **Red**: Critical (< 50%).

*   **Size (CPU Usage)**: The size of an agent node scales with its current CPU usage. Larger nodes represent agents that are performing more intensive computations.

*   **Shape (Status)**: The node's geometry changes based on its operational status.
    *   **Sphere**: The default shape for an active or busy agent.
    *   **Box (Cube)**: An agent that is currently initializing.
    *   **Tetrahedron (Pyramid)**: An agent that is in an error state.

*   **Animation (Activity)**:
    *   **Pulsing**: Indicates an agent is `active` or `busy`.
    *   **Rotating**: Indicates an agent is `busy` with a task.

*   **Telemetry Overlay**: When hovering over a node (or if it's particularly active), an information panel appears displaying detailed telemetry:
    *   **Agent Name & Type**: (e.g., Coder, Tester, Analyst).
    *   **Status**: Color-coded status badge (e.g., `active`, `busy`, `idle`).
    *   **Health & CPU Usage**: Precise percentage values.
    *   **Task Information**: The number of active and completed tasks, and the description of the current task.
    *   **Processing Logs**: A stream of the agent's latest activities.
    *   **Token Usage**: The number of tokens consumed by the agent.
---

*VisionFlow - Real-time 3D visualisation platform with AI agent orchestration*