# Quick Start Guide

Get VisionFlow running and visualize your first AI agent swarm in minutes.

## Prerequisites

- Docker 20.10+ with Docker Compose
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with CUDA 11.8+ (optional, for GPU features)
- Modern web browser (Chrome, Firefox, or Edge)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/visionflow.git
cd visionflow
```

### 2. Configure Environment

Copy the example environment file and adjust settings:

```bash
cp .env.example .env
```

Key environment variables:
- `CLAUDE_FLOW_HOST`: Host for Claude Flow MCP server (default: `multi-agent-container`)
- `CLAUDE_FLOW_PORT`: Port for Claude Flow (default: `3002`)
- `ENABLE_GPU`: Enable GPU acceleration (default: `true` if NVIDIA GPU detected)

### 3. Start the System

```bash
docker-compose up
```

This will start:
- **VisionFlow Backend**: Rust/Actix server on port 3001
- **Claude Flow MCP**: Agent orchestration service
- **Supporting Services**: RAGFlow, databases, etc.

### 4. Access the Interface

Open your browser and navigate to:
```
http://localhost:3001
```

## Spawning Your First AI Agent Swarm

### 1. Open the Control Panel

Look for the control panel in the top-left corner of the interface.

### 2. Initialize a Swarm

Click the **"Initialize Swarm"** button in the VisionFlow (MCP) section.

### 3. Configure Your Swarm

Fill in the swarm configuration:

**Task Description** (Required):
- What you want the agents to accomplish
- Examples:
  - "Build a REST API with authentication"
  - "Analyze and document this codebase"
  - "Create a machine learning pipeline"

**Topology**:
- `mesh` - All agents communicate freely (best for collaboration)
- `hierarchical` - Structured command chain
- `star` - Central coordinator model
- `ring` - Sequential processing

**Agent Types**:
- **Coordinator** - Orchestrates the team
- **Researcher** - Gathers information
- **Coder** - Implements solutions
- **Architect** - Designs systems
- **Tester** - Validates implementations
- **Reviewer** - Reviews code quality
- **Documenter** - Creates documentation

**Max Agents**: 3-20 (default: 8)

### 4. Spawn the Hive Mind

Click **"Spawn Hive Mind"** to:
1. Connect to Claude Flow
2. Initialize agent swarm
3. Begin task execution
4. Start real-time visualization

## Monitoring Your Swarm

### Visualization Features

- **3D Graph**: Real-time agent positions and connections
- **Node Colors**: Agent types and states
- **Edge Thickness**: Communication intensity
- **Physics Simulation**: GPU-accelerated force-directed layout

### Control Panel Stats

- **Active Agents**: Current agent count
- **Connections**: Inter-agent communication links
- **Token Usage**: AI model token consumption
- **Task Progress**: Completion status

### Keyboard Shortcuts

- `Space` - Pause/resume physics
- `R` - Reset view
- `F` - Toggle fullscreen
- `D` - Toggle debug info
- `G` - Toggle grid

## Example Tasks

### Simple Tasks
```
"Create a hello world REST API"
"Write unit tests for the math module"
"Document the authentication system"
```

### Complex Tasks
```
"Build a full-stack app with React and Express, including auth and database"
"Analyze the codebase and create architectural documentation with diagrams"
"Refactor legacy code to TypeScript with full test coverage"
```

### Specialized Tasks
```
"Design microservices architecture for e-commerce"
"Create ML pipeline for text classification"
"Build and compile LaTeX contract templates"
```

## Troubleshooting

### No Agents Appearing

1. Check browser console for errors (F12)
2. Verify Claude Flow is running:
   ```bash
   docker-compose ps
   ```
3. Check WebSocket connection in Network tab

### Connection Issues

The system will attempt multiple connection endpoints:
- Primary: `ws://localhost:3001/ws/mcp`
- Fallback: `ws://localhost:3001/wss`

### Performance Issues

1. Enable GPU acceleration:
   ```bash
   docker-compose -f docker-compose.gpu.yml up
   ```
2. Reduce max agents for complex tasks
3. Check system resources:
   ```bash
   docker stats
   ```

## Next Steps

- [Architecture Overview](architecture/system-overview.md) - Understand the system design
- [API Documentation](api/rest.md) - Integrate with the REST API
- [Development Setup](development/setup.md) - Set up local development
- [Configuration Guide](configuration/index.md) - Advanced configuration options

## Support

- [GitHub Issues](https://github.com/yourusername/visionflow/issues)
- [Documentation](index.md)
- [Community Discord](https://discord.gg/visionflow)