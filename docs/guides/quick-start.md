# VisionFlow Quick Start Guide

## Overview

Get VisionFlow up and running in minutes! This comprehensive guide will help you set up the real-time 3D visualisation platform for knowledge graphs and AI multi-agent systems, from installation through your first multi-agent deployment.

## Prerequisites

Before starting, ensure you have:

### System Requirements
- **Docker** 20.10+ and Docker Compose 2.0+
- **8GB RAM** minimum (16GB recommended for complex tasks)
- **Modern web browser** with WebGL support
- **Internet connection** for AI model access

### Optional Components
- **NVIDIA GPU** with CUDA 11.8+ for GPU acceleration
- **Meta Quest 3** for XR features
- **16GB+ RAM** for large multi-agent deployments

### Technical Prerequisites
- Basic understanding of command line
- Familiarity with Docker concepts
- Knowledge of web browsers and developer tools

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

Key environment variables to configure:

```bash
# Core Settings
CLAUDE_FLOW_HOST=multi-agent-container  # Host for Claude Flow MCP server
CLAUDE_FLOW_PORT=3002                   # Port for Claude Flow service

# Performance Settings
ENABLE_GPU=true                         # Enable GPU acceleration (if available)
MAX_AGENTS=20                           # Maximum concurrent agents
MEMORY_LIMIT=8g                         # Container memory limit

# Optional Features
ENABLE_XR=false                         # Meta Quest integration
DEBUG_MODE=false                        # Enable detailed logging
```

### 3. Start the System

Launch all services with Docker Compose:

```bash
# Standard deployment
docker-compose up

# GPU-accelerated deployment (recommended)
docker-compose -f docker-compose.gpu.yml up

# Background deployment
docker-compose up -d
```

This will start:
- **VisionFlow Backend**: Rust/Actix server on port 3001
- **Claude Flow MCP**: Agent orchestration service on port 3002
- **Supporting Services**: RAGFlow, databases, caching layers

### 4. Verify Installation

Check that all services are running:

```bash
docker-compose ps
```

You should see all services in "Up" status.

### 5. Access the Interface

Open your browser and navigate to:
```
http://localhost:3001
```

You should see the VisionFlow interface with a 3D visualisation area and control panel.

## Your First Multi-Agent Workflow

### 1. Open the Control Panel

Look for the control panel in the top-left corner of the VisionFlow interface. It should display connection status and system statistics.

### 2. Initialize Multi-Agent System

When no agents are active, you'll see an **"Initialize multi-agent"** button in the VisionFlow (MCP) section. Click this button to begin.

### 3. Configure Your Multi-Agent System

Fill in the multi-agent configuration form with the following details:

#### Task Description (Required)
Describe what you want the agents to accomplish. Be specific and clear:

**Simple Tasks:**
```
"Create a hello world REST API with basic endpoints"
"Write unit tests for the math utility module"
"Document the user authentication system"
```

**Complex Tasks:**
```
"Build a full-stack app with React and Express, including auth and database"
"Analyze the codebase and create architectural documentation with diagrams"
"Refactor legacy code to TypeScript with full test coverage"
```

**Specialized Tasks:**
```
"Design microservices architecture for e-commerce platform"
"Create ML pipeline for text classification with evaluation metrics"
"Build and compile LaTeX contract templates with automated generation"
```

#### Topology Selection
Choose how agents communicate and coordinate:

- **`mesh`** - All agents communicate freely (best for collaboration)
- **`hierarchical`** - Structured command chain with clear hierarchy
- **`star`** - Central coordinator manages all interactions
- **`ring`** - Sequential processing in a circular pattern

#### Agent Types
Select specialist agents based on your task requirements:

- **Coordinator** - Orchestrates the team and manages workflows
- **Researcher** - Gathers information and analyzes requirements
- **Coder** - Implements solutions and writes code
- **Architect** - Designs system architecture and patterns
- **Tester** - Creates and runs tests, validates implementations
- **Reviewer** - Reviews code quality and best practices
- **Documenter** - Creates comprehensive documentation
- **Analyst** - Analyzes data, patterns, and performance metrics
- **Optimizer** - Improves performance and efficiency

#### Configuration Options
- **Maximum Agents**: 3-20 agents (default: 8)
  - Start with 3-5 for simple tasks
  - Use 8-12 for complex projects
  - Scale to 15-20 for enterprise workflows

- **Neural Enhancement**: Enable WASM-accelerated AI processing
- **GPU Acceleration**: Leverage GPU for physics simulation and visualisation

### 4. Spawn the Multi-Agent System

Click **"Spawn Hive Mind"** to initiate the system. The process will:

1. **Connect to Claude Flow** - Establish MCP server connection
2. **Initialize Multi-Agent** - Create the agent network with your configuration
3. **Spawn Agents** - Deploy selected specialist agents
4. **Apply Task** - Distribute your task across the agent network
5. **Begin Visualization** - Start real-time 3D visualisation of agent activity

## Using the Control Panel

### Monitoring Features

#### Real-Time Statistics
- **Active Agents**: Current number of deployed agents
- **Connections**: Inter-agent communication links
- **Token Usage**: AI model token consumption tracking
- **Task Progress**: Overall completion status
- **Performance Metrics**: System resource utilization

#### 3D Visualization Controls
- **Node Colors**: Different agent types and current states
- **Edge Thickness**: Communication intensity between agents
- **Physics Simulation**: GPU-accelerated force-directed layout
- **Real-time Updates**: Live agent position and interaction updates

### Keyboard Shortcuts

Navigate and control the visualisation:
- `Space` - Pause/resume physics simulation
- `R` - Reset camera view to default position
- `F` - Toggle fullscreen mode
- `D` - Toggle debug information overlay
- `G` - Toggle grid display
- `Mouse Drag` - Rotate view
- `Mouse Wheel` - Zoom in/out
- `Shift + Mouse Drag` - Pan view

## Multi-Agent Configuration

### Topology Best Practices

#### Mesh Configuration
Best for: Collaborative tasks, brainstorming, code reviews
```javascript
{
  "topology": "mesh",
  "maxAgents": 6,
  "agentTypes": ["coordinator", "researcher", "coder", "reviewer"]
}
```

#### Hierarchical Configuration
Best for: Structured projects, enterprise workflows, complex planning
```javascript
{
  "topology": "hierarchical", 
  "maxAgents": 10,
  "agentTypes": ["coordinator", "architect", "coder", "tester", "documenter"]
}
```

#### Star Configuration
Best for: Centralized coordination, single-focus tasks
```javascript
{
  "topology": "star",
  "maxAgents": 5,
  "agentTypes": ["coordinator", "specialist", "specialist", "specialist"]
}
```

### Agent Selection Strategies

#### Development Projects
- Coordinator + Architect + Coder + Tester + Reviewer
- Focus on code quality and architecture

#### Research Tasks  
- Coordinator + Researcher + Analyst + Documenter
- Emphasis on information gathering and analysis

#### Complex System Design
- Coordinator + Architect + Multiple Specialists + Optimizer
- Comprehensive system design and optimisation

## Advanced Features

### Neural Enhancement
Enable WASM-accelerated AI processing for:
- Faster agent decision-making
- Enhanced pattern recognition
- Optimized resource allocation
- Improved coordination efficiency

### GPU Acceleration
Leverage GPU resources for:
- Real-time 3D visualisation
- Physics simulation
- Large-scale agent networks
- Complex mathematical computations

### Cross-Session Memory
Agents can maintain context across:
- Multiple task sessions
- System restarts
- Configuration changes
- Long-running projects

## Troubleshooting

### Common Issues and Solutions

#### No Agents Appearing

**Problem**: Control panel shows no active agents after spawning

**Solutions**:
1. Check browser console for JavaScript errors (F12 → Console)
2. Verify all Docker services are running:
   ```bash
   docker-compose ps
   ```
3. Check WebSocket connection in Network tab (F12 → Network)
4. Restart the backend server:
   ```bash
   docker-compose restart visionflow-backend
   ```

#### Connection Issues

**Problem**: MCP WebSocket connection fails

**The system will attempt multiple connection endpoints**:
- Primary: `ws://localhost:3001/ws/mcp`
- Secondary: `ws://localhost:3001/wss`
- Fallback: `ws://127.0.0.1:3001/ws/mcp`

**Solutions**:
1. Check firewall settings for ports 3001-3002
2. Verify Docker network configuration:
   ```bash
   docker network ls
   docker network inspect visionflow_default
   ```
3. Check container logs:
   ```bash
   docker-compose logs claude-flow
   docker-compose logs visionflow-backend
   ```

#### 404 Error When Spawning

**Problem**: "404 Not Found" when initializing multi-agent

**Solutions**:
1. Restart the backend server after adding new endpoints:
   ```bash
   docker-compose restart visionflow-backend
   ```
2. If using supervisorctl:
   ```bash
   supervisorctl restart webxr
   ```
3. Check API endpoint availability:
   ```bash
   curl http://localhost:3001/api/health
   ```

#### Performance Issues

**Problem**: Slow visualisation or unresponsive interface

**Solutions**:
1. Enable GPU acceleration:
   ```bash
   docker-compose -f docker-compose.gpu.yml up
   ```
2. Reduce maximum agents for complex tasks (try 3-5 agents first)
3. Check system resources:
   ```bash
   docker stats
   ```
4. Increase memory allocation in `.env`:
   ```bash
   MEMORY_LIMIT=16g
   ```

#### Memory Issues

**Problem**: "Out of memory" errors or system slowdowns

**Solutions**:
1. Monitor Docker memory usage:
   ```bash
   docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
   ```
2. Increase Docker memory limits
3. Reduce concurrent agents
4. Clear browser cache and restart

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set in .env file
DEBUG_MODE=true
LOG_LEVEL=debug

# Restart services
docker-compose down && docker-compose up
```

### Health Checks

Verify system health:

```bash
# Backend health
curl http://localhost:3001/api/health

# Claude Flow health  
curl http://localhost:3002/health

# WebSocket connection test
wscat -c ws://localhost:3001/ws/mcp
```

## Next Steps

Once you have your first multi-agent system running successfully:

### Explore Advanced Features
- [Architecture Overview](../architecture/system-overview.md) - Understand the system design
- [API Documentation](../api/rest.md) - Integrate with the REST API  
- [Configuration Guide](../configuration/index.md) - Advanced configuration options

### Development and Integration
- [Development Setup](../development/setup.md) - Set up local development environment
- [Testing Guide](../development/testing.md) - Test your implementations
- [Debugging Guide](../development/debugging.md) - Debug and troubleshoot

### Production Deployment
- [Docker Deployment](../deployment/docker.md) - Deploy using Docker
- [Docker MCP Integration](../deployment/docker-mcp-integration.md) - Advanced deployment patterns
- [Monitoring Guide](../deployment/monitoring.md) - Set up comprehensive monitoring

### Community and Support
- [GitHub Issues](https://github.com/yourusername/visionflow/issues) - Report bugs and request features
- [Documentation Hub](../index.md) - Comprehensive documentation
- [Community Discord](https://discord.gg/visionflow) - Connect with other users
- [Contributing Guide](../CONTRIBUTING.md) - Contribute to the project

## Example Workflows

### Rapid Prototyping
```javascript
// Quick API development
{
  "task": "Create a REST API for user management with CRUD operations",
  "topology": "mesh",
  "agents": ["coordinator", "coder", "tester"],
  "maxAgents": 4
}
```

### Enterprise Development
```javascript
// Full-scale application development
{
  "task": "Build scalable e-commerce platform with microservices architecture",
  "topology": "hierarchical", 
  "agents": ["coordinator", "architect", "coder", "tester", "reviewer", "documenter"],
  "maxAgents": 12
}
```

### Research and Analysis
```javascript
// Comprehensive codebase analysis
{
  "task": "Analyze legacy system and provide modernization roadmap",
  "topology": "star",
  "agents": ["coordinator", "researcher", "analyst", "architect", "documenter"],
  "maxAgents": 8
}
```

---

**Welcome to VisionFlow!** You now have everything needed to deploy intelligent multi-agent systems and visualise their collaborative workflows in real-time 3D environments.