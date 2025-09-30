# Quick Start Guide

*[Getting-Started](../index.md)*

Get VisionFlow running and create your first stunning 3D graph visualisation with AI multi-agent systems in just 5 minutes! This comprehensive guide takes you from installation through your first successful multi-agent deployment.

## Prerequisites

Before starting, ensure you have:

### System Requirements
- **Docker** 20.10+ and Docker Compose 2.0+
- **8GB RAM** minimum (16GB recommended for multi-agent tasks)
- **Modern web browser** with WebGL support
- **Internet connection** for AI model access

### Optional Components
- **NVIDIA GPU** with CUDA 11.8+ for GPU acceleration
- **Meta Quest 3** for XR features
- **16GB+ RAM** for large multi-agent deployments

## 5-Minute Quick Start

### Step 1: Installation and Setup (2 minutes)

```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/visionflow/visionflow.git
cd visionflow

# 2. Configure environment
cp .env.example .env

# Edit .env with essential settings:
# CLAUDE_FLOW_HOST=multi-agent-container
# MCP_TCP_PORT=9500
# ENABLE_GPU=true (if you have an NVIDIA GPU)

# 3. Start all services
docker-compose up -d

# Wait for services to initialise
echo "Waiting for VisionFlow to start..."
sleep 60

# 4. Verify services are running
docker-compose ps
```

Expected output:
```
NAME                    COMMAND                  SERVICE             STATUS
visionflow_container    "/app/scripts/start.sh"  webxr              Up
multi-agent-container   "python3 -m claude..."   claude-flow        Up
postgres                "docker-entrypoint..."   postgres           Up
redis                   "docker-entrypoint..."   redis              Up
```

### Step 2: Open VisionFlow (30 seconds)

Open your web browser and navigate to:
```
http://localhost:3001
```

You should see the VisionFlow interface with:
- A dark 3D visualisation area in the centre
- Control panels on the left and right sides
- A status indicator showing "Connected" in green

### Step 3: Load Your First Graph (1 minute)

There are several ways to quickly get started:

#### Option A: Load Demo Data (Fastest)
1. Click the **"Load Demo Graph"** button in the control panel
2. Watch as a sample knowledge graph appears with interconnected nodes
3. Use your mouse to rotate, zoom, and explore the 3D space

#### Option B: Create Empty Graph
1. Click **"New Graph"** in the control panel
2. Add nodes by clicking **"Add Node"** and entering text
3. Connect nodes by selecting two nodes and clicking **"Add Edge"**

#### Option C: Load Your Logseq Data
1. Click **"Connect to GitHub"** in the control panel
2. Authorize VisionFlow to access your Logseq repository
3. Select your markdown files to visualise

### Step 4: Explore and Interact (1.5 minutes)

#### Mouse Controls
- **Left Click + Drag**: Rotate the camera around the graph
- **Right Click + Drag**: Pan the camera position
- **Scroll Wheel**: Zoom in and out
- **Double Click**: Focus on a specific node

#### Keyboard Shortcuts
- **`Space`**: Pause/resume physics simulation
- **`R`**: Reset camera to default position
- **`F`**: Toggle fullscreen mode
- **`G`**: Toggle grid display
- **`H`**: Show/hide help overlay

#### Interactive Features
- **Node Selection**: Click nodes to select and highlight connections
- **Node Information**: Hover over nodes to see detailed information
- **Edge Filtering**: Use controls to filter relationship types
- **Physics Controls**: Adjust gravity, repulsion, and spring forces

## Your First Multi-Agent Workflow

### What are Multi-Agent Systems?

Multi-agent systems in VisionFlow are intelligent AI agents that work together to solve complex problems. Each agent has specialised capabilities:

- **Coordinator**: Orchestrates tasks and manages workflow
- **Researcher**: Gathers information and analyzes requirements
- **Coder**: Implements solutions and writes code
- **Architect**: Designs system structure and patterns
- **Tester**: Creates tests and validates implementations
- **Reviewer**: Ensures code quality and best practices

### Initialize Your First Multi-Agent System

#### Step 1: Access the Multi-Agent Panel
1. Look for the **"VisionFlow (MCP)"** section in the left control panel
2. You should see an **"Initialize multi-agent"** button if no agents are active
3. Click this button to open the configuration dialogue

#### Step 2: Configure Your Agents
Fill in the multi-agent configuration form:

**Task Description:**
Start with a simple, clear task. Here are some beginner-friendly examples:

```
Create a simple REST API with user authentication
```

```
Write unit tests for a calculator module
```

```
Build a basic React component with state management
```

**Topology Selection:**
- **Mesh**: Best for collaborative tasks (recommended for beginners)
- **Hierarchical**: Structured approach with clear leadership
- **Star**: Central coordination model
- **Ring**: Sequential processing

**Agent Selection:**
For your first workflow, try this combination:
- ✅ Coordinator (always recommended)
- ✅ Researcher
- ✅ Coder
- ✅ Tester

**Settings:**
- **Maximum Agents**: Start with 4-6 agents
- **Neural Enhancement**: Enable for better performance
- **GPU Acceleration**: Enable if you have a compatible GPU

#### Step 3: Deploy Your Agents
1. Click **"Spawn Hive Mind"** to start the agent deployment
2. Watch the 3D visualisation as agents appear as nodes
3. Observe the connections forming between agents as they communicate

#### Step 4: Monitor Agent Activity
As your agents work:
- **Node Colors**: Different colors represent agent types and states
  - 🔵 Blue: Coordinator agents
  - 🟢 Green: Researcher agents
  - 🟡 Yellow: Coder agents
  - 🟠 Orange: Tester agents
  - ⚪ White: Idle agents
  - 🔴 Red: Agents with errors

- **Edge Thickness**: Thicker connections indicate more communication
- **Node Movement**: Active agents move more dynamically
- **Pulsing**: Indicates agents are processing tasks

#### Step 5: Review Results
1. Check the **Activity Log** panel for detailed agent communications
2. View **Task Progress** to see completion status
3. Examine any **Generated Files** or **Code Output**
4. Review **Performance Metrics** to understand efficiency

### Sample Multi-Agent Workflows

#### Quick Development Task (5-10 minutes)
```
Task: "Create a simple todo list API with GET and POST endpoints"
Topology: mesh
Agents: coordinator, researcher, coder, tester
Max Agents: 4
```

Expected outcome:
- Researcher analyzes requirements
- Coder implements the API
- Tester creates validation tests
- Coordinator ensures everything works together

#### Documentation Project (10-15 minutes)
```
Task: "Document the authentication system with API examples"
Topology: hierarchical
Agents: coordinator, researcher, documenter, reviewer
Max Agents: 4
```

Expected outcome:
- Researcher gathers system information
- Documenter creates comprehensive documentation
- Reviewer ensures quality and accuracy
- Coordinator manages the overall process

#### Code Analysis Task (15-20 minutes)
```
Task: "Analyze this codebase and suggest improvements"
Topology: star
Agents: coordinator, analyst, optimiser, documenter
Max Agents: 5
```

Expected outcome:
- Analyst examines code structure and patterns
- Optimizer identifies performance improvements
- Documenter creates improvement recommendations
- Coordinator synthesizes all findings

## Understanding the Interface

### Left Control Panel

#### Graph Management
- **Load Demo Graph**: Quick sample data for testing
- **New Graph**: Create empty graph from scratch
- **Save Graph**: Persist current graph state
- **Load Graph**: Restore previously saved graph

#### Multi-Agent Systems
- **Initialize Multi-Agent**: Configure and deploy AI agents
- **Agent Status**: Monitor active agents and their states
- **Task Management**: View and control ongoing tasks
- **Performance Metrics**: Real-time system performance data

#### Data Sources
- **GitHub Integration**: Connect to Logseq repositories
- **File Upload**: Import local markdown files
- **API Connection**: Connect to external data sources
- **Real-time Sync**: Toggle automatic updates

### Right Control Panel

#### Visualisation Settings
- **Physics Controls**: Adjust gravity, repulsion, and springs
- **Rendering Options**: Quality settings and visual effects
- **Camera Settings**: Field of view and movement sensitivity
- **Performance**: Frame rate and optimisation settings

#### Analytics and Insights
- **Clustering**: Group related nodes automatically
- **Anomaly Detection**: Highlight unusual patterns
- **Semantic Analysis**: AI-powered relationship discovery
- **Export Options**: Save visualisations and data

#### Advanced Features
- **XR Mode**: Enable Quest 3 or other XR devices
- **Voice Controls**: Activate voice interaction
- **API Access**: Generate API keys and endpoints
- **Debug Tools**: Performance monitoring and troubleshooting

### 3D Visualisation Area

#### Node Types
- **Knowledge Nodes**: Represent concepts, documents, or ideas
  - 📄 Document nodes (blue)
  - 💡 Concept nodes (yellow)
  - 🏷️ Tag nodes (green)

- **Agent Nodes**: Represent AI agents and their activities
  - 🎯 Coordinator agents (purple)
  - 🔍 Researcher agents (teal)
  - ⚙️ Coder agents (orange)
  - 🧪 Tester agents (pink)

#### Edge Types
- **Solid Lines**: Strong relationships or frequent communication
- **Dashed Lines**: Weak relationships or occasional communication
- **Thick Lines**: High-bandwidth or critical connections
- **Thin Lines**: Low-bandwidth or informational connections

#### Interactive Elements
- **Selection Highlighting**: Selected nodes and their connections
- **Hover Information**: Tooltips with detailed node/edge data
- **Context Menus**: Right-click for additional options
- **Drag and Drop**: Manual node positioning

## Common Workflows

### Knowledge Graph Exploration

#### Personal Knowledge Base
1. Connect your Logseq repository via GitHub
2. Watch as your markdown files become an interactive 3D graph
3. Explore connections between your notes and ideas
4. Discover unexpected relationships through AI analysis

#### Research Project
1. Import research papers and documents
2. Use AI clustering to group related concepts
3. Identify knowledge gaps and research opportunities
4. Export findings for publication or presentation

#### Learning and Education
1. Create concept maps for complex topics
2. Visualise learning pathways and dependencies
3. Track progress through knowledge domains
4. Share interactive visualisations with students

### AI Agent Collaboration

#### Software Development
- **Frontend Development**: React components, styling, user experience
- **Backend Development**: APIs, databases, server logic
- **Testing**: Unit tests, integration tests, quality assurance
- **Documentation**: Technical writing, API docs, user guides

#### Content Creation
- **Research**: Information gathering, fact-checking, source validation
- **Writing**: Content creation, editing, style consistency
- **Review**: Quality assurance, accuracy verification
- **Publishing**: Formatting, distribution, platform optimisation

#### Data Analysis
- **Collection**: Data gathering from multiple sources
- **Processing**: Cleaning, transformation, validation
- **Analysis**: Statistical analysis, pattern recognition
- **Visualisation**: Charts, graphs, dashboard creation

## Tips for Success

### Best Practices

#### Start Simple
- Begin with small graphs (10-50 nodes)
- Use basic physics settings initially
- Focus on one feature at a time
- Gradually increase complexity

#### Optimise Performance
- Enable GPU acceleration if available
- Adjust physics quality based on your hardware
- Use clustering for large datasets
- Monitor frame rate and adjust settings accordingly

#### Effective Multi-Agent Use
- Start with 3-5 agents for simple tasks
- Clearly define your task objectives
- Choose appropriate topology for your workflow
- Monitor agent communication and performance

### Troubleshooting Tips

#### Graph Not Loading
1. Check browser console for errors (F12 → Console)
2. Verify WebSocket connection status
3. Refresh the page and try again
4. Check that all Docker services are running

#### Poor Performance
1. Reduce graph complexity or node count
2. Lower physics simulation quality
3. Disable GPU acceleration if causing issues
4. Close other browser tabs to free memory

#### Agents Not Spawning
1. Verify Claude Flow MCP connection is active
2. Check that all required services are running
3. Review task description for clarity
4. Try with fewer agents initially

#### 3D Navigation Issues
1. Ensure WebGL is enabled in your browser
2. Update graphics drivers if using GPU acceleration
3. Try different camera sensitivity settings
4. Reset camera position with 'R' key

## Next Steps

Now that you've mastered the basics, explore these advanced features:

### Advanced Visualisation
- **[Custom Shaders](../client/rendering.md)** - Create stunning visual effects
- **[XR Integration](../client/xr.md)** - Immersive Quest 3 experiences
- **[Performance Optimisation](../client/performance.md)** - Smooth 60fps with large graphs

### API Integration
- **[REST API](../api/rest.md)** - Programmatic control and automation
- **[WebSocket API](../api/websocket.md)** - Real-time data streaming
- **[GraphQL](../api/graphql.md)** - Flexible data queries

### Advanced Multi-Agent Systems
- **[Custom Agents](../features/custom-agents.md)** - Build specialised AI agents
- **[Workflow Automation](../features/workflows.md)** - Automate complex processes
- **[Performance Monitoring](../features/monitoring.md)** - Track system efficiency

### Production Deployment
- **[Docker Production](../deployment/docker.md)** - Scalable container deployment
- **[Cloud Deployment](../deployment/cloud.md)** - AWS, GCP, Azure setups
- **[Monitoring](../deployment/monitoring.md)** - Production monitoring and alerts

## Community and Support

### Get Help
- **[Documentation Hub](../index.md)** - Comprehensive guides
- **[Troubleshooting](../troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/visionflow/visionflow/issues)** - Bug reports and features
- **[Discord Community](https://discord.gg/visionflow)** - Real-time community support

### Share Your Success
- **[Community Showcase](https://showcase.visionflow.ai)** - Share your visualisations
- **[Blog](https://blog.visionflow.ai)** - Write about your experience
- **[Social Media](https://twitter.com/visionflow)** - Connect with other users

---

**Congratulations!** You've successfully created your first VisionFlow graph and deployed AI agents. The future of knowledge visualisation and AI collaboration is at your fingertips!

## Related Topics

- [Configuration Guide](configuration.md)
- [Getting Started with VisionFlow](00-index.md)
- [Installation Guide](01-installation.md)