# Your First Graph and AI Agents

*[← Getting Started](../index.md) > [Installation](01-installation.md)*

This guide takes you from a fresh installation to creating your first 3D knowledge graph and deploying multi-agent workflows in under 10 minutes.

## What You'll Learn

- Launch VisionFlow and verify installation
- Create or load your first 3D knowledge graph
- Navigate and interact with the 3D visualisation
- Initialize and deploy your first multi-agent system
- Monitor agent execution and view results

## Prerequisites

Before starting, ensure you've completed the [Installation Guide](01-installation.md) and have:

- ✅ Docker and Docker Compose running
- ✅ VisionFlow services started (`docker-compose ps` shows all services "Up")
- ✅ 8GB+ RAM available (16GB recommended for agents)
- ✅ Modern browser with WebGL support

## Part 1: Your First 3D Knowledge Graph (5 minutes)

### Step 1: Access VisionFlow (30 seconds)

Open your web browser and navigate to:

```
http://localhost:3001
```

**Expected Interface:**
- Dark 3D visualisation area in the centre
- Control panels on the left and right sides
- Status indicator showing "Connected" in green
- Bottom status bar with system information

**Troubleshooting:**
- **Blank page?** Check Docker services: `docker-compose logs -f`
- **Connection error?** Verify port 3001 is not blocked: `sudo lsof -i :3001`
- **Slow loading?** Wait 60 seconds for initial asset loading

### Step 2: Load or Create Your Graph (2 minutes)

Choose one of these options to get started:

#### Option A: Load Demo Data (Fastest - Recommended)

1. Click the **"Load Demo Graph"** button in the control panel
2. Watch as a sample knowledge graph materialises with interconnected nodes
3. The demo includes:
   - **50-100 nodes** representing concepts and entities
   - **Colour-coded relationships** showing different connection types
   - **Realistic physics** with spring forces and repulsion

#### Option B: Connect Your Logseq Data

If you use Logseq for personal knowledge management:

1. Click **"Connect to GitHub"** in the control panel
2. Authorize VisionFlow to access your repository
3. Select your Logseq markdown files
4. VisionFlow automatically:
   - Parses markdown links and references
   - Creates nodes for each page
   - Visualises bi-directional links
   - Applies physics for optimal layout

#### Option C: Create Empty Graph

Start from scratch and build manually:

1. Click **"New Graph"** in the control panel
2. Add nodes: Click **"Add Node"** and enter text (e.g., "Machine Learning", "Neural Networks")
3. Connect nodes: Select two nodes, then click **"Add Edge"**
4. Repeat to build your knowledge structure

### Step 3: Navigate the 3D Space (1 minute)

Master these essential controls:

**Mouse Controls:**
- **Left Click + Drag**: Rotate camera around the graph
- **Right Click + Drag**: Pan camera position
- **Scroll Wheel**: Zoom in and out
- **Double Click Node**: Focus camera on that node

**Keyboard Shortcuts:**
- **`Space`**: Pause/resume physics simulation
- **`R`**: Reset camera to default position
- **`F`**: Toggle fullscreen mode
- **`G`**: Toggle grid display
- **`H`**: Show/hide help overlay
- **`Ctrl+K`**: Open command palette

**Interactive Features:**
- **Click nodes** to select and highlight connections
- **Hover over nodes** to see detailed metadata
- **Edge filtering** controls to show/hide relationship types
- **Physics controls** to adjust gravity, repulsion, spring forces

### Step 4: Adjust Visualisation Settings (1.5 minutes)

Fine-tune the appearance and physics:

**Visual Settings** (Right Panel):
- **Node Colour**: Change base and highlight colours
- **Node Size**: Adjust default size (1-20)
- **Edge Thickness**: Set connection line width (0.1-5.0)
- **Glow Effects**: Add node and edge glow (0.0-1.0 intensity)
- **Bloom Effects**: Enable post-processing bloom

**Physics Settings** (Right Panel):
- **Gravity Strength**: Adjust downward force (-1.0 to 1.0)
- **Spring Strength**: Control attraction between connected nodes (0.0-1.0)
- **Repulsion Force**: Set how strongly unconnected nodes push apart (0-1000)
- **Damping**: Reduce oscillation and stabilise movement (0.0-1.0)
- **Central Force**: Add attraction towards graph centre (0.0-1.0)

**Performance Settings:**
- **Render Quality**: Low/Medium/High based on your hardware
- **Physics Quality**: Adjust simulation complexity
- **GPU Acceleration**: Enable for NVIDIA GPUs

## Part 2: Your First Multi-Agent System (5 minutes)

### Understanding Multi-Agent Workflows

VisionFlow's multi-agent system orchestrates specialized AI agents working together to solve complex problems. Each agent has specific capabilities:

**Agent Types:**
- **Coordinator**: Orchestrates tasks and manages workflow
- **Researcher**: Gathers information and analyses requirements
- **Coder**: Implements solutions and writes code
- **Architect**: Designs system structure and patterns
- **Tester**: Creates tests and validates implementations
- **Reviewer**: Ensures code quality and best practices

**Topology Options:**
- **Mesh**: All agents collaborate equally (best for complex tasks)
- **Hierarchical**: Coordinator directs specialized agents (best for structured workflows)
- **Sequential**: Agents work in defined order (best for pipeline tasks)

### Step 1: Access the Multi-Agent Panel (30 seconds)

1. Look for the **"VisionFlow (MCP)"** section in the left control panel
2. You should see connection status:
   - **Green "Connected"**: MCP bridge is active (multi-agent-container:3002)
   - **Red "Disconnected"**: Check docker services
3. Click the **"Initialize multi-agent"** button to open configuration

**Verify MCP Connection:**
```bash
# Check multi-agent container is running
docker-compose logs multi-agent-container | grep "MCP"

# Expected: "MCP Bridge listening on port 3002"
```

### Step 2: Configure Your First Agent Task (2 minutes)

Fill in the multi-agent configuration form:

**Task Description:**

Start with a simple, achievable task. Here are beginner-friendly examples:

**Example 1: Simple API Development**
```
Create a REST API with three endpoints:
- GET /users - List all users
- POST /users - Create new user
- GET /users/:id - Get specific user

Include basic validation and error handling.
```

**Example 2: Testing Task**
```
Write comprehensive unit tests for a calculator module
that supports add, subtract, multiply, and divide operations.
Include edge cases and error scenarios.
```

**Example 3: React Component**
```
Build a React component for a todo list with:
- Add new todo item
- Mark items as complete
- Delete items
- Local storage persistence
```

**Topology Selection:**

- **Mesh** (Recommended for beginners): Agents collaborate freely, best for problem-solving
- **Hierarchical**: Coordinator manages specialists, best for structured projects
- **Sequential**: Linear pipeline, best for step-by-step processes

**Strategy Selection:**

- **Consensus**: Agents vote on decisions (slower, higher quality)
- **Leader-Based**: Coordinator makes final decisions (faster, coordinator-dependent)
- **Autonomous**: Each agent works independently (fastest, may lack coordination)

**Priority:**
- **Low**: Background task, may take hours
- **Medium**: Standard priority, typically 15-30 minutes
- **High**: Urgent task, prioritised processing

**Agent Count:**
Start with **3-5 agents** for your first task:
- 1 Coordinator
- 1 Researcher (if needed)
- 1 Coder
- 1 Tester
- 1 Reviewer (optional)

### Step 3: Launch the Multi-Agent System (30 seconds)

1. Review your configuration
2. Click **"Launch Multi-Agent System"**
3. Watch the agent graph appear:
   - **Green nodes** represent active agents (bit 31 flags)
   - **Blue nodes** represent knowledge graph (bit 30 flags)
   - **Animated connections** show agent communication
   - **Node colours** indicate agent status

**Agent Status Colours:**
- **Green**: Active and processing
- **Yellow**: Waiting for input
- **Blue**: Idle/ready
- **Red**: Error or blocked
- **Grey**: Completed/terminated

### Step 4: Monitor Execution (2 minutes)

Track your agents in real-time:

**Visual Monitoring:**
- **Agent positions** update via binary WebSocket protocol (34-byte format)
- **Connection lines** show agent communication
- **Node selection** reveals detailed agent state
- **Real-time updates** at 60 FPS via BinaryWebSocketProtocol

**Status Panel:**
1. Open the **"Agent Status"** panel (left side)
2. See detailed information:
   - Current task each agent is executing
   - Progress percentage
   - Communication history
   - Resource usage

**Logs and Output:**
1. Click any agent node to view:
   - **Execution logs**: Real-time agent output
   - **Task history**: Completed subtasks
   - **Communication**: Messages between agents
   - **Errors**: Any issues encountered

**Command Palette Monitoring:**
- Press **`Ctrl+K`**
- Type "agent status" to see global overview
- Type "agent logs" for detailed logging
- Type "agent kill" to terminate if needed

### Step 5: View Results (1 minute)

When agents complete:

1. **Success notification** appears in top-right
2. **Agent nodes turn grey** indicating completion
3. **Results panel** opens automatically showing:
   - **Output files**: Code, documentation, tests
   - **Summary**: What each agent accomplished
   - **Metrics**: Execution time, token usage, costs
   - **Quality score**: Overall task quality assessment

**Export Results:**
- Click **"Download Results"** to save all output
- Click **"View in IDE"** to open code files
- Click **"Copy to Clipboard"** for quick sharing

## Part 3: External Services Integration (Advanced)

VisionFlow integrates with external AI and processing services:

### RAGFlow Integration

RAGFlow provides retrieval-augmented generation for knowledge-enhanced responses:

**Connection**: External Docker network `docker_ragflow`

**Setup:**
1. Ensure RAGFlow is running in separate docker-compose
2. VisionFlow auto-connects via shared network
3. Access RAGFlow UI at configured port (typically 8080)

**Usage:**
- Agents automatically query RAGFlow for knowledge retrieval
- Configure RAG settings in Multi-Agent panel
- View RAG queries in agent communication logs

### Voice Services (Whisper + Kokoro)

Enable voice interaction for accessibility and convenience:

**Whisper STT** (Speech-to-Text):
- Service: `whisper-webui-backend:8000`
- Purpose: Converts voice commands to text
- Supported languages: Auto-detect or specify

**Kokoro TTS** (Text-to-Speech):
- Service: `kokoro-tts-container:8880`
- Purpose: Converts agent responses to speech
- Voices: Multiple voice profiles available (af_bella, etc.)

**Enable Voice:**
1. Set `ENABLE_VOICE=true` in `.env`
2. Grant browser microphone permissions
3. Click microphone icon in control panel
4. Speak commands or queries
5. Hear agent responses via speakers

**Test Voice Pipeline:**
```bash
# Run voice pipeline test
bash scripts/voice_pipeline_test.sh

# Test individual services
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello World","voice":"af_bella"}'
```

### Vircadia XR/AR (Meta Quest 3)

For immersive 3D exploration:

**Requirements:**
- Meta Quest 3 or compatible WebXR device
- VisionFlow XR mode enabled (`ENABLE_XR=true`)
- WebXR-compatible browser on Quest (Chrome/Firefox)

**Launch XR Mode:**
1. Put on your Quest 3 headset
2. Open browser and navigate to VisionFlow URL
3. Click the **VR icon** in the interface
4. Grant permissions when prompted
5. Use hand tracking or controllers to interact

**XR Controls:**
- **Hand tracking**: Point and pinch to select nodes
- **Controllers**: Trigger to select, grip to grab/move
- **Teleport**: Point to location and click to move
- **Spatial audio**: Hear agent communication in 3D space

**XR Features:**
- **Immersive graph exploration**: Walk through your knowledge graph
- **Spatial agent monitoring**: Agents positioned in 3D space
- **Hand gesture commands**: Natural interaction
- **Collaborative viewing**: Multiple users can join (if enabled)

See [XR Setup Guide](../guides/xr-setup.md) for detailed configuration.

## Next Steps

### Beginner Path
1. ✅ Complete this guide
2. 📖 [Orchestrating Agents](../guides/orchestrating-agents.md) - Advanced agent patterns
3. 🎯 [Configuration Guide](../reference/configuration.md) - Customize your setup

### Intermediate Path
1. 🔧 [Development Workflow](../guides/development-workflow.md) - Contribute to VisionFlow
2. 🤖 [Agent Templates](../reference/agents/templates/index.md) - Create custom agents
3. 🥽 [XR Integration](../guides/xr-setup.md) - Immersive VR experiences

### Advanced Path
1. 🏗️ [System Architecture](../concepts/system-architecture.md) - Deep dive into design
2. 💻 [API Reference](../reference/api/index.md) - Build integrations
3. 🚀 [Deployment Guide](../guides/deployment.md) - Production setup

## Troubleshooting Quick Reference

### Graph Issues

**Graph not loading:**
```bash
# Check backend health
curl http://localhost:3001/api/health

# Check logs
docker-compose logs visionflow_container
```

**Poor performance:**
- Lower render quality in settings
- Reduce node count with filtering
- Enable GPU acceleration
- Check FPS in debug overlay (Ctrl+Shift+D)

### Agent Issues

**Agents not starting:**
```bash
# Verify MCP connection
docker-compose logs multi-agent-container | grep MCP

# Check available resources
docker stats multi-agent-container
```

**Agent stuck/not responding:**
- Check agent logs (click agent node)
- Verify API keys are configured
- Increase agent timeout in settings
- Restart agent: `docker-compose restart multi-agent-container`

### External Service Issues

**RAGFlow connection failed:**
```bash
# Check RAGFlow network
docker network ls | grep ragflow

# Verify VisionFlow is on network
docker network inspect docker_ragflow
```

**Voice services not working:**
```bash
# Test Whisper
curl http://localhost:8000/health

# Test Kokoro
curl http://localhost:8880/health

# Run pipeline test
bash scripts/voice_pipeline_test.sh
```

## Getting Help

- **📚 [Full Documentation](../index.md)** - Comprehensive guides
- **🐛 [Troubleshooting Guide](../guides/troubleshooting.md)** - Common issues
- **💬 GitHub Discussions** - Community support *(Repository URL to be configured)*
- **🔍 [API Reference](../reference/api/index.md)** - Technical details

---

**🎉 Congratulations!** You've successfully created your first 3D knowledge graph and deployed a multi-agent system in VisionFlow.

## Related Documentation

- [Configuration Reference](../reference/configuration.md) - Complete settings guide
- [Agent Orchestration](../guides/orchestrating-agents.md) - Advanced agent patterns
- [XR/AR Setup](../guides/xr-setup.md) - Immersive experiences
- [System Architecture](../concepts/system-architecture.md) - How it all works
