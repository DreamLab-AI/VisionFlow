# Getting Started with VisionFlow

## Quick Start Guide

VisionFlow is a real-time 3D visualisation platform combining AI agent orchestration with GPU-accelerated graph physics. Follow this guide to get up and running in minutes.

## Prerequisites

### System Requirements

**Minimum:**
- OS: Linux (Ubuntu 20.04+), macOS 12+, Windows 10+ with WSL2
- CPU: 4 cores, 2.5GHz+
- RAM: 8GB
- Docker: Version 20.10+
- Node.js: Version 18+ (for development)

**Recommended:**
- CPU: 8+ cores, 3.5GHz+
- RAM: 16GB+
- GPU: NVIDIA GPU with CUDA 11.8+ (for GPU acceleration)
- Storage: SSD with 20GB+ available

### Optional Features

**For GPU Physics:**
- NVIDIA GPU with CUDA 11.8+
- NVIDIA Container Toolkit (for Docker GPU support)

**For XR/VR:**
- Meta Quest 3 or compatible WebXR device
- WebXR-compatible browser (Chrome 90+, Firefox 98+)

## Installation Options

### Option 1: Docker (Recommended)

The fastest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd ext

# Start all services with Docker Compose
docker-compose up -d

# Access the application
open http://localhost:3002
```

**Services Started:**
- Frontend: React app on port 3002
- Backend: Rust server on port 8080
- GPU Service: CUDA physics engine

### Option 2: Development Setup

For development and customisation:

```bash
# Prerequisites
sudo apt update
sudo apt install curl build-essential

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Node.js (via nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# Clone and setup
git clone <repository-url>
cd ext

# Install dependencies
cd client && npm install
cd ../server && cargo build --release

# Start development servers
./scripts/dev.sh
```

## First Launch

### 1. Verify Installation

After starting VisionFlow, navigate to `http://localhost:3002`. You should see:

✅ **Expected:** Loading screen followed by 3D visualisation interface  
❌ **Problem:** Blank page or errors → Check Docker logs or development console

### 2. Basic Interface Tour

**Main Components:**
- **GraphCanvas**: Central 3D visualisation area with dual graph rendering
- **SettingsPanelRedesign**: Right sidebar with settings and controls
- **CommandPalette**: Press `Ctrl+K` for quick actions  
- **ConversationPane**: Bottom pane for AI interactions
- **Status Bar**: Bottom bar showing connection status

### 3. Load Sample Data

**Option A: Demo Mode**
```bash
# Load built-in demo graph
curl -X POST http://localhost:8080/api/demo/load
```

**Option B: Connect Logseq**
1. Open SettingsPanelRedesign (gear icon)
2. Navigate to "Data Sources" tab  
3. Enter your GitHub repository URL for Logseq data
4. Configure GitHub token for authentication
5. Click "Connect"

### 4. Verify GPU Acceleration (Optional)

If you have an NVIDIA GPU:

```bash
# Check GPU detection
curl http://localhost:8080/api/system/gpu-info

# Expected response:
# {"gpu_count": 1, "gpu_name": "GeForce RTX 3060", "cuda_version": "11.8"}
```

## Configuration

### Environment Variables

Create `.env` file in the project root:

```env
# Server Configuration
RUST_LOG=info
SERVER_PORT=8080
CLIENT_PORT=3002

# GPU Settings (optional)
ENABLE_GPU=true
CUDA_VISIBLE_DEVICES=0

# Data Sources
GITHUB_TOKEN=your_github_token
GITHUB_REPO=your_username/logseq-repo

# AI Services
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key

# Claude Flow MCP Integration
CLAUDE_FLOW_HOST=multi-agent-container
MCP_TCP_PORT=9500
MCP_TRANSPORT=tcp

# Features
ENABLE_XR=true
ENABLE_VOICE=false
```

### Basic Settings

Access settings via the gear icon in the top-right:

**Essential Settings:**
- **Physics Quality**: Start with "Medium" for balanced performance
- **Render Quality**: Adjust based on your hardware
- **Graph Type**: Choose "Logseq" for knowledge graphs, "Agents" for AI systems
- **XR Mode**: Enable for VR/AR experiences

## Key Features Overview

### 1. Real-time Graph Physics

VisionFlow uses GPU-accelerated physics to create dynamic, force-directed layouts:

- **Spring Forces**: Connected nodes attract each other
- **Repulsion**: Unconnected nodes push apart
- **Damping**: Prevents excessive movement
- **Boundaries**: Keeps graphs contained

**Controls:**
- Mouse: Rotate and zoom the camera
- Scroll: Zoom in/out
- Drag: Pan the view

### 2. Dual Graph System

Simultaneously visualise multiple graph types with unified physics:

- **Knowledge Graph**: Logseq data from GitHub (blue theme, bit 30 node flags)
- **Agent Graph**: AI Multi Agent telemetry from Claude Flow MCP (green theme, bit 31 node flags)  
- **Binary Protocol**: Real-time position/velocity streaming at 60 FPS
- **GraphServiceActor**: Central coordinator for both graph types

### 3. Interactive Elements

**Node Interaction:**
- Click: Select node and show metadata
- Double-click: Centre camera on node
- Hover: Show quick info tooltip

**Edge Interaction:**
- Click edge: Show relationship details
- Hover: Highlight connected nodes

### 4. XR/VR Mode (Optional)

For Quest 3 or compatible devices:

1. Connect your VR headset
2. Open VisionFlow in a WebXR browser
3. Click the VR icon in the interface
4. Use hand tracking or controllers to interact

## Common Workflows

### Exploring a Knowledge Graph

1. **Load Data**: Connect Logseq or upload graph data
2. **Adjust Physics**: Tune spring/repulsion forces for optimal layout
3. **Navigate**: Use mouse controls to explore the graph
4. **Search**: Use Command Palette (`Ctrl+K`) to find specific nodes
5. **Filter**: Toggle graph types or metadata filters

### Monitoring AI Agents

1. **Connect Agent System**: Configure Claude Flow MCP integration (TCP port 9500)
2. **View Agent Graph**: Agent nodes automatically appear with bit 31 flags
3. **Monitor Health**: Node colors indicate agent status via ClaudeFlowActor
4. **Track Tasks**: See real-time task processing through binary WebSocket updates
5. **Debug Issues**: Click problematic agents for detailed logs via API

### Performance Tuning

1. **Check FPS**: Enable performance overlay in Debug settings
2. **Adjust Quality**: Lower render quality if FPS < 30
3. **GPU Usage**: Monitor GPU utilization in System panel
4. **Network Load**: Check WebSocket message rates

## Troubleshooting

### Common Issues

**🔧 Performance Issues**
- Lower render quality in settings
- Disable advanced visual effects
- Reduce node count or use filtering
- Check GPU memory usage

**🔧 Connection Issues**
- Verify Docker containers are running: `docker-compose ps`
- Check backend health: `curl http://localhost:8080/health`
- Restart services: `docker-compose restart`

**🔧 GPU Not Detected**
- Install NVIDIA Container Toolkit
- Verify CUDA installation: `nvidia-smi`
- Check Docker GPU support: `docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi`

**🔧 XR/VR Issues**
- Use Chrome or Firefox with WebXR support
- Enable "Enable VR Developer Features" in chrome://flags
- Ensure VR headset is properly connected
- Check browser permissions for device access

### Debug Mode

Enable detailed logging:

```bash
# Backend logs
docker-compose logs -f server

# Frontend logs
# Open browser DevTools → Console tab

# Enable debug mode
curl -X POST http://localhost:8080/api/debug/enable
```

### Getting Help

**Documentation:**
- [Architecture Guide](./architecture/system-overview.md)
- [API Reference](./api/index.md)
- [Configuration Reference](./configuration/index.md)

**Community:**
- GitHub Issues: Report bugs and feature requests
- Discord: Community discussion and support
- Documentation: Comprehensive guides and tutorials

## Next Steps

### Beginner
1. ✅ Complete this getting started guide
2. 📖 Read [Basic Usage Guide](./guides/quick-start.md)
3. 🎯 Try [Tutorial: Your First Graph](./guides/quick-start.md)

### Intermediate  
1. 🔧 Explore [Configuration Options](./configuration/index.md)
2. 🤖 Set up [Agent Integration](./features/agent-orchestration.md)
3. 🥽 Try [WebXR Mode](./client/xr-integration.md)

### Advanced
1. 🏗️ Study [Architecture Documentation](./architecture/index.md)
2. 💻 Set up [Development Environment](./development/setup.md)
3. 🚀 Deploy to [Production](./deployment/index.md)

## Quick Reference

### Keyboard Shortcuts
- `Ctrl+K`: Open command palette
- `Space`: Pause/resume physics
- `R`: Reset camera view
- `T`: Toggle graph types
- `F`: Focus on selected node
- `Ctrl+/`: Toggle help overlay

### API Endpoints
- Health check: `GET /health`
- Graph data: `GET /api/graph`
- Settings: `GET/POST /api/settings`
- System info: `GET /api/system/info`

### Docker Commands
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart service
docker-compose restart server

# Stop all
docker-compose down
```

---

🎉 **Welcome to VisionFlow!** You're now ready to explore real-time 3D visualisation with AI agent orchestration.

For questions or issues, check our [Troubleshooting Guide](./guides/troubleshooting.md) or reach out via GitHub Issues.

## Related Topics

- [Configuration Guide](getting-started/configuration.md)
- [Getting Started with VisionFlow](getting-started/index.md)
- [Installation Guide](getting-started/installation.md)
- [Quick Start Guide](getting-started/quickstart.md)
