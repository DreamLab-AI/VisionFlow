# VisionFlow Documentation

## Overview

VisionFlow is a sophisticated real-time 3D visualisation platform that combines AI agent orchestration, GPU-accelerated physics, and cutting-edge XR capabilities. Built with a decoupled actor-based architecture using Rust backend and React/TypeScript frontend, it provides a powerful environment for visualizing and interacting with complex knowledge graphs and AI agents.

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
- [Docker MCP Integration](deployment/docker-mcp-integration.md) - Containerized MCP deployment
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
- **GPU-Accelerated Physics**: CUDA implementation optimized for NVIDIA hardware with dual-graph support
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

---

*VisionFlow - Real-time 3D visualisation platform with AI agent orchestration*