# VisionFlow Architecture Documentation Index

**Last Updated**: 2025-10-12
**Status**: Reflects Simplified Multi-Agent Integration

## Quick Navigation

### Core Architecture
- [**System Overview**](overview.md) - Complete system architecture with all integrations
- [**Server Architecture**](core/server.md) - Rust actor system and GPU compute pipeline
- [**Multi-Agent Integration**](multi-agent-integration.md) - HTTP API + MCP TCP integration guide

### Component Documentation
- [**GPU Architecture**](gpu/) - CUDA kernels and compute pipeline
- [**Interface Design**](interface.md) - Frontend and WebXR components
- [**Security Model**](security.md) - Authentication, authorization, and data protection

### Integration Guides
- [**Hybrid Docker/MCP**](hybrid_docker_mcp_architecture.md) - Container orchestration patterns
- [**Vircadia Integration**](vircadia-integration-analysis.md) - Metaverse platform integration

## Architecture at a Glance

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    VisionFlow System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │  VisionFlow      │          │  Agentic         │        │
│  │  Container       │◄────────►│  Workstation     │        │
│  │                  │          │                  │        │
│  │  • Rust :4000    │   HTTP   │  • Mgmt API :9090│        │
│  │  • Frontend :5173│   9090   │  • MCP TCP :9500 │        │
│  │  • Nginx :3030   │          │  • Task Isolation│        │
│  └──────────────────┘          └──────────────────┘        │
│           │                              │                   │
│           └──────────────────────────────┘                   │
│              docker_ragflow network                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Integration Pattern

**Task Management Flow (HTTP)**:
```
Web UI → Rust Backend → ManagementApiClient → Management API :9090 → Process Manager → Isolated Tasks
```

**Agent Monitoring Flow (MCP TCP)**:
```
AgentMonitorActor → TcpConnectionActor → MCP TCP :9500 → Agent Status Updates → Graph Visualization
```

## Key Architectural Changes (2025-10-12)

### What Changed
- **Container Naming**: `multi-agent-container` → `agentic-workstation`
- **Task Management**: Docker exec → HTTP Management API
- **Actor Naming**: ClaudeFlowActor → AgentMonitorActor (conceptual)
- **Integration Pattern**: Unified approach → Dual-channel (HTTP + MCP TCP)

### What Was Removed
- DockerHiveMind utility service
- McpSessionBridge
- SessionCorrelationBridge
- Direct Docker exec commands

### What Was Added
- ManagementApiClient (HTTP client)
- Management API Server (Node.js/Fastify)
- Process Manager with task isolation
- System Monitor (GPU/health/providers)

## Documentation Structure

### Core Documentation (Start Here)
1. **overview.md** - System-wide architecture, diagrams, integration flows
2. **core/server.md** - Rust server implementation details
3. **multi-agent-integration.md** - HTTP API integration guide

### Specialized Topics
- **gpu/** - CUDA kernel architecture and GPU compute
- **components/** - Frontend, WebXR, and UI components
- **security.md** - Authentication, authorization, encryption
- **interface.md** - API specifications and protocols

### Integration Guides
- **hybrid_docker_mcp_architecture.md** - Container orchestration
- **vircadia-integration-analysis.md** - Metaverse integration
- **voice-webrtc-migration-plan.md** - Voice system architecture

## Quick Reference

### Port Assignments
| Service | Port | Protocol | Container |
|---------|------|----------|-----------|
| Frontend | 5173 | HTTP | visionflow_container |
| Nginx | 3001 | HTTP/WS | visionflow_container |
| Rust Backend | 4000 | HTTP | visionflow_container |
| Management API | 9090 | HTTP | agentic-workstation |
| MCP TCP | 9500 | TCP | agentic-workstation |

### Network Configuration
- **Network Name**: docker_ragflow (bridge)
- **VisionFlow IP**: 172.18.0.2
- **Agentic Workstation**: 172.18.0.x (dynamic)

### Key Environment Variables
```bash
# VisionFlow Container
MCP_HOST=agentic-workstation
MCP_TCP_PORT=9500
MANAGEMENT_API_URL=http://agentic-workstation:9090
MANAGEMENT_API_KEY=<secret>

# Agentic Workstation Container
MANAGEMENT_API_PORT=9090
MANAGEMENT_API_HOST=0.0.0.0
MANAGEMENT_API_KEY=<secret>
```

## Architecture Principles

### Separation of Concerns
- **Task Management**: HTTP REST API (Management API)
- **Agent Monitoring**: MCP TCP (read-only polling)
- **Visualization**: WebSocket binary protocol
- **Compute**: GPU actors with CUDA kernels

### Process Isolation
- Each task runs in dedicated directory: `/workspace/tasks/{taskId}`
- Separate SQLite databases per task
- Independent log files: `/logs/tasks/{taskId}.log`
- Clean process lifecycle management

### Security by Design
- Bearer token authentication on Management API
- Rate limiting (100 req/min)
- Internal network isolation (docker_ragflow)
- No exposed Docker socket
- Structured JSON logging for audit trails

### Observability
- Prometheus metrics at `/metrics`
- Structured JSON logs
- Health check endpoints (`/health`, `/ready`)
- Real-time log streaming (SSE)
- System monitoring (GPU, CPU, memory)

## Navigation Guide

### For Frontend Developers
1. Start with [overview.md](overview.md) - System architecture
2. Read [interface.md](interface.md) - API specifications
3. Review [multi-agent-integration.md](multi-agent-integration.md) - Integration patterns

### For Backend Developers
1. Start with [core/server.md](core/server.md) - Actor system
2. Read [multi-agent-integration.md](multi-agent-integration.md) - HTTP client implementation
3. Review [security.md](security.md) - Security patterns

### For DevOps Engineers
1. Start with [overview.md](overview.md) - Container architecture
2. Read [hybrid_docker_mcp_architecture.md](hybrid_docker_mcp_architecture.md) - Docker setup
3. Review network configuration and environment variables

### For GPU/ML Engineers
1. Start with [gpu/](gpu/) - CUDA kernel documentation
2. Read [core/server.md](core/server.md) - GPU actor system
3. Review compute pipeline and kernel specifications

## Related Documentation

### External References
- [Management API README](/multi-agent-docker/management-api/README.md)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Actix Actor System](https://actix.rs/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

### Legacy Documentation
- [Archive](../archived/) - Historical documentation
- [Process Artifacts](../_archive/_process_artifacts/) - Development history

## Version History

### 2025-10-12 - Simplified Multi-Agent Integration
- Introduced HTTP Management API pattern
- Removed Docker exec complexity
- Renamed container to agentic-workstation
- Separated task management from monitoring
- Added comprehensive integration guide

### 2025-09-27 - Transitional Architecture
- Documented actor system refactoring
- GPU pipeline validation
- Voice system integration

### 2025-09-23 - Initial Architecture
- Established core documentation
- System overview and component diagrams
- Initial integration patterns

## Contributing to Documentation

When updating architecture documentation:
1. Update diagrams to reflect code changes
2. Keep overview.md synchronized with detailed docs
3. Document breaking changes prominently
4. Include migration guides for architectural shifts
5. Update this index when adding new documents

## Support

For questions about the architecture:
- Review existing documentation first
- Check sequence diagrams for integration flows
- Examine code examples in integration guides
- Refer to Management API README for API details

---

**Maintained by**: VisionFlow Architecture Team
**Next Review**: 2025-11-01
