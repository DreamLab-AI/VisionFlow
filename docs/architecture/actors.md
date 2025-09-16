# Actor System Architecture

*This file redirects to the actor model documentation.*

See [Actor Model Architecture](actor-model.md) for detailed actor system documentation.

## Actor System Overview

VisionFlow uses the Actix actor model for:
- Fault-tolerant message passing
- Concurrent state management
- Supervisor hierarchies
- Automatic failure recovery

## Quick Links

- [Actor Model Details](actor-model.md) - Complete actor system documentation
- [Actor Refactoring](actor-refactoring.md) - Recent actor system improvements
- [ClaudeFlow Actor](claude-flow-actor.md) - AI agent integration actor
- [System Architecture](index.md) - Overall system design

## Core Actors

- **GraphServiceActor** - Graph data management
- **GPUComputeActor** - Physics computation
- **ClientManagerActor** - WebSocket connections
- **SettingsActor** - Configuration management
- **EnhancedClaudeFlowActor** - AI agent coordination

---

[← Back to Architecture Documentation](README.md)