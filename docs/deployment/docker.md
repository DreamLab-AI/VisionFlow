# Docker Deployment Guide

*[Deployment](../index.md)*

## Overview

VisionFlow is containerised using Docker for consistent deployment across different environments. The system uses Docker Compose to orchestrate multiple services including the main application, Claude Flow, and supporting services.

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ext

# Start production environment
docker-compose up -d

# Start development environment
docker-compose -f docker-compose.dev.yml up

# Access the application
open http://localhost:3001
```

## Related Topics

- [Deployment Guide](../deployment/index.md)
- [Docker Compose Profiles Configuration](../deployment/docker-profiles.md)
- [Docker MCP Integration - Production Deployment Guide](../deployment/docker-mcp-integration.md)
- [Multi-Agent Container Setup](../deployment/multi-agent-setup.md)
