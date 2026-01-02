---
layout: default
title: Infrastructure
nav_order: 35
parent: Guides
has_children: true
description: Multi-agent Docker system setup and management
---

# Infrastructure Guides

Infrastructure setup and management for the multi-agent Docker environment.

## Available Guides

| Guide | Description |
|-------|-------------|
| [Architecture](architecture.md) | Multi-agent Docker system design |
| [Docker Environment](docker-environment.md) | Container setup and management |
| [Tools](tools.md) | Available MCP tools and integrations |
| [Port Configuration](port-configuration.md) | Network and service ports |
| [Troubleshooting](troubleshooting.md) | Infrastructure-specific issues |
| [Goalie Integration](goalie-integration.md) | Quality gates and automated testing |

## Quick Reference

### Service Ports

| Port | Service | Access |
|------|---------|--------|
| 22 | SSH | Public |
| 5901 | VNC | Public |
| 8080 | code-server | Public |
| 9090 | Management API | Public |
| 9600 | Z.AI | Internal |

### Common Commands

```bash
# Service status
sudo supervisorctl status

# Container diagnostics
docker stats turbo-flow-unified

# Logs
sudo supervisorctl tail -f management-api
```

## See Also

- [Main Documentation](../../README.md)
- [Deployment Guide](../deployment.md)
- [Docker Compose Guide](../docker-compose-guide.md)
- [Operations Runbook](../operations/pipeline-operator-runbook.md)

---

*Last Updated: 2025-12-19*
