---
layout: default
title: Operations
nav_order: 37
parent: Guides
has_children: true
description: Operational guides for running VisionFlow in production
---

# Operations Guides

Runbooks and operational procedures for production environments.

## Available Guides

| Guide | Description |
|-------|-------------|
| [Pipeline Operator Runbook](pipeline-operator-runbook.md) | Complete operations playbook |

## Quick Reference

### Health Checks

```bash
# System health
curl http://localhost:9090/health

# Service status
sudo supervisorctl status
```

### Log Locations

| Service | Log |
|---------|-----|
| Management API | `/var/log/management-api.log` |
| VisionFlow | `/var/log/visionflow.log` |
| Supervisor | `/var/log/supervisord.log` |

### Incident Response

1. Check service health
2. Review recent logs
3. Identify affected components
4. Execute runbook procedures
5. Document incident

## Related Documentation

- [Infrastructure](/guides/infrastructure/) - System setup
- [Troubleshooting](/guides/troubleshooting/) - Common issues
- [Telemetry Logging](/guides/telemetry-logging/) - Monitoring
