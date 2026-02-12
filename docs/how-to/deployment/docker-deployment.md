---
title: Production Docker Deployment
description: Production deployment guide for VisionFlow covering container orchestration, reverse proxy, TLS, health checks, resource limits, and scaling.
category: how-to
tags:
  - deployment
  - production
  - docker
  - tls
  - nginx
updated-date: 2026-02-12
difficulty-level: advanced
---

# Production Docker Deployment

This guide covers deploying VisionFlow to a production environment with proper TLS termination, reverse proxy configuration, health monitoring, resource constraints, and scaling considerations.

## Prerequisites

- Docker Engine 24+ with Compose V2
- NVIDIA Container Toolkit (for GPU acceleration)
- A domain name with DNS configured
- TLS certificate (via Cloudflare tunnel or Let's Encrypt)
- At least 8 GB RAM and 1 NVIDIA GPU (compute capability 8.6+)

## Production Profile

VisionFlow uses Docker Compose profiles to separate development from production. The production service is defined in `docker-compose.unified.yml` under the `visionflow-production` service.

```bash
# Start production stack
docker compose -f docker-compose.unified.yml --profile prod up -d

# With voice services
docker compose -f docker-compose.unified.yml -f docker-compose.voice.yml --profile prod up -d
```

Key differences from development:
- `Dockerfile.production` is used (optimized release build, `opt-level=3`, LTO enabled)
- No source code volume mounts -- all code is baked into the image
- No Docker socket mount
- `RUST_LOG=warn` (reduced logging)
- `NODE_ENV=production`
- Resource limits enforced (see below)

## Reverse Proxy Configuration

### Option A: Cloudflare Tunnel (Recommended)

VisionFlow ships with a Cloudflare tunnel sidecar container. Set `CLOUDFLARE_TUNNEL_TOKEN` in your `.env` and the `cloudflared` service handles TLS termination, DDoS protection, and DNS routing automatically.

```bash
# .env
CLOUDFLARE_TUNNEL_TOKEN=your-tunnel-token-here
```

No additional Nginx or Caddy configuration is needed -- Cloudflare connects directly to the internal Nginx on port 3001.

### Option B: Nginx with Let's Encrypt

If you are not using Cloudflare, place an external Nginx reverse proxy in front of VisionFlow:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:3001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /wss {
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 600m;
    }
}
```

### Option C: Caddy (Auto-TLS)

```
your-domain.com {
    reverse_proxy localhost:3001
}
```

Caddy handles certificate provisioning and renewal automatically.

## TLS Considerations

- The internal Nginx (`nginx.conf`) runs on port 3001 (dev) or 4000 and does NOT terminate TLS itself.
- TLS must be terminated at the edge: Cloudflare tunnel, external Nginx, or Caddy.
- WebSocket connections at `/wss`, `/ws/speech`, and `/ws/mcp` require the reverse proxy to support HTTP Upgrade headers.
- Set `Strict-Transport-Security` headers (already configured in the internal Nginx).

## Health Checks

All services define Docker health checks:

| Service | Endpoint | Interval | Retries | Start Period |
|---------|----------|----------|---------|--------------|
| `visionflow-production` | `GET http://localhost:3001/health` | 30s | 3 | 60s |
| `neo4j` | `wget http://localhost:7474` | 10s | 5 | 30s |
| `livekit` | `wget http://localhost:7880` | 10s | 3 | 5s |
| `turbo-whisper` | `GET http://localhost:8000/health` | 15s | 3 | 30s |
| `kokoro-tts` | `GET http://localhost:8880/health` | 15s | 3 | 20s |

Monitor health externally:

```bash
docker compose -f docker-compose.unified.yml --profile prod ps
docker inspect --format='{{.State.Health.Status}}' visionflow_prod_container
```

## Resource Limits

The production service enforces memory and CPU limits:

```yaml
deploy:
  resources:
    limits:
      memory: 8G
      cpus: '4'
    reservations:
      memory: 2G
      cpus: '1'
```

Adjust these based on your graph size. Neo4j page cache (`512M`) and heap (`1G`) are configured separately via environment variables.

## Log Management

Production uses JSON-file logging driver with rotation:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

For centralized logging, replace with a syslog or fluentd driver, or mount log volumes to a log aggregation pipeline.

## Security Hardening

1. **Change default secrets** -- Update `LIVEKIT_API_SECRET`, `NEO4J_PASSWORD`, and any MCP tokens.
2. **No Docker socket** -- The production service does not mount `/var/run/docker.sock`.
3. **Read-only code** -- All source is baked into the image; no host mounts for code.
4. **Network isolation** -- All services communicate over the `docker_ragflow` bridge network. Only the Nginx port (3001) is exposed to the host.
5. **Content Security Policy** -- Configured in `nginx.conf` with strict defaults.

## Scaling Considerations

- **Horizontal:** VisionFlow is a stateful single-instance application (GPU-bound physics). Scale by deploying separate instances for separate graph workspaces.
- **Neo4j:** For high-availability, consider Neo4j Enterprise with causal clustering. The Community edition (used here) supports a single read-write instance.
- **LiveKit:** LiveKit supports multi-node SFU clusters for larger voice sessions. Update `config/livekit.yaml` with TURN server credentials for NAT traversal.
- **GPU:** The CUDA kernels target a single GPU (`NVIDIA_VISIBLE_DEVICES=0`). Multi-GPU requires partitioning workloads across containers.

## Backup and Recovery

```bash
# Backup Neo4j data volume
docker run --rm -v visionflow-neo4j-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/neo4j-backup-$(date +%Y%m%d).tar.gz /data

# Backup application data
docker run --rm -v visionflow-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/visionflow-data-$(date +%Y%m%d).tar.gz /data
```

## See Also

- [Docker Compose Guide](./docker-compose-guide.md) -- Compose file reference and environment variables
- [Docker Environment Setup](./docker-environment-setup.md) -- Local development environment
- [Infrastructure: Docker Environment](../infrastructure/docker-environment.md) -- Container and network reference
- `nginx.production.conf` -- Production Nginx configuration
- `Dockerfile.production` -- Production multi-stage build
