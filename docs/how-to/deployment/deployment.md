---
title: Deployment Guide
description: > [Guides](./index.md) > Deployment
category: how-to
tags:
  - tutorial
  - deployment
  - api
  - api
  - docker
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Deployment Guide

 > [Guides](./index.md) > Deployment

This comprehensive guide covers deploying VisionFlow in various environments, from local development to production-ready deployments. VisionFlow integrates multiple external services including RAGFlow, Whisper, Kokoro TTS, and Vircadia XR.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Staging Deployment](#staging-deployment)
4. [Production Deployment](#production-deployment)
5. [External Services](#external-services)
6. [Troubleshooting](#troubleshooting)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Prerequisites

### System Requirements

**Development Environment**:
- CPU: 4+ cores, 2.5GHz+
- RAM: 8GB minimum, 16GB recommended
- Storage: 20GB available SSD space
- OS: Ubuntu 20.04+, macOS 12+, or Windows 10+ with WSL2
- Network: Stable broadband connection

**Production Environment**:
- CPU: 8+ cores, 3.5GHz+
- RAM: 16GB minimum, 32GB recommended
- Storage: 100GB+ SSD with 3000+ IOPS
- GPU: NVIDIA GPU with CUDA 11.8+ (optional but recommended)
- Network: 1Gbps minimum bandwidth

### Required Software

```bash
# Check Docker version (required: 20.10+)
docker --version

# Check Docker Compose version (required: 2.0+)
docker compose version

# Install Docker if needed
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose plugin
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Install NVIDIA Container Toolkit (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION-ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Network Configuration

VisionFlow requires the `docker-ragflow` external network for inter-service communication:

```bash
# Create external network
docker network create docker-ragflow

# Verify network creation
docker network ls | grep docker-ragflow
```

## Local Development

### Quick Start

The fastest way to get VisionFlow running locally:

```bash
# Clone repository
git clone https://github.com/your-org/VisionFlow.git
cd VisionFlow

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env

# Start development environment
docker compose --profile dev up -d

# Verify deployment
docker compose ps
```

### Development Configuration

Create a `.env` file with development-specific settings:

```bash
# Core Settings
ENVIRONMENT=development
DEBUG-MODE=true
RUST-LOG=debug
NODE-ENV=development

# Networking
HOST-PORT=3001
MCP-TCP-PORT=9500
CLAUDE-FLOW-HOST=multi-agent-container
BOTS-ORCHESTRATOR-URL=ws://multi-agent-container:3002

# Security (development only - change in production)
JWT-SECRET=dev-secret-change-in-production
POSTGRES-PASSWORD=dev-password
AUTH-REQUIRED=false

# Development Features
HOT-RELOAD=true
ENABLE-PROFILING=true
VITE-DEV-SERVER-PORT=5173
VITE-HMR-PORT=24678

# GPU Configuration (optional)
NVIDIA-VISIBLE-DEVICES=0
CUDA-ARCH=86  # 86 for RTX 30-series, 89 for RTX 40-series
```

### Docker Compose Commands

```bash
# Start all development services
docker compose --profile dev up -d

# Start specific service
docker compose up -d webxr

# View logs
docker compose logs -f webxr

# Restart service
docker compose restart webxr

# Stop all services
docker compose down

# Stop and remove volumes (⚠️ destroys data)
docker compose down -v

# Rebuild after code changes
docker compose --profile dev up -d --build

# Scale workers (if applicable)
docker compose up -d --scale worker=3
```

### Starting Services Individually

For debugging, you may want to start services separately:

```bash
# Start multi-agent container first
docker compose up -d multi-agent

# Start main application
docker compose up -d webxr

# Start Cloudflare tunnel (optional)
docker compose up -d cloudflared

# Check service health
docker compose ps
docker compose exec webxr curl -f http://localhost:4000/health
```

### Development Workflow

VisionFlow uses hot module replacement (HMR) for rapid development:

1. **Code Changes**: Edit files in `client/src/` or `src/`
2. **Auto-Rebuild**: Changes trigger automatic rebuilds
3. **Browser Refresh**: Frontend updates automatically via HMR
4. **Backend Restart**: Rust changes require container restart

```bash
# Watch for Rust changes
docker compose exec webxr /app/scripts/check-rust-rebuild.sh

# Manual rebuild if needed
docker compose exec webxr /app/scripts/dev-rebuild-rust.sh
```

## Staging Deployment

### Staging Environment Setup

Staging replicates production with isolated data and reduced resources:

```bash
# Create staging .env
cat > .env.staging << 'EOF'
# Staging Configuration
ENVIRONMENT=staging
DEBUG-MODE=false
RUST-LOG=info

# Security (generate secure secrets)
JWT-SECRET=$(openssl rand -hex 32)
POSTGRES-PASSWORD=$(openssl rand -hex 24)

# Network
DOMAIN=staging.yourdomain.com
HOST-PORT=3001

# Performance
MEMORY-LIMIT=8g
CPU-LIMIT=4.0
ENABLE-GPU=true
MAX-AGENTS=10

# External Services
RAGFLOW-API-BASE-URL=http://ragflow-server:9380
RAGFLOW-API-KEY=your-staging-api-key
PERPLEXITY-API-KEY=your-staging-perplexity-key
EOF

# Start staging deployment
docker compose --profile production --env-file .env.staging up -d
```

### Staging Docker Compose

Use production profile with staging environment variables:

```bash
# Deploy with staging configuration
docker compose -f docker-compose.yml --profile production --env-file .env.staging up -d

# Verify all services
docker compose ps
docker compose exec webxr-prod curl http://localhost:4000/health

# View staged application
curl https://staging.yourdomain.com
```

### Nginx Configuration for Staging

Create `nginx.staging.conf`:

```nginx
worker-processes auto;
worker-rlimit-nofile 65535;

events {
    worker-connections 4096;
    use epoll;
    multi-accept on;
}

http {
    upstream backend {
        least-conn;
        server localhost:4000;
        keepalive 32;
    }

    server {
        listen 80;
        server-name staging.yourdomain.com;
        return 301 https://$server-name$request-uri;
    }

    server {
        listen 443 ssl http2;
        server-name staging.yourdomain.com;

        ssl-certificate /etc/nginx/ssl/staging-cert.pem;
        ssl-certificate-key /etc/nginx/ssl/staging-key.pem;
        ssl-protocols TLSv1.2 TLSv1.3;
        ssl-ciphers HIGH:!aNULL:!MD5;
        ssl-prefer-server-ciphers on;

        # Security headers
        add-header Strict-Transport-Security "max-age=31536000" always;
        add-header X-Content-Type-Options "nosniff" always;
        add-header X-Frame-Options "SAMEORIGIN" always;

        location / {
            proxy-pass http://backend;
            proxy-set-header Host $host;
            proxy-set-header X-Real-IP $remote-addr;
            proxy-set-header X-Forwarded-For $proxy-add-x-forwarded-for;
            proxy-set-header X-Forwarded-Proto $scheme;
            proxy-http-version 1.1;
            proxy-set-header Upgrade $http-upgrade;
            proxy-set-header Connection "upgrade";
        }

        # Rate limiting
        limit-req zone=api burst=20 nodelay;
    }

    # Rate limiting zones
    limit-req-zone $binary-remote-addr zone=api:10m rate=100r/m;
}
```

## Production Deployment

### Production Environment Configuration

Generate secure secrets and configure production environment:

```bash
# Generate secure credentials
JWT-SECRET=$(openssl rand -hex 32)
POSTGRES-PASSWORD=$(openssl rand -hex 24)
WS-AUTH-TOKEN=$(openssl rand -hex 32)
API-KEY=$(openssl rand -hex 32)

# Create production .env
cat > .env.production << EOF
# Production Configuration
ENVIRONMENT=production
DEBUG-MODE=false
RUST-LOG=warn
NODE-ENV=production

# Security
JWT-SECRET=$JWT-SECRET
POSTGRES-PASSWORD=$POSTGRES-PASSWORD
WS-AUTH-TOKEN=$WS-AUTH-TOKEN
API-KEY=$API-KEY

# Authentication
WS-AUTH-ENABLED=true
RATE-LIMIT-ENABLED=true
CORS-ENABLED=true
CORS-ALLOWED-ORIGINS=https://yourdomain.com

# Network
DOMAIN=yourdomain.com
HOST-PORT=4000
CLOUDFLARE-TUNNEL-TOKEN=your-cloudflare-tunnel-token

# Performance
ENABLE-GPU=true
MEMORY-LIMIT=32g
CPU-LIMIT=16.0
MAX-AGENTS=50

# External Services (see External Services section)
RAGFLOW-API-BASE-URL=http://ragflow-server:9380
RAGFLOW-API-KEY=your-production-api-key
PERPLEXITY-API-KEY=your-production-perplexity-key
OPENAI-API-KEY=your-production-openai-key

# Resource Limits
MAX-REQUEST-SIZE=10485760
MAX-MESSAGE-SIZE=1048576
WS-MAX-CONNECTIONS=100
TCP-MAX-CONNECTIONS=50

# Monitoring
ENABLE-METRICS=true
HEALTH-CHECK-ENABLED=true
SECURITY-AUDIT-LOG=true
PERFORMANCE-MONITORING=true
EOF

# Secure the file
chmod 600 .env.production
```

### Production Deployment Commands

```bash
# Deploy production configuration
docker compose --profile production --env-file .env.production up -d

# Verify deployment
docker compose ps
docker compose exec webxr-prod curl -f http://localhost:4000/health

# Check logs
docker compose logs -f webxr-prod

# Monitor resource usage
docker stats webxr-prod-container
```

### Production with Cloudflare Tunnel

VisionFlow uses Cloudflare Tunnel for secure public access:

```bash
# 1. Create Cloudflare Tunnel
cloudflared tunnel create visionflow

# 2. Configure tunnel in config.yml
cat > config.yml << 'EOF'
tunnel: visionflow
credentials-file: /etc/cloudflared/credentials.json

ingress:
  - hostname: yourdomain.com
    service: http://webxr-prod:4000
  - hostname: api.yourdomain.com
    service: http://webxr-prod:4000
  - service: http-status:404
EOF

# 3. Start with tunnel
docker compose --profile production up -d cloudflared

# 4. Verify tunnel
docker compose logs -f cloudflared
```

### Production Nginx Configuration

For non-Cloudflare deployments, use Nginx as reverse proxy:

```nginx
worker-processes auto;
worker-rlimit-nofile 65535;

events {
    worker-connections 8192;
    use epoll;
    multi-accept on;
}

http {
    # Rate limiting
    limit-req-zone $binary-remote-addr zone=api:10m rate=100r/m;
    limit-req-zone $binary-remote-addr zone=websocket:10m rate=50r/m;
    limit-conn-zone $binary-remote-addr zone=addr:10m;

    # Upstream
    upstream visionflow {
        least-conn;
        server webxr-prod:4000 max-fails=3 fail-timeout=30s;
        keepalive 64;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server-name yourdomain.com;
        return 301 https://$server-name$request-uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server-name yourdomain.com;

        # SSL configuration
        ssl-certificate /etc/nginx/ssl/cert.pem;
        ssl-certificate-key /etc/nginx/ssl/key.pem;
        ssl-protocols TLSv1.2 TLSv1.3;
        ssl-ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl-prefer-server-ciphers on;
        ssl-session-cache shared:SSL:10m;
        ssl-session-timeout 10m;

        # Security headers
        add-header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add-header X-Content-Type-Options "nosniff" always;
        add-header X-Frame-Options "SAMEORIGIN" always;
        add-header X-XSS-Protection "1; mode=block" always;
        add-header Referrer-Policy "strict-origin-when-cross-origin" always;

        # Main application
        location / {
            proxy-pass http://visionflow;
            proxy-set-header Host $host;
            proxy-set-header X-Real-IP $remote-addr;
            proxy-set-header X-Forwarded-For $proxy-add-x-forwarded-for;
            proxy-set-header X-Forwarded-Proto $scheme;
            proxy-http-version 1.1;
            proxy-set-header Connection "";

            # Rate limiting
            limit-req zone=api burst=20 nodelay;
            limit-conn addr 10;
        }

        # WebSocket endpoints
        location /ws {
            proxy-pass http://visionflow;
            proxy-http-version 1.1;
            proxy-set-header Upgrade $http-upgrade;
            proxy-set-header Connection "upgrade";
            proxy-set-header Host $host;
            proxy-set-header X-Real-IP $remote-addr;
            proxy-set-header X-Forwarded-For $proxy-add-x-forwarded-for;
            proxy-read-timeout 86400;

            limit-req zone=websocket burst=10 nodelay;
        }

        # Health check (internal only)
        location /health {
            proxy-pass http://visionflow;
            access-log off;
            allow 127.0.0.1;
            deny all;
        }
    }
}
```

### Security Hardening

```bash
# 1. Container security
cat >> docker-compose.prod.yml << 'EOF'
services:
  webxr-prod:
    security-opt:
      - no-new-privileges:true
    read-only: true
    tmpfs:
      - /tmp
    cap-drop:
      - ALL
    cap-add:
      - NET-BIND-SERVICE
EOF

# 2. Firewall configuration
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# 3. Restrict Docker daemon
echo '{"hosts": ["unix:///var/run/docker.sock"], "tls": true}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker

# 4. SSL/TLS with Let's Encrypt
sudo certbot --nginx -d yourdomain.com
```

## External Services

VisionFlow integrates with multiple external services that must be deployed separately.

### RAGFlow

RAGFlow provides knowledge retrieval and RAG capabilities.

**Deployment**:
```bash
# Clone RAGFlow
git clone https://github.com/infiniflow/ragflow.git
cd ragflow

# Configure environment
cp .env.example .env
nano .env  # Set API keys and configuration

# Start RAGFlow
docker compose up -d

# Verify deployment
curl http://localhost:9380/api/health
```

**VisionFlow Configuration**:
```bash
# In VisionFlow .env
RAGFLOW-API-BASE-URL=http://ragflow-server:9380
RAGFLOW-API-KEY=your-ragflow-api-key
RAGFLOW-AGENT-ID=your-agent-id
```

**Network Integration**:
```bash
# Connect RAGFlow to VisionFlow network
docker network connect docker-ragflow ragflow-server

# Verify connectivity
docker exec visionflow-container curl http://ragflow-server:9380/api/health
```

### Whisper Speech-to-Text

Whisper provides speech recognition for voice commands.

**Deployment**:
```bash
# Using OpenAI Whisper container
docker run -d \
  --name whisper-stt \
  --network docker-ragflow \
  -p 8080:8080 \
  -e MODEL-SIZE=base \
  --gpus all \
  onerahmet/openai-whisper-asr-webservice:latest

# Verify deployment
curl http://localhost:8080/health
```

**VisionFlow Configuration**:
```bash
# Whisper is accessed via fixed IP in docker-ragflow network
# Default: 172.18.0.5:8080
# No additional configuration needed if using default network
```

**Custom Configuration**:
```bash
# For custom Whisper endpoint, update src/config/mod.rs
# Or set environment variable
WHISPER-STT-ENDPOINT=http://172.18.0.5:8080
```

### Kokoro Text-to-Speech

Kokoro provides neural text-to-speech for voice responses.

**Deployment**:
```bash
# Using Kokoro TTS container
docker run -d \
  --name kokoro-tts \
  --network docker-ragflow \
  -p 5000:5000 \
  -e PORT=5000 \
  --gpus all \
  hexgrad/kokoro-tts:latest

# Verify deployment
curl http://localhost:5000/health
```

**Network Configuration**:
```bash
# Kokoro should be accessible at 172.18.0.9:5000 on docker-ragflow network
# Verify connectivity
docker exec visionflow-container curl http://172.18.0.9:5000/health

# Fix network if needed
docker network connect docker-ragflow kokoro-tts
```

**VisionFlow Configuration**:
```bash
# Kokoro endpoint is hardcoded to 172.18.0.9:5000
# Ensure Kokoro container has this IP on docker-ragflow network

# Verify in settings.yaml
cat data/settings.yaml | grep -A 5 voice
```

### Vircadia XR Server

Vircadia provides multi-user XR/VR capabilities.

**Quick Start**:
```bash
# Navigate to Vircadia directory
cd vircadia/server/vircadia-world/server/service

# Create network
docker network create vircadia-network

# Start PostgreSQL
docker run -d \
  --name vircadia-world-postgres \
  --network vircadia-network \
  -e POSTGRES-DB=vircadia-world \
  -e POSTGRES-USER=postgres \
  -e POSTGRES-PASSWORD=vircadia-password \
  -p 127.0.0.1:5432:5432 \
  -v vircadia-world-server-postgres-data:/var/lib/postgresql/data \
  postgres:17.5-alpine3.21

# Start PGWeb UI
docker run -d \
  --name vircadia-world-pgweb \
  --network vircadia-network \
  -p 127.0.0.1:5437:8081 \
  -e "PGWEB-DATABASE-URL=postgres://postgres:vircadia-password@vircadia-world-postgres:5432/vircadia-world?sslmode=disable" \
  sosedoff/pgweb:0.16.2

# Verify services
docker ps | grep vircadia
```

**Production Configuration**:
```bash
# Environment configuration (.env)
VRCA-SERVER-SERVICE-POSTGRES-HOST-CONTAINER-BIND-EXTERNAL=127.0.0.1
VRCA-SERVER-SERVICE-POSTGRES-PORT-CONTAINER-BIND-EXTERNAL=5432
VRCA-SERVER-SERVICE-POSTGRES-DATABASE=vircadia-world
VRCA-SERVER-SERVICE-POSTGRES-SUPER-USER-PASSWORD=secure-password-here

VRCA-SERVER-SERVICE-WORLD-API-MANAGER-PORT-CONTAINER-BIND-EXTERNAL=3020
VRCA-SERVER-SERVICE-WORLD-STATE-MANAGER-PORT-CONTAINER-BIND-EXTERNAL=3021
```

**Service Endpoints**:
- PostgreSQL: `127.0.0.1:5432` (localhost only)
- PGWeb UI: `http://localhost:5437` (localhost only)
- World API: `0.0.0.0:3020` (public for Quest 3 access)
- State Manager: `0.0.0.0:3021` (public for XR sync)

**Integration with VisionFlow**:
```bash
# VisionFlow accesses Vircadia via API endpoints
# Configure in data/settings.yaml
cat >> data/settings.yaml << 'EOF'
xr:
  enabled: true
  vircadia-api-url: http://localhost:3020
  vircadia-state-url: http://localhost:3021
EOF
```

### Multi-Agent Container

The multi-agent container provides MCP tools and agent orchestration.

**Deployment**:
```bash
# Start multi-agent container
docker compose up -d multi-agent

# Start GUI tools (Blender, QGIS, PBR)
docker compose up -d gui-tools-service

# Verify MCP tools
docker exec multi-agent-container ./mcp-helper.sh list-tools
```

**Configuration**:
```bash
# Environment variables
MCP-TCP-PORT=9500
MCP-BRIDGE-PORT=3002
BLENDER-HOST=gui-tools-service
BLENDER-PORT=9876
QGIS-HOST=gui-tools-service
QGIS-PORT=9877
PBR-HOST=gui-tools-service
PBR-PORT=9878
```

**Service Endpoints**:
- Claude Flow UI: `http://localhost:3000`
- WebSocket Bridge: `ws://localhost:3002`
- MCP TCP Server: `tcp://localhost:9500`
- Blender MCP: `tcp://localhost:9876`
- QGIS MCP: `tcp://localhost:9877`
- PBR Generator: `tcp://localhost:9878`

## Troubleshooting

### Local Development Issues

**Port Conflicts**:
```bash
# Check port usage
sudo netstat -tulpn | grep -E '3001|4000|9500'

# Change ports in .env
HOST-PORT=3003
MCP-TCP-PORT=9501

# Restart services
docker compose down
docker compose --profile dev up -d
```

**Container Won't Start**:
```bash
# Check logs
docker compose logs -f webxr

# Check for errors
docker compose ps

# Remove and recreate
docker compose down
docker compose --profile dev up -d --force-recreate
```

**GPU Not Detected**:
```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

# Check NVIDIA runtime
docker info | grep -i nvidia

# Restart Docker with GPU support
sudo systemctl restart docker
docker compose --profile dev up -d --force-recreate
```

### Staging Issues

**Service Communication Failures**:
```bash
# Check network connectivity
docker network inspect docker-ragflow

# Verify services are on same network
docker inspect webxr-prod | grep NetworkMode

# Reconnect to network
docker network connect docker-ragflow webxr-prod
```

**SSL Certificate Issues**:
```bash
# Regenerate self-signed certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -subj "/CN=staging.yourdomain.com"

# Update Nginx configuration
docker compose restart nginx
```

### Production Issues

**High Memory Usage**:
```bash
# Check memory metrics
docker stats

# Increase memory limits
# Edit .env.production
MEMORY-LIMIT=64g

# Apply changes
docker compose --profile production up -d --force-recreate
```

**Database Connection Failures**:
```bash
# Check database connectivity
docker compose exec webxr-prod curl -f http://localhost:4000/health

# Verify credentials
docker compose exec webxr-prod env | grep POSTGRES

# Restart database
docker compose restart postgres
```

**Rate Limiting Triggered**:
```bash
# Check current limits
curl -I https://yourdomain.com/api/health

# Adjust rate limits in .env
RATE-LIMIT-MAX-REQUESTS=200
RATE-LIMIT-WINDOW-MS=60000

# Restart service
docker compose --profile production up -d --force-recreate
```

### External Service Issues

**RAGFlow Connection Failed**:
```bash
# Verify RAGFlow is running
docker ps | grep ragflow

# Check connectivity
docker exec visionflow-container curl http://ragflow-server:9380/api/health

# Verify network
docker network inspect docker-ragflow | grep ragflow-server

# Reconnect if needed
docker network connect docker-ragflow ragflow-server
```

**Whisper STT Not Responding**:
```bash
# Check Whisper container
docker ps | grep whisper

# Test endpoint
curl -X POST http://172.18.0.5:8080/asr \
  -F "audio-file=@test.wav"

# Check logs
docker logs whisper-stt

# Restart if needed
docker restart whisper-stt
```

**Kokoro TTS Network Issues**:
```bash
# Verify Kokoro IP address
docker inspect kokoro-tts | grep IPAddress

# Should be 172.18.0.9 on docker-ragflow network
# If not, reconnect
docker network disconnect docker-ragflow kokoro-tts
docker network connect docker-ragflow kokoro-tts --ip 172.18.0.9

# Verify connectivity
docker exec visionflow-container curl http://172.18.0.9:5000/health
```

**Vircadia PostgreSQL Won't Start**:
```bash
# Check logs
docker logs vircadia-world-postgres

# Check volume permissions
docker volume inspect vircadia-world-server-postgres-data

# Reset volume (⚠️ destroys data)
docker stop vircadia-world-postgres
docker rm vircadia-world-postgres
docker volume rm vircadia-world-server-postgres-data
# Re-run start script
```

## Monitoring and Maintenance

### Health Checks

```bash
# VisionFlow health
curl http://localhost:3030/health

# Multi-agent container health
curl http://localhost:3002/health

# RAGFlow health
curl http://localhost:9380/api/health

# Whisper health
curl http://172.18.0.5:8080/health

# Kokoro health
curl http://172.18.0.9:5000/health

# Vircadia health
curl http://localhost:3020/health
```

### Automated Backups

```bash
#!/bin/bash
# backup.sh
BACKUP-DIR="/backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p $BACKUP-DIR

# Backup VisionFlow data
docker exec visionflow-container tar czf - /app/data | cat > $BACKUP-DIR/visionflow-data.tar.gz

# Backup Vircadia database
docker exec vircadia-world-postgres pg-dump -U postgres vircadia-world | gzip > $BACKUP-DIR/vircadia-db.sql.gz

# Backup configurations
cp .env* $BACKUP-DIR/
cp docker-compose*.yml $BACKUP-DIR/

# Clean old backups (keep last 7 days)
find /backups -type d -mtime +7 -exec rm -rf {} \;

echo "Backup complete: $BACKUP-DIR"
```

**Schedule Backups**:
```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh >> /var/log/visionflow-backup.log 2>&1
```

### Log Management

```bash
# Configure log rotation
cat > /etc/logrotate.d/visionflow << 'EOF'
/var/log/visionflow/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0640 visionflow visionflow
    postrotate
        docker compose kill -s USR1 webxr
    endscript
}
EOF

# Apply rotation
logrotate -f /etc/logrotate.d/visionflow
```

### Performance Monitoring

```bash
# Real-time container stats
docker stats

# Resource usage over time
docker stats --no-stream > /var/log/docker-stats-$(date +%Y%m%d).log

# GPU utilisation
nvidia-smi dmon -s pucvmet -c 100 > /var/log/gpu-stats-$(date +%Y%m%d).log
```

### System Updates

```bash
# Update images
docker compose pull

# Recreate containers with new images
docker compose --profile production up -d --force-recreate

# Prune unused resources
docker system prune -a --volumes

# Verify deployment
docker compose ps
curl -f https://yourdomain.com/health
```

---

## Related Articles

-  - Comprehensive environment variable and settings documentation
- [Configuration Guide](./configuration.md) - Practical configuration scenarios
- [Development Workflow](./development-workflow.md) - Development best practices and workflows
- [Orchestrating Agents](./orchestrating-agents.md) - Multi-agent coordination guide

---

*Last Updated: 2025-10-03*
*Document Version: 3.0.0*
