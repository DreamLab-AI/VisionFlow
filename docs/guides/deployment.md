# Deployment Guide

[← Knowledge Base](../index.md) > [Guides](./index.md) > Deployment

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
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Network Configuration

VisionFlow requires the `docker_ragflow` external network for inter-service communication:

```bash
# Create external network
docker network create docker_ragflow

# Verify network creation
docker network ls | grep docker_ragflow
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
DEBUG_MODE=true
RUST_LOG=debug
NODE_ENV=development

# Networking
HOST_PORT=3001
MCP_TCP_PORT=9500
CLAUDE_FLOW_HOST=multi-agent-container
BOTS_ORCHESTRATOR_URL=ws://multi-agent-container:3002

# Security (development only - change in production)
JWT_SECRET=dev_secret_change_in_production
POSTGRES_PASSWORD=dev_password
AUTH_REQUIRED=false

# Development Features
HOT_RELOAD=true
ENABLE_PROFILING=true
VITE_DEV_SERVER_PORT=5173
VITE_HMR_PORT=24678

# GPU Configuration (optional)
NVIDIA_VISIBLE_DEVICES=0
CUDA_ARCH=86  # 86 for RTX 30-series, 89 for RTX 40-series
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
DEBUG_MODE=false
RUST_LOG=info

# Security (generate secure secrets)
JWT_SECRET=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 24)

# Network
DOMAIN=staging.yourdomain.com
HOST_PORT=3001

# Performance
MEMORY_LIMIT=8g
CPU_LIMIT=4.0
ENABLE_GPU=true
MAX_AGENTS=10

# External Services
RAGFLOW_API_BASE_URL=http://ragflow-server:9380
RAGFLOW_API_KEY=your_staging_api_key
PERPLEXITY_API_KEY=your_staging_perplexity_key
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
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    upstream backend {
        least_conn;
        server localhost:4000;
        keepalive 32;
    }

    server {
        listen 80;
        server_name staging.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name staging.yourdomain.com;

        ssl_certificate /etc/nginx/ssl/staging-cert.pem;
        ssl_certificate_key /etc/nginx/ssl/staging-key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-Frame-Options "SAMEORIGIN" always;

        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }

    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
}
```

## Production Deployment

### Production Environment Configuration

Generate secure secrets and configure production environment:

```bash
# Generate secure credentials
JWT_SECRET=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 24)
WS_AUTH_TOKEN=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 32)

# Create production .env
cat > .env.production << EOF
# Production Configuration
ENVIRONMENT=production
DEBUG_MODE=false
RUST_LOG=warn
NODE_ENV=production

# Security
JWT_SECRET=$JWT_SECRET
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
WS_AUTH_TOKEN=$WS_AUTH_TOKEN
API_KEY=$API_KEY

# Authentication
WS_AUTH_ENABLED=true
RATE_LIMIT_ENABLED=true
CORS_ENABLED=true
CORS_ALLOWED_ORIGINS=https://yourdomain.com

# Network
DOMAIN=yourdomain.com
HOST_PORT=4000
CLOUDFLARE_TUNNEL_TOKEN=your_cloudflare_tunnel_token

# Performance
ENABLE_GPU=true
MEMORY_LIMIT=32g
CPU_LIMIT=16.0
MAX_AGENTS=50

# External Services (see External Services section)
RAGFLOW_API_BASE_URL=http://ragflow-server:9380
RAGFLOW_API_KEY=your_production_api_key
PERPLEXITY_API_KEY=your_production_perplexity_key
OPENAI_API_KEY=your_production_openai_key

# Resource Limits
MAX_REQUEST_SIZE=10485760
MAX_MESSAGE_SIZE=1048576
WS_MAX_CONNECTIONS=100
TCP_MAX_CONNECTIONS=50

# Monitoring
ENABLE_METRICS=true
HEALTH_CHECK_ENABLED=true
SECURITY_AUDIT_LOG=true
PERFORMANCE_MONITORING=true
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
docker stats webxr_prod_container
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
  - service: http_status:404
EOF

# 3. Start with tunnel
docker compose --profile production up -d cloudflared

# 4. Verify tunnel
docker compose logs -f cloudflared
```

### Production Nginx Configuration

For non-Cloudflare deployments, use Nginx as reverse proxy:

```nginx
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 8192;
    use epoll;
    multi_accept on;
}

http {
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
    limit_req_zone $binary_remote_addr zone=websocket:10m rate=50r/m;
    limit_conn_zone $binary_remote_addr zone=addr:10m;

    # Upstream
    upstream visionflow {
        least_conn;
        server webxr-prod:4000 max_fails=3 fail_timeout=30s;
        keepalive 64;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        # Main application
        location / {
            proxy_pass http://visionflow;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_http_version 1.1;
            proxy_set_header Connection "";

            # Rate limiting
            limit_req zone=api burst=20 nodelay;
            limit_conn addr 10;
        }

        # WebSocket endpoints
        location /ws {
            proxy_pass http://visionflow;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 86400;

            limit_req zone=websocket burst=10 nodelay;
        }

        # Health check (internal only)
        location /health {
            proxy_pass http://visionflow;
            access_log off;
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
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
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
RAGFLOW_API_BASE_URL=http://ragflow-server:9380
RAGFLOW_API_KEY=your_ragflow_api_key
RAGFLOW_AGENT_ID=your_agent_id
```

**Network Integration**:
```bash
# Connect RAGFlow to VisionFlow network
docker network connect docker_ragflow ragflow-server

# Verify connectivity
docker exec visionflow_container curl http://ragflow-server:9380/api/health
```

### Whisper Speech-to-Text

Whisper provides speech recognition for voice commands.

**Deployment**:
```bash
# Using OpenAI Whisper container
docker run -d \
  --name whisper-stt \
  --network docker_ragflow \
  -p 8080:8080 \
  -e MODEL_SIZE=base \
  --gpus all \
  onerahmet/openai-whisper-asr-webservice:latest

# Verify deployment
curl http://localhost:8080/health
```

**VisionFlow Configuration**:
```bash
# Whisper is accessed via fixed IP in docker_ragflow network
# Default: 172.18.0.5:8080
# No additional configuration needed if using default network
```

**Custom Configuration**:
```bash
# For custom Whisper endpoint, update src/config/mod.rs
# Or set environment variable
WHISPER_STT_ENDPOINT=http://172.18.0.5:8080
```

### Kokoro Text-to-Speech

Kokoro provides neural text-to-speech for voice responses.

**Deployment**:
```bash
# Using Kokoro TTS container
docker run -d \
  --name kokoro-tts \
  --network docker_ragflow \
  -p 5000:5000 \
  -e PORT=5000 \
  --gpus all \
  hexgrad/kokoro-tts:latest

# Verify deployment
curl http://localhost:5000/health
```

**Network Configuration**:
```bash
# Kokoro should be accessible at 172.18.0.9:5000 on docker_ragflow network
# Verify connectivity
docker exec visionflow_container curl http://172.18.0.9:5000/health

# Fix network if needed
docker network connect docker_ragflow kokoro-tts
```

**VisionFlow Configuration**:
```bash
# Kokoro endpoint is hardcoded to 172.18.0.9:5000
# Ensure Kokoro container has this IP on docker_ragflow network

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
docker network create vircadia_network

# Start PostgreSQL
docker run -d \
  --name vircadia_world_postgres \
  --network vircadia_network \
  -e POSTGRES_DB=vircadia_world \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=vircadia_password \
  -p 127.0.0.1:5432:5432 \
  -v vircadia_world_server_postgres_data:/var/lib/postgresql/data \
  postgres:17.5-alpine3.21

# Start PGWeb UI
docker run -d \
  --name vircadia_world_pgweb \
  --network vircadia_network \
  -p 127.0.0.1:5437:8081 \
  -e "PGWEB_DATABASE_URL=postgres://postgres:vircadia_password@vircadia_world_postgres:5432/vircadia_world?sslmode=disable" \
  sosedoff/pgweb:0.16.2

# Verify services
docker ps | grep vircadia
```

**Production Configuration**:
```bash
# Environment configuration (.env)
VRCA_SERVER_SERVICE_POSTGRES_HOST_CONTAINER_BIND_EXTERNAL=127.0.0.1
VRCA_SERVER_SERVICE_POSTGRES_PORT_CONTAINER_BIND_EXTERNAL=5432
VRCA_SERVER_SERVICE_POSTGRES_DATABASE=vircadia_world
VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD=secure_password_here

VRCA_SERVER_SERVICE_WORLD_API_MANAGER_PORT_CONTAINER_BIND_EXTERNAL=3020
VRCA_SERVER_SERVICE_WORLD_STATE_MANAGER_PORT_CONTAINER_BIND_EXTERNAL=3021
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
  vircadia_api_url: http://localhost:3020
  vircadia_state_url: http://localhost:3021
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
MCP_TCP_PORT=9500
MCP_BRIDGE_PORT=3002
BLENDER_HOST=gui-tools-service
BLENDER_PORT=9876
QGIS_HOST=gui-tools-service
QGIS_PORT=9877
PBR_HOST=gui-tools-service
PBR_PORT=9878
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
HOST_PORT=3003
MCP_TCP_PORT=9501

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
docker network inspect docker_ragflow

# Verify services are on same network
docker inspect webxr-prod | grep NetworkMode

# Reconnect to network
docker network connect docker_ragflow webxr-prod
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
MEMORY_LIMIT=64g

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
RATE_LIMIT_MAX_REQUESTS=200
RATE_LIMIT_WINDOW_MS=60000

# Restart service
docker compose --profile production up -d --force-recreate
```

### External Service Issues

**RAGFlow Connection Failed**:
```bash
# Verify RAGFlow is running
docker ps | grep ragflow

# Check connectivity
docker exec visionflow_container curl http://ragflow-server:9380/api/health

# Verify network
docker network inspect docker_ragflow | grep ragflow-server

# Reconnect if needed
docker network connect docker_ragflow ragflow-server
```

**Whisper STT Not Responding**:
```bash
# Check Whisper container
docker ps | grep whisper

# Test endpoint
curl -X POST http://172.18.0.5:8080/asr \
  -F "audio_file=@test.wav"

# Check logs
docker logs whisper-stt

# Restart if needed
docker restart whisper-stt
```

**Kokoro TTS Network Issues**:
```bash
# Verify Kokoro IP address
docker inspect kokoro-tts | grep IPAddress

# Should be 172.18.0.9 on docker_ragflow network
# If not, reconnect
docker network disconnect docker_ragflow kokoro-tts
docker network connect docker_ragflow kokoro-tts --ip 172.18.0.9

# Verify connectivity
docker exec visionflow_container curl http://172.18.0.9:5000/health
```

**Vircadia PostgreSQL Won't Start**:
```bash
# Check logs
docker logs vircadia_world_postgres

# Check volume permissions
docker volume inspect vircadia_world_server_postgres_data

# Reset volume (⚠️ destroys data)
docker stop vircadia_world_postgres
docker rm vircadia_world_postgres
docker volume rm vircadia_world_server_postgres_data
# Re-run start script
```

## Monitoring and Maintenance

### Health Checks

```bash
# VisionFlow health
curl http://localhost:3001/health

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
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup VisionFlow data
docker exec visionflow_container tar czf - /app/data | cat > $BACKUP_DIR/visionflow_data.tar.gz

# Backup Vircadia database
docker exec vircadia_world_postgres pg_dump -U postgres vircadia_world | gzip > $BACKUP_DIR/vircadia_db.sql.gz

# Backup configurations
cp .env* $BACKUP_DIR/
cp docker-compose*.yml $BACKUP_DIR/

# Clean old backups (keep last 7 days)
find /backups -type d -mtime +7 -exec rm -rf {} \;

echo "Backup complete: $BACKUP_DIR"
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

- [Configuration Reference](../reference/configuration.md) - Comprehensive environment variable and settings documentation
- [Troubleshooting Guide](./06-troubleshooting.md) - Detailed problem-solving procedures
- [Development Workflow](./02-development-workflow.md) - Development best practices and workflows
- [System Architecture](../architecture/overview.md) - Technical architecture overview
- [Voice API Reference](../reference/api/voice-api.md) - Voice system integration details
- [Multi-Agent Docker](../multi-agent-docker/README.md) - MCP tools and agent orchestration

---

*Last Updated: 2025-10-03*
*Document Version: 3.0.0*
