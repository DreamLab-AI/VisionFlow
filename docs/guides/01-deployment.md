# Deployment Guide

*[← Back to Guides](index.md)*

This comprehensive guide covers deploying VisionFlow in various environments, from local development to production-ready cloud deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Deployment](#local-development-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Multi-Agent Container Setup](#multi-agent-container-setup)
7. [GPU-Accelerated Deployment](#gpu-accelerated-deployment)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.5GHz+
- RAM: 8GB
- Storage: 20GB available SSD space
- OS: Ubuntu 20.04+, macOS 12+, or Windows 10+ with WSL2
- Docker: Version 20.10+
- Docker Compose: Version 2.0+

**Recommended Requirements:**
- CPU: 8+ cores, 3.5GHz+
- RAM: 16GB+
- Storage: 50GB+ SSD
- GPU: NVIDIA GPU with CUDA 11.8+ (optional)
- Network: Stable broadband connection

### Software Dependencies

```bash
# Check Docker version
docker --version  # Should be 20.10+

# Check Docker Compose version
docker compose version  # Should be 2.0+

# Install Docker (if needed)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose (if needed)
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

## Local Development Deployment

### Quick Start

The fastest way to get VisionFlow running locally:

```bash
# Clone the repository
git clone https://github.com/your-org/VisionsFlow.git
cd VisionsFlow

# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env

# Start all services
docker-compose up -d

# Verify deployment
docker-compose ps
```

### Development Configuration

Create a development-specific `.env` file:

```bash
# Development Environment
ENVIRONMENT=development
DEBUG_MODE=true
RUST_LOG=debug
HOT_RELOAD=true

# Networking
HOST_PORT=3001
MCP_TCP_PORT=9500
CLAUDE_FLOW_HOST=multi-agent-container

# Security (development only)
JWT_SECRET=dev_secret_change_in_production
POSTGRES_PASSWORD=dev_password
AUTH_REQUIRED=false

# Development features
ENABLE_PROFILING=true
MOCK_SERVICES=true
```

### Starting Services Individually

For debugging, you may want to start services separately:

```bash
# Start core infrastructure
docker-compose up -d postgres redis

# Start backend services
docker-compose up -d server

# Start frontend
docker-compose up -d client

# Start multi-agent system
docker-compose up -d multi-agent-container
```

## Docker Deployment

### Standard Docker Compose Deployment

```bash
# Production-ready deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale specific services
docker-compose up -d --scale worker=3

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Docker Compose Configuration

The system uses multiple compose files for different scenarios:

#### Base Configuration (docker-compose.yml)
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: visionflow
      POSTGRES_USER: visionflow
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U visionflow"]
      interval: 30s
      timeout: 10s
      retries: 5

  server:
    build: ./server
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgres://visionflow:${POSTGRES_PASSWORD}@postgres/visionflow
      RUST_LOG: ${RUST_LOG:-info}
    ports:
      - "${HOST_PORT:-3001}:8080"

  client:
    build: ./client
    environment:
      REACT_APP_API_URL: http://server:8080
    ports:
      - "3002:80"

  multi-agent-container:
    build: ./multi-agent-docker
    environment:
      CLAUDE_FLOW_HOST: ${CLAUDE_FLOW_HOST}
      MCP_TCP_PORT: ${MCP_TCP_PORT}
    volumes:
      - agent_workspace:/workspace
    ports:
      - "3000:3000"  # Claude Flow UI
      - "3002:3002"  # WebSocket Bridge
      - "9876:9876"  # Blender MCP
      - "9877:9877"  # QGIS MCP
      - "9878:9878"  # PBR Generator

volumes:
  postgres_data:
  agent_workspace:

networks:
  default:
    name: docker_ragflow
```

### Multi-Stage Build Optimization

Optimize Docker images with multi-stage builds:

```dockerfile
# server/Dockerfile
# Build stage
FROM rust:1.75 as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
COPY . .
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/visionflow /usr/local/bin/
EXPOSE 8080
CMD ["visionflow"]
```

## Production Deployment

### Production Environment Setup

1. **Generate Secure Secrets**
```bash
# Generate secure passwords and tokens
JWT_SECRET=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 24)
REDIS_PASSWORD=$(openssl rand -hex 16)

# Save to production .env
cat > .env.production << EOF
ENVIRONMENT=production
DEBUG_MODE=false
RUST_LOG=warn

# Security
JWT_SECRET=$JWT_SECRET
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD

# Performance
ENABLE_GPU=true
MAX_AGENTS=50
MEMORY_LIMIT=32g
CPU_LIMIT=16.0

# Networking
DOMAIN=visionflow.example.com
ENABLE_SSL=true
EOF
```

2. **Production Docker Compose Override**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  server:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G

  postgres:
    restart: always
    command: >
      postgres
      -c max_connections=200
      -c shared_buffers=2GB
      -c effective_cache_size=6GB
      -c maintenance_work_mem=512MB

  redis:
    restart: always
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - server
      - client
```

3. **Deploy to Production**
```bash
# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify all services are healthy
docker-compose ps
docker-compose exec server curl http://localhost:8080/health

# Set up SSL certificates
sudo certbot --nginx -d visionflow.example.com
```

### Security Hardening

1. **Network Security**
```bash
# Configure firewall
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw enable

# Restrict Docker daemon
echo '{"hosts": ["unix:///var/run/docker.sock"], "tls": true}' | sudo tee /etc/docker/daemon.json
```

2. **Container Security**
```yaml
# Add to docker-compose.prod.yml
services:
  server:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    user: "1000:1000"
```

## Cloud Deployment

### AWS Deployment

1. **ECS with Fargate**
```bash
# Install AWS CLI and ECS CLI
pip install awscli ecs-cli

# Configure credentials
aws configure

# Create ECS cluster
ecs-cli configure --cluster visionflow --region us-west-2
ecs-cli up --capability-iam --size 2 --instance-type t3.large

# Deploy task definition
ecs-cli compose --file docker-compose.yml --ecs-params ecs-params.yml up
```

2. **ECS Parameters** (ecs-params.yml)
```yaml
version: 1
task_definition:
  task_execution_role: ecsTaskExecutionRole
  ecs_network_mode: awsvpc
  cpu_limit: 4096
  mem_limit: 16384
  services:
    server:
      cpu_shares: 2048
      mem_reservation: 8192
```

### Google Cloud Platform

```bash
# Configure gcloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create GKE cluster
gcloud container clusters create visionflow \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4

# Deploy with Kubernetes
kubectl apply -f k8s/
```

### Azure Container Instances

```bash
# Login to Azure
az login

# Create resource group
az group create --name visionflow-rg --location eastus

# Deploy container group
az container create \
  --resource-group visionflow-rg \
  --file docker-compose.yml
```

## Multi-Agent Container Setup

### Architecture Overview

The multi-agent container provides a complete development environment with MCP tools:

```mermaid
graph TB
    subgraph "Multi-Agent Container"
        SUPERVISOR[Supervisord]
        WS_BRIDGE[WebSocket Bridge<br/>Port 3002]
        CLAUDE_FLOW[Claude Flow]

        subgraph "MCP Tools"
            BLENDER[Blender MCP]
            QGIS[QGIS MCP]
            IMAGEMAGICK[ImageMagick]
            KICAD[KiCad MCP]
            NGSPICE[NGSpice MCP]
            PBR[PBR Generator]
        end
    end

    subgraph "GUI Tools Container"
        EXT_BLENDER[Blender GUI]
        EXT_QGIS[QGIS GUI]
        EXT_PBR[PBR Service]
    end

    CLAUDE_FLOW --> BLENDER
    BLENDER -.-> EXT_BLENDER
    QGIS -.-> EXT_QGIS
    PBR -.-> EXT_PBR
```

### Deployment Steps

1. **Build the Multi-Agent Container**
```bash
cd multi-agent-docker
docker build -t visionflow/multi-agent:latest .
```

2. **Start with GUI Tools**
```bash
# Start GUI tools container first
docker-compose up -d gui-tools-service

# Start multi-agent container
docker-compose up -d multi-agent-container

# Verify MCP tools
docker exec multi-agent-container ./mcp-helper.sh list-tools
```

3. **Configure MCP Tools**
```bash
# Inside the container
docker exec -it multi-agent-container bash

# Initialize workspace
/app/setup-workspace.sh --force

# Test MCP tools
./mcp-helper.sh test-all
```

### Connecting External Tools

Configure environment variables for external tool connections:

```bash
# .env
BLENDER_HOST=gui-tools-service
QGIS_HOST=gui-tools-service
PBR_HOST=gui-tools-service

# Port mappings
BLENDER_PORT=9876
QGIS_PORT=9877
PBR_PORT=9878
```

## GPU-Accelerated Deployment

### NVIDIA GPU Setup

1. **Install NVIDIA Container Toolkit**
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

2. **Verify GPU Access**
```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

3. **GPU-Enabled Docker Compose**
```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  server:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      ENABLE_GPU: "true"
      CUDA_VISIBLE_DEVICES: "0"
```

### GPU Configuration

```bash
# Environment variables for GPU
ENABLE_GPU=true
NVIDIA_VISIBLE_DEVICES=0
CUDA_ARCH=89  # For RTX 40xx
GPU_MEMORY_LIMIT=8g

# Start with GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Monitoring and Maintenance

### Health Monitoring

1. **Set up Prometheus and Grafana**
```yaml
# monitoring/docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3003:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}

volumes:
  prometheus_data:
  grafana_data:
```

2. **Configure Prometheus Scraping**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'visionflow'
    static_configs:
      - targets: ['server:8080']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Backup and Recovery

1. **Automated Backups**
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup database
docker exec postgres pg_dump -U visionflow visionflow | gzip > $BACKUP_DIR/postgres.sql.gz

# Backup volumes
docker run --rm -v visionflow_agent_workspace:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/workspace.tar.gz -C /data .

# Backup configurations
cp .env* $BACKUP_DIR/
cp docker-compose*.yml $BACKUP_DIR/

# Clean old backups (keep last 7 days)
find /backups -type d -mtime +7 -exec rm -rf {} \;
```

2. **Schedule Backups**
```bash
# Add to crontab
0 2 * * * /path/to/backup.sh >> /var/log/visionflow-backup.log 2>&1
```

### Maintenance Tasks

1. **Regular Updates**
```bash
# Update images
docker-compose pull

# Recreate containers with new images
docker-compose up -d --force-recreate

# Prune unused resources
docker system prune -a --volumes
```

2. **Log Management**
```bash
# Configure log rotation
cat > /etc/logrotate.d/visionflow << EOF
/var/log/visionflow/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0640 visionflow visionflow
    postrotate
        docker-compose kill -s USR1 server
    endscript
}
EOF
```

## Troubleshooting Deployment Issues

### Common Problems

1. **Port Conflicts**
```bash
# Check port usage
sudo netstat -tulpn | grep -E '3001|3002|8080'

# Change ports in .env
HOST_PORT=3003
MCP_TCP_PORT=9501
```

2. **Memory Issues**
```bash
# Check memory usage
docker stats

# Adjust memory limits
docker-compose up -d --memory="8g" server
```

3. **Network Issues**
```bash
# Recreate network
docker network rm docker_ragflow
docker-compose up -d
```

### Debug Commands

```bash
# Check service health
docker-compose ps
docker-compose exec server curl http://localhost:8080/health

# View logs
docker-compose logs -f --tail=100 server

# Access container shell
docker-compose exec server /bin/bash

# Test database connection
docker-compose exec postgres psql -U visionflow -c "SELECT version();"
```

## Next Steps

- Continue to [Development Workflow](02-development-workflow.md) for development best practices
- See [Using the GUI Sandbox](03-using-the-gui-sandbox.md) for MCP tool usage
- Check [Troubleshooting](06-troubleshooting.md) for more detailed problem-solving

---

*[← Back to Guides](index.md) | [Development Workflow →](02-development-workflow.md)*