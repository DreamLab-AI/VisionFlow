# Docker Deployment Guide

Complete guide for deploying VisionFlow using Docker and Docker Compose.

## Prerequisites

- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with CUDA support (optional but recommended)
- 10GB free disk space

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/visionflow/visionflow.git
cd visionflow
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Core Configuration
NODE_ENV=production
PORT=8080
FRONTEND_PORT=3001

# Claude Flow MCP
CLAUDE_FLOW_HOST=multi-agent-container
MCP_TCP_PORT=9500
MCP_TRANSPORT=tcp

# API Keys (optional)
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key
GITHUB_TOKEN=your_github_token

# GPU Configuration
NO_GPU_COMPUTE=false  # Set to true if no GPU available
CUDA_VISIBLE_DEVICES=0

# Database
DATABASE_URL=postgresql://user:pass@db:5432/visionflow
```

### 3. Deploy with Docker Compose

```bash
# Production deployment
docker-compose up -d

# Development deployment
docker-compose -f docker-compose.dev.yml up

# With GPU support
docker-compose -f docker-compose.gpu.yml up -d
```

### 4. Verify Deployment

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Test health endpoints
curl http://localhost:8080/health
curl http://localhost:9501/health
```

## Docker Compose Configurations

### Production (docker-compose.yml)

```yaml
version: '3.8'

services:
  visionflow:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8080:8080"
      - "3001:3001"
    environment:
      - NODE_ENV=production
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  multi-agent:
    image: visionflow/multi-agent:latest
    ports:
      - "9500:9500"
      - "9501:9501"
    environment:
      - MCP_TCP_PORT=9500
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - visionflow
    restart: unless-stopped
```

### Development (docker-compose.dev.yml)

```yaml
version: '3.8'

services:
  visionflow-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8080:8080"
      - "3001:3001"
      - "9229:9229"  # Debug port
    environment:
      - NODE_ENV=development
      - RUST_LOG=debug
    volumes:
      - .:/app
      - /app/target
      - /app/client/node_modules
    command: cargo watch -x run

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=visionflow
      - POSTGRES_USER=dev
      - POSTGRES_PASSWORD=dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### GPU Support (docker-compose.gpu.yml)

```yaml
version: '3.8'

services:
  visionflow-gpu:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Dockerfile Configuration

### Multi-Stage Production Build

```dockerfile
# Build stage - Rust backend
FROM rust:1.75 as rust-builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Build stage - React frontend
FROM node:20-alpine as node-builder
WORKDIR /app
COPY client/package*.json ./
RUN npm ci
COPY client ./
RUN npm run build

# Production stage
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=rust-builder /app/target/release/visionflow /app/
COPY --from=node-builder /app/build /app/static/

EXPOSE 8080 3001
CMD ["./visionflow"]
```

## NVIDIA GPU Setup

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Test GPU access in container
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Nginx Configuration

### Basic Configuration

```nginx
upstream visionflow {
    server visionflow:8080;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://visionflow;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /wss {
        proxy_pass http://visionflow;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

### SSL Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # ... rest of configuration
}
```

## Volume Management

### Persistent Data

```bash
# Create named volumes
docker volume create visionflow_data
docker volume create visionflow_logs

# Backup volumes
docker run --rm -v visionflow_data:/data -v $(pwd):/backup ubuntu tar cvf /backup/data.tar /data

# Restore volumes
docker run --rm -v visionflow_data:/data -v $(pwd):/backup ubuntu tar xvf /backup/data.tar
```

## Health Monitoring

### Health Check Endpoints

- Main application: `http://localhost:8080/health`
- MCP service: `http://localhost:9501/health`
- Frontend: `http://localhost:3001/health`

### Docker Health Checks

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Logging

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f visionflow

# Last 100 lines
docker-compose logs --tail=100
```

### Log Configuration

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
sudo lsof -i :8080

# Kill process
sudo kill -9 <PID>
```

#### Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### Out of Memory
```bash
# Increase Docker memory limit
docker system prune -a
```

#### GPU Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Debug Commands

```bash
# Container shell access
docker-compose exec visionflow bash

# View resource usage
docker stats

# Inspect container
docker inspect visionflow

# Clean up
docker-compose down -v
docker system prune -a
```

## Production Recommendations

1. **Use Docker Swarm or Kubernetes** for orchestration
2. **Enable log aggregation** with ELK stack
3. **Set up monitoring** with Prometheus/Grafana
4. **Configure automatic backups** for volumes
5. **Use secrets management** for API keys
6. **Enable rate limiting** with nginx
7. **Set up SSL certificates** with Let's Encrypt
8. **Configure firewall rules** for security

## Support

For deployment issues:
- GitHub Issues: [Report problems](https://github.com/visionflow/issues)
- Documentation: [Full docs](../README.md)
- Email: support@visionflow.ai

---

*Last updated: January 2025*