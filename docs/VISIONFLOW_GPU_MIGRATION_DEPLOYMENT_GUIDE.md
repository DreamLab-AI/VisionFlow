# VisionFlow GPU Physics Migration - Deployment Guide

## Overview

This deployment guide provides comprehensive instructions for deploying the VisionFlow GPU Physics Migration to production environments. The guide covers prerequisites, configuration, deployment procedures, verification, and rollback strategies.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Infrastructure Requirements](#infrastructure-requirements)
- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Environment Configuration](#environment-configuration)
- [Deployment Procedure](#deployment-procedure)
- [Post-Deployment Verification](#post-deployment-verification)
- [Monitoring Setup](#monitoring-setup)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, or RHEL 8+)
- **CPU**: 4 cores, 2.4 GHz
- **Memory**: 8 GB RAM
- **Storage**: 50 GB available disk space
- **Network**: 1 Gbps network interface

#### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **CPU**: 8 cores, 3.0 GHz
- **Memory**: 16 GB RAM
- **Storage**: 100 GB SSD storage
- **Network**: 10 Gbps network interface
- **GPU**: NVIDIA GPU with CUDA support (optional)

### Software Dependencies

#### Runtime Dependencies
```bash
# Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Node.js 18+ (for Claude Flow MCP)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Rust toolchain (for backend compilation)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### Optional GPU Support
```bash
# NVIDIA Container Toolkit (for GPU acceleration)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Network Configuration

#### Firewall Rules
```bash
# Open required ports
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8080/tcp  # Backend API
sudo ufw allow 3000/tcp  # Claude Flow MCP
sudo ufw allow 3002/tcp  # WebSocket connections
```

#### DNS Configuration
- Ensure proper DNS resolution for your domain
- Configure SSL certificates (Let's Encrypt recommended)
- Set up CDN if using global distribution

## Infrastructure Requirements

### Container Architecture

```yaml
# docker-compose.production.yml structure
services:
  visionflow_backend:     # Rust backend server
  visionflow_frontend:    # React/Vite frontend
  claude_flow_mcp:        # Claude Flow MCP service
  nginx:                  # Reverse proxy and load balancer
  postgres:               # Database (if needed)
  redis:                  # Caching layer (optional)
```

### Resource Allocation

#### Backend Container
- **CPU**: 2-4 cores
- **Memory**: 4-8 GB
- **Storage**: 20 GB
- **Network**: High bandwidth for WebSocket connections

#### Frontend Container
- **CPU**: 1-2 cores
- **Memory**: 2-4 GB
- **Storage**: 5 GB
- **Network**: CDN recommended for static assets

#### Claude Flow MCP Container
- **CPU**: 1-2 cores
- **Memory**: 2-4 GB
- **Storage**: 10 GB
- **Network**: Stable connection to backend

## Pre-Deployment Checklist

### Code Quality Verification
- [ ] All unit tests passing (>90% coverage)
- [ ] Integration tests completed successfully
- [ ] Performance benchmarks meet requirements
- [ ] Security scan completed with no critical issues
- [ ] Code review approved by technical lead

### Configuration Validation
- [ ] Environment variables configured for production
- [ ] SSL certificates installed and validated
- [ ] Database connections tested
- [ ] External API keys configured and tested
- [ ] Monitoring and logging configured

### Infrastructure Preparation
- [ ] Production servers provisioned and accessible
- [ ] DNS records configured and propagated
- [ ] Load balancer configured (if applicable)
- [ ] Backup systems configured and tested
- [ ] Monitoring systems deployed and configured

### Migration Verification
- [ ] Mock data elimination verified
- [ ] Binary protocol implementation tested
- [ ] MCP integration functional
- [ ] GPU simulation performance validated
- [ ] Error handling scenarios tested

## Environment Configuration

### Environment Variables

#### Backend Configuration
```bash
# Production environment file (.env.production)
RUST_LOG=info
RUST_BACKTRACE=1

# Database configuration
DATABASE_URL=postgresql://user:password@localhost/visionflow

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
WEBSOCKET_PORT=3002

# Claude Flow MCP
CLAUDE_FLOW_HOST=powerdev
CLAUDE_FLOW_PORT=3000
MCP_WEBSOCKET_URL=ws://powerdev:3000/ws

# External Services
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key
GITHUB_TOKEN=your_github_token

# Performance Configuration
MAX_AGENTS=400
UPDATE_FREQUENCY=60
GPU_ACCELERATION=false  # Set to true if GPU available
BINARY_PROTOCOL_ENABLED=true

# Security
CORS_ORIGIN=https://yourdomain.com
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
```

#### Frontend Configuration
```bash
# Frontend environment (.env.production)
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_WEBSOCKET_URL=wss://yourdomain.com/ws
VITE_MCP_WEBSOCKET_URL=wss://yourdomain.com/mcp-ws

# Feature Flags
VITE_GPU_PHYSICS_ENABLED=true
VITE_MOCK_DATA_ENABLED=false
VITE_DEBUG_MODE=false

# Performance Settings
VITE_MAX_AGENTS=200
VITE_DEFAULT_FPS=60
VITE_BINARY_PROTOCOL=true

# Analytics and Monitoring
VITE_ANALYTICS_ENABLED=true
VITE_ERROR_REPORTING_URL=https://errors.yourdomain.com
```

### Docker Compose Configuration

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  visionflow_backend:
    image: visionflow/backend:latest
    container_name: visionflow_backend
    environment:
      - RUST_LOG=info
      - API_PORT=8080
      - WEBSOCKET_PORT=3002
      - CLAUDE_FLOW_HOST=claude_flow_mcp
      - CLAUDE_FLOW_PORT=3000
    ports:
      - "8080:8080"
      - "3002:3002"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - claude_flow_mcp
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  claude_flow_mcp:
    image: visionflow/claude-flow-mcp:latest
    container_name: claude_flow_mcp
    environment:
      - NODE_ENV=production
      - PORT=3000
      - MCP_PORT=3001
    ports:
      - "3000:3000"
      - "3001:3001"
    volumes:
      - ./claude-flow-data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: visionflow_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - visionflow_backend
    restart: unless-stopped

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  visionflow_data:
    driver: local
  claude_flow_data:
    driver: local
```

### NGINX Configuration

```nginx
# nginx/nginx.conf
upstream backend {
    server visionflow_backend:8080;
}

upstream websocket {
    server visionflow_backend:3002;
}

upstream mcp_service {
    server claude_flow_mcp:3000;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend static files
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
        
        # Caching for static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CORS headers
        add_header Access-Control-Allow-Origin "https://yourdomain.com" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization" always;
    }

    # WebSocket connections
    location /ws {
        proxy_pass http://websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific settings
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
        proxy_connect_timeout 86400;
    }

    # MCP WebSocket
    location /mcp-ws {
        proxy_pass http://mcp_service;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health checks
    location /health {
        proxy_pass http://backend/health;
        access_log off;
    }
}
```

## Deployment Procedure

### Step 1: Infrastructure Preparation

```bash
# 1. Create deployment directory
mkdir -p /opt/visionflow
cd /opt/visionflow

# 2. Clone repository
git clone https://github.com/your-org/visionflow.git .
git checkout production

# 3. Create directory structure
mkdir -p {data,logs,nginx/ssl,backups}
chmod 755 data logs backups
chmod 700 nginx/ssl
```

### Step 2: Configuration Setup

```bash
# 1. Copy production configuration
cp docker-compose.production.yml docker-compose.yml
cp .env.production .env

# 2. Generate secrets
openssl rand -hex 32 > .secrets/jwt_secret
openssl rand -hex 32 > .secrets/encryption_key

# 3. Install SSL certificates
# Option A: Let's Encrypt
certbot certonly --standalone -d yourdomain.com
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/key.pem

# Option B: Custom certificates
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem
chmod 600 nginx/ssl/*
```

### Step 3: Build and Deploy

```bash
# 1. Build application images
docker-compose build --no-cache

# 2. Pull external dependencies
docker-compose pull

# 3. Start services
docker-compose up -d

# 4. Verify deployment
docker-compose ps
docker-compose logs -f
```

### Step 4: Database Migration (if applicable)

```bash
# 1. Run database migrations
docker-compose exec visionflow_backend cargo run --bin migrate

# 2. Seed initial data
docker-compose exec visionflow_backend cargo run --bin seed
```

### Step 5: SSL and Security Setup

```bash
# 1. Test SSL configuration
curl -I https://yourdomain.com

# 2. Verify security headers
curl -I https://yourdomain.com | grep -E "(Strict-Transport|Content-Security|X-Frame)"

# 3. Test WebSocket connections
wscat -c wss://yourdomain.com/ws
```

## Post-Deployment Verification

### Health Check Script

```bash
#!/bin/bash
# health-check.sh

echo "=== VisionFlow Health Check ==="

# 1. Container Status
echo "1. Checking container status..."
docker-compose ps | grep -E "(Up|healthy)" || echo "❌ Some containers are not healthy"

# 2. API Health
echo "2. Checking API health..."
curl -f https://yourdomain.com/api/health || echo "❌ API health check failed"

# 3. WebSocket Connection
echo "3. Checking WebSocket..."
timeout 5 wscat -c wss://yourdomain.com/ws -x '{"type":"ping"}' || echo "❌ WebSocket connection failed"

# 4. MCP Service
echo "4. Checking MCP service..."
curl -f http://localhost:3000/health || echo "❌ MCP service health check failed"

# 5. Frontend Loading
echo "5. Checking frontend..."
curl -f https://yourdomain.com/ | grep -q "VisionFlow" || echo "❌ Frontend not loading correctly"

# 6. SSL Certificate
echo "6. Checking SSL certificate..."
echo | openssl s_client -servername yourdomain.com -connect yourdomain.com:443 2>/dev/null | \
  openssl x509 -noout -dates || echo "❌ SSL certificate issue"

echo "=== Health Check Complete ==="
```

### Performance Verification

```bash
#!/bin/bash
# performance-test.sh

echo "=== Performance Verification ==="

# 1. Load test with 50 concurrent agents
echo "1. Testing 50 concurrent agents..."
curl -X POST https://yourdomain.com/api/bots/load-test \
  -H "Content-Type: application/json" \
  -d '{"agentCount": 50, "duration": 60}'

# 2. WebSocket throughput test
echo "2. Testing WebSocket throughput..."
node test-websocket-throughput.js

# 3. Memory usage check
echo "3. Checking memory usage..."
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# 4. Binary protocol validation
echo "4. Validating binary protocol..."
curl -X GET https://yourdomain.com/api/bots/binary-test

echo "=== Performance Test Complete ==="
```

### Functional Testing

```bash
#!/bin/bash
# functional-test.sh

echo "=== Functional Testing ==="

# 1. Agent visualization test
echo "1. Testing agent visualization..."
curl -X POST https://yourdomain.com/api/bots/spawn \
  -H "Content-Type: application/json" \
  -d '{"count": 10, "type": "test"}'

# 2. Real-time updates test
echo "2. Testing real-time updates..."
wscat -c wss://yourdomain.com/ws -x '{"type":"subscribe","topic":"agents"}'

# 3. GPU simulation test
echo "3. Testing GPU simulation..."
curl -X POST https://yourdomain.com/api/bots/gpu-test \
  -H "Content-Type: application/json" \
  -d '{"agentCount": 100}'

# 4. Error handling test
echo "4. Testing error handling..."
curl -X POST https://yourdomain.com/api/bots/error-test

echo "=== Functional Test Complete ==="
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'visionflow-backend'
    static_configs:
      - targets: ['visionflow_backend:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'claude-flow-mcp'
    static_configs:
      - targets: ['claude_flow_mcp:3000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']
    scrape_interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "VisionFlow GPU Migration Dashboard",
    "panels": [
      {
        "title": "Agent Count",
        "type": "stat",
        "targets": [
          {
            "expr": "visionflow_active_agents"
          }
        ]
      },
      {
        "title": "Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(visionflow_processing_duration_seconds_sum[5m]) / rate(visionflow_processing_duration_seconds_count[5m])"
          }
        ]
      },
      {
        "title": "WebSocket Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "visionflow_websocket_connections"
          }
        ]
      },
      {
        "title": "Binary Protocol Efficiency",
        "type": "graph",
        "targets": [
          {
            "expr": "visionflow_binary_protocol_bytes_per_second"
          }
        ]
      }
    ]
  }
}
```

### Log Monitoring

```bash
# Configure log aggregation
# filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_docker_metadata:
        host: "unix:///var/run/docker.sock"

output.elasticsearch:
  hosts: ["elasticsearch:9200"]

# Logstash configuration for VisionFlow logs
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [container][name] == "visionflow_backend" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} \[%{DATA:module}\] %{GREEDYDATA:msg}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "visionflow-logs-%{+YYYY.MM.dd}"
  }
}
```

## Rollback Procedures

### Immediate Rollback (Emergency)

```bash
#!/bin/bash
# emergency-rollback.sh

echo "=== EMERGENCY ROLLBACK INITIATED ==="

# 1. Stop current services
docker-compose down

# 2. Restore previous version
git checkout previous-stable-tag
docker-compose pull

# 3. Restore database backup (if needed)
if [ -f backups/database-backup-$(date -d "1 day ago" +%Y%m%d).sql ]; then
  echo "Restoring database backup..."
  docker-compose exec postgres psql -U postgres -d visionflow < backups/database-backup-$(date -d "1 day ago" +%Y%m%d).sql
fi

# 4. Start services
docker-compose up -d

# 5. Verify rollback
./health-check.sh

echo "=== ROLLBACK COMPLETE ==="
```

### Gradual Rollback

```bash
#!/bin/bash
# gradual-rollback.sh

echo "=== GRADUAL ROLLBACK INITIATED ==="

# 1. Switch traffic to backup instance
echo "Switching traffic to backup..."
# Update load balancer configuration
# Or use blue-green deployment switch

# 2. Monitor for 5 minutes
echo "Monitoring for 5 minutes..."
for i in {1..30}; do
  sleep 10
  ./health-check.sh > /dev/null || echo "Health check failed at iteration $i"
done

# 3. Complete rollback if stable
echo "Completing rollback..."
docker-compose down
git checkout previous-stable-tag
docker-compose up -d

echo "=== GRADUAL ROLLBACK COMPLETE ==="
```

### Rollback Verification

```bash
#!/bin/bash
# verify-rollback.sh

echo "=== ROLLBACK VERIFICATION ==="

# 1. Check application version
VERSION=$(curl -s https://yourdomain.com/api/version | jq -r '.version')
echo "Current version: $VERSION"

# 2. Verify functionality
./functional-test.sh

# 3. Check performance
./performance-test.sh

# 4. Monitor error rates
echo "Monitoring error rates for 10 minutes..."
ERRORS_BEFORE=$(curl -s https://yourdomain.com/api/metrics | grep error_count | cut -d' ' -f2)
sleep 600
ERRORS_AFTER=$(curl -s https://yourdomain.com/api/metrics | grep error_count | cut -d' ' -f2)
ERROR_INCREASE=$((ERRORS_AFTER - ERRORS_BEFORE))

if [ $ERROR_INCREASE -lt 10 ]; then
  echo "✅ Rollback successful - error rate within acceptable limits"
else
  echo "❌ Rollback may have issues - error rate increased by $ERROR_INCREASE"
fi

echo "=== ROLLBACK VERIFICATION COMPLETE ==="
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Container Startup Failures

**Symptoms**: Containers exit immediately or fail to start
```bash
# Diagnosis
docker-compose logs <service_name>
docker inspect <container_name>

# Common solutions
# - Check environment variables
# - Verify volume mounts
# - Check port conflicts
# - Validate configuration files
```

#### 2. WebSocket Connection Issues

**Symptoms**: Real-time updates not working, connection timeouts
```bash
# Diagnosis
curl -I https://yourdomain.com/ws
wscat -c wss://yourdomain.com/ws

# Solutions
# - Check NGINX WebSocket configuration
# - Verify backend WebSocket handler
# - Check firewall rules
# - Validate SSL certificates for WSS
```

#### 3. Performance Degradation

**Symptoms**: Slow agent updates, high latency, memory issues
```bash
# Diagnosis
docker stats
curl https://yourdomain.com/api/metrics
htop

# Solutions
# - Increase container resources
# - Check for memory leaks
# - Optimize database queries
# - Review network configuration
```

#### 4. MCP Service Connectivity

**Symptoms**: Agent data not loading, MCP connection errors
```bash
# Diagnosis
curl http://localhost:3000/health
docker-compose logs claude_flow_mcp

# Solutions
# - Verify MCP service configuration
# - Check network connectivity between containers
# - Validate MCP protocol implementation
# - Review authentication settings
```

### Debug Mode Activation

```bash
# Enable debug logging
export RUST_LOG=debug
export NODE_ENV=development
export VITE_DEBUG_MODE=true

# Restart services with debug logging
docker-compose down
docker-compose up -d

# Monitor debug logs
docker-compose logs -f --tail=100
```

### Performance Profiling

```bash
# CPU profiling
docker-compose exec visionflow_backend cargo flamegraph --bin visionflow

# Memory profiling
docker-compose exec visionflow_backend valgrind --tool=massif ./target/release/visionflow

# Network profiling
tcpdump -i any -w network-trace.pcap port 8080 or port 3002
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily Tasks
- [ ] Check system health and monitoring alerts
- [ ] Review error logs for any issues
- [ ] Verify backup completion
- [ ] Monitor resource usage trends

#### Weekly Tasks
- [ ] Update SSL certificates if needed
- [ ] Review security logs
- [ ] Analyze performance metrics
- [ ] Test rollback procedures

#### Monthly Tasks
- [ ] Security updates and patches
- [ ] Performance optimization review
- [ ] Capacity planning assessment
- [ ] Documentation updates

### Backup Procedures

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/visionflow/backups"

# 1. Database backup
docker-compose exec postgres pg_dump -U postgres visionflow > $BACKUP_DIR/database_$DATE.sql

# 2. Configuration backup
tar -czf $BACKUP_DIR/config_$DATE.tar.gz docker-compose.yml .env nginx/

# 3. Data volumes backup
docker run --rm -v visionflow_data:/data -v $BACKUP_DIR:/backup alpine tar -czf /backup/data_$DATE.tar.gz -C /data .

# 4. Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

## Security Considerations

### Security Checklist
- [ ] SSL/TLS certificates properly configured
- [ ] CORS policies configured correctly
- [ ] API authentication and authorization implemented
- [ ] Database connections encrypted
- [ ] Secrets managed securely (not in environment files)
- [ ] Container images scanned for vulnerabilities
- [ ] Network segmentation implemented
- [ ] Monitoring and alerting for security events

### Security Updates

```bash
# Regular security update procedure
apt update && apt list --upgradable
docker-compose pull  # Update container images
docker system prune -f  # Clean unused images

# Update SSL certificates
certbot renew --dry-run
certbot renew
systemctl reload nginx
```

## Support and Escalation

### Support Contacts
- **Technical Lead**: [Contact Information]
- **DevOps Team**: [Contact Information]
- **Security Team**: [Contact Information]
- **On-Call Engineer**: [Contact Information]

### Escalation Procedures
1. **Level 1**: Application issues, configuration problems
2. **Level 2**: Performance issues, integration problems
3. **Level 3**: Security incidents, data corruption
4. **Emergency**: System outages, security breaches

### Documentation Updates
This deployment guide should be updated whenever:
- New configuration options are added
- Deployment procedures change
- New troubleshooting scenarios are discovered
- Security requirements are updated

---

**Document Version**: 1.0  
**Last Updated**: July 31, 2025  
**Next Review**: August 31, 2025  
**Maintained By**: VisionFlow DevOps Team