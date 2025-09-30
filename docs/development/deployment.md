# Deployment Guide

**Version**: 2.2.0
**Last Updated**: 2025-09-30
**Target Environments**: Development, Staging, Production

## Overview

This guide provides comprehensive instructions for deploying VisionFlow in various environments, from local development to production Kubernetes clusters.

## Prerequisites

### Required Software
- Docker 24.0+ with Compose V2
- Node.js 18+ and npm 9+
- Rust 1.70+ (for development builds)
- PostgreSQL 15+ (or use Docker)
- Redis 7.0+ (or use Docker)
- NVIDIA GPU with CUDA 12.0+ (for GPU features)

### System Requirements

**Development Environment**:
- CPU: 4+ cores
- RAM: 8 GB minimum, 16 GB recommended
- GPU: NVIDIA GTX 1060+ (optional but recommended)
- Storage: 20 GB available space

**Production Environment**:
- CPU: 8+ cores per node
- RAM: 16 GB minimum, 32 GB recommended
- GPU: NVIDIA RTX 3080+ or data centre GPU
- Storage: 100 GB+ SSD with IOPS 3000+
- Network: 1 Gbps minimum bandwidth

## Development Deployment

### Quick Start with Docker Compose

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/visionflow.git
cd visionflow/ext
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start services**:
```bash
docker-compose up -d
```

4. **Verify deployment**:
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Access application
open http://localhost:3000
```

### Development Configuration

**File**: `docker-compose.yml`
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: visionflow
      POSTGRES_USER: visionflow
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    environment:
      DATABASE_URL: postgresql://visionflow:${DB_PASSWORD}@postgres:5432/visionflow
      REDIS_URL: redis://redis:6379
      RUST_LOG: info
    ports:
      - "8080:8080"
      - "3002:3002"
    volumes:
      - ./src:/app/src
      - cargo-cache:/usr/local/cargo
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: ./client
      dockerfile: Dockerfile
    environment:
      VITE_API_URL: http://localhost:8080
      VITE_WS_URL: ws://localhost:3002
    ports:
      - "3000:3000"
    volumes:
      - ./client/src:/app/src
    depends_on:
      - backend

  mcp-server:
    image: node:18-alpine
    working_dir: /app
    command: npx claude-flow@alpha mcp start
    ports:
      - "9500:9500"
    environment:
      MCP_PORT: 9500
      MCP_HOST: 0.0.0.0
    volumes:
      - ./multi-agent-docker:/app

volumes:
  postgres-data:
  redis-data:
  cargo-cache:
```

### Environment Variables

**File**: `.env`
```bash
# Database Configuration
DB_PASSWORD=secure_password_here
DATABASE_URL=postgresql://visionflow:${DB_PASSWORD}@postgres:5432/visionflow

# Redis Configuration
REDIS_URL=redis://redis:6379

# Backend Configuration
RUST_LOG=info
SERVER_PORT=8080
WEBSOCKET_PORT=3002
GPU_ENABLED=true

# Frontend Configuration
VITE_API_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:3002

# Security
JWT_SECRET=generate_secure_secret_here
SESSION_SECRET=generate_secure_secret_here

# Rate Limiting
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX_REQUESTS=100

# MCP Configuration
MCP_PORT=9500
MCP_HOST=0.0.0.0
```

### Local Development Without Docker

**Backend**:
```bash
# Install dependencies
cargo build --release

# Run database migrations
diesel migration run

# Start backend server
cargo run --release
```

**Frontend**:
```bash
cd client
npm install
npm run dev
```

## Staging Deployment

### Docker Compose for Staging

**File**: `docker-compose.staging.yml`
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: visionflow_staging
      POSTGRES_USER: visionflow
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres-staging:/var/lib/postgresql/data
    networks:
      - staging-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    networks:
      - staging-network
    volumes:
      - redis-staging:/data
    restart: unless-stopped

  backend:
    image: your-registry/visionflow-backend:staging
    environment:
      DATABASE_URL: postgresql://visionflow:${DB_PASSWORD}@postgres:5432/visionflow_staging
      REDIS_URL: redis://redis:6379
      RUST_LOG: info
      ENVIRONMENT: staging
    networks:
      - staging-network
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  frontend:
    image: your-registry/visionflow-frontend:staging
    environment:
      VITE_API_URL: https://staging-api.yourdomain.com
      VITE_WS_URL: wss://staging-ws.yourdomain.com
    networks:
      - staging-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx/staging.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - staging-network
    depends_on:
      - backend
      - frontend
    restart: unless-stopped

networks:
  staging-network:
    driver: bridge

volumes:
  postgres-staging:
  redis-staging:
```

### Nginx Configuration

**File**: `nginx/staging.conf`
```nginx
upstream backend {
    least_conn;
    server backend:8080;
}

upstream websocket {
    server backend:3002;
}

server {
    listen 80;
    server_name staging.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name staging.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend
    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Rate limiting
        limit_req zone=api burst=20 nodelay;
    }

    # WebSocket
    location /ws {
        proxy_pass http://websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/m;
```

## Production Deployment

### Kubernetes Deployment

**Prerequisites**:
- Kubernetes 1.25+
- Helm 3.10+
- kubectl configured

**Namespace**: `visionflow-prod`

#### PostgreSQL StatefulSet

**File**: `k8s/postgres-statefulset.yaml`
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: visionflow-prod
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: visionflow
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        ports:
        - containerPort: 5432
          name: postgres
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: visionflow-prod
spec:
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
  selector:
    app: postgres
```

#### Backend Deployment

**File**: `k8s/backend-deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: visionflow-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: your-registry/visionflow-backend:v2.2.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 3002
          name: websocket
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          value: redis://redis:6379
        - name: RUST_LOG
          value: info
        - name: ENVIRONMENT
          value: production
        resources:
          requests:
            cpu: "2000m"
            memory: "4Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "4000m"
            memory: "8Gi"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: backend
  namespace: visionflow-prod
spec:
  selector:
    app: backend
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: websocket
    port: 3002
    targetPort: 3002
  type: ClusterIP
```

#### Ingress Configuration

**File**: `k8s/ingress.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: visionflow-ingress
  namespace: visionflow-prod
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/websocket-services: "backend"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - app.yourdomain.com
    - api.yourdomain.com
    secretName: visionflow-tls
  rules:
  - host: app.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
  - host: api.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8080
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 3002
```

### Helm Deployment

**Chart Structure**:
```
visionflow-chart/
├── Chart.yaml
├── values.yaml
├── values-prod.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   └── secret.yaml
```

**Install**:
```bash
helm install visionflow ./visionflow-chart \
  -f values-prod.yaml \
  --namespace visionflow-prod \
  --create-namespace
```

## Database Migrations

### Development

```bash
# Run migrations
diesel migration run

# Rollback
diesel migration revert

# Create new migration
diesel migration generate add_new_table
```

### Production

```bash
# Run migrations with backup
pg_dump -h postgres -U visionflow visionflow > backup_$(date +%Y%m%d).sql

# Apply migrations
diesel migration run --database-url ${DATABASE_URL}

# Verify
psql -h postgres -U visionflow -d visionflow -c "\dt"
```

## Monitoring and Logging

### Prometheus Metrics

**ServiceMonitor**:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: backend-metrics
  namespace: visionflow-prod
spec:
  selector:
    matchLabels:
      app: backend
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### Logging with ELK

**Filebeat Configuration**:
```yaml
filebeat.inputs:
- type: container
  paths:
    - /var/log/containers/*visionflow*.log
  processors:
    - add_kubernetes_metadata:
        host: ${NODE_NAME}
        matchers:
        - logs_path:
            logs_path: "/var/log/containers/"

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "visionflow-%{+yyyy.MM.dd}"
```

## Backup and Recovery

### Database Backup

**Automated Backup**:
```bash
#!/bin/bash
# backup-db.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"

# Create backup
pg_dump -h postgres -U visionflow -F c visionflow > "${BACKUP_DIR}/visionflow_${DATE}.dump"

# Compress
gzip "${BACKUP_DIR}/visionflow_${DATE}.dump"

# Clean old backups (keep last 30 days)
find ${BACKUP_DIR} -name "*.dump.gz" -mtime +30 -delete

# Upload to S3
aws s3 cp "${BACKUP_DIR}/visionflow_${DATE}.dump.gz" s3://your-bucket/backups/
```

**Cron Schedule**:
```cron
0 2 * * * /scripts/backup-db.sh
```

### Database Restore

```bash
# Restore from backup
gunzip -c visionflow_20250930.dump.gz | pg_restore -h postgres -U visionflow -d visionflow -c
```

## Security Hardening

### TLS Configuration

**Generate Certificates**:
```bash
# Self-signed for staging
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -subj "/CN=staging.yourdomain.com"

# Production: Use Let's Encrypt with cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Network Policies

**File**: `k8s/network-policy.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
  namespace: visionflow-prod
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 3002
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Troubleshooting

### Common Issues

**Issue**: Backend fails to connect to database
```bash
# Check connectivity
kubectl exec -it backend-pod -- nc -zv postgres 5432

# Verify credentials
kubectl get secret db-credentials -o yaml

# Check logs
kubectl logs -f deployment/backend
```

**Issue**: GPU not available
```bash
# Verify GPU node
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"

# Check device plugin
kubectl get pods -n kube-system | grep nvidia-device-plugin

# Pod GPU allocation
kubectl describe pod backend-pod | grep nvidia.com/gpu
```

**Issue**: High memory usage
```bash
# Check memory metrics
kubectl top pods

# Increase limits if needed
kubectl set resources deployment/backend --limits=memory=16Gi
```

## Performance Tuning

### Database Optimisation

**postgresql.conf**:
```ini
# Connections
max_connections = 200

# Memory
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
work_mem = 20MB

# Query Optimisation
random_page_cost = 1.1  # For SSD
effective_io_concurrency = 200

# Checkpoints
checkpoint_completion_target = 0.9
wal_buffers = 16MB
```

### Redis Optimisation

**redis.conf**:
```ini
maxmemory 2gb
maxmemory-policy allkeys-lru
save ""  # Disable RDB snapshots for cache use
```

### Backend Tuning

**Environment Variables**:
```bash
# Actix worker threads
ACTIX_WORKERS=16

# Database connection pool
DATABASE_POOL_SIZE=20

# GPU batch size
GPU_BATCH_SIZE=100

# WebSocket buffer
WS_BUFFER_SIZE=8192
```

## Health Checks

### Endpoint Verification

```bash
# Health check
curl https://api.yourdomain.com/health

# Readiness check
curl https://api.yourdomain.com/ready

# Metrics
curl https://api.yourdomain.com/metrics
```

### Smoke Tests

```bash
#!/bin/bash
# smoke-test.sh

API_URL="https://api.yourdomain.com"

# Test health endpoint
if curl -f "${API_URL}/health" > /dev/null 2>&1; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    exit 1
fi

# Test API endpoint
if curl -f "${API_URL}/api/graph/data" -H "Authorization: Bearer ${TOKEN}" > /dev/null 2>&1; then
    echo "✓ API check passed"
else
    echo "✗ API check failed"
    exit 1
fi

# Test WebSocket
# ... WebSocket connection test

echo "All smoke tests passed"
```

## Rollback Procedures

### Kubernetes Rollback

```bash
# View deployment history
kubectl rollout history deployment/backend -n visionflow-prod

# Rollback to previous version
kubectl rollout undo deployment/backend -n visionflow-prod

# Rollback to specific revision
kubectl rollout undo deployment/backend --to-revision=3 -n visionflow-prod

# Monitor rollback
kubectl rollout status deployment/backend -n visionflow-prod
```

### Database Rollback

```bash
# Restore from backup
gunzip -c backup_20250930.sql.gz | psql -h postgres -U visionflow visionflow

# Run migration rollback
diesel migration revert
```

## References

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
