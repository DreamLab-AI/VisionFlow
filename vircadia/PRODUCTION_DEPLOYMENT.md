# Vircadia Production Deployment Guide

## Phase 6: Production Infrastructure & Deployment

### Prerequisites

- Docker 20.10+
- PostgreSQL 14+
- HTTPS/WSS certificates
- CDN for 3D assets
- Load balancer (nginx/HAProxy)

---

## 1. Production Server Setup

### 1.1 Update Docker Compose for Production

Create `/ext/vircadia/server.production.docker.compose.yml`:

```yaml
version: '3.8'

services:
  vircadia_world_postgres:
    image: postgres:14-alpine
    container_name: vircadia_world_postgres_prod
    environment:
      POSTGRES_DB: vircadia_world
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - vircadia_postgres_data_prod:/var/lib/postgresql/data
      - ./server/vircadia-world/migrations:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - vircadia_network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  vircadia_world_api:
    build:
      context: ./server/vircadia-world
      dockerfile: Dockerfile.production
    container_name: vircadia_world_api_prod
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@vircadia_world_postgres:5432/vircadia_world
      NODE_ENV: production
      ENABLE_CORS: "true"
      LOG_LEVEL: "info"
    ports:
      - "3020:3020"
    depends_on:
      - vircadia_world_postgres
    networks:
      - vircadia_network
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  vircadia_state_manager:
    build:
      context: ./server/vircadia-world
      dockerfile: Dockerfile.state
    container_name: vircadia_state_manager_prod
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@vircadia_world_postgres:5432/vircadia_world
      NODE_ENV: production
    ports:
      - "3021:3021"
    depends_on:
      - vircadia_world_postgres
    networks:
      - vircadia_network
    restart: unless-stopped

networks:
  vircadia_network:
    driver: bridge

volumes:
  vircadia_postgres_data_prod:
```

### 1.2 Production Environment Variables

Create `/ext/vircadia/.env.production`:

```bash
# Database
POSTGRES_USER=vircadia_admin
POSTGRES_PASSWORD=<secure-password>
POSTGRES_DB=vircadia_world

# Server
NODE_ENV=production
PORT=3020
STATE_PORT=3021

# SSL/TLS
SSL_ENABLED=true
SSL_CERT_PATH=/etc/ssl/certs/vircadia.crt
SSL_KEY_PATH=/etc/ssl/private/vircadia.key

# CORS
CORS_ORIGIN=https://visionflow.app
ENABLE_CORS=true

# Performance
MAX_CONNECTIONS=1000
QUERY_TIMEOUT_MS=5000
CONNECTION_POOL_SIZE=20

# Monitoring
LOG_LEVEL=info
METRICS_ENABLED=true
SENTRY_DSN=<sentry-dsn>

# CDN
ASSET_CDN_URL=https://cdn.visionflow.app
```

---

## 2. Load Balancer Configuration (nginx)

Create `/etc/nginx/sites-available/vircadia`:

```nginx
upstream vircadia_backend {
    least_conn;
    server localhost:3020;
    server localhost:3021;
    server localhost:3022;
}

upstream vircadia_websocket {
    ip_hash; # Sticky sessions for WebSocket
    server localhost:3020;
    server localhost:3021;
    server localhost:3022;
}

server {
    listen 80;
    server_name vircadia.visionflow.app;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name vircadia.visionflow.app;

    ssl_certificate /etc/ssl/certs/vircadia.crt;
    ssl_certificate_key /etc/ssl/private/vircadia.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # WebSocket endpoint
    location /world/ws {
        proxy_pass http://vircadia_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 86400; # 24 hours
        proxy_send_timeout 86400;
    }

    # REST API
    location /world/api {
        proxy_pass http://vircadia_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check
    location /health {
        proxy_pass http://vircadia_backend;
        access_log off;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/vircadia /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 3. Database Optimization

### 3.1 Create Indexes

```sql
-- Performance indexes
CREATE INDEX idx_entities_entity_name ON entity.entities(general__entity_name);
CREATE INDEX idx_entities_sync_group ON entity.entities(group__sync);
CREATE INDEX idx_entities_created_at ON entity.entities(general__created_at);
CREATE INDEX idx_entities_metadata_type ON entity.entities((meta__data->>'type'));
CREATE INDEX idx_entities_metadata_agentid ON entity.entities((meta__data->>'agentId'));

-- Full-text search for entity names
CREATE INDEX idx_entities_name_trgm ON entity.entities USING gin(general__entity_name gin_trgm_ops);

-- Composite indexes for common queries
CREATE INDEX idx_entities_sync_created ON entity.entities(group__sync, general__created_at DESC);
```

### 3.2 PostgreSQL Configuration

Update `/etc/postgresql/14/main/postgresql.conf`:

```conf
# Memory
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 16MB
maintenance_work_mem = 512MB

# Connections
max_connections = 200
max_worker_processes = 8

# Write-Ahead Log
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 2GB

# Query Planning
random_page_cost = 1.1  # For SSD
effective_io_concurrency = 200

# Monitoring
log_min_duration_statement = 1000  # Log queries > 1 second
log_checkpoints = on
log_connections = on
log_disconnections = on
```

---

## 4. CDN Setup for 3D Assets

### 4.1 Asset Structure

```
cdn.visionflow.app/
├── avatars/
│   ├── default-avatar.glb
│   ├── male-avatar-1.glb
│   └── female-avatar-1.glb
├── models/
│   ├── graph-node.glb
│   └── graph-edge.glb
└── textures/
    ├── skybox/
    └── ui/
```

### 4.2 CloudFront Configuration (AWS)

```json
{
  "DistributionConfig": {
    "Origins": [
      {
        "Id": "vircadia-assets",
        "DomainName": "vircadia-assets.s3.amazonaws.com",
        "S3OriginConfig": {
          "OriginAccessIdentity": ""
        }
      }
    ],
    "DefaultCacheBehavior": {
      "TargetOriginId": "vircadia-assets",
      "ViewerProtocolPolicy": "redirect-to-https",
      "AllowedMethods": ["GET", "HEAD", "OPTIONS"],
      "CachedMethods": ["GET", "HEAD"],
      "Compress": true,
      "DefaultTTL": 86400,
      "MaxTTL": 31536000,
      "MinTTL": 0
    },
    "Enabled": true,
    "PriceClass": "PriceClass_All"
  }
}
```

---

## 5. Monitoring & Logging

### 5.1 Prometheus Metrics

Create `/ext/vircadia/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vircadia-api'
    static_configs:
      - targets: ['localhost:3020', 'localhost:3021', 'localhost:3022']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']
```

### 5.2 Grafana Dashboard

Import dashboard configuration:

```json
{
  "dashboard": {
    "title": "Vircadia Multi-User XR",
    "panels": [
      {
        "title": "Active Users",
        "targets": [
          {
            "expr": "vircadia_active_users"
          }
        ]
      },
      {
        "title": "WebSocket Connections",
        "targets": [
          {
            "expr": "vircadia_websocket_connections"
          }
        ]
      },
      {
        "title": "Query Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, vircadia_query_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Entity Sync Rate",
        "targets": [
          {
            "expr": "rate(vircadia_entity_updates_total[5m])"
          }
        ]
      }
    ]
  }
}
```

---

## 6. Client-Side Production Build

### 6.1 Update Vite Config

Add to `/ext/client/vite.config.ts`:

```typescript
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'babylon': ['@babylonjs/core', '@babylonjs/loaders'],
          'vircadia': [
            './src/services/vircadia/VircadiaClientCore',
            './src/services/vircadia/AvatarManager',
            './src/services/vircadia/SpatialAudioManager'
          ]
        }
      }
    },
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  },
  define: {
    'import.meta.env.VITE_VIRCADIA_SERVER_URL': JSON.stringify(
      'wss://vircadia.visionflow.app/world/ws'
    )
  }
});
```

### 6.2 Production Environment

Create `/ext/client/.env.production`:

```bash
VITE_VIRCADIA_SERVER_URL=wss://vircadia.visionflow.app/world/ws
VITE_VIRCADIA_AUTH_PROVIDER=oauth
VITE_VIRCADIA_ENABLED=true
VITE_VIRCADIA_ENABLE_MULTI_USER=true
VITE_VIRCADIA_ENABLE_SPATIAL_AUDIO=true
VITE_QUEST3_ENABLE_HAND_TRACKING=true
VITE_BABYLON_INSTANCED_RENDERING=true
VITE_BABYLON_ENABLE_LOD=true
```

---

## 7. Deployment Scripts

### 7.1 Production Deployment Script

Create `/ext/vircadia/deploy-production.sh`:

```bash
#!/bin/bash
set -e

echo "=== Vircadia Production Deployment ==="

# Load environment
source .env.production

# Build Docker images
echo "Building Docker images..."
docker-compose -f server.production.docker.compose.yml build

# Run database migrations
echo "Running database migrations..."
docker-compose -f server.production.docker.compose.yml run --rm vircadia_world_api npm run migrate

# Start services
echo "Starting services..."
docker-compose -f server.production.docker.compose.yml up -d

# Wait for health check
echo "Waiting for services to be healthy..."
sleep 10

# Verify deployment
echo "Verifying deployment..."
curl -f https://vircadia.visionflow.app/health || exit 1

echo "Deployment complete!"
echo "Monitor at: https://grafana.visionflow.app"
```

### 7.2 Rollback Script

Create `/ext/vircadia/rollback.sh`:

```bash
#!/bin/bash
set -e

echo "=== Rolling back to previous version ==="

# Get previous version tag
PREVIOUS_VERSION=$(git describe --tags --abbrev=0 HEAD^)

echo "Rolling back to: $PREVIOUS_VERSION"

# Checkout previous version
git checkout $PREVIOUS_VERSION

# Rebuild and redeploy
./deploy-production.sh

echo "Rollback complete to $PREVIOUS_VERSION"
```

---

## 8. Security Hardening

### 8.1 Firewall Rules (UFW)

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (redirect to HTTPS)
sudo ufw allow 443/tcp   # HTTPS/WSS
sudo ufw enable
```

### 8.2 Rate Limiting (nginx)

Add to nginx config:

```nginx
limit_req_zone $binary_remote_addr zone=vircadia_api:10m rate=100r/s;
limit_conn_zone $binary_remote_addr zone=vircadia_conn:10m;

server {
    # ...
    location /world/api {
        limit_req zone=vircadia_api burst=50 nodelay;
        limit_conn vircadia_conn 10;
        # ...
    }
}
```

---

## 9. Backup & Recovery

### 9.1 Automated Database Backups

Create `/ext/vircadia/backup-db.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/var/backups/vircadia"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/vircadia_world_$DATE.sql.gz"

mkdir -p $BACKUP_DIR

docker exec vircadia_world_postgres_prod pg_dump -U vircadia_admin vircadia_world | gzip > $BACKUP_FILE

# Keep only last 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup created: $BACKUP_FILE"
```

Add to crontab:
```bash
0 2 * * * /ext/vircadia/backup-db.sh
```

---

## 10. Feature Flag Rollout Plan

### Week 1: Internal Testing (10%)
```typescript
featureFlags.updateConfig({
  rolloutPercentage: 10,
  allowedUserIds: ['internal-team-ids']
});
```

### Week 2: Beta Users (25%)
```typescript
featureFlags.updateConfig({
  rolloutPercentage: 25,
  allowedUserIds: [] // Open to percentage-based rollout
});
```

### Week 3: General Rollout (50%)
```typescript
featureFlags.updateConfig({
  rolloutPercentage: 50
});
```

### Week 4: Full Deployment (100%)
```typescript
featureFlags.updateConfig({
  rolloutPercentage: 100
});
```

---

## Success Metrics

Monitor these KPIs:

- **Uptime**: 99.9% availability target
- **Latency**: <100ms for position updates
- **Concurrent Users**: Support 100+ simultaneous users
- **Frame Rate**: Maintain 90 FPS on Quest 3
- **Bandwidth**: <5 Mbps per user average
- **Error Rate**: <1% of sessions encounter critical errors

---

## Troubleshooting

### WebSocket Connection Issues
```bash
# Check nginx WebSocket config
sudo nginx -t
sudo tail -f /var/log/nginx/error.log

# Verify backend health
curl https://vircadia.visionflow.app/health
```

### Database Performance
```sql
-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check connection pool
SELECT count(*) FROM pg_stat_activity;
```

### Memory Issues
```bash
# Check Docker container memory
docker stats vircadia_world_api_prod

# Restart with increased memory
docker-compose -f server.production.docker.compose.yml up -d --force-recreate
```

---

**Deployment Complete! 🚀**

Monitor dashboard: https://grafana.visionflow.app
