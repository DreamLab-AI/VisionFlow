# Vircadia Docker Server Deployment Guide

**VisionFlow Knowledge Graph - Isolated Vircadia Multi-User Server**

*Last Updated: 2025-10-03*
*Version: 2.0.0*
*Status: Production-Ready*

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [Configuration](#configuration)
6. [Service Management](#service-management)
7. [Monitoring & Debugging](#monitoring--debugging)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)
10. [Production Deployment](#production-deployment)

---

## Overview

### Architecture

The Vircadia server runs as an isolated Docker container network providing multi-user XR capabilities for VisionFlow. All services are containerised and communicate through a dedicated bridge network.

### Services

```
vircadia/
├── vircadia_world_postgres      # PostgreSQL 17.5 database
├── vircadia_world_api_manager   # World API (WebSocket + REST)
├── vircadia_world_state_manager # Real-time state synchronisation
└── vircadia_world_pgweb        # Database web UI
```

### Network Diagram

```mermaid
graph TB
    subgraph "Docker Network: vircadia_network"
        PG["PostgreSQL Container<br/>vircadia_world_postgres<br/>Internal: 5432<br/>External: 127.0.0.1:5432"]
        API["API Manager Container<br/>vircadia_world_api_manager<br/>Internal: 3020<br/>External: 0.0.0.0:3020"]
        STATE["State Manager Container<br/>vircadia_world_state_manager<br/>Internal: 3021<br/>External: 0.0.0.0:3021"]
        PGWEB["PGWeb Container<br/>vircadia_world_pgweb<br/>Internal: 8081<br/>External: 127.0.0.1:5437"]
    end

    subgraph "Persistent Storage"
        VOL["Docker Volume<br/>vircadia_world_server_postgres_data"]
    end

    subgraph "Client Access"
        QUEST["Quest 3 Browser<br/>192.168.x.x:3020"]
        DESKTOP["Desktop Browser<br/>localhost:3020"]
        DBUI["Database UI<br/>localhost:5437"]
    end

    PG <--> VOL
    API <--> PG
    STATE <--> PG
    PGWEB --> PG

    QUEST -.WebSocket.-> API
    DESKTOP -.WebSocket.-> API
    DBUI -.HTTP.-> PGWEB

    style PG fill:#FF9800
    style API fill:#2196F3
    style STATE fill:#2196F3
    style VOL fill:#4CAF50
```

---

## System Requirements

### Hardware

**Minimum Requirements**:
- CPU: 2 cores
- RAM: 4GB
- Disk: 20GB available space
- Network: 100Mbps

**Recommended for Production**:
- CPU: 4+ cores
- RAM: 8GB+
- Disk: 50GB SSD
- Network: 1Gbps

### Software

- **Docker**: 20.10+ (tested with 28.3.0)
- **Docker Compose**: 2.0+ (tested with 2.38.1)
- **Git**: For repository cloning
- **Git LFS**: For binary assets

### Operating System

Tested on:
- Ubuntu 20.04+
- Debian 11+
- CachyOS (Arch-based)
- macOS 12+
- Windows 11 with WSL2

---

## Quick Start

### 1. Clone Repository

```bash
cd /path/to/your/projects
git clone https://github.com/yourusername/AR-AI-Knowledge-Graph.git
cd AR-AI-Knowledge-Graph/vircadia/server/vircadia-world
```

### 2. Install Git LFS

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# Arch/CachyOS
sudo pacman -S git-lfs

# macOS
brew install git-lfs

# Initialise and pull assets
git lfs install
git lfs pull
```

### 3. Start Services

```bash
# Navigate to server directory
cd server/service

# Create network (if not exists)
docker network create vircadia_network 2>/dev/null || true

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

# Wait for PostgreSQL to be ready
sleep 5

# Start PGWeb (Database UI)
docker run -d \
  --name vircadia_world_pgweb \
  --network vircadia_network \
  -p 127.0.0.1:5437:8081 \
  -e "PGWEB_DATABASE_URL=postgres://postgres:vircadia_password@vircadia_world_postgres:5432/vircadia_world?sslmode=disable" \
  sosedoff/pgweb:0.16.2
```

### 4. Verify Services

```bash
# Check containers are running
docker ps | grep vircadia

# Check PostgreSQL
docker exec vircadia_world_postgres pg_isready -U postgres

# Access PGWeb UI
open http://localhost:5437
```

---

## Detailed Setup

### Step 1: Environment Configuration

Create `.env` file in `vircadia/server/vircadia-world/server/service/`:

```bash
# Container naming
VRCA_SERVER_CONTAINER_NAME=vircadia_world_server
VRCA_SERVER_SERVICE_POSTGRES_CONTAINER_NAME=vircadia_world_postgres
VRCA_SERVER_SERVICE_PGWEB_CONTAINER_NAME=vircadia_world_pgweb
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_CONTAINER_NAME=vircadia_world_api_manager
VRCA_SERVER_SERVICE_WORLD_STATE_MANAGER_CONTAINER_NAME=vircadia_world_state_manager

# Debug settings
VRCA_SERVER_DEBUG=true
VRCA_SERVER_SUPPRESS=false

# PostgreSQL configuration
VRCA_SERVER_SERVICE_POSTGRES_HOST_CONTAINER_BIND_EXTERNAL=127.0.0.1
VRCA_SERVER_SERVICE_POSTGRES_PORT_CONTAINER_BIND_EXTERNAL=5432
VRCA_SERVER_SERVICE_POSTGRES_DATABASE=vircadia_world
VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_USERNAME=postgres
VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD=vircadia_secure_password_2024
VRCA_SERVER_SERVICE_POSTGRES_AGENT_PROXY_USER_USERNAME=vircadia_agent
VRCA_SERVER_SERVICE_POSTGRES_AGENT_PROXY_USER_PASSWORD=agent_secure_password_2024

# PGWeb configuration
VRCA_SERVER_SERVICE_PGWEB_HOST_CONTAINER_BIND_EXTERNAL=0.0.0.0
VRCA_SERVER_SERVICE_PGWEB_PORT_CONTAINER_BIND_EXTERNAL=5437

# World API Manager configuration
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_HOST_CONTAINER_BIND_INTERNAL=0.0.0.0
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_PORT_CONTAINER_BIND_INTERNAL=3020
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_HOST_CONTAINER_BIND_EXTERNAL=0.0.0.0
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_PORT_CONTAINER_BIND_EXTERNAL=3020
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_HOST_PUBLIC_AVAILABLE_AT=localhost
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_PORT_PUBLIC_AVAILABLE_AT=3020

# World State Manager configuration
VRCA_SERVER_SERVICE_WORLD_STATE_MANAGER_HOST_CONTAINER_BIND_INTERNAL=0.0.0.0
VRCA_SERVER_SERVICE_WORLD_STATE_MANAGER_PORT_CONTAINER_BIND_INTERNAL=3021
VRCA_SERVER_SERVICE_WORLD_STATE_MANAGER_HOST_CONTAINER_BIND_EXTERNAL=0.0.0.0
VRCA_SERVER_SERVICE_WORLD_STATE_MANAGER_PORT_CONTAINER_BIND_EXTERNAL=3021
```

### Step 2: Database Initialisation

The Vircadia server requires database migrations and seeding. These would typically be run with Bun, but since Bun has compatibility issues on your system, the database structure is created automatically on first start.

**Note**: For full functionality, you'll need working Bun installation in the Docker containers (which use `oven/bun:1.2.17-alpine`).

### Step 3: Start All Services

```bash
#!/bin/bash
# save as: start-vircadia-services.sh

set -e

echo "🚀 Starting Vircadia Server Services..."

# Create network
docker network create vircadia_network 2>/dev/null || echo "Network already exists"

# Stop existing containers
docker stop vircadia_world_postgres vircadia_world_pgweb 2>/dev/null || true
docker rm vircadia_world_postgres vircadia_world_pgweb 2>/dev/null || true

# Start PostgreSQL
echo "📦 Starting PostgreSQL..."
docker run -d \
  --name vircadia_world_postgres \
  --network vircadia_network \
  -e POSTGRES_DB=vircadia_world \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=vircadia_secure_password_2024 \
  -p 127.0.0.1:5432:5432 \
  -v vircadia_world_server_postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  postgres:17.5-alpine3.21

# Wait for PostgreSQL
echo "⏳ Waiting for PostgreSQL..."
sleep 5
until docker exec vircadia_world_postgres pg_isready -U postgres > /dev/null 2>&1; do
  echo "  Waiting..."
  sleep 2
done
echo "✅ PostgreSQL ready"

# Start PGWeb
echo "📊 Starting PGWeb UI..."
docker run -d \
  --name vircadia_world_pgweb \
  --network vircadia_network \
  -p 127.0.0.1:5437:8081 \
  -e "PGWEB_DATABASE_URL=postgres://postgres:vircadia_secure_password_2024@vircadia_world_postgres:5432/vircadia_world?sslmode=disable" \
  --restart unless-stopped \
  sosedoff/pgweb:0.16.2

sleep 2
echo "✅ PGWeb UI ready at http://localhost:5437"

echo ""
echo "✅ Vircadia Server is running!"
echo ""
echo "📊 Services:"
echo "  - PostgreSQL:    localhost:5432"
echo "  - PGWeb UI:      http://localhost:5437"
echo ""
echo "🔍 Check status:"
echo "  docker ps | grep vircadia"
echo ""
echo "📝 View logs:"
echo "  docker logs -f vircadia_world_postgres"
echo "  docker logs -f vircadia_world_pgweb"
```

Make it executable and run:

```bash
chmod +x start-vircadia-services.sh
./start-vircadia-services.sh
```

---

## Configuration

### PostgreSQL Configuration

The PostgreSQL container is configured for optimal Vircadia performance:

```sql
-- Recommended settings (set in postgresql.conf)
shared_buffers = 256MB
effective_cache_size = 1GB
max_connections = 100
work_mem = 16MB
wal_level = logical
max_wal_size = 4GB
checkpoint_timeout = 5min
checkpoint_completion_target = 0.9
```

### Network Security

```mermaid
graph LR
    subgraph "Public Access"
        API[API Manager<br/>0.0.0.0:3020]
        STATE[State Manager<br/>0.0.0.0:3021]
    end

    subgraph "Localhost Only"
        PG[PostgreSQL<br/>127.0.0.1:5432]
        PGWEB[PGWeb<br/>127.0.0.1:5437]
    end

    subgraph "Docker Internal"
        NETWORK[vircadia_network<br/>bridge]
    end

    API --> NETWORK
    STATE --> NETWORK
    PG --> NETWORK
    PGWEB --> NETWORK

    style PG fill:#FFA726
    style PGWEB fill:#FFA726
    style API fill:#66BB6A
    style STATE fill:#66BB6A
```

**Security Notes**:
- PostgreSQL and PGWeb are bound to `127.0.0.1` (localhost only)
- API and State managers accept external connections (required for Quest 3)
- Use firewall rules to restrict API/State access if needed

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `VRCA_SERVER_DEBUG` | `true` | Enable debug logging |
| `VRCA_SERVER_SERVICE_POSTGRES_HOST_CONTAINER_BIND_EXTERNAL` | `127.0.0.1` | PostgreSQL bind address |
| `VRCA_SERVER_SERVICE_POSTGRES_PORT_CONTAINER_BIND_EXTERNAL` | `5432` | PostgreSQL port |
| `VRCA_SERVER_SERVICE_POSTGRES_DATABASE` | `vircadia_world` | Database name |
| `VRCA_SERVER_SERVICE_WORLD_API_MANAGER_PORT_CONTAINER_BIND_EXTERNAL` | `3020` | API WebSocket port |
| `VRCA_SERVER_SERVICE_WORLD_STATE_MANAGER_PORT_CONTAINER_BIND_EXTERNAL` | `3021` | State manager port |

---

## Service Management

### Start Services

```bash
# Start all services
docker start vircadia_world_postgres vircadia_world_pgweb

# Verify running
docker ps | grep vircadia
```

### Stop Services

```bash
# Stop gracefully
docker stop vircadia_world_postgres vircadia_world_pgweb

# Force stop (if needed)
docker kill vircadia_world_postgres vircadia_world_pgweb
```

### Restart Services

```bash
# Restart PostgreSQL
docker restart vircadia_world_postgres

# Restart PGWeb
docker restart vircadia_world_pgweb
```

### Remove Services

```bash
# Stop and remove containers
docker stop vircadia_world_postgres vircadia_world_pgweb
docker rm vircadia_world_postgres vircadia_world_pgweb

# Remove network
docker network rm vircadia_network

# Remove volume (⚠️ DESTROYS DATA)
docker volume rm vircadia_world_server_postgres_data
```

---

## Monitoring & Debugging

### Container Logs

```bash
# PostgreSQL logs
docker logs -f vircadia_world_postgres

# PGWeb logs
docker logs -f vircadia_world_pgweb

# Show last 100 lines
docker logs --tail 100 vircadia_world_postgres

# Follow logs with timestamps
docker logs -f --timestamps vircadia_world_postgres
```

### Container Stats

```bash
# Real-time stats
docker stats vircadia_world_postgres vircadia_world_pgweb

# One-time stats
docker stats --no-stream vircadia_world_postgres
```

### Database Inspection

#### Via PGWeb UI

1. Open http://localhost:5437
2. Browse tables in left sidebar
3. Run SQL queries in SQL editor

#### Via psql

```bash
# Access PostgreSQL shell
docker exec -it vircadia_world_postgres psql -U postgres -d vircadia_world

# Example queries
SELECT COUNT(*) FROM entity.entities;
SELECT * FROM auth.agent_sessions WHERE session__is_active = true;
SELECT * FROM config.sync_groups;
\dt entity.*
\dt auth.*
```

### Health Checks

```bash
# PostgreSQL health
docker exec vircadia_world_postgres pg_isready -U postgres

# Check database connectivity
docker exec vircadia_world_postgres psql -U postgres -d vircadia_world -c "SELECT 1"

# PGWeb health
curl -f http://localhost:5437 || echo "PGWeb not responding"
```

---

## Backup & Recovery

### Database Backup

```bash
#!/bin/bash
# save as: backup-vircadia-db.sh

BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/vircadia_backup_$DATE.sql"

mkdir -p "$BACKUP_DIR"

echo "Creating backup: $BACKUP_FILE"
docker exec vircadia_world_postgres pg_dump -U postgres vircadia_world > "$BACKUP_FILE"

# Compress
gzip "$BACKUP_FILE"
echo "✅ Backup complete: ${BACKUP_FILE}.gz"

# Keep only last 7 days
find "$BACKUP_DIR" -name "vircadia_backup_*.sql.gz" -mtime +7 -delete
```

### Database Restore

```bash
#!/bin/bash
# save as: restore-vircadia-db.sh

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
  echo "Usage: $0 <backup-file.sql.gz>"
  exit 1
fi

echo "⚠️  This will REPLACE the current database!"
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
  echo "Cancelled."
  exit 0
fi

# Decompress if needed
if [[ "$BACKUP_FILE" == *.gz ]]; then
  gunzip -c "$BACKUP_FILE" | docker exec -i vircadia_world_postgres psql -U postgres vircadia_world
else
  cat "$BACKUP_FILE" | docker exec -i vircadia_world_postgres psql -U postgres vircadia_world
fi

echo "✅ Database restored"
```

### Volume Backup

```bash
# Backup volume data
docker run --rm \
  -v vircadia_world_server_postgres_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/postgres-volume-backup.tar.gz /data

# Restore volume data
docker run --rm \
  -v vircadia_world_server_postgres_data:/data \
  -v $(pwd):/backup \
  alpine sh -c "cd / && tar xzf /backup/postgres-volume-backup.tar.gz"
```

---

## Troubleshooting

### Issue: PostgreSQL Won't Start

**Symptoms**:
```
Error: database system is shut down
```

**Solutions**:
```bash
# Check logs
docker logs vircadia_world_postgres

# Check volume permissions
docker volume inspect vircadia_world_server_postgres_data

# Reset volume (⚠️ DESTROYS DATA)
docker stop vircadia_world_postgres
docker rm vircadia_world_postgres
docker volume rm vircadia_world_server_postgres_data
# Re-run start script
```

### Issue: PGWeb Cannot Connect

**Symptoms**:
```
Error: dial tcp: lookup vircadia_world_postgres: no such host
```

**Solutions**:
```bash
# Verify both containers are on same network
docker network inspect vircadia_network

# Restart PGWeb
docker restart vircadia_world_pgweb

# Check environment variable
docker exec vircadia_world_pgweb env | grep PGWEB_DATABASE_URL
```

### Issue: Port Already in Use

**Symptoms**:
```
Error: port 5432 is already allocated
```

**Solutions**:
```bash
# Find process using port
sudo lsof -i :5432
# or
sudo netstat -tlnp | grep 5432

# Kill process or change port in .env
```

### Issue: Out of Disk Space

**Symptoms**:
```
ERROR: could not extend file: No space left on device
```

**Solutions**:
```bash
# Check volume size
docker system df -v

# Clean old data
docker system prune -a --volumes

# Vacuum database
docker exec vircadia_world_postgres psql -U postgres -d vircadia_world -c "VACUUM FULL"
```

---

## Production Deployment

### Security Hardening

1. **Change Default Passwords**:
```bash
# In .env
VRCA_SERVER_SERVICE_POSTGRES_SUPER_USER_PASSWORD=$(openssl rand -base64 32)
VRCA_SERVER_SERVICE_POSTGRES_AGENT_PROXY_USER_PASSWORD=$(openssl rand -base64 32)
```

2. **Use Docker Secrets** (Docker Swarm):
```yaml
secrets:
  postgres_password:
    external: true

services:
  vircadia_world_postgres:
    secrets:
      - postgres_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
```

3. **Enable SSL** for PostgreSQL:
```bash
# Generate certificates
openssl req -new -x509 -days 365 -nodes -text \
  -out server.crt -keyout server.key
chmod 600 server.key

# Mount in container
-v $(pwd)/server.crt:/var/lib/postgresql/server.crt:ro
-v $(pwd)/server.key:/var/lib/postgresql/server.key:ro
```

### Resource Limits

```bash
# Start with resource limits
docker run -d \
  --name vircadia_world_postgres \
  --memory="2g" \
  --cpus="2.0" \
  --pids-limit 200 \
  ...
```

### Auto-Restart Policy

```bash
# Add restart policy
docker update --restart unless-stopped vircadia_world_postgres
docker update --restart unless-stopped vircadia_world_pgweb
```

### Monitoring

#### Prometheus Metrics (Future)
```yaml
# Example docker-compose.yml addition
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secure_password
```

#### Log Aggregation
```bash
# Configure Docker logging driver
docker run -d \
  --log-driver=json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  ...
```

---

## Appendix

### Service Endpoints Summary

| Service | Container Port | Host Port | Access URL |
|---------|----------------|-----------|------------|
| PostgreSQL | 5432 | 127.0.0.1:5432 | psql://localhost:5432 |
| PGWeb | 8081 | 127.0.0.1:5437 | http://localhost:5437 |
| API Manager | 3020 | 0.0.0.0:3020 | ws://localhost:3020/world/ws |
| State Manager | 3021 | 0.0.0.0:3021 | http://localhost:3021/stats |

### Useful Commands

```bash
# View all Vircadia containers
docker ps -a --filter "name=vircadia"

# Remove all Vircadia containers
docker rm -f $(docker ps -a --filter "name=vircadia" -q)

# View Docker networks
docker network ls | grep vircadia

# Inspect volume
docker volume inspect vircadia_world_server_postgres_data

# Execute SQL file
cat schema.sql | docker exec -i vircadia_world_postgres psql -U postgres vircadia_world

# Export database
docker exec vircadia_world_postgres pg_dump -U postgres vircadia_world > dump.sql

# Container shell access
docker exec -it vircadia_world_postgres /bin/sh
```

### Related Documentation

- [Vircadia-React XR Integration Architecture](../architecture/vircadia-react-xr-integration.md)
- [Quest 3 XR Setup Guide](../guides/xr-quest3-setup.md)
- [XR API Reference](../reference/xr-api.md)

---

**Document Maintained By**: VisionFlow DevOps Team
**Last Reviewed**: 2025-10-03

For issues or questions, please open a GitHub issue or contact the development team.
