# VisionFlow Migration Guide

*Version: 2.0.0 | Last Updated: 2025-09-18*

## Overview

This guide provides step-by-step instructions for migrating existing VisionFlow deployments to the latest version with enhanced multi-agent coordination capabilities, persistent TCP connections, and improved client visualisation.

## 🚨 Breaking Changes

### 1. API Changes
- All swarm operations now require `swarm_id` parameter
- WebSocket protocol updated to v2 with binary format
- Authentication required for all endpoints (previously optional)

### 2. Configuration Changes
- MCP server address uses hostname (`multi-agent-container`) instead of IP
- New required environment variables for connection pooling
- Updated Docker network configuration

### 3. Database Schema
- New `swarms` table for multi-swarm management
- Updated `agents` table with `swarm_id` foreign key
- Additional indices for performance optimisation

## 📋 Pre-Migration Checklist

Before starting the migration:

- [ ] Backup existing database
- [ ] Export current configuration
- [ ] Document active swarms and tasks
- [ ] Schedule maintenance window
- [ ] Notify all connected clients
- [ ] Review new system requirements

## 🔧 Migration Steps

### Step 1: Backup Current System

```bash
# 1. Stop the current system
docker-compose down

# 2. Backup database (if using external database)
pg_dump -h localhost -U visionflow -d visionflow > backup_$(date +%Y%m%d).sql

# 3. Backup configuration
cp -r ./config ./config.backup
cp .env .env.backup

# 4. Backup logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz ./logs
```

### Step 2: Update Environment Configuration

Update your `.env` file with new required variables:

```bash
# Connection pooling
MCP_CONNECTION_POOL_SIZE=10
MCP_CONNECTION_TIMEOUT=30000
MCP_KEEPALIVE_INTERVAL=5000

# Client configuration
CLIENT_POLL_RATE_MS=1000
MAX_SWARMS_PER_CLIENT=5
POSITION_UPDATE_FPS=60

# Swarm management
DEFAULT_SWARM_TOPOLOGY=mesh
MAX_AGENTS_PER_SWARM=200
SWARM_IDLE_TIMEOUT=3600000

# GPU optimisation
GPU_COMPUTE_BATCH_SIZE=64
FORCE_CALCULATION_THREADS=256

# Security
JWT_SECRET=<generate-new-secret>
CORS_ALLOWED_ORIGINS=http://localhost:3001
```

### Step 3: Update Docker Configuration

1. **Pull latest images**:
```bash
docker-compose pull
```

2. **Update docker-compose.yml** (if using custom configuration):
```yaml
version: '3.8'
services:
  visionflow:
    image: visionflow:2.0.0
    environment:
      - MCP_CONNECTION_POOL_SIZE=${MCP_CONNECTION_POOL_SIZE}
      - CLIENT_POLL_RATE_MS=${CLIENT_POLL_RATE_MS}
    networks:
      - visionflow-network
    
  multi-agent-container:
    image: multi-agent-container:2.0.0
    ports:
      - "9500:9500"
    networks:
      - visionflow-network

networks:
  visionflow-network:
    driver: bridge
```

### Step 4: Database Migration

Run the migration script:

```bash
# Using Docker
docker-compose run --rm visionflow cargo run --bin migrate

# Or manually
cargo run --bin migrate -- --database-url $DATABASE_URL
```

The migration will:
- Create new `swarms` table
- Add `swarm_id` to existing agents
- Create performance indices
- Migrate existing data to new schema

### Step 5: Update Client Applications

#### Browser Clients

1. **Clear browser cache**:
   - Chrome: Settings → Privacy → Clear browsing data
   - Firefox: Settings → Privacy & Security → Clear Data
   - Safari: Develop → Empty Caches

2. **Update client configuration**:
```javascript
// Update connection settings
const config = {
  apiUrl: 'https://your-domain/api',
  wsUrl: 'wss://your-domain/wss',
  wsProtocolVersion: 2,
  binaryProtocol: true,
  pollInterval: 1000,
};
```

#### API Clients

Update API calls to include swarm_id:

```javascript
// Old API call
POST /api/agents/spawn
{
  "type": "researcher",
  "capabilities": ["analysis"]
}

// New API call
POST /api/swarms/{swarm_id}/agents/spawn
{
  "type": "researcher",
  "capabilities": ["analysis"]
}
```

### Step 6: Start Updated System

```bash
# Start services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check service health
curl http://localhost:3001/health
curl http://localhost:9501/health

# Monitor logs
docker-compose logs -f
```

### Step 7: Verify Migration

Run the verification script:

```bash
docker-compose run --rm visionflow cargo run --bin verify-migration
```

This will check:
- Database schema integrity
- MCP connection persistence
- WebSocket protocol compatibility
- API endpoint responses
- Client connection handling

## 🔍 Post-Migration Validation

### 1. Connection Verification

```bash
# Test MCP connection persistence
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}' | nc localhost 9500

# Should maintain connection for multiple requests
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | nc localhost 9500
```

### 2. Swarm Management Testing

```bash
# Create test swarm
curl -X POST http://localhost:3001/api/swarms \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"topology": "mesh", "name": "test-swarm"}'

# List swarms
curl http://localhost:3001/api/swarms \
  -H "Authorization: Bearer $TOKEN"
```

### 3. Performance Benchmarks

```bash
# Run performance tests
docker-compose run --rm visionflow cargo bench

# Expected results:
# - Connection latency: < 5ms
# - Position updates: 60 FPS
# - Binary protocol: 85% bandwidth reduction
```

## 🚨 Rollback Procedure

If issues are encountered:

### 1. Immediate Rollback

```bash
# Stop new version
docker-compose down

# Restore previous version
docker-compose -f docker-compose.backup.yml up -d

# Restore database
psql -h localhost -U visionflow -d visionflow < backup_$(date +%Y%m%d).sql

# Restore configuration
cp .env.backup .env
```

### 2. Partial Rollback

For specific component rollback:

```bash
# Rollback only VisionFlow backend
docker-compose stop visionflow
docker-compose rm -f visionflow
docker-compose up -d visionflow
```

## 🐛 Common Issues and Solutions

### Issue 1: Connection Timeouts

**Symptom**: Clients receive connection timeout errors

**Solution**:
```bash
# Increase connection pool size
MCP_CONNECTION_POOL_SIZE=20

# Increase timeout
MCP_CONNECTION_TIMEOUT=60000
```

### Issue 2: WebSocket Protocol Mismatch

**Symptom**: WebSocket connections fail with protocol error

**Solution**:
1. Ensure all clients are updated
2. Clear CDN cache if using one
3. Force protocol version in client:
```javascript
const ws = new WebSocket(wsUrl, [], {
  protocolVersion: 2
});
```

### Issue 3: Performance Degradation

**Symptom**: Lower frame rates or increased latency

**Solution**:
```bash
# Optimise GPU settings
GPU_COMPUTE_BATCH_SIZE=128
FORCE_CALCULATION_THREADS=512

# Reduce polling frequency if needed
CLIENT_POLL_RATE_MS=2000
```

## 📊 Migration Metrics

Track these metrics during migration:

| Metric | Target | Alert Threshold |
|--------|---------|-----------------|
| Connection Success Rate | > 99.9% | < 99% |
| API Response Time | < 100ms | > 200ms |
| WebSocket Latency | < 5ms | > 10ms |
| GPU Utilisation | < 80% | > 90% |
| Memory Usage | < 4GB | > 6GB |
| Error Rate | < 0.1% | > 1% |

## 📚 Additional Resources

- [Implementation Report](../IMPLEMENTATION_REPORT.md)
- [API Documentation](../api/README.md)
- [Architecture Overview](../architecture/overview.md)
- [Troubleshooting Guide](../troubleshooting.md)

## 🆘 Support

If you encounter issues during migration:

1. Check logs: `docker-compose logs -f`
2. Review [GitHub Issues](https://github.com/visionflow/issues)
3. Contact support with:
   - Migration logs
   - Error messages
   - System configuration

---

*Remember to test the migration in a staging environment before applying to production.*