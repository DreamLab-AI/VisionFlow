# VisionFlow Unified Docker Setup Guide

**Date**: 2025-11-06
**Status**: Ready for deployment

---

## Quick Start

### 1. Check if Neo4j Already Exists

```bash
# Run the check script
./check-neo4j.sh
```

If Neo4j is found in the `docker_ragflow` network, update your `.env`:
```env
NEO4J_URI=bolt://[neo4j-container-ip]:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

If Neo4j is **NOT** found, proceed to step 2.

---

### 2. Use Updated Docker Compose (Includes Neo4j)

Replace your current docker-compose:
```bash
# Backup original
cp docker-compose.unified.yml docker-compose.unified.yml.backup

# Use the version with Neo4j
cp docker-compose.unified-with-neo4j.yml docker-compose.unified.yml
```

---

### 3. Configure Environment Variables

Update your `.env` file:
```bash
# Copy from example if you don't have one
cp .env.example .env

# Edit with your actual values
nano .env
```

**Required Neo4j settings**:
```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YOUR_SECURE_PASSWORD_HERE
NEO4J_DATABASE=neo4j
```

**⚠️ IMPORTANT**: Change `NEO4J_PASSWORD` to a secure password!

---

### 4. Start the Stack

```bash
# Development profile (includes Nginx on port 3001)
docker-compose --profile dev up -d

# Or production profile
docker-compose --profile prod up -d
```

---

### 5. Verify Services

```bash
# Check all containers are running
docker-compose ps

# Check Neo4j is healthy
docker logs visionflow-neo4j

# Check backend startup
docker logs visionflow_container | grep -i neo4j
```

**Expected output**:
```
✅ Connected to Neo4j at bolt://neo4j:7687
✅ Neo4jSettingsRepository initialized successfully
✅ Starting HTTP server on 0.0.0.0:4000
```

---

### 6. Access Services

- **Frontend**: http://192.168.0.51:3001 (development only)
- **Backend API**: http://192.168.0.51:4000
- **Neo4j Browser**: http://192.168.0.51:7474
  - Username: `neo4j`
  - Password: Your `NEO4J_PASSWORD` from `.env`

---

## Testing Graph Sync

### Trigger Synchronization

```bash
curl -X POST http://192.168.0.51:3001/api/admin/sync/streaming
```

### Monitor Progress

```bash
# Watch sync logs
docker logs -f visionflow_container | grep StreamingSync
```

**Expected output**:
```
🚀 Starting streaming GitHub sync with 8 workers
📁 Found XX markdown files in repository
✅ KG file PageName.md: N nodes, M edges
✅ Ontology file OntologyName.md: X classes, Y properties, Z axioms
🎉 Streaming GitHub sync complete
```

### Verify Graph Data

Access Neo4j Browser at http://192.168.0.51:7474:

```cypher
// Count all nodes
MATCH (n) RETURN labels(n) AS type, count(n) AS count;

// Check ontology classes
MATCH (c:OwlClass) RETURN c.iri, c.label LIMIT 10;

// Check knowledge graph nodes
MATCH (n:Node) WHERE n.public = "true" RETURN n.metadata_id LIMIT 10;
```

---

## Troubleshooting

### 502 Bad Gateway Error

**Cause**: Backend failed to start, usually due to Neo4j connection issues.

**Check**:
```bash
docker logs visionflow_container | grep -i "neo4j\|error"
```

**Common errors**:

1. **"Failed to create Neo4j settings repository"**
   - Neo4j is not running or not accessible
   - Check: `docker ps | grep neo4j`
   - Fix: Ensure Neo4j container is healthy

2. **"Connection refused"**
   - Wrong `NEO4J_URI` in `.env`
   - Fix: Use `bolt://neo4j:7687` for unified docker

3. **"Authentication failed"**
   - Wrong password in `.env`
   - Fix: Check `NEO4J_PASSWORD` matches container

---

### Graph Sync Returns 404 Errors

**Cause**: Fixed in commit `3993a0c` - GitHub API URL encoding bug

**Verify fix**:
```bash
docker logs visionflow_container | grep "get_contents_url"
```

**Should see**:
```
✓ Final GitHub API URL: '.../contents/ontologies/subfolder/File.md'
```

**Should NOT see**:
```
✗ Final GitHub API URL: '.../contents/ontologies%2Fsubfolder%2FFile.md'
```

---

### Ontology Files Return 0 Classes

**Cause**: Fixed in commit `3993a0c` - Inconsistent marker check

**Verify fix**:
```bash
docker logs visionflow_container | grep "Parsed.*OntologyBlock"
```

**Should see**:
```
✓ Parsed MyOntology.md in XXms: N classes, M properties, P axioms
```

**Should NOT see all zeros**:
```
✗ Parsed MyOntology.md in XXms: 0 classes, 0 properties, 0 axioms
```

---

### Private Repo Returns 401/403 Errors

**Cause**: Fixed in commit `3993a0c` - Missing authentication

**Verify fix**:
```bash
docker logs visionflow_container | grep "fetch_with_retry"
```

**Should see**:
```
✓ Successfully fetched XXXX bytes
```

**Should NOT see**:
```
✗ Failed to fetch: 401 Unauthorized
✗ Failed to fetch: 403 Forbidden
```

---

## Configuration Options

### Neo4j Memory Settings

Adjust in `.env` for your hardware:
```env
# Small server (8GB RAM)
NEO4J_PAGECACHE_SIZE=256M
NEO4J_HEAP_INIT=256M
NEO4J_HEAP_MAX=512M

# Medium server (16GB RAM)
NEO4J_PAGECACHE_SIZE=512M
NEO4J_HEAP_INIT=512M
NEO4J_HEAP_MAX=1G

# Large server (32GB+ RAM)
NEO4J_PAGECACHE_SIZE=2G
NEO4J_HEAP_INIT=1G
NEO4J_HEAP_MAX=4G
```

### Worker Count

Control sync parallelism via backend environment (future feature):
```env
GITHUB_SYNC_WORKERS=8  # 4-8 recommended
```

---

## Profiles

The unified docker-compose supports two profiles:

### Development (`--profile dev`)
- Includes Nginx on port 3001
- Includes Vite dev server on port 5173
- Hot reload enabled
- Debug logging
- Direct API access on port 4000

### Production (`--profile prod`)
- Minimal runtime
- Only port 4000 exposed
- Optimized binary
- Warn-level logging
- No source mounts

---

## Backup and Restore

### Backup Neo4j Data

```bash
# Stop VisionFlow (but keep Neo4j running)
docker stop visionflow_container

# Create backup
docker exec visionflow-neo4j neo4j-admin database dump neo4j \
  --to-path=/var/lib/neo4j/data/dumps

# Copy backup to host
docker cp visionflow-neo4j:/var/lib/neo4j/data/dumps/neo4j.dump ./backup-$(date +%Y%m%d).dump
```

### Restore Neo4j Data

```bash
# Stop both containers
docker-compose --profile dev down

# Start only Neo4j
docker-compose up -d neo4j

# Wait for healthy
docker exec visionflow-neo4j cypher-shell -u neo4j -p YOUR_PASSWORD "RETURN 1"

# Load backup
docker cp ./backup-20251106.dump visionflow-neo4j:/tmp/restore.dump
docker exec visionflow-neo4j neo4j-admin database load neo4j --from-path=/tmp

# Start VisionFlow
docker-compose --profile dev up -d visionflow
```

---

## Network Configuration

### External Network: docker_ragflow

This network is shared with other services (e.g., RAGFlow). Create if missing:

```bash
docker network create docker_ragflow
```

### Check Network Connectivity

```bash
# List all containers in network
docker network inspect docker_ragflow

# Test connectivity from VisionFlow to Neo4j
docker exec visionflow_container nc -zv neo4j 7687
```

---

## Performance Tuning

### Neo4j Indexes (Recommended)

After first sync, create indexes for better performance:

```cypher
// Knowledge graph indexes
CREATE INDEX node_metadata_id IF NOT EXISTS FOR (n:Node) ON (n.metadata_id);
CREATE INDEX node_public IF NOT EXISTS FOR (n:Node) ON (n.public);

// Ontology indexes
CREATE INDEX owl_class_iri IF NOT EXISTS FOR (c:OwlClass) ON (c.iri);
CREATE INDEX owl_property_iri IF NOT EXISTS FOR (p:OwlProperty) ON (p.iri);

// Show indexes
SHOW INDEXES;
```

### Database Constraints

```cypher
// Ensure unique IRIs
CREATE CONSTRAINT owl_class_iri_unique IF NOT EXISTS
FOR (c:OwlClass) REQUIRE c.iri IS UNIQUE;

CREATE CONSTRAINT owl_property_iri_unique IF NOT EXISTS
FOR (p:OwlProperty) REQUIRE p.iri IS UNIQUE;
```

---

## Related Documentation

- **502 Error Diagnosis**: `502_ERROR_DIAGNOSIS.md`
- **Graph Sync Fixes**: `GRAPH_SYNC_FIXES.md`
- **Implementation Status**: `docs/reference/implementation-status.md`
- **Neo4j Migration**: `docs/guides/neo4j-migration.md`

---

## Support

### Check Logs

```bash
# VisionFlow backend
docker logs visionflow_container

# Neo4j
docker logs visionflow-neo4j

# Both together
docker-compose logs -f
```

### Common Log Locations

- VisionFlow logs: `/app/logs/` (inside container)
- Neo4j logs: `/logs/` (inside container)
- Nginx logs: `/var/log/nginx/` (inside container)

Access container:
```bash
docker exec -it visionflow_container bash
```

---

**Document Version**: 1.0
**Created**: 2025-11-06
**Deployment**: Unified Docker with Neo4j
**Status**: ✅ Ready for use
