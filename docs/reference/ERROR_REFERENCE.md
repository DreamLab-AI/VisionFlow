---
title: Error Reference and Troubleshooting
description: Complete error code reference with solutions, troubleshooting guides, and diagnostic procedures
category: reference
tags:
  - api
  - api
  - docker
updated-date: 2025-12-18
difficulty-level: intermediate
version: 2.0
---


# Error Reference and Troubleshooting

**Version**: 2.0
**Last Updated**: December 18, 2025

Complete reference for all VisionFlow error codes, common issues, and troubleshooting procedures.

---

## Table of Contents

1. [Error Code System](#error-code-system)
2. [API Layer Errors](#api-layer-errors)
3. [Database Layer Errors](#database-layer-errors)
4. [Graph/Ontology Errors](#graphontology-errors)
5. [GPU/Physics Errors](#gpuphysics-errors)
6. [WebSocket Errors](#websocket-errors)
7. [Common Issues & Solutions](#common-issues--solutions)
8. [Diagnostic Procedures](#diagnostic-procedures)

---

## Error Code System

### Format Pattern

**Structure**: `[SYSTEM][SEVERITY][NUMBER]`

```
AP-E-001
│  │ └── Number (000-999)
│  └──── Severity (E/F/W/I)
└─────── System (2-char)
```

### System Identifiers

| Code | System | Description |
|------|--------|-------------|
| `AP` | API/Application | REST API, HTTP handlers |
| `DB` | Database | SQLite, PostgreSQL, Redis |
| `GR` | Graph/Reasoning | Ontology, inference engine |
| `GP` | GPU/Physics | CUDA kernels, physics simulation |
| `WS` | WebSocket | Real-time communication |
| `AU` | Authentication | JWT, Nostr, OAuth |
| `ST` | Storage | File management, S3 |

### Severity Levels

| Level | Code | Meaning | Action Required |
|-------|------|---------|-----------------|
| **Error** | `E` | Operation failed, recoverable | Retry or fix input |
| **Fatal** | `F` | Unrecoverable failure | Restart service |
| **Warning** | `W` | Degraded performance | Monitor and optimize |
| **Info** | `I` | Informational only | None |

---

## API Layer Errors

### AP-E-001 to AP-E-099: Request Validation

#### AP-E-001: Invalid Request Format

**Message**: `Request body is malformed JSON`

**Cause**: Invalid JSON syntax in request body.

**Solution**:
```bash
# Validate JSON before sending
echo '{"key": "value"}' | jq .

# Correct malformed JSON
curl -X POST http://localhost:9090/api/graph/update \
  -H "Content-Type: application/json" \
  -d '{"source": "valid-json"}'  # ✓ Valid
```

**Example Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "AP-E-001",
    "message": "Invalid request format",
    "details": [
      {
        "line": 3,
        "column": 15,
        "message": "Expected ',' or '}' after property value"
      }
    ]
  }
}
```

---

#### AP-E-002: Missing Required Field

**Message**: `Required field '{field}' is missing`

**Cause**: Request missing mandatory field.

**Solution**:
```json
// ✗ Missing 'source' field
{
  "format": "rdf/xml"
}

// ✓ Include all required fields
{
  "source": "https://example.org/ontology.owl",
  "format": "rdf/xml"
}
```

---

#### AP-E-010: Duplicate Value

**Message**: `Value '{value}' already exists for {field}`

**Cause**: Attempting to create resource with duplicate unique identifier.

**Solution**:
1. Check existing resources: `GET /api/resource`
2. Use unique identifier or delete existing resource
3. Use `PUT` for updates instead of `POST`

---

### AP-E-100 to AP-E-199: Authentication/Authorization

#### AP-E-101: Missing Auth Token

**Message**: `Authorization header missing`

**Cause**: No JWT token provided in request.

**Solution**:
```bash
# Obtain token first
TOKEN=$(curl -X POST http://localhost:9090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"pass"}' \
  | jq -r '.data.token')

# Use token in subsequent requests
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:9090/api/graph/data
```

---

#### AP-E-102: Invalid Token

**Message**: `Token is invalid or expired`

**Cause**: JWT token is malformed, expired, or has invalid signature.

**Solution**:
```bash
# Check token expiration
echo $TOKEN | cut -d'.' -f2 | base64 -d | jq '.exp'

# Refresh token if expired
curl -X POST http://localhost:9090/api/auth/refresh \
  -H "Authorization: Bearer $OLD_TOKEN"

# Or re-authenticate
curl -X POST http://localhost:9090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"pass"}'
```

---

#### AP-E-104: Insufficient Permissions

**Message**: `User lacks permission for {action}`

**Cause**: User role does not have required permissions.

**Solution**:
1. Check user role: `GET /api/auth/me`
2. Request elevated permissions from administrator
3. Use account with appropriate role (admin, editor, viewer)

**Permission Matrix**:

| Action | Viewer | Editor | Admin |
|--------|--------|--------|-------|
| Read Graph | ✓ | ✓ | ✓ |
| Update Graph | ✗ | ✓ | ✓ |
| Manage Users | ✗ | ✗ | ✓ |
| System Config | ✗ | ✗ | ✓ |

---

### AP-E-200 to AP-E-299: Resource Not Found

#### AP-E-201: Resource Not Found

**Message**: `{resource} with ID '{id}' not found`

**Cause**: Requested resource does not exist in database.

**Solution**:
```bash
# Verify resource exists
curl http://localhost:9090/api/graph/nodes/$NODE_ID

# List available resources
curl http://localhost:9090/api/graph/data

# Check for typos in ID
```

---

### AP-E-300 to AP-E-399: Business Logic Errors

#### AP-E-305: Rate Limit Exceeded

**Message**: `Too many requests, retry after {seconds}s`

**Cause**: Exceeded rate limit for endpoint.

**Solution**:
```bash
# Check rate limit headers
curl -I http://localhost:9090/api/health

# Wait for reset time
sleep 60

# Implement exponential backoff
for i in {1..5}; do
  curl http://localhost:9090/api/endpoint && break
  sleep $((2**i))
done
```

**Rate Limits** (see [API_REFERENCE.md](./API_REFERENCE.md#rate-limiting))

---

## Database Layer Errors

### DB-E-001 to DB-E-099: Connection Errors

#### DB-E-001: Connection Failed

**Message**: `Cannot connect to database`

**Cause**: Database server unreachable or not running.

**Diagnostic Steps**:
```bash
# Check database container status
docker ps | grep postgres

# Check database is listening
docker exec postgres pg_isready

# Check connection from app container
docker exec visionflow-container nc -zv postgres 5432

# Check logs
docker logs postgres
docker logs visionflow-container | grep -i "database"
```

**Solutions**:
```bash
# Restart database container
docker-compose restart postgres

# Check connection string
echo $POSTGRES_HOST:$POSTGRES_PORT

# Verify credentials
docker exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB
```

---

#### DB-E-003: Connection Pool Exhausted

**Message**: `No available connections in pool`

**Cause**: All database connections in use.

**Solution**:
```bash
# Increase pool size in .env
POSTGRES_MAX_CONNECTIONS=200

# Check active connections
docker exec postgres psql -U visionflow -c \
  "SELECT count(*) FROM pg_stat_activity WHERE state='active';"

# Kill idle connections
docker exec postgres psql -U visionflow -c \
  "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
   WHERE state='idle' AND state_change < now() - interval '5 minutes';"
```

---

### DB-E-100 to DB-E-199: Query Errors

#### DB-E-103: Deadlock Detected

**Message**: `Query deadlock, retrying...`

**Cause**: Two transactions waiting for each other's locks.

**Solution**:
- Automatic retry implemented in application
- If persistent, check for long-running transactions
- Optimise query order to acquire locks consistently

**Diagnostic Query**:
```sql
-- Check for locks
SELECT
    blocked_locks.pid AS blocked_pid,
    blocking_locks.pid AS blocking_pid,
    blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity
  ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
  ON blocking_locks.locktype = blocked_locks.locktype
WHERE NOT blocked_locks.granted;
```

---

## Graph/Ontology Errors

### GR-E-001 to GR-E-099: Parsing & Validation

#### GR-E-001: Invalid OWL Syntax

**Message**: `OWL file has syntax error: {error}`

**Cause**: Malformed OWL/RDF XML.

**Solution**:
```bash
# Validate OWL file
xmllint --noout ontology.owl

# Check for common issues
# - Unclosed tags
# - Invalid XML characters
# - Missing namespace declarations

# Use OWL validator
curl -X POST http://mowl-power.cs.man.ac.uk:8080/validator/upload \
  -F file=@ontology.owl
```

**Common Issues**:
```xml
<!-- ✗ Missing namespace -->
<owl:Class rdf:about="#Person"/>

<!-- ✓ Correct namespace -->
<owl:Class rdf:about="http://example.org/ontology#Person"/>

<!-- ✗ Unclosed tag -->
<owl:Class rdf:about="#Person">

<!-- ✓ Properly closed -->
<owl:Class rdf:about="#Person"/>
```

---

#### GR-E-102: Inconsistent Ontology

**Message**: `Ontology is inconsistent`

**Cause**: Logical contradictions in axioms.

**Example**:
```turtle
# Contradiction: Person and Organization are disjoint,
# but Student is a subclass of both
:Person owl:disjointWith :Organization .
:Student rdfs:subClassOf :Person .
:Student rdfs:subClassOf :Organization .  # ✗ Contradiction
```

**Solution**:
1. Run consistency check: `POST /api/ontology/validate`
2. Review validation report: `GET /api/ontology/report/{id}`
3. Fix contradictions:
   ```turtle
   # Remove conflicting axiom
   # :Student rdfs:subClassOf :Organization .

   # Or remove disjoint axiom if both are valid
   # :Person owl:disjointWith :Organization .
   ```

---

## GPU/Physics Errors

### GP-E-001 to GP-E-099: GPU Initialization

#### GP-E-001: No GPU Found

**Message**: `No CUDA-capable GPU detected`

**Cause**: No NVIDIA GPU available or driver not installed.

**Solution**:
```bash
# Check GPU availability
nvidia-smi

# If not found, install drivers
sudo ubuntu-drivers autoinstall

# Verify CUDA installation
nvcc --version

# Use CPU fallback
ENABLE_GPU=false docker-compose up
```

---

#### GP-E-002: GPU Memory Insufficient

**Message**: `GPU memory insufficient ({available}/{required})`

**Cause**: Not enough VRAM for graph size.

**Solution**:
```bash
# Check current VRAM usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Reduce graph size
curl -X POST http://localhost:9090/api/graph/data \
  -d '{"maxNodes": 50000}'

# Reduce GPU batch size
GPU_BATCH_SIZE=10000  # in .env

# Use smaller precision
GPU_PRECISION=float16
```

---

### GP-E-100 to GP-E-199: Computation Errors

#### GP-E-101: Kernel Launch Failed

**Message**: `CUDA kernel launch failed: {error}`

**Cause**: Invalid kernel parameters or GPU state.

**Diagnostic**:
```bash
# Check CUDA error log
docker logs visionflow-container | grep -i "cuda"

# Reset GPU
sudo nvidia-smi --gpu-reset

# Check for out-of-bounds memory access
# Enable CUDA error checking in code
```

---

## WebSocket Errors

### WS-E-001 to WS-E-099: Connection Errors

#### WS-E-001: Connection Refused

**Message**: `WebSocket connection refused`

**Cause**: WebSocket server not running or port blocked.

**Solution**:
```bash
# Check server is running
curl http://localhost:9090/api/health

# Check WebSocket port is open
nc -zv localhost 9090

# Check firewall rules
sudo ufw status

# Test WebSocket connection
wscat -c ws://localhost:9090/ws?token=$TOKEN
```

---

#### WS-E-004: Invalid Protocol Version

**Message**: `Protocol version mismatch`

**Cause**: Client using incompatible binary protocol version.

**Solution**:
```typescript
// Update client to use Protocol V2
ws.binaryType = 'arraybuffer';

// Check first byte of binary message
const version = new Uint8Array(buffer)[0];
if (version !== 2) {
  console.error('Expected protocol V2, got V' + version);
}
```

---

### WS-E-100 to WS-E-199: Message Errors

#### WS-E-102: Message Too Large

**Message**: `Message size ({size}) exceeds limit ({limit})`

**Cause**: Binary message too large for WebSocket frame.

**Solution**:
```bash
# Increase WebSocket message limit
WS_MAX_MESSAGE_SIZE=100MB  # in .env

# Enable compression
WS_COMPRESSION=true

# Reduce update rate
curl -X POST http://localhost:9090/api/config \
  -d '{"websocket": {"maxUpdateRate": 30}}'
```

---

## Common Issues & Solutions

### Issue: High Memory Usage

**Symptoms**: Container using > 90% memory, slow performance.

**Diagnostic**:
```bash
# Check container memory
docker stats visionflow-container

# Check process memory
docker exec visionflow-container ps aux --sort=-%mem | head -10

# Check database size
du -sh data/*.db
```

**Solutions**:
```bash
# Increase memory limit
MEMORY_LIMIT=32g

# Enable memory optimization
ENABLE_MEMORY_OPTIMIZATION=true

# Reduce cache size
REDIS_MAX_MEMORY=1gb
POSTGRES_SHARED_BUFFERS=2GB

# Run vacuum on database
docker exec postgres vacuumdb -U visionflow -d visionflow
```

---

### Issue: Slow Graph Rendering

**Symptoms**: < 30 FPS, laggy interactions.

**Diagnostic**:
```bash
# Check WebSocket latency
wscat -c ws://localhost:9090/ws --execute "ping"

# Check GPU utilization
nvidia-smi -l 1

# Check browser performance
# Chrome DevTools → Performance tab
```

**Solutions**:
```bash
# Enable GPU acceleration
ENABLE_GPU=true

# Use binary protocol
ws.binaryType = 'arraybuffer';

# Reduce node count
curl -X POST http://localhost:9090/api/graph/filter \
  -d '{"quality": 0.7, "maxNodes": 50000}'

# Enable instanced rendering
ENABLE_INSTANCED_RENDERING=true
```

---

### Issue: Authentication Failures

**Symptoms**: All API requests return 401 Unauthorized.

**Diagnostic**:
```bash
# Test authentication
curl -X POST http://localhost:9090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test"}'

# Check JWT secret is set
echo $JWT_SECRET

# Verify token format
echo $TOKEN | cut -d'.' -f2 | base64 -d | jq .
```

**Solutions**:
```bash
# Regenerate JWT secret
JWT_SECRET=$(openssl rand -hex 32)

# Restart authentication service
docker-compose restart visionflow-container

# Clear and re-authenticate
rm ~/.visionflow/token
curl -X POST http://localhost:9090/api/auth/login ...
```

---

## Diagnostic Procedures

### Full System Health Check

```bash
#!/bin/bash
# scripts/health-check.sh

echo "=== VisionFlow Health Check ==="

# 1. Container Status
echo "Containers:"
docker ps --filter name=visionflow --format "table {{.Names}}\t{{.Status}}"

# 2. API Health
echo -e "\nAPI Health:"
curl -s http://localhost:9090/api/health | jq .

# 3. Database Connectivity
echo -e "\nDatabase:"
docker exec postgres pg_isready

# 4. Redis Connectivity
echo -e "\nRedis:"
docker exec redis redis-cli ping

# 5. GPU Status (if enabled)
if [ "$ENABLE_GPU" = "true" ]; then
  echo -e "\nGPU:"
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
fi

# 6. WebSocket
echo -e "\nWebSocket:"
nc -zv localhost 9090 2>&1 | grep -q succeeded && echo "OK" || echo "FAIL"

# 7. Disk Space
echo -e "\nDisk Space:"
df -h data/

# 8. Memory Usage
echo -e "\nMemory:"
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}"

echo -e "\n=== Health Check Complete ==="
```

---

### Performance Profiling

```bash
#!/bin/bash
# scripts/profile-performance.sh

# Enable profiling
curl -X POST http://localhost:9090/api/config \
  -d '{"system": {"profiling": {"enabled": true}}}'

# Run test workload
curl -X POST http://localhost:9090/api/graph/update

# Collect metrics
curl http://localhost:9090/metrics > metrics.txt

# Analyze with Prometheus
# See http://localhost:9090/graph

# Generate flamegraph
docker exec visionflow-container perf record -F 99 -p $(pidof visionflow) -g -- sleep 30
docker exec visionflow-container perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

---

---

---

## Related Documentation

- [Configuration Reference](CONFIGURATION_REFERENCE.md)
- [Complete API Reference](API_REFERENCE.md)
- [WebSocket Binary Protocol Reference](websocket-protocol.md)
- [WebSocket Binary Protocol - Complete System Documentation](../diagrams/infrastructure/websocket/binary-protocol-complete.md)
- [ASCII Diagram Deprecation Audit](../audits/ascii-diagram-deprecation-audit.md)

## Cross-Reference Index

### Related Documentation

| Topic | Documentation | Link |
|-------|---------------|------|
| API Errors | API Reference | [API_REFERENCE.md](./API_REFERENCE.md#error-responses) |
| Configuration | Configuration Reference | [CONFIGURATION_REFERENCE.md](./CONFIGURATION_REFERENCE.md) |
| Database Issues | Database Schema Reference | [DATABASE_SCHEMA_REFERENCE.md](./DATABASE_SCHEMA_REFERENCE.md) |
| Troubleshooting | Troubleshooting Guide | [troubleshooting.md](../guides/troubleshooting.md) |

---

**Error Reference Version**: 2.0
**VisionFlow Version**: v0.1.0
**Maintainer**: VisionFlow Support Team
**Last Updated**: December 18, 2025
