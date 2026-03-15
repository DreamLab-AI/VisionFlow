# RuVector PostgreSQL — External AI Memory

## Overview

RuVector PostgreSQL is the centralized persistent memory for all agentic orchestration. It runs as a companion container (`ruvector-postgres`) alongside the agentic workstation, connected via the `docker_ragflow` network. All agents, swarms, and sessions share this single store. Data survives container rebuilds via an external Docker volume.

## Architecture

```
agentic-workstation                    ruvector-postgres
+-------------------+                  +------------------+
| Claude Code       |  MCP memory_*   | PostgreSQL 17    |
| claude-flow       | --------------> | RuVector v2.0.0  |
| ruflo             |  port 5432      | (112 SQL funcs)  |
| Agent Teams       |                 |                  |
+-------------------+                  +------------------+
        |                                      |
        v                                      v
  client-side ONNX                    ruvector_postgres_data_v2
  (all-MiniLM-L6-v2)                  (external Docker volume)
  384-dim embeddings
```

## Connection

| Property | Value |
|----------|-------|
| Container | `ruvector-postgres` |
| Image | `ruvnet/ruvector-postgres:latest` |
| Host | `ruvector-postgres` (Docker DNS) |
| Port | `5432` |
| Database | `ruvector` |
| User | `ruvector` |
| Password | `ruvector` |
| ConnInfo | `$RUVECTOR_PG_CONNINFO` |
| Volume | `ruvector_postgres_data_v2` (external, persistent) |

## Extension: RuVector v2.0.0

RuVector replaces standard pgvector with an enhanced extension providing 112 SQL functions and AVX-512 SIMD acceleration for sub-millisecond vector operations.

### Vector Search
- **Type**: `ruvector(384)` — 384-dimensional vectors
- **Index**: HNSW (m=16, ef_construction=64)
- **Distance**: Cosine similarity via `<=>` operator
- **Embeddings**: Generated client-side by claude-flow's ONNX runtime (all-MiniLM-L6-v2 model)
- **Performance**: Sub-millisecond search across 19,000+ entries

### Extended Capabilities

| Category | Functions |
|----------|-----------|
| **Cypher graph queries** | `ruvector_cypher('graph', 'MATCH (n) RETURN n', '{}')` |
| **SPARQL** | `ruvector_sparql('store', 'SELECT ?s ?p ?o WHERE { ?s ?p ?o }', '{}')` |
| **Agent routing** | `ruvector_register_agent()`, `ruvector_route()`, `ruvector_find_agents_by_capability()` |
| **Self-learning** | `ruvector_enable_learning()`, `ruvector_record_feedback()`, `ruvector_extract_patterns()` |
| **Graph operations** | `ruvector_add_node()`, `ruvector_add_edge()`, `ruvector_shortest_path()` |
| **Attention** | `attention_score()`, `attention_softmax()`, `attention_weighted_add()` |
| **Hyperbolic geometry** | `ruvector_poincare_distance()`, `ruvector_lorentz_distance()`, `ruvector_mobius_add()` |
| **Temporal** | `temporal_velocity()`, `temporal_drift()`, `temporal_ema_update()` |

## Database Schema

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `memory_entries` | Primary KV + vector store | `key`, `namespace`, `value` (JSONB), `embedding` ruvector(384), `project_id`, `agent_id` |
| `projects` | Project registry | `name`, `path`, `git_remote`, `total_entries` |
| `patterns` | Learned code/workflow patterns | `type`, `pattern`, `confidence`, `embedding` ruvector(384) |
| `reasoning_patterns` | ReasoningBank trajectories | `pattern_key`, `confidence`, `success_count`, `failure_count` |
| `sona_trajectories` | SONA self-optimization tracking | `trajectory_id`, `agent_id`, `steps` (JSONB), `success` |
| `session_state` | Session persistence | `session_id`, `state` (JSONB), `agents`, `tasks` |

## Usage

### Via MCP Tools (Preferred)

```javascript
// Store after successful task
mcp__claude-flow__memory_store({
  namespace: "patterns",
  key: "descriptive-key",
  value: JSON.stringify({description: "what worked", category: "...", confidence: 0.9})
})

// Recall before starting task
mcp__claude-flow__memory_search({query: "[task keywords]", namespace: "patterns", limit: 10})
```

### Via SQL (Advanced)

```bash
# Store
PGPASSWORD=ruvector psql -h ruvector-postgres -U ruvector -d ruvector -c "
INSERT INTO memory_entries (id, project_id, namespace, key, value, metadata, source_type)
VALUES (gen_random_uuid()::text, <project_id>, '<namespace>', '<key>', '<json>'::jsonb, '<meta>'::jsonb, 'claude')
ON CONFLICT ON CONSTRAINT memory_entries_pkey DO UPDATE SET value = EXCLUDED.value, updated_at = now();
"

# Vector similarity search (HNSW)
PGPASSWORD=ruvector psql -h ruvector-postgres -U ruvector -d ruvector -c "
WITH query AS (SELECT embedding FROM memory_entries WHERE key = '<known-key>')
SELECT me.key, me.namespace, me.value,
       (1 - (me.embedding <=> q.embedding))::numeric(6,4) as similarity
FROM memory_entries me, query q
WHERE me.embedding IS NOT NULL
ORDER BY me.embedding <=> q.embedding LIMIT 10;
"

# JSONB content search
PGPASSWORD=ruvector psql -h ruvector-postgres -U ruvector -d ruvector -c "
SELECT key, value->>'description' FROM memory_entries
WHERE namespace = 'patterns' AND value->>'category' = 'architecture'
ORDER BY updated_at DESC LIMIT 20;
"
```

### Via Python

```python
import psycopg
import os

with psycopg.connect(os.getenv('RUVECTOR_PG_CONNINFO')) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT key, value FROM memory_entries WHERE namespace = %s LIMIT 10",
            ('patterns',)
        )
        for row in cur:
            print(row)
```

## Docker Compose Integration

RuVector PostgreSQL is defined in `docker-compose.unified.yml` and starts before the agentic workstation:

```yaml
services:
  ruvector-postgres:
    image: ruvnet/ruvector-postgres:latest
    container_name: ruvector-postgres
    hostname: ruvector-postgres
    environment:
      POSTGRES_USER: ${RUVECTOR_PG_USER:-ruvector}
      POSTGRES_PASSWORD: ${RUVECTOR_PG_PASSWORD:-ruvector}
      POSTGRES_DB: ${RUVECTOR_PG_DATABASE:-ruvector}
    volumes:
      - ruvector-data:/var/lib/postgresql/data
    networks:
      docker_ragflow:
        aliases:
          - ruvector-postgres
    shm_size: 256m
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ruvector -d ruvector"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '4'
        reservations:
          memory: 512M
          cpus: '1'

  agentic-workstation:
    depends_on:
      ruvector-postgres:
        condition: service_healthy
    # ...

volumes:
  ruvector-data:
    external: true
    name: ruvector_postgres_data_v2
```

The `ruvector_postgres_data_v2` volume is external, meaning it must exist before starting. Create it once:
```bash
docker volume create ruvector_postgres_data_v2
```

## Known Quirks

1. **`count(*)` returns 0** — The RuVector extension overrides the `count` aggregate. Use `count(id)` or `count(column_name)` instead.
2. **`ruvector_embed()` unavailable** — The SQL embedding function signatures exist but the ONNX model runtime is not bundled in the Docker image. Embeddings must be generated client-side by claude-flow's Node.js ONNX runtime.
3. **Graph nodes may not persist** — `ruvector_add_node()` returns IDs but `ruvector_graph_stats()` may show 0 nodes. Graph operations may require specific API patterns.

## Memory Protocol

### Session Start
1. `bd ready` — check project state
2. `mcp__claude-flow__memory_search({query: "[context]", limit: 10})` — recall from external memory
3. Check native tasks from prior sessions

### During Work
- **Project state** (blockers, decisions, dependencies) → Beads (`bd add`)
- **Current task progress** → Native Tasks
- **Learned patterns, routing weights** → External memory via MCP (persists in RuVector)
- **Cross-agent coordination** → External memory `coordination` namespace

### Session End
- `bd add --type issue "..."` — file discovered work
- `bd add --type decision "..."` — record decisions
- `mcp__claude-flow__memory_store(...)` — store session outcomes

## Mandatory Rules

1. **NEVER** use `claude-flow memory *` CLI commands — they bypass the external store
2. **ALWAYS** use MCP memory tools (`mcp__claude-flow__memory_*`) for standard ops
3. Use SQL only for bulk operations, analytics, or cross-namespace vector search
4. Embeddings are client-side only (all-MiniLM-L6-v2, 384 dimensions)
5. Use `count(id)` not `count(*)` for row counts
