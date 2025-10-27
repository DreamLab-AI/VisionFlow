# VisionFlow Database Analysis Tools

This directory contains tools and scripts for analyzing and managing VisionFlow databases.

## Files

### Analysis Scripts
- **`analyze_databases.py`** - Comprehensive Python analysis script
- **`verify_graph.sql`** - SQL queries to verify knowledge graph integrity
- **`add_mock_credentials.sql`** - Insert mock credentials for development

### Database Copies
- `settings.db` - Local copy of settings database
- `knowledge_graph.db` - Local copy of knowledge graph
- `ontology.db` - Local copy of ontology database

## Usage

### 1. Analyze All Databases
```bash
cd /home/devuser/workspace/project/db_analysis
python3 analyze_databases.py
```

### 2. Add Mock Credentials (Development Only)
```bash
# Method 1: Via SQL file
docker exec -i visionflow_container sqlite3 /app/data/settings.db < add_mock_credentials.sql

# Method 2: Direct SQL
docker exec -i visionflow_container sqlite3 /app/data/settings.db << 'SQL'
INSERT INTO api_keys (service_name, api_key_encrypted, key_name, is_active)
VALUES ('nostr', 'wss://relay.damus.io', 'Mock Nostr', 1);
SQL
```

### 3. Verify Knowledge Graph
```bash
# After building graph from markdown files
docker exec visionflow_container sqlite3 /app/data/knowledge_graph.db < verify_graph.sql
```

### 4. Quick Checks
```bash
# Node count
docker exec visionflow_container sqlite3 /app/data/knowledge_graph.db \
  "SELECT COUNT(*) FROM nodes"

# Edge count
docker exec visionflow_container sqlite3 /app/data/knowledge_graph.db \
  "SELECT COUNT(*) FROM edges"

# API keys configured
docker exec visionflow_container sqlite3 /app/data/settings.db \
  "SELECT service_name, key_name, is_active FROM api_keys"
```

## Key Findings

### Current State (2025-10-23)
- ✅ **Settings DB**: Properly configured, 7 settings, 4 feature flags enabled
- ⚠️ **Knowledge Graph**: Schema initialized but EMPTY (0/185 nodes, 0/4014 edges)
- ⚠️ **Ontology DB**: Schema initialized but EMPTY (12 tables, no data)
- ✅ **Source Files**: 185 markdown files available for import
- ⚠️ **Credentials**: No API keys configured (api_keys table empty)

### Required Actions
1. Build knowledge graph from markdown files
2. Add API credentials (mock for dev, real for prod)
3. Optionally initialize ontology database

## Database Schema

### Settings Database
```
- settings (7 records) - Core configuration
- api_keys (0 records) - Service credentials
- feature_flags (4 records) - Feature toggles
- physics_settings (5 records) - Physics simulation profiles
- users, sessions, rate_limits - User management
```

### Knowledge Graph Database
```
- nodes (0/185) - Graph nodes from markdown
- edges (0/4014) - Relationships between nodes
- kg_nodes, kg_edges - Alternative graph representation
- file_metadata, file_topics - Source tracking
- graph_metadata (11 records) - Graph configuration
- graph_snapshots, graph_clusters - Analysis data
```

### Ontology Database
```
- owl_classes, owl_properties - OWL definitions
- owl_axioms, owl_class_hierarchy - Logical structure
- ontologies, namespaces - Metadata
- class_mappings, property_mappings - Alignments
- inference_results, validation_reports - Reasoning
```

## Credential Setup

### Mock Credentials (Development)
Services that need credentials:
- **Nostr**: Relay URL + private key (decentralized social)
- **GitHub**: Personal access token (repo integration)
- **RAGFlow**: API endpoint + key (RAG system)
- **Anthropic**: API key (Claude integration)

Use `add_mock_credentials.sql` for testing.

### Production Credentials
1. Generate real API keys from respective services
2. Use VisionFlow admin UI to add encrypted credentials
3. Never store production keys in plain text

## Graph Building

The knowledge graph needs to be built from the 185 markdown files:

```bash
# Option 1: Via VisionFlow UI
# Navigate to http://localhost:8080 -> Graph Management -> Rebuild

# Option 2: Via API
curl -X POST http://localhost:8080/api/graph/rebuild \
  -H "Content-Type: application/json" \
  -d '{"source": "markdown"}'

# Option 3: Via container (if available)
docker exec visionflow_container python -m visionflow.graph.builder \
  --source /app/data/markdown --rebuild
```

## Verification Queries

After building the graph:

```sql
-- Expected results:
SELECT COUNT(*) FROM nodes;        -- Should be 185
SELECT COUNT(*) FROM edges;        -- Should be ~4014

-- Check for issues:
SELECT COUNT(*) FROM nodes
WHERE id NOT IN (SELECT source FROM edges)
  AND id NOT IN (SELECT target FROM edges);  -- Should be 0 (no orphans)
```

## Reference

See `/home/devuser/workspace/project/docs/VisionFlow_Database_Integrity_Report.md` for the complete analysis report.
