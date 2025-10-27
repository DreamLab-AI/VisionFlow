# Migration Guide: VisionFlow v0.x → v1.0.0

## 📋 Overview

This guide provides step-by-step instructions for migrating from VisionFlow v0.x to v1.0.0, which introduces a complete architectural transformation to hexagonal architecture with CQRS pattern.

**Estimated Migration Time**: 2-4 hours (depending on customizations)

---

## ⚠️ Breaking Changes Summary

### Critical Breaking Changes
1. **Database Architecture**: Single database → Three separate databases
2. **Configuration**: File-based (YAML/TOML) → Database-stored settings
3. **API Patterns**: Direct SQL → Repository ports with CQRS
4. **WebSocket Protocol**: JSON → Binary protocol V2 (36 bytes)

### API Breaking Changes
| v0.x Pattern | v1.0.0 Replacement | Impact |
|--------------|-------------------|--------|
| `database.execute_query()` | `repository.get_graph()` | HIGH |
| `actor.send_message()` | `adapter.execute_command()` | MEDIUM |
| `load_config_file()` | `settings_repo.get_setting()` | HIGH |
| JSON WebSocket | Binary protocol | MEDIUM |

---

## 🗂️ Pre-Migration Checklist

### 1. System Requirements
- [ ] Rust 1.75.0 or higher (`rustc --version`)
- [ ] SQLite 3.35.0+ (`sqlite3 --version`)
- [ ] At least 2GB free disk space for database migration
- [ ] Backup of existing data directory

### 2. Backup Your Data
```bash
# Create backup directory
mkdir -p backups/v0-migration-$(date +%Y%m%d)

# Backup databases
cp data/*.db backups/v0-migration-$(date +%Y%m%d)/

# Backup configuration files
cp config.yml backups/v0-migration-$(date +%Y%m%d)/
cp ontology_physics.toml backups/v0-migration-$(date +%Y%m%d)/ 2>/dev/null || true

# Verify backups
ls -lh backups/v0-migration-$(date +%Y%m%d)/
```

### 3. Review Current Configuration
```bash
# Document current environment variables
env | grep -E 'DATABASE|CONFIG|VISIONFLOW' > backups/v0-env-vars.txt

# Document current settings
sqlite3 data/visionflow.db '.schema' > backups/v0-schema.sql
```

---

## 🚀 Migration Steps

### Step 1: Update Codebase

#### Option A: Docker Deployment
```bash
# Pull latest v1.0.0 image
docker pull visionflow/visionflow:1.0.0

# Stop existing container
docker-compose down

# Update docker-compose.yml
# Change image: visionflow/visionflow:0.x → visionflow/visionflow:1.0.0
```

#### Option B: Native Deployment
```bash
# Pull latest code
git fetch --all --tags
git checkout tags/v1.0.0

# Update dependencies
cargo update

# Build with features
cargo build --release --features gpu,ontology
```

### Step 2: Database Migration

#### Automated Migration Script
```bash
# Run migration tool
cargo run --bin migrate_legacy_configs

# Expected output:
# ✓ Creating settings.db...
# ✓ Creating knowledge_graph.db...
# ✓ Creating ontology.db...
# ✓ Migrating settings from config.yml...
# ✓ Migrating graph data...
# ✓ Migrating ontology data...
# ✓ Migration complete!
```

#### Manual Migration (If Script Fails)
```bash
# Create new database structure
sqlite3 data/settings.db < migrations/settings_schema.sql
sqlite3 data/knowledge_graph.db < migrations/knowledge_graph_schema.sql
sqlite3 data/ontology.db < migrations/ontology_schema.sql

# Migrate data manually
python3 scripts/manual_migration.py --input data/visionflow.db
```

#### Verify Migration
```bash
# Check database schemas
sqlite3 data/settings.db '.schema'
sqlite3 data/knowledge_graph.db '.schema'
sqlite3 data/ontology.db '.schema'

# Verify data integrity
cargo test --test migration_integrity
```

### Step 3: Update Environment Variables

#### Old Environment Variables (v0.x)
```bash
# REMOVE THESE
DATABASE_URL=data/visionflow.db
CONFIG_FILE=config.yml
ONTOLOGY_CONFIG=ontology_physics.toml
```

#### New Environment Variables (v1.0.0)
```bash
# ADD THESE to .env
SETTINGS_DB_PATH=data/settings.db
KNOWLEDGE_GRAPH_DB_PATH=data/knowledge_graph.db
ONTOLOGY_DB_PATH=data/ontology.db

# Optional: Connection pooling
DB_POOL_SIZE=10
DB_CONNECTION_TIMEOUT=30

# Optional: Performance tuning
DB_WAL_MODE=true
DB_CACHE_SIZE=10000
```

#### Update .env File
```bash
# Backup old .env
cp .env .env.v0.backup

# Create new .env from template
cp .env.v1.example .env

# Edit with your settings
nano .env
```

### Step 4: Migrate Configuration Files

#### Settings Migration
```bash
# Extract settings from config.yml
cat config.yml

# Insert into settings.db using CLI tool
cargo run --bin settings-import -- \
  --yaml config.yml \
  --db data/settings.db

# Verify settings
sqlite3 data/settings.db "SELECT * FROM settings;"
```

#### Physics Settings Migration
```bash
# Migrate ontology_physics.toml
cargo run --bin physics-import -- \
  --toml ontology_physics.toml \
  --db data/settings.db

# Verify physics settings
sqlite3 data/settings.db "SELECT * FROM physics_settings;"
```

#### Remove Legacy Files (After Verification)
```bash
# DO NOT REMOVE until migration is verified!
# After successful testing:
# mv config.yml backups/
# mv ontology_physics.toml backups/
```

### Step 5: Update Application Code

#### Database Access Migration

**v0.x Pattern (Deprecated)**
```rust
// OLD - Direct SQL
let conn = database.get_connection()?;
let mut stmt = conn.prepare("SELECT * FROM nodes WHERE id = ?")?;
let node = stmt.query_row([id], |row| {
    Ok(Node {
        id: row.get(0)?,
        label: row.get(1)?,
        // ...
    })
})?;
```

**v1.0.0 Pattern (New)**
```rust
// NEW - Repository port
let node = knowledge_graph_repo
    .get_node(node_id)
    .await?;

// Or using CQRS query
let query = GetNodeQuery { id: node_id };
let node = query_bus.execute(query).await?;
```

#### Actor Message Migration

**v0.x Pattern (Deprecated)**
```rust
// OLD - Direct actor message
use actix::prelude::*;

let msg = GraphMessage::SaveGraph(graph_data);
graph_actor.send(msg).await?;
```

**v1.0.0 Pattern (New)**
```rust
// NEW - Adapter + Command
let command = SaveGraphCommand {
    graph_id,
    nodes,
    edges,
};
let result = graph_service.save_graph(command).await?;
```

#### Configuration Access Migration

**v0.x Pattern (Deprecated)**
```rust
// OLD - File-based config
let config = std::fs::read_to_string("config.yml")?;
let settings: Settings = serde_yaml::from_str(&config)?;
```

**v1.0.0 Pattern (New)**
```rust
// NEW - Database-backed settings
let setting_value = settings_repo
    .get_setting("physics.gravity")
    .await?;

// Or using query
let query = GetSettingQuery {
    key: "physics.gravity".to_string(),
};
let value = query_bus.execute(query).await?;
```

### Step 6: Client-Side Updates

#### WebSocket Protocol Update

**v0.x JSON Protocol (Deprecated)**
```javascript
// OLD - JSON messages
const message = {
  type: 'node_update',
  node_id: 123,
  position: { x: 1.0, y: 2.0, z: 3.0 }
};
websocket.send(JSON.stringify(message));
```

**v1.0.0 Binary Protocol (New)**
```javascript
// NEW - Binary protocol (36 bytes)
const buffer = new ArrayBuffer(36);
const view = new DataView(buffer);

// Header (4 bytes)
view.setUint32(0, MESSAGE_TYPE_NODE_UPDATE, true);

// Node ID (4 bytes)
view.setUint32(4, nodeId, true);

// Position (12 bytes: 3 x float32)
view.setFloat32(8, position.x, true);
view.setFloat32(12, position.y, true);
view.setFloat32(16, position.z, true);

// Send binary message
websocket.send(buffer);
```

#### Remove Client-Side Caching

**v0.x (Deprecated - Remove)**
```javascript
// OLD - Client-side cache (causes sync issues)
class GraphCache {
  constructor() {
    this.nodeCache = new Map();
    this.edgeCache = new Map();
  }

  updateNode(node) {
    this.nodeCache.set(node.id, node);
  }
}
```

**v1.0.0 (New - Server-Authoritative)**
```javascript
// NEW - Always use server state
class GraphState {
  constructor(websocket) {
    this.websocket = websocket;
    this.nodes = new Map(); // Updated only from server
    this.edges = new Map();

    // Listen for server updates
    this.websocket.onmessage = (msg) => {
      this.handleServerUpdate(msg.data);
    };
  }

  handleServerUpdate(binaryData) {
    // Parse binary message and update local state
    // No caching - server is source of truth
  }
}
```

### Step 7: Test Migration

#### Run Test Suite
```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test '*'

# Performance benchmarks
cargo bench

# End-to-end tests
cargo test --test e2e
```

#### Verify Functionality
```bash
# Start server
cargo run --release

# Check health endpoint
curl http://localhost:3030/api/health

# Expected response:
# {
#   "status": "healthy",
#   "databases": {
#     "settings": "connected",
#     "knowledge_graph": "connected",
#     "ontology": "connected"
#   }
# }
```

#### Performance Validation
```bash
# Run performance validation suite
cargo run --bin validate-performance

# Check metrics:
# ✓ Database operations: <10ms (p99)
# ✓ API latency: <100ms (p99)
# ✓ WebSocket latency: <50ms (p99)
```

---

## 🔧 Common Migration Issues

### Issue 1: Database Connection Errors

**Symptom:**
```
Error: Failed to open database: unable to open database file
```

**Solution:**
```bash
# Ensure data directory exists
mkdir -p data

# Check file permissions
chmod 755 data
chmod 644 data/*.db

# Verify database paths in .env
cat .env | grep DB_PATH
```

### Issue 2: Settings Not Found

**Symptom:**
```
Error: Setting not found: physics.gravity
```

**Solution:**
```bash
# Re-run settings import
cargo run --bin settings-import -- --yaml config.yml --db data/settings.db

# Manually insert missing setting
sqlite3 data/settings.db "INSERT INTO settings (key, value, value_type) VALUES ('physics.gravity', '9.81', 'float');"
```

### Issue 3: WebSocket Connection Refused

**Symptom:**
```
WebSocket connection failed: Connection refused
```

**Solution:**
```bash
# Check server is running
curl http://localhost:3030/api/health

# Verify WebSocket endpoint
wscat -c ws://localhost:3030/ws

# Check firewall rules
sudo ufw status
```

### Issue 4: GPU Acceleration Not Working

**Symptom:**
```
Warning: CUDA not available, falling back to CPU
```

**Solution:**
```bash
# Verify CUDA installation
nvcc --version

# Check GPU drivers
nvidia-smi

# Rebuild with GPU features
cargo build --release --features gpu
```

---

## 📊 Performance Comparison

### Before Migration (v0.x)
| Operation | Latency | Throughput |
|-----------|---------|------------|
| Node Insert | 15ms | 66 ops/sec |
| Graph Query | 100ms | 10 queries/sec |
| WebSocket Msg | 25ms | 40 msg/sec |

### After Migration (v1.0.0)
| Operation | Latency | Throughput |
|-----------|---------|------------|
| Node Insert | 2ms | 500 ops/sec |
| Graph Query | 8ms | 125 queries/sec |
| WebSocket Msg | <10ms | 100+ msg/sec |

**Expected Improvements:**
- 87% faster node insertion
- 92% faster graph queries
- 60% lower WebSocket latency
- 80% bandwidth reduction

---

## 🔄 Rollback Procedure

If migration fails or issues arise:

### 1. Stop v1.0.0 Server
```bash
# Docker
docker-compose down

# Native
pkill webxr
```

### 2. Restore v0.x Backup
```bash
# Restore databases
cp backups/v0-migration-YYYYMMDD/*.db data/

# Restore configuration
cp backups/v0-migration-YYYYMMDD/config.yml .
cp backups/v0-migration-YYYYMMDD/ontology_physics.toml .

# Restore .env
cp .env.v0.backup .env
```

### 3. Rollback Code
```bash
# Docker
# Edit docker-compose.yml: change image to v0.x
docker-compose up -d

# Native
git checkout tags/v0.9.0
cargo build --release
```

### 4. Verify Rollback
```bash
# Test server
curl http://localhost:3030/api/health

# Check logs
tail -f logs/visionflow.log
```

---

## 📞 Support

### Need Help?

- **Documentation**: [docs/](https://docs.visionflow.io)
- **GitHub Issues**: [github.com/yourusername/VisionFlow/issues](https://github.com/yourusername/VisionFlow/issues)
- **Discussions**: [github.com/yourusername/VisionFlow/discussions](https://github.com/yourusername/VisionFlow/discussions)
- **Enterprise Support**: support@visionflow.io

### Migration Assistance

If you encounter issues not covered in this guide:

1. **Check Logs**: `tail -f logs/migration.log`
2. **Run Diagnostics**: `cargo run --bin diagnose-migration`
3. **Report Issue**: Include logs, error messages, and system info

---

## ✅ Post-Migration Checklist

After successful migration:

- [ ] All databases accessible and healthy
- [ ] Settings correctly migrated
- [ ] Graph data intact (node/edge counts match)
- [ ] Ontology data validated
- [ ] Test suite passes (100%)
- [ ] Performance benchmarks meet targets
- [ ] WebSocket connection stable
- [ ] Client updates deployed
- [ ] Legacy config files backed up
- [ ] Documentation updated
- [ ] Team notified of changes

---

## 🎉 Migration Complete!

Congratulations on successfully migrating to VisionFlow v1.0.0!

**Next Steps:**
1. Review [Architecture Guide](../architecture/) for new patterns
2. Explore [API Documentation](../api/) for updated endpoints
3. Read [Performance Guide](../performance/) for optimization tips
4. Join our [Discord community](https://discord.gg/visionflow)

**Enjoy the improved performance and maintainability of hexagonal architecture!**

---

**VisionFlow v1.0.0 Migration Guide**
Last Updated: 2025-10-27
