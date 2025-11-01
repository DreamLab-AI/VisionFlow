# Database & Schema Cleanup Plan

## Current State Analysis

### Database Files

**PRODUCTION (data/)**
- ✅ `data/unified.db` (408K) - **ACTIVE** authoritative unified database
- ⚠️ `data/knowledge_graph.db` (288K) - LEGACY, superseded by unified.db
- ⚠️ `data/ontology.db` (192K) - LEGACY, superseded by unified.db
- ⚠️ `data/settings.db` (212K) - LEGACY, superseded by unified.db

**ROOT DIRECTORY CLUTTER**
- 🗑️ `unified.db` (0 bytes) - Empty duplicate, DELETE
- 🗑️ `agentdb.db` (340K) - MCP agent db, MOVE to `.swarm/` or `data/`

**TEST ARTIFACTS**
- 🗑️ `tests/db_analysis/*.db` - Test databases, safe to DELETE
- 🗑️ `scripts/knowledge_graph.db` - Script testing artifact, DELETE

**MCP/SWARM DATABASES (Keep)**
- ✅ `.swarm/memory.db` - MCP system (multiple locations)
- ✅ `.hive-mind/*.db` - Agent coordination system

### Schema Files

**ACTIVE SCHEMAS**
- ✅ `migration/unified_schema.sql` - **AUTHORITATIVE** unified DB schema
- ⚠️ `migration/control_center_schema.sql` - Partial duplicate

**LEGACY SCHEMAS (schema/)**
⚠️ **BUILD DEPENDENCY** - `database_service.rs:164-166` uses `include_str!()` for:
- `schema/settings_db.sql`
- `schema/knowledge_graph_db.sql`
- `schema/ontology_metadata_db.sql`

These are embedded at compile time for legacy three-database system.

**STATUS**: `database_service.rs` manages the OLD architecture but may still be used for backward compatibility.

### Code Architecture

**CURRENT (Unified)**
- `src/repositories/unified_graph_repository.rs` → `data/unified.db`
- `src/repositories/unified_ontology_repository.rs` → `data/unified.db`
- Uses `migration/unified_schema.sql` as authoritative schema

**LEGACY (Three-Database)**
- `src/services/database_service.rs` → `data/{settings,knowledge_graph,ontology}.db`
- Embeds schemas from `schema/*.sql` at compile time
- Used by: `app_state.rs`, `settings_watcher.rs`, `migrate.rs`

## Cleanup Actions

### Phase 1: Safe Deletions (No Code Changes)

```bash
# Delete empty duplicate
rm unified.db

# Move agent db to proper location
mv agentdb.db data/

# Delete test artifacts
rm -rf tests/db_analysis/*.db
rm scripts/knowledge_graph.db
```

### Phase 2: Schema Organization

**Option A: Keep schema/ for build (RECOMMENDED)**
- Leave `schema/` in place (required for `include_str!()` compile-time embedding)
- Add README explaining it's for legacy compatibility
- Update paths would break builds

**Option B: Move and update paths**
```bash
mkdir -p data/schema
mv schema/* data/schema/
```
Then update `database_service.rs`:
```rust
const SETTINGS_SCHEMA: &str = include_str!("../../data/schema/settings_db.sql");
const KNOWLEDGE_GRAPH_SCHEMA: &str = include_str!("../../data/schema/knowledge_graph_db.sql");
const ONTOLOGY_SCHEMA: &str = include_str!("../../data/schema/ontology_metadata_db.sql");
```

### Phase 3: Deprecate Legacy Databases

**Check Usage**
```bash
# Find references to old databases
grep -r "knowledge_graph\.db" src/
grep -r "ontology\.db" src/ | grep -v unified
grep -r "settings\.db" src/
```

**If safe:**
```bash
# Archive old databases
mkdir -p data/archive
mv data/knowledge_graph.db data/archive/
mv data/ontology.db data/archive/
mv data/settings.db data/archive/
```

### Phase 4: Consolidate Schemas

**Merge control_center_schema.sql into unified_schema.sql**

The control center schema defines:
- `physics_settings`
- `constraint_settings`
- `rendering_settings`
- `settings_profiles`

Check if these tables already exist in `unified_schema.sql`. If not, append them.

## Recommended Action

**Immediate (Safe)**
1. Delete empty `unified.db` in root
2. Move `agentdb.db` to `data/`
3. Delete test database artifacts
4. Add `schema/README.md` explaining legacy build dependency

**Future (Requires Testing)**
5. Move `schema/` to `data/schema/` and update `database_service.rs` paths
6. Deprecate `database_service.rs` three-database system
7. Archive legacy database files
8. Consolidate control_center_schema into unified_schema

## Risk Assessment

**LOW RISK**
- Deleting empty/test databases
- Moving agentdb.db
- Adding documentation

**MEDIUM RISK**
- Moving schema/ directory (breaks build if paths not updated)
- Archiving legacy databases (need to verify no code references them)

**HIGH RISK**
- Removing database_service.rs (need full codebase audit)
- Deleting schema files (embedded at compile time)

## Verification Steps

After cleanup:
```bash
# Verify build succeeds
cargo build --release

# Verify unified.db schema is correct
sqlite3 data/unified.db ".schema" | head -50

# Check server starts
./scripts/launch.sh up dev

# Verify no broken references
grep -r "knowledge_graph\.db\|ontology\.db\|settings\.db" src/ | grep -v unified | grep -v archive
```

## Files Created/Modified

- [x] `/migrations/` → `/scripts/migrations/`
- [x] `/frontend/` → `/client/src/components/ControlCenter/`
- [x] `Cargo.toml` - Added example configurations
- [x] `/launch.unified.sh` → `/scripts/launch.sh`
- [x] Removed `/-/` accidental directory
- [ ] Database cleanup pending user approval

