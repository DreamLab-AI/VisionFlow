# Database Migrations

## Overview

This directory contains database migration scripts for the settings database.

## Migration 001 - Add Missing Settings

**Status**: ✅ EXECUTED (2025-10-22)
**File**: `001_add_missing_settings.sql`
**Settings Added**: 73

### What Changed

Extended the settings database from 5 to 78 settings, adding:

- **Analytics Settings (11)**: Clustering algorithms, metrics collection, graph analysis
- **Dashboard Settings (8)**: Status displays, auto-refresh, compute mode tracking
- **Performance Settings (11)**: FPS targeting, GPU memory, physics optimization
- **GPU Visualization (8)**: Heatmaps, particle trails, visual effects
- **Bloom Effects (4)**: Post-processing bloom parameters
- **Developer Settings (11)**: Debug mode, profiling, logging, validation
- **Agent Control (20)**: Multi-agent coordination, learning, workflows

### Quick Start

#### Execute Migration

```bash
# Using shell script (recommended)
./scripts/run_migration.sh

# Using SQLite directly
sqlite3 data/settings.db < scripts/migrations/001_add_missing_settings.sql
```

#### Verify Results

```bash
# Count total settings
sqlite3 data/settings.db "SELECT COUNT(*) FROM settings"
# Expected: 78

# Check for duplicates
sqlite3 data/settings.db "SELECT key, COUNT(*) FROM settings GROUP BY key HAVING COUNT(*) > 1"
# Expected: (no output)

# View category breakdown
sqlite3 data/settings.db "
  SELECT
    CASE
      WHEN key LIKE 'analytics.%' THEN 'analytics'
      WHEN key LIKE 'dashboard.%' THEN 'dashboard'
      WHEN key LIKE 'performance.%' THEN 'performance'
      WHEN key LIKE 'gpu.%' THEN 'gpu'
      WHEN key LIKE 'effects.%' THEN 'effects'
      WHEN key LIKE 'dev.%' THEN 'developer'
      WHEN key LIKE 'agents.%' THEN 'agents'
      ELSE 'other'
    END as category,
    COUNT(*) as count
  FROM settings
  WHERE parent_key = 'app_full_settings'
  GROUP BY category
"
```

### Documentation

- **Results Report**: `docs/MIGRATION_001_RESULTS.md` - Detailed analysis and validation
- **Summary**: `docs/MIGRATION_SUMMARY.md` - Executive summary and next steps
- **Quick Reference**: `docs/SETTINGS_QUICK_REFERENCE.md` - All settings catalogued with examples

### Rollback

If you need to rollback this migration:

```sql
DELETE FROM settings WHERE parent_key = 'app_full_settings';
```

This will remove all 73 settings added by this migration, restoring the database to 5 settings.

### Schema

The migration uses the following schema:

```sql
CREATE TABLE settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    parent_key TEXT,
    value_type TEXT NOT NULL CHECK(value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    value_text TEXT,
    value_integer INTEGER,
    value_float REAL,
    value_boolean INTEGER CHECK(value_boolean IN (0, 1)),
    value_json TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Example Queries

```sql
-- Get all analytics settings
SELECT key, value_type,
       COALESCE(value_text, CAST(value_integer AS TEXT),
                CAST(value_float AS TEXT),
                CASE value_boolean WHEN 1 THEN 'true' ELSE 'false' END) as value,
       description
FROM settings
WHERE key LIKE 'analytics.%'
ORDER BY key;

-- Get all agent settings
SELECT key, value_type,
       COALESCE(value_text, CAST(value_integer AS TEXT),
                CAST(value_float AS TEXT),
                CASE value_boolean WHEN 1 THEN 'true' ELSE 'false' END) as value,
       description
FROM settings
WHERE key LIKE 'agents.%'
ORDER BY key;

-- Update a setting
UPDATE settings
SET value_integer = 16, updated_at = CURRENT_TIMESTAMP
WHERE key = 'agents.maxConcurrent';

-- Get recently modified settings
SELECT key, updated_at
FROM settings
WHERE updated_at > created_at
ORDER BY updated_at DESC;
```

## Future Migrations

### Migration Naming Convention

- Format: `###_description.sql`
- Examples: `002_add_visualization_presets.sql`, `003_add_user_preferences.sql`

### Migration Template

```sql
-- Migration ###: Description
-- Date: YYYY-MM-DD
-- Description: What this migration does

-- Add your SQL statements here

-- Validation comments at end
-- Expected: X settings/tables/rows affected
```

### Best Practices

1. **Always backup** the database before running migrations
2. **Test migrations** on a copy of the database first
3. **Document changes** in a results file
4. **Include rollback** instructions
5. **Validate results** after execution
6. **Use transactions** for complex migrations

### Creating New Migrations

1. Create migration file: `scripts/migrations/###_description.sql`
2. Write SQL statements following schema
3. Test on development database
4. Document expected results
5. Execute on production database
6. Verify with validation queries
7. Create results documentation

## Support

For issues or questions about migrations:

1. Check documentation in `docs/MIGRATION_*.md`
2. Review schema: `sqlite3 data/settings.db ".schema"`
3. Examine current state: `sqlite3 data/settings.db "SELECT * FROM settings"`
4. Review migration SQL for syntax

---

**Last Updated**: 2025-10-22
**Migration Count**: 1
**Database Version**: 1.0.0
