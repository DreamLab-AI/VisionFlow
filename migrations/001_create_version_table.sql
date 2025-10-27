-- Migration: 001_create_version_table
-- Description: Create schema migrations tracking table
-- Author: Database Migration System
-- Date: 2025-10-27

-- === UP MIGRATION ===

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT NOT NULL,
    execution_time_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at
    ON schema_migrations(applied_at);

-- === DOWN MIGRATION ===

DROP INDEX IF EXISTS idx_schema_migrations_applied_at;
DROP TABLE IF EXISTS schema_migrations;
