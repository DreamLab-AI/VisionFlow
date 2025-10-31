-- Migration: Settings Management Tables
-- Version: 006
-- Description: Create tables for persistent settings storage

-- Physics settings table (single row with ID=1 for current settings)
CREATE TABLE IF NOT EXISTS physics_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    settings_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Constraint settings table (single row with ID=1 for current settings)
CREATE TABLE IF NOT EXISTS constraint_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    settings_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Rendering settings table (single row with ID=1 for current settings)
CREATE TABLE IF NOT EXISTS rendering_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    settings_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Settings profiles table (for saving multiple named configurations)
CREATE TABLE IF NOT EXISTS settings_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    physics_json TEXT NOT NULL,
    constraints_json TEXT NOT NULL,
    rendering_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_settings_profiles_name ON settings_profiles(name);
CREATE INDEX IF NOT EXISTS idx_settings_profiles_updated ON settings_profiles(updated_at DESC);

-- Insert default settings (will only insert if table is empty)
INSERT OR IGNORE INTO physics_settings (id, settings_json, updated_at)
VALUES (1, '{}', datetime('now'));

INSERT OR IGNORE INTO constraint_settings (id, settings_json, updated_at)
VALUES (1, '{}', datetime('now'));

INSERT OR IGNORE INTO rendering_settings (id, settings_json, updated_at)
VALUES (1, '{}', datetime('now'));
