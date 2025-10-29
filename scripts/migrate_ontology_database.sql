-- Database Migration Script for Raw Markdown Storage
-- Adds markdown_content, file_sha1, and last_synced columns to owl_classes table
--
-- This migration enables the new architecture where:
-- 1. GitHub downloader stores raw markdown in database
-- 2. Downstream components parse OWL blocks with horned-owl
-- 3. Zero semantic loss from source to database
--
-- Run with: sqlite3 ontology.db < scripts/migrate_ontology_database.sql

BEGIN TRANSACTION;

-- Add new columns to owl_classes table
ALTER TABLE owl_classes ADD COLUMN markdown_content TEXT;
ALTER TABLE owl_classes ADD COLUMN file_sha1 TEXT;
ALTER TABLE owl_classes ADD COLUMN last_synced DATETIME;

-- Create index on file_sha1 for efficient change detection
CREATE INDEX IF NOT EXISTS idx_owl_classes_sha1 ON owl_classes(file_sha1);

-- Verify migration
SELECT
    COUNT(*) as total_classes,
    COUNT(markdown_content) as classes_with_markdown,
    COUNT(file_sha1) as classes_with_sha1
FROM owl_classes;

COMMIT;

-- Migration complete
-- Next steps:
-- 1. Run GitHub sync to populate markdown_content and file_sha1
-- 2. Use owl_extractor_service to parse OWL blocks from markdown
-- 3. Verify all semantic data is preserved
