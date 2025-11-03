-- Migration: Add inference_cache table for storing reasoning results
-- Created: 2025-11-03

-- Create inference_cache table for caching reasoning results
CREATE TABLE IF NOT EXISTS inference_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_id TEXT NOT NULL,
    ontology_checksum TEXT NOT NULL,
    inferred_axioms TEXT NOT NULL,  -- JSON serialized InferredAxiom array
    timestamp INTEGER NOT NULL,      -- Unix timestamp
    inference_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ontology_id, ontology_checksum)
);

CREATE INDEX IF NOT EXISTS idx_inference_cache_ontology_id
    ON inference_cache(ontology_id);

CREATE INDEX IF NOT EXISTS idx_inference_cache_timestamp
    ON inference_cache(timestamp);

-- Add user_defined column to owl_axioms to distinguish inferred vs explicit axioms
-- This is a best-effort migration - column may already exist
ALTER TABLE owl_axioms ADD COLUMN user_defined BOOLEAN DEFAULT 1;

-- Create index for querying inferred axioms
CREATE INDEX IF NOT EXISTS idx_owl_axioms_user_defined
    ON owl_axioms(user_defined);

-- Clean up expired cache entries (older than 7 days)
-- This will be handled by application logic, but we create a view for monitoring
CREATE VIEW IF NOT EXISTS expired_inference_cache AS
SELECT * FROM inference_cache
WHERE timestamp < (strftime('%s', 'now') - 604800);

COMMENT ON TABLE inference_cache IS 'Caches reasoning results to avoid recomputation';
COMMENT ON COLUMN inference_cache.ontology_checksum IS 'Blake3 hash of ontology state';
COMMENT ON COLUMN inference_cache.inferred_axioms IS 'JSON array of InferredAxiom objects';
