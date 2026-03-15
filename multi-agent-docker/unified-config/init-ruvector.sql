-- RuVector PostgreSQL Schema Extensions
-- Replaces pgvector types with native ruvector types for full embedding support

-- Ensure ruvector extension is loaded (replaces pgvector)
CREATE EXTENSION IF NOT EXISTS ruvector;

-- Fix 8: HNSW Vector Indexes (ruvector native types)
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS embedding ruvector(384);
ALTER TABLE patterns ADD COLUMN IF NOT EXISTS embedding ruvector(384);

CREATE INDEX IF NOT EXISTS idx_memory_embedding_hnsw ON memory_entries
  USING hnsw (embedding ruvector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_patterns_embedding_hnsw ON patterns
  USING hnsw (embedding ruvector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Fix 9: Project Identification Schema
ALTER TABLE projects ADD COLUMN IF NOT EXISTS git_remote TEXT;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS pkg_name TEXT;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS sig_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_projects_git_remote ON projects(git_remote);
CREATE INDEX IF NOT EXISTS idx_projects_pkg_name ON projects(pkg_name);

-- Verify in-DB embedding generation works (requires --features embeddings build)
DO $$
BEGIN
  PERFORM ruvector_embed_vec('test embedding generation', 'all-MiniLM-L6-v2');
  RAISE NOTICE 'ruvector_embed_vec: OK — in-database embeddings available';
EXCEPTION WHEN OTHERS THEN
  RAISE WARNING 'ruvector_embed_vec not available: % — falling back to client-side ONNX', SQLERRM;
END $$;
