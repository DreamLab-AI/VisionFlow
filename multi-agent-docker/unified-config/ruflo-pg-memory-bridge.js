/**
 * RuVector PostgreSQL Memory Bridge
 *
 * Drop-in backend that routes MCP memory operations (store/search/list/get/delete)
 * directly to the RuVector PostgreSQL database instead of local sql.js.
 *
 * When RUVECTOR_EMBEDDINGS_AVAILABLE=true, uses ruvector_embed_vec() for in-DB
 * embedding generation. Otherwise falls back to client-side Xenova ONNX.
 */

import pg from 'pg';
const { Pool } = pg;

// Connection pool — uses RUVECTOR_PG_CONNINFO or individual env vars
const pool = new Pool({
  host: process.env.RUVECTOR_PG_HOST || 'ruvector-postgres',
  port: parseInt(process.env.RUVECTOR_PG_PORT || '5432', 10),
  user: process.env.RUVECTOR_PG_USER || 'ruvector',
  password: process.env.RUVECTOR_PG_PASSWORD || 'ruvector',
  database: process.env.RUVECTOR_PG_DATABASE || 'ruvector',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000,
});

const EMBEDDINGS_AVAILABLE = process.env.RUVECTOR_EMBEDDINGS_AVAILABLE === 'true';
const EMBED_MODEL = 'all-MiniLM-L6-v2';
// Default project_id — 13 = project-claude (primary workspace)
const DEFAULT_PROJECT_ID = parseInt(process.env.RUVECTOR_PROJECT_ID || '13', 10);

// Lazy-loaded client-side embedding pipeline (Xenova ONNX fallback)
let _pipeline = null;
async function getClientEmbedding(text) {
  if (!_pipeline) {
    try {
      const transformers = await import('@xenova/transformers');
      _pipeline = await transformers.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    } catch {
      return null;
    }
  }
  const output = await _pipeline(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

/**
 * Store a memory entry in PostgreSQL.
 */
async function storeEntry({ key, value, namespace = 'default', tags, ttl, upsert = true, generateEmbedding = true, projectId = DEFAULT_PROJECT_ID }) {
  const client = await pool.connect();
  try {
    let valueJson;
    if (typeof value === 'string') {
      try { JSON.parse(value); valueJson = value; } catch { valueJson = JSON.stringify(value); }
    } else {
      valueJson = JSON.stringify(value);
    }
    const metadataJson = JSON.stringify({ tags: tags || [], source: 'ruflo-pg-bridge' });

    let embeddingExpr = 'NULL';
    let embeddingJsonExpr = 'NULL';
    const params = [key, namespace, valueJson, metadataJson, projectId];
    let paramIdx = 6;

    if (generateEmbedding) {
      if (EMBEDDINGS_AVAILABLE) {
        embeddingExpr = `ruvector_embed_vec($${paramIdx}, '${EMBED_MODEL}')`;
        params.push(valueJson);
        paramIdx++;
      } else {
        const embedding = await getClientEmbedding(valueJson);
        if (embedding) {
          embeddingExpr = `$${paramIdx}::ruvector`;
          params.push(`[${embedding.join(',')}]`);
          paramIdx++;
          embeddingJsonExpr = `$${paramIdx}::jsonb`;
          params.push(JSON.stringify(embedding));
          paramIdx++;
        }
      }
    }

    // Unique constraint is (project_id, namespace, key)
    const sql = upsert
      ? `INSERT INTO memory_entries (id, project_id, key, namespace, source_type, value, metadata, embedding, embedding_json, updated_at)
         VALUES (gen_random_uuid()::text, $5, $1, $2, 'claude', $3::jsonb, $4::jsonb, ${embeddingExpr}, ${embeddingJsonExpr}, now())
         ON CONFLICT (project_id, namespace, key) DO UPDATE SET
           value = EXCLUDED.value,
           metadata = EXCLUDED.metadata,
           embedding = EXCLUDED.embedding,
           embedding_json = EXCLUDED.embedding_json,
           updated_at = now()
         RETURNING id, key, namespace, (embedding IS NOT NULL) as "hasEmbedding"`
      : `INSERT INTO memory_entries (id, project_id, key, namespace, source_type, value, metadata, embedding, embedding_json)
         VALUES (gen_random_uuid()::text, $5, $1, $2, 'claude', $3::jsonb, $4::jsonb, ${embeddingExpr}, ${embeddingJsonExpr})
         RETURNING id, key, namespace, (embedding IS NOT NULL) as "hasEmbedding"`;

    const result = await client.query(sql, params);
    return result.rows[0] || { key, namespace, hasEmbedding: false };
  } finally {
    client.release();
  }
}

/**
 * Search memory entries by vector similarity (KNN via HNSW index).
 */
async function searchEntries({ query, namespace, limit = 10, threshold = 0.0 }) {
  const client = await pool.connect();
  try {
    let embeddingExpr;
    const params = [];
    let paramIdx = 1;

    if (EMBEDDINGS_AVAILABLE) {
      embeddingExpr = `ruvector_embed_vec($${paramIdx}, '${EMBED_MODEL}')`;
      params.push(query);
      paramIdx++;
    } else {
      const embedding = await getClientEmbedding(query);
      if (!embedding) {
        // No embedding available — fall back to text search
        return textSearch({ query, namespace, limit });
      }
      embeddingExpr = `$${paramIdx}::ruvector`;
      params.push(`[${embedding.join(',')}]`);
      paramIdx++;
    }

    let whereClause = `WHERE embedding IS NOT NULL`;
    if (namespace) {
      whereClause += ` AND namespace = $${paramIdx}`;
      params.push(namespace);
      paramIdx++;
    }

    params.push(limit);

    const sql = `
      SELECT key, namespace, value, metadata,
             (1 - (embedding <=> ${embeddingExpr}))::numeric(6,4) as similarity,
             created_at, updated_at
      FROM memory_entries
      ${whereClause}
      ORDER BY embedding <=> ${embeddingExpr}
      LIMIT $${paramIdx}`;

    const result = await client.query(sql, params);
    return result.rows.filter(r => parseFloat(r.similarity) >= threshold);
  } finally {
    client.release();
  }
}

/**
 * Fallback text search when embeddings unavailable.
 */
async function textSearch({ query, namespace, limit = 10 }) {
  const client = await pool.connect();
  try {
    const params = [`%${query}%`];
    let paramIdx = 2;
    let whereClause = `WHERE (key ILIKE $1 OR value::text ILIKE $1)`;

    if (namespace) {
      whereClause += ` AND namespace = $${paramIdx}`;
      params.push(namespace);
      paramIdx++;
    }

    params.push(limit);

    const sql = `
      SELECT key, namespace, value, metadata, 0.5 as similarity, created_at, updated_at
      FROM memory_entries
      ${whereClause}
      ORDER BY updated_at DESC
      LIMIT $${paramIdx}`;

    const result = await client.query(sql, params);
    return result.rows;
  } finally {
    client.release();
  }
}

/**
 * List memory entries by namespace.
 */
async function listEntries({ namespace, limit = 50, offset = 0 }) {
  const client = await pool.connect();
  try {
    const params = [];
    let paramIdx = 1;
    let whereClause = '';

    if (namespace) {
      whereClause = `WHERE namespace = $${paramIdx}`;
      params.push(namespace);
      paramIdx++;
    }

    params.push(limit, offset);

    const sql = `
      SELECT key, namespace, value, metadata,
             (embedding IS NOT NULL) as "hasEmbedding",
             created_at, updated_at
      FROM memory_entries
      ${whereClause}
      ORDER BY updated_at DESC
      LIMIT $${paramIdx} OFFSET $${paramIdx + 1}`;

    const result = await client.query(sql, params);
    return result.rows;
  } finally {
    client.release();
  }
}

/**
 * Get a single entry by key and namespace.
 */
async function getEntry({ key, namespace = 'default', projectId = DEFAULT_PROJECT_ID }) {
  const client = await pool.connect();
  try {
    const sql = `
      SELECT key, namespace, value, metadata,
             (embedding IS NOT NULL) as "hasEmbedding",
             created_at, updated_at
      FROM memory_entries
      WHERE key = $1 AND namespace = $2 AND project_id = $3`;

    const result = await client.query(sql, [key, namespace, projectId]);
    return result.rows[0] || null;
  } finally {
    client.release();
  }
}

/**
 * Delete a memory entry by key and namespace.
 */
async function deleteEntry({ key, namespace = 'default', projectId = DEFAULT_PROJECT_ID }) {
  const client = await pool.connect();
  try {
    const sql = `DELETE FROM memory_entries WHERE key = $1 AND namespace = $2 AND project_id = $3 RETURNING key, namespace`;
    const result = await client.query(sql, [key, namespace, projectId]);
    return { deleted: result.rowCount > 0, key, namespace };
  } finally {
    client.release();
  }
}

/**
 * Get memory statistics.
 */
async function getStats({ namespace } = {}) {
  const client = await pool.connect();
  try {
    const params = [];
    let whereClause = '';
    if (namespace) {
      whereClause = 'WHERE namespace = $1';
      params.push(namespace);
    }

    const sql = `
      SELECT
        count(id) as total_entries,
        count(id) FILTER (WHERE embedding IS NOT NULL) as embedded_entries,
        count(DISTINCT namespace) as namespaces,
        pg_size_pretty(pg_total_relation_size('memory_entries')) as table_size
      FROM memory_entries ${whereClause}`;

    const result = await client.query(sql, params);
    const stats = result.rows[0];
    return {
      totalEntries: parseInt(stats.total_entries, 10),
      embeddedEntries: parseInt(stats.embedded_entries, 10),
      embeddingCoverage: stats.total_entries > 0
        ? ((stats.embedded_entries / stats.total_entries) * 100).toFixed(1) + '%'
        : '0%',
      namespaces: parseInt(stats.namespaces, 10),
      tableSize: stats.table_size,
      backend: 'ruvector-postgres',
      embeddingsProvider: EMBEDDINGS_AVAILABLE ? 'ruvector-native' : 'xenova-onnx-client',
    };
  } finally {
    client.release();
  }
}

/**
 * Test connection to PostgreSQL.
 */
async function testConnection() {
  const client = await pool.connect();
  try {
    await client.query('SELECT 1');

    // Test embedding capability
    let embeddingsWork = false;
    try {
      await client.query(`SELECT ruvector_embed_vec('test', '${EMBED_MODEL}')::text LIMIT 1`);
      embeddingsWork = true;
    } catch {
      embeddingsWork = false;
    }

    return { connected: true, embeddingsAvailable: embeddingsWork };
  } finally {
    client.release();
  }
}

// Graceful shutdown
process.on('SIGTERM', () => pool.end());
process.on('SIGINT', () => pool.end());

export {
  storeEntry,
  searchEntries,
  listEntries,
  getEntry,
  deleteEntry,
  getStats,
  testConnection,
  pool,
};
