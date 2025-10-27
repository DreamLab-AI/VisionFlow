-- Clean ALL graph data (nodes and edges) from knowledge_graph.db
-- WARNING: This removes ALL data, not just GitHub data
-- Use this before a complete re-sync to ensure a clean state

BEGIN TRANSACTION;

-- Delete all edges first (foreign key constraints)
DELETE FROM kg_edges;

-- Delete all nodes
DELETE FROM kg_nodes;

-- Delete metadata if it exists
DELETE FROM kg_metadata WHERE 1=1;

COMMIT;

-- Optimize database
VACUUM;

-- Verify clean state
SELECT 'Final node count:' as info, COUNT(*) as count FROM kg_nodes
UNION ALL
SELECT 'Final edge count:' as info, COUNT(*) as count FROM kg_edges;
