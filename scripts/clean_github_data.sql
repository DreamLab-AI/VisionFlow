-- Clean existing GitHub sync data before re-sync
-- This script removes all GitHub-sourced nodes and edges from the knowledge graph database

BEGIN TRANSACTION;

-- Delete all edges first (foreign key constraints)
DELETE FROM kg_edges WHERE source IN (
    SELECT id FROM kg_nodes WHERE metadata LIKE '%"source":"github"%'
) OR target IN (
    SELECT id FROM kg_nodes WHERE metadata LIKE '%"source":"github"%'
);

-- Delete all GitHub-sourced nodes
DELETE FROM kg_nodes WHERE metadata LIKE '%"source":"github"%' OR metadata LIKE '%github%';

-- Optional: Clean up orphaned edges (edges pointing to non-existent nodes)
DELETE FROM kg_edges WHERE source NOT IN (SELECT id FROM kg_nodes);
DELETE FROM kg_edges WHERE target NOT IN (SELECT id FROM kg_nodes);

COMMIT;

-- Optimize database
VACUUM;

-- Show final counts
SELECT 'Remaining nodes:' as info, COUNT(*) as count FROM kg_nodes
UNION ALL
SELECT 'Remaining edges:' as info, COUNT(*) as count FROM kg_edges;
