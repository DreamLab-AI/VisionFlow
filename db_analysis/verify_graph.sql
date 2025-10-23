-- VisionFlow Knowledge Graph Verification Script
-- ================================================
-- Run after graph build to verify data integrity
-- Usage: docker exec visionflow_container sqlite3 /app/data/knowledge_graph.db < verify_graph.sql

.mode column
.headers on

-- Basic counts
SELECT 'Node Count' as Metric, COUNT(*) as Value FROM nodes
UNION ALL
SELECT 'Edge Count', COUNT(*) FROM edges
UNION ALL
SELECT 'KG Nodes', COUNT(*) FROM kg_nodes
UNION ALL
SELECT 'KG Edges', COUNT(*) FROM kg_edges
UNION ALL
SELECT 'Files Processed', COUNT(*) FROM file_metadata;

.print ''
.print '=== Node Type Distribution ==='
SELECT
    COALESCE(node_type, 'NULL') as NodeType,
    COUNT(*) as Count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM nodes), 2) || '%' as Percentage
FROM nodes
GROUP BY node_type
ORDER BY Count DESC;

.print ''
.print '=== Edge Type Distribution ==='
SELECT
    COALESCE(edge_type, 'NULL') as EdgeType,
    COUNT(*) as Count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM edges), 2) || '%' as Percentage
FROM edges
GROUP BY edge_type
ORDER BY Count DESC
LIMIT 10;

.print ''
.print '=== Top 10 Most Connected Nodes (Hubs) ==='
SELECT
    n.label as NodeLabel,
    n.node_type as Type,
    COUNT(DISTINCT e.id) as Connections
FROM nodes n
LEFT JOIN edges e ON n.id = e.source OR n.id = e.target
GROUP BY n.id, n.label, n.node_type
ORDER BY Connections DESC
LIMIT 10;

.print ''
.print '=== Orphaned Nodes (No Connections) ==='
SELECT COUNT(*) as OrphanedCount
FROM nodes
WHERE id NOT IN (SELECT source FROM edges)
  AND id NOT IN (SELECT target FROM edges);

.print ''
.print '=== Files with Most Nodes ==='
SELECT
    COALESCE(source_file, 'Unknown') as SourceFile,
    COUNT(*) as NodeCount
FROM nodes
WHERE source_file IS NOT NULL
GROUP BY source_file
ORDER BY NodeCount DESC
LIMIT 15;

.print ''
.print '=== Graph Metadata ==='
SELECT
    key as MetricKey,
    value as MetricValue,
    data_type as DataType,
    updated_at as LastUpdated
FROM graph_metadata
ORDER BY key;

.print ''
.print '=== Graph Statistics Summary ==='
SELECT
    (SELECT COUNT(*) FROM nodes) as TotalNodes,
    (SELECT COUNT(*) FROM edges) as TotalEdges,
    (SELECT COUNT(DISTINCT source_file) FROM nodes WHERE source_file IS NOT NULL) as UniqueFiles,
    (SELECT COUNT(*) FROM file_metadata) as ProcessedFiles,
    (SELECT COUNT(*) FROM graph_snapshots) as SnapshotCount,
    ROUND((SELECT AVG(degree) FROM (
        SELECT COUNT(*) as degree FROM edges GROUP BY source
    )), 2) as AvgOutDegree;

.print ''
.print '=== Recent Updates ==='
SELECT
    'Nodes' as Table,
    MAX(updated_at) as LastUpdate
FROM nodes
UNION ALL
SELECT
    'Edges',
    MAX(created_at)
FROM edges;
