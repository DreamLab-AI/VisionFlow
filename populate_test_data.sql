-- Test data to verify complete data flow pipeline
-- GitHub → Database → GPU → Client

-- Insert test nodes
INSERT OR REPLACE INTO nodes (id, metadata_id, label, x, y, z, vx, vy, vz, ax, ay, az, mass, charge, damping, is_pinned, node_type, color, size_value, parent_id, depth, created_at, updated_at)
VALUES
(1, 'test-node-1', 'Data Flow Test Node 1', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0, 'concept', '#FF5733', 1.0, NULL, 0, datetime('now'), datetime('now')),
(2, 'test-node-2', 'GPU Pipeline Test', 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0, 'concept', '#33FF57', 1.0, NULL, 0, datetime('now'), datetime('now')),
(3, 'test-node-3', 'Client Render Test', -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0, 'concept', '#3357FF', 1.0, NULL, 0, datetime('now'), datetime('now')),
(4, 'test-node-4', 'WebGL Test Node', 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0, 'concept', '#FF33F5', 1.0, NULL, 0, datetime('now'), datetime('now')),
(5, 'test-node-5', 'Physics Simulation', 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.9, 0, 'concept', '#F5FF33', 1.0, NULL, 0, datetime('now'), datetime('now'));

-- Insert test edges
INSERT OR REPLACE INTO edges (id, source, target, edge_type, weight, created_at, updated_at)
VALUES
(1, 1, 2, 'related', 1.0, datetime('now'), datetime('now')),
(2, 2, 3, 'related', 1.0, datetime('now'), datetime('now')),
(3, 3, 4, 'related', 1.0, datetime('now'), datetime('now')),
(4, 4, 5, 'related', 1.0, datetime('now'), datetime('now')),
(5, 5, 1, 'related', 1.0, datetime('now'), datetime('now'));

-- Verify inserts
SELECT 'Nodes inserted:', COUNT(*) FROM nodes;
SELECT 'Edges inserted:', COUNT(*) FROM edges;
