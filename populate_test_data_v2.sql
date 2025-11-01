-- Test data matching actual schema
INSERT OR REPLACE INTO nodes (metadata_id, label, x, y, z, color, size)
VALUES
('test-1', 'Database', 0.0, 0.0, 0.0, '#FF5733', 15.0),
('test-2', 'GPU', 50.0, 0.0, 0.0, '#33FF57', 15.0),
('test-3', 'Client', -50.0, 0.0, 0.0, '#3357FF', 15.0),
('test-4', 'WebGL', 0.0, 50.0, 0.0, '#FF33F5', 15.0),
('test-5', 'Physics', 0.0, -50.0, 0.0, '#F5FF33', 15.0);

INSERT OR REPLACE INTO edges (id, source, target, edge_type, weight)
VALUES
('test-edge-1', 1, 2, 'related', 1.0),
('test-edge-2', 2, 3, 'related', 1.0),
('test-edge-3', 3, 4, 'related', 1.0),
('test-edge-4', 4, 5, 'related', 1.0),
('test-edge-5', 5, 1, 'related', 1.0);

SELECT 'Nodes:', COUNT(*) FROM nodes;
SELECT 'Edges:', COUNT(*) FROM edges;
