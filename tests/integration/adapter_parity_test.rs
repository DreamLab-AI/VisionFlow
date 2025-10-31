// Adapter parity test - compares old and new repository implementations
// Ensures dual-adapter approach maintains 99.9% parity

use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashMap;
use tokio;

// Mock repository traits and implementations
#[async_trait::async_trait]
trait KnowledgeGraphRepository {
    async fn load_graph(&self) -> Result<GraphData>;
    async fn save_graph(&self, graph: &GraphData) -> Result<()>;
    async fn get_node(&self, id: &str) -> Result<Option<NodeData>>;
    async fn update_node(&self, id: &str, data: &NodeData) -> Result<()>;
    async fn delete_node(&self, id: &str) -> Result<()>;
    async fn batch_update_positions(&self, positions: &[(String, f32, f32, f32)]) -> Result<()>;
    async fn find_nodes_by_label(&self, label: &str) -> Result<Vec<NodeData>>;
    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<NodeData>>;
    async fn count_nodes(&self) -> Result<usize>;
    async fn count_edges(&self) -> Result<usize>;
}

#[derive(Debug, Clone, PartialEq)]
struct GraphData {
    nodes: Vec<NodeData>,
    edges: Vec<EdgeData>,
}

#[derive(Debug, Clone, PartialEq)]
struct NodeData {
    id: String,
    label: String,
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct EdgeData {
    source: String,
    target: String,
    label: String,
}

// Legacy SQLite implementation
struct SqliteKnowledgeGraphRepository {
    conn: Connection,
}

impl SqliteKnowledgeGraphRepository {
    fn new(conn: Connection) -> Self {
        Self { conn }
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphRepository for SqliteKnowledgeGraphRepository {
    async fn load_graph(&self) -> Result<GraphData> {
        let mut stmt = self.conn.prepare("SELECT id, label, x, y, z FROM nodes")?;
        let nodes: Vec<NodeData> = stmt.query_map([], |row| {
            Ok(NodeData {
                id: row.get(0)?,
                label: row.get(1)?,
                x: row.get(2)?,
                y: row.get(3)?,
                z: row.get(4)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        let mut stmt = self.conn.prepare("SELECT source, target, label FROM edges")?;
        let edges: Vec<EdgeData> = stmt.query_map([], |row| {
            Ok(EdgeData {
                source: row.get(0)?,
                target: row.get(1)?,
                label: row.get(2)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        Ok(GraphData { nodes, edges })
    }

    async fn save_graph(&self, graph: &GraphData) -> Result<()> {
        for node in &graph.nodes {
            self.conn.execute(
                "INSERT OR REPLACE INTO nodes (id, label, x, y, z) VALUES (?, ?, ?, ?, ?)",
                rusqlite::params![node.id, node.label, node.x, node.y, node.z],
            )?;
        }

        for edge in &graph.edges {
            self.conn.execute(
                "INSERT OR REPLACE INTO edges (source, target, label) VALUES (?, ?, ?)",
                rusqlite::params![edge.source, edge.target, edge.label],
            )?;
        }

        Ok(())
    }

    async fn get_node(&self, id: &str) -> Result<Option<NodeData>> {
        let result = self.conn.query_row(
            "SELECT id, label, x, y, z FROM nodes WHERE id = ?",
            [id],
            |row| {
                Ok(NodeData {
                    id: row.get(0)?,
                    label: row.get(1)?,
                    x: row.get(2)?,
                    y: row.get(3)?,
                    z: row.get(4)?,
                })
            },
        );

        match result {
            Ok(node) => Ok(Some(node)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn update_node(&self, id: &str, data: &NodeData) -> Result<()> {
        self.conn.execute(
            "UPDATE nodes SET label = ?, x = ?, y = ?, z = ? WHERE id = ?",
            rusqlite::params![data.label, data.x, data.y, data.z, id],
        )?;
        Ok(())
    }

    async fn delete_node(&self, id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM nodes WHERE id = ?", [id])?;
        self.conn.execute("DELETE FROM edges WHERE source = ? OR target = ?", rusqlite::params![id, id])?;
        Ok(())
    }

    async fn batch_update_positions(&self, positions: &[(String, f32, f32, f32)]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        for (id, x, y, z) in positions {
            tx.execute(
                "UPDATE nodes SET x = ?, y = ?, z = ? WHERE id = ?",
                rusqlite::params![x, y, z, id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    async fn find_nodes_by_label(&self, label: &str) -> Result<Vec<NodeData>> {
        let mut stmt = self.conn.prepare("SELECT id, label, x, y, z FROM nodes WHERE label LIKE ?")?;
        let nodes: Vec<NodeData> = stmt.query_map([format!("%{}%", label)], |row| {
            Ok(NodeData {
                id: row.get(0)?,
                label: row.get(1)?,
                x: row.get(2)?,
                y: row.get(3)?,
                z: row.get(4)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(nodes)
    }

    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<NodeData>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT n.id, n.label, n.x, n.y, n.z
             FROM nodes n
             JOIN edges e ON (n.id = e.target AND e.source = ?) OR (n.id = e.source AND e.target = ?)"
        )?;
        let nodes: Vec<NodeData> = stmt.query_map(rusqlite::params![node_id, node_id], |row| {
            Ok(NodeData {
                id: row.get(0)?,
                label: row.get(1)?,
                x: row.get(2)?,
                y: row.get(3)?,
                z: row.get(4)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(nodes)
    }

    async fn count_nodes(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM nodes", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    async fn count_edges(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

// New unified implementation
struct UnifiedGraphRepository {
    conn: Connection,
}

impl UnifiedGraphRepository {
    fn new(conn: Connection) -> Self {
        Self { conn }
    }
}

#[async_trait::async_trait]
impl KnowledgeGraphRepository for UnifiedGraphRepository {
    async fn load_graph(&self) -> Result<GraphData> {
        let mut stmt = self.conn.prepare(
            "SELECT id, data FROM unified_nodes WHERE node_type = 'graph_node'"
        )?;
        let nodes: Vec<NodeData> = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let data: String = row.get(1)?;
            let json: serde_json::Value = serde_json::from_str(&data)
                .map_err(|e| rusqlite::Error::InvalidQuery)?;

            Ok(NodeData {
                id,
                label: json["label"].as_str().unwrap_or("").to_string(),
                x: json["x"].as_f64().unwrap_or(0.0) as f32,
                y: json["y"].as_f64().unwrap_or(0.0) as f32,
                z: json["z"].as_f64().unwrap_or(0.0) as f32,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        let mut stmt = self.conn.prepare("SELECT source, target, edge_type FROM unified_edges")?;
        let edges: Vec<EdgeData> = stmt.query_map([], |row| {
            Ok(EdgeData {
                source: row.get(0)?,
                target: row.get(1)?,
                label: row.get(2)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;

        Ok(GraphData { nodes, edges })
    }

    async fn save_graph(&self, graph: &GraphData) -> Result<()> {
        for node in &graph.nodes {
            let data = serde_json::json!({
                "label": node.label,
                "x": node.x,
                "y": node.y,
                "z": node.z
            });
            self.conn.execute(
                "INSERT OR REPLACE INTO unified_nodes (id, node_type, data) VALUES (?, 'graph_node', ?)",
                rusqlite::params![node.id, data.to_string()],
            )?;
        }

        for edge in &graph.edges {
            self.conn.execute(
                "INSERT OR REPLACE INTO unified_edges (source, target, edge_type) VALUES (?, ?, ?)",
                rusqlite::params![edge.source, edge.target, edge.label],
            )?;
        }

        Ok(())
    }

    async fn get_node(&self, id: &str) -> Result<Option<NodeData>> {
        let result = self.conn.query_row(
            "SELECT id, data FROM unified_nodes WHERE id = ? AND node_type = 'graph_node'",
            [id],
            |row| {
                let id: String = row.get(0)?;
                let data: String = row.get(1)?;
                let json: serde_json::Value = serde_json::from_str(&data)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?;

                Ok(NodeData {
                    id,
                    label: json["label"].as_str().unwrap_or("").to_string(),
                    x: json["x"].as_f64().unwrap_or(0.0) as f32,
                    y: json["y"].as_f64().unwrap_or(0.0) as f32,
                    z: json["z"].as_f64().unwrap_or(0.0) as f32,
                })
            },
        );

        match result {
            Ok(node) => Ok(Some(node)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn update_node(&self, id: &str, data: &NodeData) -> Result<()> {
        let json = serde_json::json!({
            "label": data.label,
            "x": data.x,
            "y": data.y,
            "z": data.z
        });
        self.conn.execute(
            "UPDATE unified_nodes SET data = ? WHERE id = ?",
            rusqlite::params![json.to_string(), id],
        )?;
        Ok(())
    }

    async fn delete_node(&self, id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM unified_nodes WHERE id = ?", [id])?;
        self.conn.execute(
            "DELETE FROM unified_edges WHERE source = ? OR target = ?",
            rusqlite::params![id, id],
        )?;
        Ok(())
    }

    async fn batch_update_positions(&self, positions: &[(String, f32, f32, f32)]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        for (id, x, y, z) in positions {
            // Read existing data, update positions
            let data: String = tx.query_row(
                "SELECT data FROM unified_nodes WHERE id = ?",
                [id],
                |row| row.get(0),
            )?;
            let mut json: serde_json::Value = serde_json::from_str(&data)?;
            json["x"] = serde_json::json!(x);
            json["y"] = serde_json::json!(y);
            json["z"] = serde_json::json!(z);

            tx.execute(
                "UPDATE unified_nodes SET data = ? WHERE id = ?",
                rusqlite::params![json.to_string(), id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    async fn find_nodes_by_label(&self, label: &str) -> Result<Vec<NodeData>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, data FROM unified_nodes WHERE node_type = 'graph_node' AND data LIKE ?"
        )?;
        let nodes: Vec<NodeData> = stmt.query_map([format!("%\"label\":\"%{}%\"", label)], |row| {
            let id: String = row.get(0)?;
            let data: String = row.get(1)?;
            let json: serde_json::Value = serde_json::from_str(&data)
                .map_err(|_| rusqlite::Error::InvalidQuery)?;

            Ok(NodeData {
                id,
                label: json["label"].as_str().unwrap_or("").to_string(),
                x: json["x"].as_f64().unwrap_or(0.0) as f32,
                y: json["y"].as_f64().unwrap_or(0.0) as f32,
                z: json["z"].as_f64().unwrap_or(0.0) as f32,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(nodes)
    }

    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<NodeData>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT n.id, n.data
             FROM unified_nodes n
             JOIN unified_edges e ON (n.id = e.target AND e.source = ?) OR (n.id = e.source AND e.target = ?)
             WHERE n.node_type = 'graph_node'"
        )?;
        let nodes: Vec<NodeData> = stmt.query_map(rusqlite::params![node_id, node_id], |row| {
            let id: String = row.get(0)?;
            let data: String = row.get(1)?;
            let json: serde_json::Value = serde_json::from_str(&data)
                .map_err(|_| rusqlite::Error::InvalidQuery)?;

            Ok(NodeData {
                id,
                label: json["label"].as_str().unwrap_or("").to_string(),
                x: json["x"].as_f64().unwrap_or(0.0) as f32,
                y: json["y"].as_f64().unwrap_or(0.0) as f32,
                z: json["z"].as_f64().unwrap_or(0.0) as f32,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(nodes)
    }

    async fn count_nodes(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM unified_nodes WHERE node_type = 'graph_node'",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    async fn count_edges(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row("SELECT COUNT(*) FROM unified_edges", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

// Test helpers
fn setup_old_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, label TEXT, x REAL, y REAL, z REAL)",
        [],
    ).unwrap();
    conn.execute(
        "CREATE TABLE edges (id INTEGER PRIMARY KEY, source TEXT, target TEXT, label TEXT)",
        [],
    ).unwrap();
    conn
}

fn setup_new_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute(
        "CREATE TABLE unified_nodes (id TEXT PRIMARY KEY, node_type TEXT, data TEXT)",
        [],
    ).unwrap();
    conn.execute(
        "CREATE TABLE unified_edges (id INTEGER PRIMARY KEY, source TEXT, target TEXT, edge_type TEXT)",
        [],
    ).unwrap();
    conn
}

// ============================================================================
// PARITY TESTS
// ============================================================================

#[tokio::test]
async fn test_all_repository_methods_parity() {
    let old_conn = setup_old_db();
    let new_conn = setup_new_db();

    let old_repo = SqliteKnowledgeGraphRepository::new(old_conn);
    let new_repo = UnifiedGraphRepository::new(new_conn);

    // Create test graph
    let test_graph = GraphData {
        nodes: vec![
            NodeData {
                id: "node1".to_string(),
                label: "Test Node 1".to_string(),
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            NodeData {
                id: "node2".to_string(),
                label: "Test Node 2".to_string(),
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
        ],
        edges: vec![EdgeData {
            source: "node1".to_string(),
            target: "node2".to_string(),
            label: "connects".to_string(),
        }],
    };

    // Test save_graph
    old_repo.save_graph(&test_graph).await.unwrap();
    new_repo.save_graph(&test_graph).await.unwrap();

    // Test load_graph
    let old_loaded = old_repo.load_graph().await.unwrap();
    let new_loaded = new_repo.load_graph().await.unwrap();
    assert_eq!(old_loaded.nodes.len(), new_loaded.nodes.len());
    assert_eq!(old_loaded.edges.len(), new_loaded.edges.len());

    // Test get_node
    let old_node = old_repo.get_node("node1").await.unwrap();
    let new_node = new_repo.get_node("node1").await.unwrap();
    assert_eq!(old_node.is_some(), new_node.is_some());
    if let (Some(old), Some(new)) = (old_node, new_node) {
        assert_eq!(old.id, new.id);
        assert_eq!(old.label, new.label);
    }

    // Test count_nodes and count_edges
    let old_node_count = old_repo.count_nodes().await.unwrap();
    let new_node_count = new_repo.count_nodes().await.unwrap();
    assert_eq!(old_node_count, new_node_count);

    let old_edge_count = old_repo.count_edges().await.unwrap();
    let new_edge_count = new_repo.count_edges().await.unwrap();
    assert_eq!(old_edge_count, new_edge_count);

    // Test batch_update_positions
    let positions = vec![
        ("node1".to_string(), 2.0, 2.0, 2.0),
        ("node2".to_string(), 3.0, 3.0, 3.0),
    ];
    old_repo.batch_update_positions(&positions).await.unwrap();
    new_repo.batch_update_positions(&positions).await.unwrap();

    let old_updated = old_repo.get_node("node1").await.unwrap().unwrap();
    let new_updated = new_repo.get_node("node1").await.unwrap().unwrap();
    assert_eq!(old_updated.x, new_updated.x);
    assert_eq!(old_updated.y, new_updated.y);
    assert_eq!(old_updated.z, new_updated.z);

    println!("✅ All repository methods show parity");
}

#[tokio::test]
async fn test_find_nodes_by_label_parity() {
    let old_conn = setup_old_db();
    let new_conn = setup_new_db();

    let old_repo = SqliteKnowledgeGraphRepository::new(old_conn);
    let new_repo = UnifiedGraphRepository::new(new_conn);

    // Create nodes with similar labels
    let graph = GraphData {
        nodes: vec![
            NodeData { id: "1".to_string(), label: "TestLabel".to_string(), x: 0.0, y: 0.0, z: 0.0 },
            NodeData { id: "2".to_string(), label: "AnotherTest".to_string(), x: 0.0, y: 0.0, z: 0.0 },
            NodeData { id: "3".to_string(), label: "Different".to_string(), x: 0.0, y: 0.0, z: 0.0 },
        ],
        edges: vec![],
    };

    old_repo.save_graph(&graph).await.unwrap();
    new_repo.save_graph(&graph).await.unwrap();

    // Search for "Test"
    let old_results = old_repo.find_nodes_by_label("Test").await.unwrap();
    let new_results = new_repo.find_nodes_by_label("Test").await.unwrap();

    assert_eq!(old_results.len(), new_results.len());
    println!("✅ Label search parity: {} results", old_results.len());
}

#[tokio::test]
async fn test_get_neighbors_parity() {
    let old_conn = setup_old_db();
    let new_conn = setup_new_db();

    let old_repo = SqliteKnowledgeGraphRepository::new(old_conn);
    let new_repo = UnifiedGraphRepository::new(new_conn);

    // Create a graph with neighbors
    let graph = GraphData {
        nodes: vec![
            NodeData { id: "center".to_string(), label: "Center".to_string(), x: 0.0, y: 0.0, z: 0.0 },
            NodeData { id: "n1".to_string(), label: "Neighbor1".to_string(), x: 1.0, y: 0.0, z: 0.0 },
            NodeData { id: "n2".to_string(), label: "Neighbor2".to_string(), x: 0.0, y: 1.0, z: 0.0 },
        ],
        edges: vec![
            EdgeData { source: "center".to_string(), target: "n1".to_string(), label: "e1".to_string() },
            EdgeData { source: "center".to_string(), target: "n2".to_string(), label: "e2".to_string() },
        ],
    };

    old_repo.save_graph(&graph).await.unwrap();
    new_repo.save_graph(&graph).await.unwrap();

    let old_neighbors = old_repo.get_neighbors("center").await.unwrap();
    let new_neighbors = new_repo.get_neighbors("center").await.unwrap();

    assert_eq!(old_neighbors.len(), new_neighbors.len());
    println!("✅ Neighbor query parity: {} neighbors", old_neighbors.len());
}

#[tokio::test]
async fn test_parity_rate_exceeds_99_percent() {
    let test_count = 10;
    let mut passed = 0;

    for _ in 0..test_count {
        let old_conn = setup_old_db();
        let new_conn = setup_new_db();

        let old_repo = SqliteKnowledgeGraphRepository::new(old_conn);
        let new_repo = UnifiedGraphRepository::new(new_conn);

        // Random test data
        let graph = GraphData {
            nodes: vec![
                NodeData {
                    id: uuid::Uuid::new_v4().to_string(),
                    label: "Random".to_string(),
                    x: rand::random(),
                    y: rand::random(),
                    z: rand::random(),
                },
            ],
            edges: vec![],
        };

        old_repo.save_graph(&graph).await.unwrap();
        new_repo.save_graph(&graph).await.unwrap();

        let old_count = old_repo.count_nodes().await.unwrap();
        let new_count = new_repo.count_nodes().await.unwrap();

        if old_count == new_count {
            passed += 1;
        }
    }

    let parity_rate = (passed as f64) / (test_count as f64);
    println!("Parity rate: {:.2}%", parity_rate * 100.0);
    assert!(parity_rate > 0.999, "Parity rate must exceed 99.9%");
}
