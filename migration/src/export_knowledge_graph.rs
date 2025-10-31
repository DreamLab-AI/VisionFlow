use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use sqlx::{sqlite::SqlitePool, Row};
use std::fs::File;
use std::io::Write;

/// Node with complete physics state from knowledge_graph.db
#[derive(Debug, Serialize, Deserialize)]
struct GraphNode {
    id: i64,
    metadata_id: String,
    label: String,
    // Physics state
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    mass: f64,
    // Additional metadata
    node_type: Option<String>,
    category: Option<String>,
}

/// Edge with weight from knowledge_graph.db
#[derive(Debug, Serialize, Deserialize)]
struct GraphEdge {
    id: i64,
    source_id: i64,
    target_id: i64,
    relationship_type: String,
    weight: f64,
    metadata: Option<String>,
}

/// Clustering results
#[derive(Debug, Serialize, Deserialize)]
struct ClusterResult {
    node_id: i64,
    cluster_id: i32,
    algorithm: String,
    timestamp: String,
}

/// Pathfinding cache entry
#[derive(Debug, Serialize, Deserialize)]
struct PathfindingCache {
    source_id: i64,
    target_id: i64,
    path_nodes: String, // JSON array
    distance: f64,
    cached_at: String,
}

/// Complete export result with checksums
#[derive(Debug, Serialize, Deserialize)]
struct KnowledgeGraphExport {
    export_timestamp: String,
    database_path: String,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    clustering: Vec<ClusterResult>,
    pathfinding_cache: Vec<PathfindingCache>,
    // Checksums for verification
    nodes_sha1: String,
    edges_sha1: String,
    clustering_sha1: String,
    pathfinding_sha1: String,
    total_sha1: String,
}

/// Compute SHA1 checksum of serialized data
fn compute_sha1<T: Serialize>(data: &[T]) -> Result<String> {
    let json = serde_json::to_string(data)?;
    let mut hasher = Sha1::new();
    hasher.update(json.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 VisionFlow Knowledge Graph Export");
    println!("=====================================\n");

    // Database path (adjust as needed)
    let db_path = std::env::var("KG_DATABASE_URL")
        .unwrap_or_else(|_| "sqlite:///home/devuser/workspace/project/data/knowledge_graph.db".to_string());

    println!("📊 Connecting to: {}", db_path);
    let pool = SqlitePool::connect(&db_path)
        .await
        .context("Failed to connect to knowledge_graph.db")?;

    // Export nodes with all physics state
    println!("📦 Exporting nodes...");
    let nodes: Vec<GraphNode> = sqlx::query(
        r#"
        SELECT
            id, metadata_id, label,
            x, y, z, vx, vy, vz, mass,
            node_type, category
        FROM nodes
        ORDER BY id
        "#
    )
    .map(|row: sqlx::sqlite::SqliteRow| GraphNode {
        id: row.get("id"),
        metadata_id: row.get("metadata_id"),
        label: row.get("label"),
        x: row.get("x"),
        y: row.get("y"),
        z: row.get("z"),
        vx: row.get("vx"),
        vy: row.get("vy"),
        vz: row.get("vz"),
        mass: row.get("mass"),
        node_type: row.get("node_type"),
        category: row.get("category"),
    })
    .fetch_all(&pool)
    .await
    .context("Failed to export nodes")?;

    println!("   ✅ Exported {} nodes", nodes.len());

    // Export edges with weights
    println!("📦 Exporting edges...");
    let edges: Vec<GraphEdge> = sqlx::query(
        r#"
        SELECT
            id, source_id, target_id,
            relationship_type, weight, metadata
        FROM edges
        ORDER BY id
        "#
    )
    .map(|row: sqlx::sqlite::SqliteRow| GraphEdge {
        id: row.get("id"),
        source_id: row.get("source_id"),
        target_id: row.get("target_id"),
        relationship_type: row.get("relationship_type"),
        weight: row.get("weight"),
        metadata: row.get("metadata"),
    })
    .fetch_all(&pool)
    .await
    .context("Failed to export edges")?;

    println!("   ✅ Exported {} edges", edges.len());

    // Export clustering results
    println!("📦 Exporting clustering results...");
    let clustering: Vec<ClusterResult> = sqlx::query(
        r#"
        SELECT
            node_id, cluster_id, algorithm,
            datetime(timestamp) as timestamp
        FROM clustering_results
        ORDER BY node_id
        "#
    )
    .map(|row: sqlx::sqlite::SqliteRow| ClusterResult {
        node_id: row.get("node_id"),
        cluster_id: row.get("cluster_id"),
        algorithm: row.get("algorithm"),
        timestamp: row.get("timestamp"),
    })
    .fetch_all(&pool)
    .await
    .context("Failed to export clustering results")?;

    println!("   ✅ Exported {} clustering results", clustering.len());

    // Export pathfinding cache
    println!("📦 Exporting pathfinding cache...");
    let pathfinding_cache: Vec<PathfindingCache> = sqlx::query(
        r#"
        SELECT
            source_id, target_id, path_nodes,
            distance, datetime(cached_at) as cached_at
        FROM pathfinding_cache
        ORDER BY source_id, target_id
        "#
    )
    .map(|row: sqlx::sqlite::SqliteRow| PathfindingCache {
        source_id: row.get("source_id"),
        target_id: row.get("target_id"),
        path_nodes: row.get("path_nodes"),
        distance: row.get("distance"),
        cached_at: row.get("cached_at"),
    })
    .fetch_all(&pool)
    .await
    .context("Failed to export pathfinding cache")?;

    println!("   ✅ Exported {} pathfinding entries", pathfinding_cache.len());

    // Compute checksums
    println!("\n🔐 Computing checksums...");
    let nodes_sha1 = compute_sha1(&nodes)?;
    let edges_sha1 = compute_sha1(&edges)?;
    let clustering_sha1 = compute_sha1(&clustering)?;
    let pathfinding_sha1 = compute_sha1(&pathfinding_cache)?;

    println!("   ✅ Nodes checksum: {}", nodes_sha1);
    println!("   ✅ Edges checksum: {}", edges_sha1);
    println!("   ✅ Clustering checksum: {}", clustering_sha1);
    println!("   ✅ Pathfinding checksum: {}", pathfinding_sha1);

    // Create export structure
    let export = KnowledgeGraphExport {
        export_timestamp: format!("{}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")),
        database_path: db_path.clone(),
        nodes,
        edges,
        clustering,
        pathfinding_cache,
        nodes_sha1: nodes_sha1.clone(),
        edges_sha1: edges_sha1.clone(),
        clustering_sha1: clustering_sha1.clone(),
        pathfinding_sha1: pathfinding_sha1.clone(),
        total_sha1: {
            let combined = format!("{}{}{}{}", nodes_sha1, edges_sha1, clustering_sha1, pathfinding_sha1);
            let mut hasher = Sha1::new();
            hasher.update(combined.as_bytes());
            format!("{:x}", hasher.finalize())
        },
    };

    // Write to JSON
    let output_path = "/home/devuser/workspace/project/migration/knowledge_graph_export.json";
    println!("\n💾 Writing export to: {}", output_path);

    let json = serde_json::to_string_pretty(&export)
        .context("Failed to serialize export")?;

    let mut file = File::create(output_path)
        .context("Failed to create output file")?;

    file.write_all(json.as_bytes())
        .context("Failed to write output file")?;

    println!("\n✅ Export complete!");
    println!("   Total checksum: {}", export.total_sha1);
    println!("   Output: {}", output_path);
    println!("   Size: {} bytes", json.len());

    pool.close().await;
    Ok(())
}
