use anyhow::{Context, Result};
use serde::Deserialize;
use sqlx::{sqlite::SqlitePool, Row};
use std::fs::File;
use std::io::Read;

// Import types from transform script
#[derive(Debug, Deserialize)]
struct TransformedData {
    unified_nodes: Vec<UnifiedNode>,
    unified_edges: Vec<GraphEdge>,
    owl_classes: Vec<OwlClass>,
    owl_axioms: Vec<OwlAxiom>,
    owl_properties: Vec<OwlProperty>,
    clustering: Vec<ClusterResult>,
    pathfinding_cache: Vec<PathfindingCache>,
    reasoning_cache: Vec<ReasoningCache>,
    #[allow(dead_code)]
    checksum: String,
}

#[derive(Debug, Deserialize)]
struct UnifiedNode {
    id: i64,
    metadata_id: String,
    label: String,
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    mass: f64,
    owl_class_iri: Option<String>,
    node_type: Option<String>,
    category: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GraphEdge {
    id: i64,
    source_id: i64,
    target_id: i64,
    relationship_type: String,
    weight: f64,
    metadata: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OwlClass {
    id: i64,
    iri: String,
    label: Option<String>,
    parent_class_iri: Option<String>,
    markdown_content: Option<String>,
    file_path: Option<String>,
    file_sha1: Option<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Debug, Deserialize)]
struct OwlAxiom {
    id: i64,
    axiom_type: String,
    subject_iri: Option<String>,
    object_iri: Option<String>,
    property_iri: Option<String>,
    strength: f64,
    priority: i32,
    user_defined: bool,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct OwlProperty {
    id: i64,
    iri: String,
    property_type: String,
    label: Option<String>,
    domain_iri: Option<String>,
    range_iri: Option<String>,
    functional: bool,
    inverse_functional: bool,
    transitive: bool,
    symmetric: bool,
}

#[derive(Debug, Deserialize)]
struct ClusterResult {
    node_id: i64,
    cluster_id: i32,
    algorithm: String,
    timestamp: String,
}

#[derive(Debug, Deserialize)]
struct PathfindingCache {
    source_id: i64,
    target_id: i64,
    path_nodes: String,
    distance: f64,
    cached_at: String,
}

#[derive(Debug, Deserialize)]
struct ReasoningCache {
    ontology_checksum: String,
    inferred_axiom_type: String,
    subject_iri: String,
    object_iri: String,
    cached_at: String,
}

const BATCH_SIZE: usize = 10_000;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 VisionFlow Unified Import");
    println!("=============================\n");

    // Load transformed data
    let input_path = "/home/devuser/workspace/project/migration/unified_transform.json";
    println!("📖 Loading transformed data from: {}", input_path);
    let mut file = File::open(input_path)
        .context("Failed to open transform file")?;
    let mut json = String::new();
    file.read_to_string(&mut json)?;
    let data: TransformedData = serde_json::from_str(&json)
        .context("Failed to parse transform file")?;
    println!("   ✅ Loaded data");

    // Connect to unified.db (create if not exists)
    let db_path = std::env::var("UNIFIED_DATABASE_URL")
        .unwrap_or_else(|_| "sqlite:///home/devuser/workspace/project/data/unified.db".to_string());

    println!("\n📊 Connecting to: {}", db_path);
    let pool = SqlitePool::connect(&db_path)
        .await
        .context("Failed to connect to unified.db")?;

    // Create schema
    println!("\n🏗️  Creating unified schema...");

    // OWL Classes table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS owl_classes (
            id INTEGER PRIMARY KEY,
            iri TEXT UNIQUE NOT NULL,
            label TEXT,
            parent_class_iri TEXT,
            markdown_content TEXT,
            file_path TEXT,
            file_sha1 TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_class_iri) REFERENCES owl_classes(iri)
        )
        "#
    )
    .execute(&pool)
    .await?;
    println!("   ✅ Created owl_classes table");

    // Graph nodes table (with OWL linkage)
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id INTEGER PRIMARY KEY,
            metadata_id TEXT UNIQUE NOT NULL,
            label TEXT NOT NULL,
            x REAL DEFAULT 0.0,
            y REAL DEFAULT 0.0,
            z REAL DEFAULT 0.0,
            vx REAL DEFAULT 0.0,
            vy REAL DEFAULT 0.0,
            vz REAL DEFAULT 0.0,
            mass REAL DEFAULT 1.0,
            owl_class_iri TEXT,
            node_type TEXT,
            category TEXT,
            FOREIGN KEY (owl_class_iri) REFERENCES owl_classes(iri)
        )
        "#
    )
    .execute(&pool)
    .await?;
    println!("   ✅ Created graph_nodes table");

    // Other tables
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS graph_edges (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relationship_type TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            metadata TEXT,
            FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
            FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
        )
        "#
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS owl_axioms (
            id INTEGER PRIMARY KEY,
            axiom_type TEXT NOT NULL,
            subject_iri TEXT,
            object_iri TEXT,
            property_iri TEXT,
            strength REAL DEFAULT 1.0,
            priority INTEGER DEFAULT 5,
            user_defined BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        "#
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS owl_properties (
            id INTEGER PRIMARY KEY,
            iri TEXT UNIQUE NOT NULL,
            property_type TEXT NOT NULL,
            label TEXT,
            domain_iri TEXT,
            range_iri TEXT,
            functional BOOLEAN DEFAULT 0,
            inverse_functional BOOLEAN DEFAULT 0,
            transitive BOOLEAN DEFAULT 0,
            symmetric BOOLEAN DEFAULT 0
        )
        "#
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS clustering_results (
            node_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            FOREIGN KEY (node_id) REFERENCES graph_nodes(id)
        )
        "#
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS pathfinding_cache (
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            path_nodes TEXT NOT NULL,
            distance REAL NOT NULL,
            cached_at TIMESTAMP NOT NULL,
            PRIMARY KEY (source_id, target_id)
        )
        "#
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS reasoning_cache (
            ontology_checksum TEXT NOT NULL,
            inferred_axiom_type TEXT NOT NULL,
            subject_iri TEXT NOT NULL,
            object_iri TEXT NOT NULL,
            cached_at TIMESTAMP NOT NULL
        )
        "#
    )
    .execute(&pool)
    .await?;

    // Import data in batches
    println!("\n💾 Importing data (batch size: {})...", BATCH_SIZE);

    // Import OWL classes
    println!("\n📦 Importing OWL classes...");
    for chunk in data.owl_classes.chunks(BATCH_SIZE) {
        for class in chunk {
            sqlx::query(
                r#"
                INSERT INTO owl_classes
                (id, iri, label, parent_class_iri, markdown_content, file_path, file_sha1, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#
            )
            .bind(class.id)
            .bind(&class.iri)
            .bind(&class.label)
            .bind(&class.parent_class_iri)
            .bind(&class.markdown_content)
            .bind(&class.file_path)
            .bind(&class.file_sha1)
            .bind(&class.created_at)
            .bind(&class.updated_at)
            .execute(&pool)
            .await?;
        }
    }
    println!("   ✅ Imported {} OWL classes", data.owl_classes.len());

    // Import nodes
    println!("📦 Importing nodes...");
    for chunk in data.unified_nodes.chunks(BATCH_SIZE) {
        for node in chunk {
            sqlx::query(
                r#"
                INSERT INTO graph_nodes
                (id, metadata_id, label, x, y, z, vx, vy, vz, mass, owl_class_iri, node_type, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                "#
            )
            .bind(node.id)
            .bind(&node.metadata_id)
            .bind(&node.label)
            .bind(node.x)
            .bind(node.y)
            .bind(node.z)
            .bind(node.vx)
            .bind(node.vy)
            .bind(node.vz)
            .bind(node.mass)
            .bind(&node.owl_class_iri)
            .bind(&node.node_type)
            .bind(&node.category)
            .execute(&pool)
            .await?;
        }
    }
    println!("   ✅ Imported {} nodes", data.unified_nodes.len());

    // Import edges
    println!("📦 Importing edges...");
    for chunk in data.unified_edges.chunks(BATCH_SIZE) {
        for edge in chunk {
            sqlx::query(
                r#"
                INSERT INTO graph_edges
                (id, source_id, target_id, relationship_type, weight, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                "#
            )
            .bind(edge.id)
            .bind(edge.source_id)
            .bind(edge.target_id)
            .bind(&edge.relationship_type)
            .bind(edge.weight)
            .bind(&edge.metadata)
            .execute(&pool)
            .await?;
        }
    }
    println!("   ✅ Imported {} edges", data.unified_edges.len());

    // Import axioms
    println!("📦 Importing axioms...");
    for axiom in &data.owl_axioms {
        sqlx::query(
            r#"
            INSERT INTO owl_axioms
            (id, axiom_type, subject_iri, object_iri, property_iri, strength, priority, user_defined, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(axiom.id)
        .bind(&axiom.axiom_type)
        .bind(&axiom.subject_iri)
        .bind(&axiom.object_iri)
        .bind(&axiom.property_iri)
        .bind(axiom.strength)
        .bind(axiom.priority)
        .bind(axiom.user_defined)
        .bind(&axiom.created_at)
        .execute(&pool)
        .await?;
    }
    println!("   ✅ Imported {} axioms", data.owl_axioms.len());

    // Create indexes AFTER import (faster)
    println!("\n🔍 Creating indexes...");
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_nodes_metadata ON graph_nodes(metadata_id)")
        .execute(&pool).await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_nodes_owl_iri ON graph_nodes(owl_class_iri)")
        .execute(&pool).await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id)")
        .execute(&pool).await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id)")
        .execute(&pool).await?;
    println!("   ✅ Created indexes");

    // Verify foreign keys
    println!("\n🔍 Verifying foreign key integrity...");
    let fk_violations: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM graph_nodes WHERE owl_class_iri IS NOT NULL
         AND owl_class_iri NOT IN (SELECT iri FROM owl_classes)"
    )
    .fetch_one(&pool)
    .await?;

    if fk_violations > 0 {
        println!("   ⚠️  Warning: {} nodes with invalid owl_class_iri", fk_violations);
    } else {
        println!("   ✅ All foreign keys valid");
    }

    println!("\n✅ Import complete!");
    println!("   Database: {}", db_path);

    pool.close().await;
    Ok(())
}
