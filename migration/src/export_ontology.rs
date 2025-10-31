use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use sqlx::{sqlite::SqlitePool, Row};
use std::fs::File;
use std::io::Write;

/// OWL Class from ontology.db
#[derive(Debug, Serialize, Deserialize)]
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

/// OWL Axiom (SubClassOf, DisjointClasses, etc.)
#[derive(Debug, Serialize, Deserialize)]
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

/// OWL Property (ObjectProperty, DataProperty)
#[derive(Debug, Serialize, Deserialize)]
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

/// Reasoning cache entry
#[derive(Debug, Serialize, Deserialize)]
struct ReasoningCache {
    ontology_checksum: String,
    inferred_axiom_type: String,
    subject_iri: String,
    object_iri: String,
    cached_at: String,
}

/// Complete ontology export with checksums
#[derive(Debug, Serialize, Deserialize)]
struct OntologyExport {
    export_timestamp: String,
    database_path: String,
    owl_classes: Vec<OwlClass>,
    owl_axioms: Vec<OwlAxiom>,
    owl_properties: Vec<OwlProperty>,
    reasoning_cache: Vec<ReasoningCache>,
    // Checksums for verification
    classes_sha1: String,
    axioms_sha1: String,
    properties_sha1: String,
    reasoning_sha1: String,
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
    println!("🚀 VisionFlow Ontology Export");
    println!("==============================\n");

    // Database path (adjust as needed)
    let db_path = std::env::var("ONT_DATABASE_URL")
        .unwrap_or_else(|_| "sqlite:///home/devuser/workspace/project/data/ontology.db".to_string());

    println!("📊 Connecting to: {}", db_path);
    let pool = SqlitePool::connect(&db_path)
        .await
        .context("Failed to connect to ontology.db")?;

    // Export OWL classes
    println!("📦 Exporting OWL classes...");
    let owl_classes: Vec<OwlClass> = sqlx::query(
        r#"
        SELECT
            id, iri, label, parent_class_iri,
            markdown_content, file_path, file_sha1,
            datetime(created_at) as created_at,
            datetime(updated_at) as updated_at
        FROM owl_classes
        ORDER BY id
        "#
    )
    .map(|row: sqlx::sqlite::SqliteRow| OwlClass {
        id: row.get("id"),
        iri: row.get("iri"),
        label: row.get("label"),
        parent_class_iri: row.get("parent_class_iri"),
        markdown_content: row.get("markdown_content"),
        file_path: row.get("file_path"),
        file_sha1: row.get("file_sha1"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
    .fetch_all(&pool)
    .await
    .context("Failed to export OWL classes")?;

    println!("   ✅ Exported {} OWL classes", owl_classes.len());

    // Export OWL axioms
    println!("📦 Exporting OWL axioms...");
    let owl_axioms: Vec<OwlAxiom> = sqlx::query(
        r#"
        SELECT
            id, axiom_type, subject_iri, object_iri, property_iri,
            strength, priority, user_defined,
            datetime(created_at) as created_at
        FROM owl_axioms
        ORDER BY id
        "#
    )
    .map(|row: sqlx::sqlite::SqliteRow| OwlAxiom {
        id: row.get("id"),
        axiom_type: row.get("axiom_type"),
        subject_iri: row.get("subject_iri"),
        object_iri: row.get("object_iri"),
        property_iri: row.get("property_iri"),
        strength: row.get("strength"),
        priority: row.get("priority"),
        user_defined: row.get("user_defined"),
        created_at: row.get("created_at"),
    })
    .fetch_all(&pool)
    .await
    .context("Failed to export OWL axioms")?;

    println!("   ✅ Exported {} OWL axioms", owl_axioms.len());

    // Export OWL properties
    println!("📦 Exporting OWL properties...");
    let owl_properties: Vec<OwlProperty> = sqlx::query(
        r#"
        SELECT
            id, iri, property_type, label,
            domain_iri, range_iri,
            functional, inverse_functional, transitive, symmetric
        FROM owl_properties
        ORDER BY id
        "#
    )
    .map(|row: sqlx::sqlite::SqliteRow| OwlProperty {
        id: row.get("id"),
        iri: row.get("iri"),
        property_type: row.get("property_type"),
        label: row.get("label"),
        domain_iri: row.get("domain_iri"),
        range_iri: row.get("range_iri"),
        functional: row.get("functional"),
        inverse_functional: row.get("inverse_functional"),
        transitive: row.get("transitive"),
        symmetric: row.get("symmetric"),
    })
    .fetch_all(&pool)
    .await
    .context("Failed to export OWL properties")?;

    println!("   ✅ Exported {} OWL properties", owl_properties.len());

    // Export reasoning cache
    println!("📦 Exporting reasoning cache...");
    let reasoning_cache: Vec<ReasoningCache> = sqlx::query(
        r#"
        SELECT
            ontology_checksum, inferred_axiom_type,
            subject_iri, object_iri,
            datetime(cached_at) as cached_at
        FROM reasoning_cache
        ORDER BY cached_at DESC
        "#
    )
    .map(|row: sqlx::sqlite::SqliteRow| ReasoningCache {
        ontology_checksum: row.get("ontology_checksum"),
        inferred_axiom_type: row.get("inferred_axiom_type"),
        subject_iri: row.get("subject_iri"),
        object_iri: row.get("object_iri"),
        cached_at: row.get("cached_at"),
    })
    .fetch_all(&pool)
    .await
    .context("Failed to export reasoning cache")?;

    println!("   ✅ Exported {} reasoning cache entries", reasoning_cache.len());

    // Compute checksums
    println!("\n🔐 Computing checksums...");
    let classes_sha1 = compute_sha1(&owl_classes)?;
    let axioms_sha1 = compute_sha1(&owl_axioms)?;
    let properties_sha1 = compute_sha1(&owl_properties)?;
    let reasoning_sha1 = compute_sha1(&reasoning_cache)?;

    println!("   ✅ Classes checksum: {}", classes_sha1);
    println!("   ✅ Axioms checksum: {}", axioms_sha1);
    println!("   ✅ Properties checksum: {}", properties_sha1);
    println!("   ✅ Reasoning checksum: {}", reasoning_sha1);

    // Create export structure
    let export = OntologyExport {
        export_timestamp: format!("{}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")),
        database_path: db_path.clone(),
        owl_classes,
        owl_axioms,
        owl_properties,
        reasoning_cache,
        classes_sha1: classes_sha1.clone(),
        axioms_sha1: axioms_sha1.clone(),
        properties_sha1: properties_sha1.clone(),
        reasoning_sha1: reasoning_sha1.clone(),
        total_sha1: {
            let combined = format!("{}{}{}{}", classes_sha1, axioms_sha1, properties_sha1, reasoning_sha1);
            let mut hasher = Sha1::new();
            hasher.update(combined.as_bytes());
            format!("{:x}", hasher.finalize())
        },
    };

    // Write to JSON
    let output_path = "/home/devuser/workspace/project/migration/ontology_export.json";
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
