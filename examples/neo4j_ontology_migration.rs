// examples/neo4j_ontology_migration.rs
//! Example: Migrate ontology data from SQLite to Neo4j with rich metadata
//!
//! This example demonstrates:
//! 1. Reading OWL classes from SQLite with rich metadata
//! 2. Batch writing to Neo4j for optimal performance
//! 3. Verifying data integrity after migration
//! 4. Using advanced query features

use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("=== Neo4j Ontology Migration Example ===\n");

    // Example 1: Simple migration
    simple_migration().await?;

    // Example 2: Migration with verification
    migration_with_verification().await?;

    // Example 3: Dual-write pattern
    dual_write_pattern().await?;

    // Example 4: Query examples
    query_examples().await?;

    Ok(())
}

/// Example 1: Simple migration from SQLite to Neo4j
async fn simple_migration() -> Result<(), Box<dyn std::error::Error>> {
    use webxr::adapters::neo4j_ontology_repository::{Neo4jOntologyRepository, Neo4jOntologyConfig};
    use webxr::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
    use webxr::ports::ontology_repository::OntologyRepository;

    println!("Example 1: Simple Migration\n");

    // 1. Connect to SQLite
    let sqlite_repo = SqliteOntologyRepository::new("data/ontology.db")?;
    println!("✓ Connected to SQLite");

    // 2. Connect to Neo4j
    let neo4j_config = Neo4jOntologyConfig::default();
    let neo4j_repo = Neo4jOntologyRepository::new(neo4j_config).await?;
    println!("✓ Connected to Neo4j");

    // 3. Read all classes from SQLite
    let start = Instant::now();
    let classes = sqlite_repo.list_owl_classes().await?;
    println!("✓ Read {} classes from SQLite in {:?}", classes.len(), start.elapsed());

    // 4. Batch write to Neo4j (100 classes per batch)
    let start = Instant::now();
    let iris = neo4j_repo.batch_add_classes(&classes).await?;
    println!("✓ Wrote {} classes to Neo4j in {:?}", iris.len(), start.elapsed());

    // 5. Migrate properties
    let properties = sqlite_repo.list_owl_properties().await?;
    for property in properties {
        neo4j_repo.add_owl_property(&property).await?;
    }
    println!("✓ Migrated properties\n");

    Ok(())
}

/// Example 2: Migration with verification
async fn migration_with_verification() -> Result<(), Box<dyn std::error::Error>> {
    use webxr::adapters::neo4j_ontology_repository::{Neo4jOntologyRepository, Neo4jOntologyConfig};
    use webxr::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
    use webxr::ports::ontology_repository::OntologyRepository;

    println!("Example 2: Migration with Verification\n");

    let sqlite_repo = SqliteOntologyRepository::new("data/ontology.db")?;
    let neo4j_config = Neo4jOntologyConfig::default();
    let neo4j_repo = Neo4jOntologyRepository::new(neo4j_config).await?;

    // Read from SQLite
    let sqlite_classes = sqlite_repo.list_owl_classes().await?;

    // Write to Neo4j
    neo4j_repo.batch_add_classes(&sqlite_classes).await?;

    // Verify: Read back from Neo4j
    let neo4j_classes = neo4j_repo.list_owl_classes().await?;

    // Compare counts
    assert_eq!(sqlite_classes.len(), neo4j_classes.len());
    println!("✓ Verification passed: {} classes", neo4j_classes.len());

    // Spot check: Verify rich metadata is preserved
    if let Some(first_class) = sqlite_classes.first() {
        let neo4j_class = neo4j_repo.get_owl_class(&first_class.iri).await?.unwrap();

        assert_eq!(first_class.term_id, neo4j_class.term_id);
        assert_eq!(first_class.quality_score, neo4j_class.quality_score);
        assert_eq!(first_class.maturity, neo4j_class.maturity);
        assert_eq!(first_class.owl_physicality, neo4j_class.owl_physicality);

        println!("✓ Rich metadata preserved:");
        println!("  - term_id: {:?}", neo4j_class.term_id);
        println!("  - quality_score: {:?}", neo4j_class.quality_score);
        println!("  - maturity: {:?}", neo4j_class.maturity);
        println!("  - physicality: {:?}\n", neo4j_class.owl_physicality);
    }

    Ok(())
}

/// Example 3: Dual-write pattern for consistency
async fn dual_write_pattern() -> Result<(), Box<dyn std::error::Error>> {
    use webxr::adapters::neo4j_ontology_repository::{Neo4jOntologyRepository, Neo4jOntologyConfig};
    use webxr::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
    use webxr::ports::ontology_repository::{OntologyRepository, OwlClass};

    println!("Example 3: Dual-Write Pattern\n");

    let sqlite_repo = SqliteOntologyRepository::new("data/ontology.db")?;
    let neo4j_config = Neo4jOntologyConfig::default();
    let neo4j_repo = Neo4jOntologyRepository::new(neo4j_config).await?;

    // Create a new class with rich metadata
    let class = OwlClass {
        iri: "http://example.org/blockchain/SmartContract".to_string(),
        term_id: Some("BC-0001".to_string()),
        preferred_term: Some("Smart Contract".to_string()),
        label: Some("Smart Contract".to_string()),
        description: Some("Self-executing contract with terms in code".to_string()),

        // Classification
        source_domain: Some("blockchain".to_string()),
        version: Some("1.0".to_string()),
        class_type: Some("concept".to_string()),

        // Quality metrics
        status: Some("approved".to_string()),
        maturity: Some("stable".to_string()),
        quality_score: Some(0.95),
        authority_score: Some(0.90),
        public_access: Some(true),
        content_status: Some("published".to_string()),

        // OWL2 properties
        owl_physicality: Some("virtual".to_string()),
        owl_role: Some("agent".to_string()),

        // Domain relationships
        belongs_to_domain: Some("blockchain".to_string()),
        bridges_to_domain: Some("ai".to_string()),

        parent_classes: vec![],
        properties: Default::default(),
        source_file: None,
        file_sha1: None,
        markdown_content: None,
        last_synced: None,
        additional_metadata: None,
    };

    // Write to both repositories
    let sqlite_iri = sqlite_repo.add_owl_class(&class).await?;
    let neo4j_iri = neo4j_repo.add_owl_class(&class).await?;

    assert_eq!(sqlite_iri, neo4j_iri);
    println!("✓ Dual-write successful: {}\n", neo4j_iri);

    Ok(())
}

/// Example 4: Query examples with rich metadata
async fn query_examples() -> Result<(), Box<dyn std::error::Error>> {
    use webxr::adapters::neo4j_ontology_repository::{Neo4jOntologyRepository, Neo4jOntologyConfig};

    println!("Example 4: Advanced Query Examples\n");

    let neo4j_config = Neo4jOntologyConfig::default();
    let repo = Neo4jOntologyRepository::new(neo4j_config).await?;

    // Query 1: High-quality classes
    println!("Query 1: High-quality classes (score >= 0.8)");
    let high_quality = repo.query_by_quality(0.8).await?;
    println!("Found {} high-quality classes", high_quality.len());
    for class in high_quality.iter().take(5) {
        println!("  - {} (quality: {:?}, authority: {:?})",
            class.label.as_ref().unwrap_or(&"unnamed".to_string()),
            class.quality_score,
            class.authority_score
        );
    }
    println!();

    // Query 2: Cross-domain bridges
    println!("Query 2: Cross-domain bridges");
    let bridges = repo.query_cross_domain_bridges().await?;
    println!("Found {} cross-domain bridge classes", bridges.len());
    for class in bridges.iter().take(5) {
        println!("  - {} bridges from {} to {}",
            class.label.as_ref().unwrap_or(&"unnamed".to_string()),
            class.belongs_to_domain.as_ref().unwrap_or(&"?".to_string()),
            class.bridges_to_domain.as_ref().unwrap_or(&"?".to_string())
        );
    }
    println!();

    // Query 3: Classes by domain
    println!("Query 3: Classes in blockchain domain");
    let blockchain = repo.query_by_domain("blockchain").await?;
    println!("Found {} blockchain classes", blockchain.len());
    println!();

    // Query 4: Classes by maturity
    println!("Query 4: Stable vs experimental classes");
    let stable = repo.query_by_maturity("stable").await?;
    let experimental = repo.query_by_maturity("experimental").await?;
    println!("  Stable: {} classes", stable.len());
    println!("  Experimental: {} classes", experimental.len());
    println!();

    // Query 5: Classes by physicality
    println!("Query 5: Physical vs virtual vs abstract");
    let physical = repo.query_by_physicality("physical").await?;
    let virtual_classes = repo.query_by_physicality("virtual").await?;
    let abstract_classes = repo.query_by_physicality("abstract").await?;
    println!("  Physical: {} classes", physical.len());
    println!("  Virtual: {} classes", virtual_classes.len());
    println!("  Abstract: {} classes", abstract_classes.len());
    println!();

    // Query 6: Clustering by physicality and role
    println!("Query 6: Physicality-role clustering");
    let clustering = repo.get_physicality_role_clustering().await?;
    for (physicality, roles) in clustering {
        println!("  Physicality: {}", physicality);
        for (role, classes) in roles {
            println!("    Role {}: {} classes", role, classes.len());
        }
    }
    println!();

    // Query 7: Add and query relationships
    println!("Query 7: Semantic relationships");
    repo.add_relationship(
        "http://example.org/blockchain/SmartContract",
        "uses",
        "http://example.org/blockchain/Blockchain",
        0.95,
        false
    ).await?;

    let uses_rels = repo.query_relationships_by_type("uses").await?;
    println!("Found {} 'uses' relationships", uses_rels.len());
    for (source, target, confidence, is_inferred) in uses_rels.iter().take(5) {
        println!("  - {} uses {} (confidence: {}, inferred: {})",
            source, target, confidence, is_inferred);
    }

    Ok(())
}
