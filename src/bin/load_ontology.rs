// src/bin/load_ontology.rs
//! Ontology Loader Binary
//!
//! Loads OWL ontology data from GitHub repository markdown files
//! and populates the unified.db database.

use std::env;
use std::sync::Arc;
use log::{info, error};

use visionflow::repositories::unified_ontology_repository::UnifiedOntologyRepository;
use visionflow::services::parsers::ontology_parser::OntologyParser;
use visionflow::ports::ontology_repository::OntologyRepository;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    info!("Starting ontology loader...");

    // 1. Initialize repository
    let db_path = env::var("DATABASE_PATH").unwrap_or_else(|_| "data/unified.db".to_string());
    info!("Using database: {}", db_path);

    let ontology_repo = Arc::new(
        UnifiedOntologyRepository::new(&db_path)?
    );

    // 2. Initialize parser
    let parser = OntologyParser::new();

    // 3. Load sample ontology data for testing
    info!("Loading sample ontology classes...");

    // Create sample OWL classes for testing
    let sample_classes = vec![
        ("mv:Person", "Person", "A human individual"),
        ("mv:Company", "Company", "A business organization"),
        ("mv:Project", "Project", "A collaborative endeavor"),
        ("mv:Concept", "Concept", "An abstract idea"),
        ("mv:Technology", "Technology", "A technical tool or system"),
    ];

    let mut total_classes = 0;

    for (iri, label, desc) in sample_classes {
        let class = visionflow::ports::ontology_repository::OwlClass {
            id: None,
            ontology_id: "default".to_string(),
            iri: iri.to_string(),
            label: Some(label.to_string()),
            description: Some(desc.to_string()),
            parent_class_iri: None,
            file_sha1: None,
            last_synced: None,
            markdown_content: None,
        };

        ontology_repo.save_owl_class(&class).await?;
        total_classes += 1;
        info!("Saved class: {} ({})", label, iri);
    }

    // 4. Create sample hierarchy
    info!("Creating class hierarchy...");
    ontology_repo.save_class_hierarchy("mv:Company", "mv:Concept").await?;
    ontology_repo.save_class_hierarchy("mv:Project", "mv:Concept").await?;

    // 5. Create sample properties
    info!("Creating sample properties...");
    let prop = visionflow::ports::ontology_repository::OwlProperty {
        id: None,
        ontology_id: "default".to_string(),
        iri: "mv:worksAt".to_string(),
        label: Some("works at".to_string()),
        property_type: visionflow::ports::ontology_repository::PropertyType::ObjectProperty,
        domain: Some("mv:Person".to_string()),
        range: Some("mv:Company".to_string()),
    };
    ontology_repo.save_owl_property(&prop).await?;

    // 6. Create sample axioms
    info!("Creating sample axioms...");
    let axiom = visionflow::ports::ontology_repository::OwlAxiom {
        id: None,
        ontology_id: "default".to_string(),
        axiom_type: visionflow::ports::ontology_repository::AxiomType::SubClassOf,
        subject: "mv:Company".to_string(),
        object: "mv:Concept".to_string(),
        annotations: None,
        strength: Some(1.0),
        priority: Some(5),
        distance: Some(50.0),
    };
    ontology_repo.save_owl_axiom(&axiom).await?;

    // 7. Verify data
    let class_count = ontology_repo.count_classes().await?;
    info!("\nOntology loaded successfully!");
    info!("Classes: {}", class_count);
    info!("Database: {}", db_path);

    Ok(())
}
