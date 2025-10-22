//! Basic ontology validation example
//!
//! This example demonstrates how to:
//! - Load an OWL ontology
//! - Validate a property graph against the ontology
//! - Handle validation results and violations
//! - Apply inferred relationships to the graph

use std::collections::HashMap;

use crate::services::owl_validator::{
    GraphEdge, GraphNode, OwlValidatorService, PropertyGraph, RdfTriple, Severity, ValidationConfig,
};

/// Main example function demonstrating ontology validation workflow
pub async fn demonstrate_ontology_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦉 Ontology Validation Example\n");

    // Step 1: Create the validator service with custom configuration
    println!("📋 Step 1: Initializing OWL validator service...");
    let config = ValidationConfig {
        enable_reasoning: true,
        reasoning_timeout_seconds: 30,
        enable_inference: true,
        max_inference_depth: 3,
        enable_caching: true,
        cache_ttl_seconds: 3600,
        validate_cardinality: true,
        validate_domains_ranges: true,
        validate_disjoint_classes: true,
    };
    let validator = OwlValidatorService::with_config(config);
    println!("✅ Validator initialized\n");

    // Step 2: Load an ontology
    println!("📋 Step 2: Loading ontology from file...");
    let ontology_content = create_sample_ontology();
    let ontology_id = validator.load_ontology(&ontology_content).await?;
    println!("✅ Ontology loaded with ID: {}\n", ontology_id);

    // Step 3: Create a property graph to validate
    println!("📋 Step 3: Creating sample property graph...");
    let graph = create_sample_graph();
    println!(
        "✅ Graph created with {} nodes and {} edges\n",
        graph.nodes.len(),
        graph.edges.len()
    );

    // Step 4: Validate the graph against the ontology
    println!("📋 Step 4: Running validation...");
    let report = validator.validate(&ontology_id, &graph).await?;
    println!("✅ Validation completed in {} ms\n", report.duration_ms);

    // Step 5: Analyze validation results
    println!("📋 Step 5: Analyzing validation results...");
    println!("📊 Validation Statistics:");
    println!("   • Total triples: {}", report.total_triples);
    println!(
        "   • Classes checked: {}",
        report.statistics.classes_checked
    );
    println!(
        "   • Properties checked: {}",
        report.statistics.properties_checked
    );
    println!(
        "   • Individuals checked: {}",
        report.statistics.individuals_checked
    );
    println!(
        "   • Constraints evaluated: {}",
        report.statistics.constraints_evaluated
    );
    println!(
        "   • Inference rules applied: {}",
        report.statistics.inference_rules_applied
    );
    println!("   • Cache hits: {}", report.statistics.cache_hits);
    println!();

    // Step 6: Display violations
    println!("📋 Step 6: Checking for violations...");
    if report.violations.is_empty() {
        println!("✅ No violations found - graph is valid!\n");
    } else {
        println!("⚠️  Found {} violations:\n", report.violations.len());
        for (idx, violation) in report.violations.iter().enumerate() {
            let severity_icon = match violation.severity {
                Severity::Error => "❌",
                Severity::Warning => "⚠️ ",
                Severity::Info => "ℹ️ ",
            };
            println!(
                "   {}. {} [{:?}] {}",
                idx + 1,
                severity_icon,
                violation.severity,
                violation.rule
            );
            println!("      Message: {}", violation.message);
            if let Some(subject) = &violation.subject {
                println!("      Subject: {}", subject);
            }
            if let Some(predicate) = &violation.predicate {
                println!("      Predicate: {}", predicate);
            }
            println!();
        }
    }

    // Step 7: Display inferred triples
    println!("📋 Step 7: Checking inferred relationships...");
    if report.inferred_triples.is_empty() {
        println!("ℹ️  No new relationships inferred\n");
    } else {
        println!(
            "🔍 Inferred {} new relationships:\n",
            report.inferred_triples.len()
        );
        for (idx, triple) in report.inferred_triples.iter().take(5).enumerate() {
            println!(
                "   {}. {} --[{}]--> {}",
                idx + 1,
                shorten_iri(&triple.subject),
                shorten_iri(&triple.predicate),
                shorten_iri(&triple.object)
            );
        }
        if report.inferred_triples.len() > 5 {
            println!("   ... and {} more", report.inferred_triples.len() - 5);
        }
        println!();
    }

    // Step 8: Apply inferences (if any)
    if !report.inferred_triples.is_empty() {
        println!("📋 Step 8: Applying inferences...");
        println!("ℹ️  In production, these would be applied to the graph database");
        println!("   Example: Adding inverse relationship 'worksFor' when 'employs' exists\n");
    }

    println!("🎉 Ontology validation example completed successfully!");

    Ok(())
}

/// Create a sample OWL ontology in Turtle format
fn create_sample_ontology() -> String {
    r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

# Ontology declaration
ex:CompanyOntology a owl:Ontology ;
    rdfs:label "Company Knowledge Graph Ontology" ;
    rdfs:comment "Defines classes and properties for a corporate knowledge graph" .

# Class definitions
ex:Person a owl:Class ;
    rdfs:label "Person" ;
    rdfs:comment "A human individual" .

ex:Employee a owl:Class ;
    rdfs:subClassOf ex:Person ;
    rdfs:label "Employee" ;
    rdfs:comment "A person employed by an organization" .

ex:Manager a owl:Class ;
    rdfs:subClassOf ex:Employee ;
    rdfs:label "Manager" ;
    rdfs:comment "An employee with management responsibilities" .

ex:Company a owl:Class ;
    rdfs:label "Company" ;
    rdfs:comment "A business organization" .

ex:Department a owl:Class ;
    rdfs:label "Department" ;
    rdfs:comment "A division within a company" .

# Disjoint classes - people and companies are different
ex:Person owl:disjointWith ex:Company .

# Property definitions
ex:employs a owl:ObjectProperty ;
    rdfs:domain ex:Company ;
    rdfs:range ex:Employee ;
    rdfs:label "employs" ;
    rdfs:comment "Company employs an employee" .

ex:worksFor a owl:ObjectProperty ;
    rdfs:domain ex:Employee ;
    rdfs:range ex:Company ;
    rdfs:label "works for" ;
    owl:inverseOf ex:employs .

ex:manages a owl:ObjectProperty ;
    rdfs:domain ex:Manager ;
    rdfs:range ex:Employee ;
    rdfs:label "manages" .

ex:partOf a owl:ObjectProperty, owl:TransitiveProperty ;
    rdfs:domain ex:Department ;
    rdfs:range ex:Company ;
    rdfs:label "part of" .

ex:knows a owl:ObjectProperty, owl:SymmetricProperty ;
    rdfs:domain ex:Person ;
    rdfs:range ex:Person ;
    rdfs:label "knows" .

# Data properties
ex:name a owl:DatatypeProperty ;
    rdfs:domain ex:Person ;
    rdfs:range rdfs:Literal .

ex:email a owl:DatatypeProperty, owl:FunctionalProperty ;
    rdfs:domain ex:Person ;
    rdfs:range rdfs:Literal .

ex:age a owl:DatatypeProperty, owl:FunctionalProperty ;
    rdfs:domain ex:Person ;
    rdfs:range xsd:integer .
"#
    .to_string()
}

/// Create a sample property graph for validation
fn create_sample_graph() -> PropertyGraph {
    let mut graph = PropertyGraph {
        nodes: Vec::new(),
        edges: Vec::new(),
        metadata: HashMap::new(),
    };

    // Create person nodes
    graph.nodes.push(GraphNode {
        id: "person_john".to_string(),
        labels: vec!["Person".to_string(), "Employee".to_string()],
        properties: {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!("John Smith"));
            props.insert("email".to_string(), serde_json::json!("john@example.com"));
            props.insert("age".to_string(), serde_json::json!(35));
            props
        },
    });

    graph.nodes.push(GraphNode {
        id: "person_jane".to_string(),
        labels: vec![
            "Person".to_string(),
            "Employee".to_string(),
            "Manager".to_string(),
        ],
        properties: {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!("Jane Doe"));
            props.insert("email".to_string(), serde_json::json!("jane@example.com"));
            props.insert("age".to_string(), serde_json::json!(42));
            props
        },
    });

    graph.nodes.push(GraphNode {
        id: "person_bob".to_string(),
        labels: vec!["Person".to_string(), "Employee".to_string()],
        properties: {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!("Bob Wilson"));
            props.insert("email".to_string(), serde_json::json!("bob@example.com"));
            props.insert("age".to_string(), serde_json::json!(28));
            props
        },
    });

    // Create company node
    graph.nodes.push(GraphNode {
        id: "company_acme".to_string(),
        labels: vec!["Company".to_string()],
        properties: {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!("Acme Corporation"));
            props
        },
    });

    // Create department nodes
    graph.nodes.push(GraphNode {
        id: "dept_engineering".to_string(),
        labels: vec!["Department".to_string()],
        properties: {
            let mut props = HashMap::new();
            props.insert("name".to_string(), serde_json::json!("Engineering"));
            props
        },
    });

    // Create relationships
    graph.edges.push(GraphEdge {
        id: "edge_1".to_string(),
        source: "company_acme".to_string(),
        target: "person_john".to_string(),
        relationship_type: "employs".to_string(),
        properties: HashMap::new(),
    });

    graph.edges.push(GraphEdge {
        id: "edge_2".to_string(),
        source: "company_acme".to_string(),
        target: "person_jane".to_string(),
        relationship_type: "employs".to_string(),
        properties: HashMap::new(),
    });

    graph.edges.push(GraphEdge {
        id: "edge_3".to_string(),
        source: "company_acme".to_string(),
        target: "person_bob".to_string(),
        relationship_type: "employs".to_string(),
        properties: HashMap::new(),
    });

    graph.edges.push(GraphEdge {
        id: "edge_4".to_string(),
        source: "person_jane".to_string(),
        target: "person_john".to_string(),
        relationship_type: "manages".to_string(),
        properties: HashMap::new(),
    });

    graph.edges.push(GraphEdge {
        id: "edge_5".to_string(),
        source: "person_jane".to_string(),
        target: "person_bob".to_string(),
        relationship_type: "manages".to_string(),
        properties: HashMap::new(),
    });

    graph.edges.push(GraphEdge {
        id: "edge_6".to_string(),
        source: "person_john".to_string(),
        target: "person_bob".to_string(),
        relationship_type: "knows".to_string(),
        properties: HashMap::new(),
    });

    graph.edges.push(GraphEdge {
        id: "edge_7".to_string(),
        source: "dept_engineering".to_string(),
        target: "company_acme".to_string(),
        relationship_type: "partOf".to_string(),
        properties: HashMap::new(),
    });

    graph
}

/// Helper function to shorten IRIs for display
fn shorten_iri(iri: &str) -> String {
    if let Some(last_part) = iri.split('/').last().or_else(|| iri.split('#').last()) {
        last_part.to_string()
    } else {
        iri.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ontology_validation_example() {
        // Test that the example runs without errors
        assert!(demonstrate_ontology_validation().await.is_ok());
    }

    #[test]
    fn test_sample_ontology_creation() {
        let ontology = create_sample_ontology();
        assert!(ontology.contains("@prefix"));
        assert!(ontology.contains("owl:Ontology"));
        assert!(ontology.contains("ex:Person"));
        assert!(ontology.contains("ex:Company"));
    }

    #[test]
    fn test_sample_graph_creation() {
        let graph = create_sample_graph();
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.edges.len(), 7);

        // Verify we have the expected node types
        let person_nodes = graph
            .nodes
            .iter()
            .filter(|n| n.labels.contains(&"Person".to_string()))
            .count();
        assert_eq!(person_nodes, 3);

        let company_nodes = graph
            .nodes
            .iter()
            .filter(|n| n.labels.contains(&"Company".to_string()))
            .count();
        assert_eq!(company_nodes, 1);
    }

    #[test]
    fn test_iri_shortening() {
        assert_eq!(shorten_iri("http://example.org/Person"), "Person");
        assert_eq!(shorten_iri("http://xmlns.com/foaf/0.1/knows"), "knows");
        assert_eq!(shorten_iri("simple"), "simple");
    }
}
