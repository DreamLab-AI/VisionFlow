//! Example demonstrating how to use the OntologyConstraintTranslator
//! to convert OWL axioms into physics constraints for graph layout

use std::collections::HashMap;

use crate::models::{
    constraints::{Constraint, ConstraintKind, ConstraintSet},
    graph::GraphData,
    node::Node,
};
use crate::physics::ontology_constraints::{
    ConsistencyCheck, OWLAxiom, OWLAxiomType, OntologyConstraintTranslator, OntologyInference,
    OntologyReasoningReport,
};
use crate::types::vec3::Vec3Data;
use crate::utils::socket_flow_messages::BinaryNodeData;

/// Example function showing how to create and use the ontology constraints translator
pub fn demonstrate_ontology_constraints() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample nodes representing different types
    let nodes = create_sample_nodes();

    // Create sample OWL axioms
    let axioms = create_sample_axioms();

    // Create sample inferences
    let inferences = create_sample_inferences();

    // Create reasoning report
    let reasoning_report = OntologyReasoningReport {
        axioms,
        inferences,
        consistency_checks: vec![ConsistencyCheck {
            is_consistent: true,
            conflicting_axioms: vec![],
            suggested_resolution: None,
        }],
        reasoning_time_ms: 150,
    };

    // Create graph data
    let graph_data = GraphData {
        nodes,
        edges: vec![], // Empty for this example
        metadata: Default::default(),
        id_to_metadata: HashMap::new(),
    };

    // Create and use the translator
    let mut translator = OntologyConstraintTranslator::new();

    println!("🔧 Applying ontology constraints to graph layout...");

    // Apply ontology constraints to generate physics constraints
    let constraint_set = translator.apply_ontology_constraints(&graph_data, &reasoning_report)?;

    println!(
        "✅ Generated {} constraints from {} axioms and {} inferences",
        constraint_set.constraints.len(),
        reasoning_report.axioms.len(),
        reasoning_report.inferences.len()
    );

    // Display constraint groups
    for (group_name, indices) in &constraint_set.groups {
        println!(
            "📦 Group '{}' contains {} constraints",
            group_name,
            indices.len()
        );
    }

    // Analyze constraint types
    analyze_constraint_types(&constraint_set.constraints);

    // Show cache statistics
    let cache_stats = translator.get_cache_stats();
    println!(
        "📊 Cache Stats: {} entries, {} cached constraints",
        cache_stats.total_cache_entries, cache_stats.total_cached_constraints
    );

    Ok(())
}

/// Create sample nodes for demonstration
fn create_sample_nodes() -> Vec<Node> {
    vec![
        // Animal instances
        Node {
            id: 0,
            metadata_id: "dog1".to_string(),
            label: "Buddy (Dog)".to_string(),
            data: BinaryNodeData {
                position: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                velocity: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                acceleration: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                mass: 1.0,
                radius: 5.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("species".to_string(), "Canis lupus".to_string());
                meta
            },
            file_size: 0,
            node_type: Some("Animal".to_string()),
            size: None,
            color: Some("#4CAF50".to_string()),
            weight: None,
            group: Some("Mammals".to_string()),
            user_data: None,
        },
        Node {
            id: 1,
            metadata_id: "cat1".to_string(),
            label: "Whiskers (Cat)".to_string(),
            data: BinaryNodeData {
                position: Vec3Data {
                    x: 10.0,
                    y: 0.0,
                    z: 0.0,
                },
                velocity: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                acceleration: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                mass: 1.0,
                radius: 5.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("species".to_string(), "Felis catus".to_string());
                meta
            },
            file_size: 0,
            node_type: Some("Animal".to_string()),
            size: None,
            color: Some("#4CAF50".to_string()),
            weight: None,
            group: Some("Mammals".to_string()),
            user_data: None,
        },
        // Plant instances
        Node {
            id: 2,
            metadata_id: "oak1".to_string(),
            label: "Old Oak".to_string(),
            data: BinaryNodeData {
                position: Vec3Data {
                    x: 0.0,
                    y: 20.0,
                    z: 0.0,
                },
                velocity: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                acceleration: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                mass: 1.0,
                radius: 8.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("species".to_string(), "Quercus robur".to_string());
                meta
            },
            file_size: 0,
            node_type: Some("Plant".to_string()),
            size: None,
            color: Some("#8BC34A".to_string()),
            weight: None,
            group: Some("Trees".to_string()),
            user_data: None,
        },
        Node {
            id: 3,
            metadata_id: "rose1".to_string(),
            label: "Garden Rose".to_string(),
            data: BinaryNodeData {
                position: Vec3Data {
                    x: 15.0,
                    y: 25.0,
                    z: 0.0,
                },
                velocity: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                acceleration: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                mass: 1.0,
                radius: 3.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("species".to_string(), "Rosa damascena".to_string());
                meta
            },
            file_size: 0,
            node_type: Some("Plant".to_string()),
            size: None,
            color: Some("#E91E63".to_string()),
            weight: None,
            group: Some("Flowers".to_string()),
            user_data: None,
        },
        // Person instances
        Node {
            id: 4,
            metadata_id: "john_smith".to_string(),
            label: "John Smith".to_string(),
            data: BinaryNodeData {
                position: Vec3Data {
                    x: -10.0,
                    y: -10.0,
                    z: 0.0,
                },
                velocity: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                acceleration: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                mass: 1.0,
                radius: 6.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("age".to_string(), "35".to_string());
                meta.insert("profession".to_string(), "Engineer".to_string());
                meta
            },
            file_size: 0,
            node_type: Some("Person".to_string()),
            size: None,
            color: Some("#2196F3".to_string()),
            weight: None,
            group: Some("People".to_string()),
            user_data: None,
        },
        Node {
            id: 5,
            metadata_id: "jane_doe".to_string(),
            label: "Jane Doe".to_string(),
            data: BinaryNodeData {
                position: Vec3Data {
                    x: -5.0,
                    y: -15.0,
                    z: 0.0,
                },
                velocity: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                acceleration: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                mass: 1.0,
                radius: 6.0,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("age".to_string(), "28".to_string());
                meta.insert("profession".to_string(), "Scientist".to_string());
                meta
            },
            file_size: 0,
            node_type: Some("Person".to_string()),
            size: None,
            color: Some("#2196F3".to_string()),
            weight: None,
            group: Some("People".to_string()),
            user_data: None,
        },
    ]
}

/// Create sample OWL axioms for demonstration
fn create_sample_axioms() -> Vec<OWLAxiom> {
    vec![
        // Disjoint classes: Animals and Plants are separate
        OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "Animal".to_string(),
            object: Some("Plant".to_string()),
            property: None,
            confidence: 1.0,
        },
        // Disjoint classes: Persons and Animals are separate
        OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "Person".to_string(),
            object: Some("Animal".to_string()),
            property: None,
            confidence: 0.9,
        },
        // SubClass relationship: Mammals are Animals
        OWLAxiom {
            axiom_type: OWLAxiomType::SubClassOf,
            subject: "Mammals".to_string(),
            object: Some("Animal".to_string()),
            property: None,
            confidence: 1.0,
        },
        // SubClass relationship: Trees are Plants
        OWLAxiom {
            axiom_type: OWLAxiomType::SubClassOf,
            subject: "Trees".to_string(),
            object: Some("Plant".to_string()),
            property: None,
            confidence: 1.0,
        },
        // Same individual example
        OWLAxiom {
            axiom_type: OWLAxiomType::SameAs,
            subject: "john_smith".to_string(),
            object: Some("john_smith".to_string()),
            property: None,
            confidence: 1.0,
        },
        // Functional property: each person has exactly one age
        OWLAxiom {
            axiom_type: OWLAxiomType::FunctionalProperty,
            subject: "hasAge".to_string(),
            object: None,
            property: Some("hasAge".to_string()),
            confidence: 0.8,
        },
    ]
}

/// Create sample ontology inferences
fn create_sample_inferences() -> Vec<OntologyInference> {
    vec![
        OntologyInference {
            inferred_axiom: OWLAxiom {
                axiom_type: OWLAxiomType::DisjointClasses,
                subject: "Mammals".to_string(),
                object: Some("Plant".to_string()),
                property: None,
                confidence: 0.95,
            },
            premise_axioms: vec![
                "DisjointClasses(Animal, Plant)".to_string(),
                "SubClassOf(Mammals, Animal)".to_string(),
            ],
            reasoning_confidence: 0.9,
            is_derived: true,
        },
        OntologyInference {
            inferred_axiom: OWLAxiom {
                axiom_type: OWLAxiomType::SubClassOf,
                subject: "People".to_string(),
                object: Some("Animal".to_string()),
                property: None,
                confidence: 0.7,
            },
            premise_axioms: vec![
                "SubClassOf(Person, LivingBeing)".to_string(),
                "SubClassOf(Animal, LivingBeing)".to_string(),
            ],
            reasoning_confidence: 0.6,
            is_derived: true,
        },
    ]
}

/// Analyze and display constraint types generated
fn analyze_constraint_types(constraints: &[Constraint]) {
    let mut type_counts = HashMap::new();
    let mut total_weight = 0.0;

    for constraint in constraints {
        let count = type_counts.entry(constraint.kind).or_insert(0);
        *count += 1;
        total_weight += constraint.weight;
    }

    println!("📋 Constraint Analysis:");
    for (kind, count) in type_counts {
        let percentage = (count as f32 / constraints.len() as f32) * 100.0;
        println!("  • {:?}: {} constraints ({:.1}%)", kind, count, percentage);
    }

    let avg_weight = if constraints.is_empty() {
        0.0
    } else {
        total_weight / constraints.len() as f32
    };
    println!("  • Average weight: {:.3}", avg_weight);

    // Find strongest constraints
    if let Some(strongest) = constraints
        .iter()
        .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
    {
        println!(
            "  • Strongest constraint: {:?} (weight: {:.3})",
            strongest.kind, strongest.weight
        );
    }

    // Count nodes affected
    let mut all_nodes = std::collections::HashSet::new();
    for constraint in constraints {
        for &node_id in &constraint.node_indices {
            all_nodes.insert(node_id);
        }
    }
    println!("  • Total nodes affected: {}", all_nodes.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ontology_constraints_example() {
        // Test that the example runs without errors
        assert!(demonstrate_ontology_constraints().is_ok());
    }

    #[test]
    fn test_disjoint_classes_constraint_generation() {
        let mut translator = OntologyConstraintTranslator::new();
        let nodes = create_sample_nodes();

        let disjoint_axiom = OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "Animal".to_string(),
            object: Some("Plant".to_string()),
            property: None,
            confidence: 1.0,
        };

        let constraints = translator
            .axioms_to_constraints(&[disjoint_axiom], &nodes)
            .unwrap();

        // Should create separation constraints between Animal and Plant instances
        assert!(!constraints.is_empty());
        assert!(constraints
            .iter()
            .all(|c| c.kind == ConstraintKind::Separation));

        // Should have 2 Animal nodes * 2 Plant nodes = 4 separation constraints
        assert_eq!(constraints.len(), 4);
    }

    #[test]
    fn test_subclass_constraint_generation() {
        let mut translator = OntologyConstraintTranslator::new();
        let nodes = create_sample_nodes();

        let subclass_axiom = OWLAxiom {
            axiom_type: OWLAxiomType::SubClassOf,
            subject: "Mammals".to_string(),
            object: Some("Animal".to_string()),
            property: None,
            confidence: 1.0,
        };

        let constraints = translator
            .axioms_to_constraints(&[subclass_axiom], &nodes)
            .unwrap();

        // Should create clustering constraints to group mammals with animals
        assert!(!constraints.is_empty());
        assert!(constraints
            .iter()
            .all(|c| c.kind == ConstraintKind::Clustering));
    }

    #[test]
    fn test_constraint_strength_calculation() {
        let translator = OntologyConstraintTranslator::new();

        // Test different axiom types have appropriate strengths
        let disjoint_strength = translator.get_constraint_strength(&OWLAxiomType::DisjointClasses);
        let sameas_strength = translator.get_constraint_strength(&OWLAxiomType::SameAs);
        let subclass_strength = translator.get_constraint_strength(&OWLAxiomType::SubClassOf);

        // SameAs should be strongest (high co-location force)
        assert!(sameas_strength > disjoint_strength);
        assert!(sameas_strength > subclass_strength);

        // All strengths should be positive and reasonable
        assert!(disjoint_strength > 0.0 && disjoint_strength <= 1.0);
        assert!(sameas_strength > 0.0 && sameas_strength <= 1.0);
        assert!(subclass_strength > 0.0 && subclass_strength <= 1.0);
    }
}
