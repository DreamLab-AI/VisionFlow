// Demo validation test showing how the test fixtures work together
// This file demonstrates integration between sample.ttl, test_mapping.toml, and sample_graph.json

#[cfg(test)]
mod fixture_integration_tests {
    use std::fs;
    use serde_json::{Value, Map};
    use std::collections::HashMap;

    /// Load and parse the sample graph data
    fn load_sample_graph() -> Result<Value, Box<dyn std::error::Error>> {
        let content = fs::read_to_string("tests/fixtures/ontology/sample_graph.json")?;
        let graph: Value = serde_json::from_str(&content)?;
        Ok(graph)
    }

    /// Load the mapping configuration
    fn load_mapping_config() -> Result<String, Box<dyn std::error::Error>> {
        let content = fs::read_to_string("tests/fixtures/ontology/test_mapping.toml")?;
        Ok(content)
    }

    /// Load the ontology content
    fn load_ontology() -> Result<String, Box<dyn std::error::Error>> {
        let content = fs::read_to_string("tests/fixtures/ontology/sample.ttl")?;
        Ok(content)
    }

    #[test]
    fn test_fixture_files_exist_and_loadable() {
        // Verify all fixture files can be loaded
        assert!(load_sample_graph().is_ok(), "Should load sample_graph.json");
        assert!(load_mapping_config().is_ok(), "Should load test_mapping.toml");
        assert!(load_ontology().is_ok(), "Should load sample.ttl");
    }

    #[test]
    fn test_graph_data_structure() {
        let graph = load_sample_graph().unwrap();

        // Verify metadata structure
        assert!(graph["metadata"].is_object(), "Should have metadata object");
        assert_eq!(graph["metadata"]["version"], "1.0.0", "Should have correct version");

        // Verify nodes and edges arrays exist
        assert!(graph["nodes"].is_array(), "Should have nodes array");
        assert!(graph["edges"].is_array(), "Should have edges array");
        assert!(graph["test_scenarios"].is_array(), "Should have test_scenarios array");

        // Verify expected counts match statistics
        let stats = &graph["expected_statistics"];
        assert_eq!(graph["nodes"].as_array().unwrap().len(), 12, "Should have 12 nodes");
        assert_eq!(graph["edges"].as_array().unwrap().len(), 29, "Should have 29 edges");
    }

    #[test]
    fn test_node_types_distribution() {
        let graph = load_sample_graph().unwrap();
        let nodes = graph["nodes"].as_array().unwrap();

        let mut type_counts = HashMap::new();
        for node in nodes {
            let node_type = node["type"].as_str().unwrap();
            *type_counts.entry(node_type).or_insert(0) += 1;
        }

        // Verify expected distribution
        assert_eq!(*type_counts.get("person").unwrap_or(&0), 5, "Should have 5 person nodes");
        assert_eq!(*type_counts.get("company").unwrap_or(&0), 3, "Should have 3 company nodes");
        assert_eq!(*type_counts.get("department").unwrap_or(&0), 4, "Should have 4 department nodes");
    }

    #[test]
    fn test_edge_types_distribution() {
        let graph = load_sample_graph().unwrap();
        let edges = graph["edges"].as_array().unwrap();

        let mut type_counts = HashMap::new();
        for edge in edges {
            let edge_type = edge["type"].as_str().unwrap();
            *type_counts.entry(edge_type).or_insert(0) += 1;
        }

        // Verify employment relationships
        assert!(*type_counts.get("works_for").unwrap_or(&0) > 0, "Should have works_for relationships");
        assert!(*type_counts.get("employs").unwrap_or(&0) > 0, "Should have employs relationships");

        // Verify department relationships
        assert!(*type_counts.get("has_department").unwrap_or(&0) > 0, "Should have has_department relationships");
        assert!(*type_counts.get("belongs_to").unwrap_or(&0) > 0, "Should have belongs_to relationships");

        // Verify colleague relationships
        assert!(*type_counts.get("colleague_of").unwrap_or(&0) > 0, "Should have colleague_of relationships");
    }

    #[test]
    fn test_ontology_class_references() {
        let graph = load_sample_graph().unwrap();
        let ontology_content = load_ontology().unwrap();
        let nodes = graph["nodes"].as_array().unwrap();

        // Verify that ontology classes referenced in graph exist in ontology
        for node in nodes {
            let ontology_class = node["ontology_class"].as_str().unwrap();
            assert!(
                ontology_content.contains(ontology_class),
                "Ontology should contain class: {}", ontology_class
            );
        }
    }

    #[test]
    fn test_property_references() {
        let graph = load_sample_graph().unwrap();
        let ontology_content = load_ontology().unwrap();
        let nodes = graph["nodes"].as_array().unwrap();

        // Check that properties used in graph are defined in ontology
        for node in nodes {
            if let Some(properties) = node["properties"].as_object() {
                for property_name in properties.keys() {
                    if property_name.starts_with("test:") {
                        assert!(
                            ontology_content.contains(property_name),
                            "Ontology should contain property: {}", property_name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_mapping_configuration_structure() {
        let mapping_content = load_mapping_config().unwrap();

        // Basic structure checks
        assert!(mapping_content.contains("[metadata]"), "Should have metadata section");
        assert!(mapping_content.contains("[node_mappings]"), "Should have node_mappings section");
        assert!(mapping_content.contains("[edge_mappings]"), "Should have edge_mappings section");
        assert!(mapping_content.contains("[validation_rules]"), "Should have validation_rules section");

        // Specific mapping checks
        assert!(mapping_content.contains("[node_mappings.person]"), "Should have person node mapping");
        assert!(mapping_content.contains("[node_mappings.company]"), "Should have company node mapping");
        assert!(mapping_content.contains("[edge_mappings.employs]"), "Should have employs edge mapping");
        assert!(mapping_content.contains("[edge_mappings.works_for]"), "Should have works_for edge mapping");
    }

    #[test]
    fn test_symmetric_relationships() {
        let graph = load_sample_graph().unwrap();
        let edges = graph["edges"].as_array().unwrap();

        // Find colleague relationships and verify symmetry
        let mut colleague_pairs = Vec::new();
        for edge in edges {
            if edge["type"] == "colleague_of" {
                let from = edge["from"].as_str().unwrap();
                let to = edge["to"].as_str().unwrap();
                colleague_pairs.push((from, to));
            }
        }

        // For each colleague relationship, verify reverse exists
        for (person_a, person_b) in &colleague_pairs {
            let reverse_exists = colleague_pairs.iter()
                .any(|(from, to)| from == person_b && to == person_a);
            assert!(reverse_exists, "Colleague relationship should be symmetric: {} <-> {}", person_a, person_b);
        }
    }

    #[test]
    fn test_inverse_relationships() {
        let graph = load_sample_graph().unwrap();
        let edges = graph["edges"].as_array().unwrap();

        // Collect employment relationships
        let mut works_for = Vec::new();
        let mut employs = Vec::new();

        for edge in edges {
            match edge["type"].as_str().unwrap() {
                "works_for" => {
                    let from = edge["from"].as_str().unwrap();
                    let to = edge["to"].as_str().unwrap();
                    works_for.push((from, to));
                },
                "employs" => {
                    let from = edge["from"].as_str().unwrap();
                    let to = edge["to"].as_str().unwrap();
                    employs.push((from, to));
                },
                _ => {}
            }
        }

        // Verify inverse relationships
        for (person, company) in &works_for {
            let inverse_exists = employs.iter()
                .any(|(comp, pers)| comp == company && pers == person);
            assert!(inverse_exists, "Works_for should have inverse employs: {} -> {}", person, company);
        }
    }

    #[test]
    fn test_test_scenarios_completeness() {
        let graph = load_sample_graph().unwrap();
        let test_scenarios = graph["test_scenarios"].as_array().unwrap();

        // Verify we have comprehensive test scenarios
        assert!(!test_scenarios.is_empty(), "Should have test scenarios");

        let scenario_names: Vec<&str> = test_scenarios.iter()
            .map(|s| s["name"].as_str().unwrap())
            .collect();

        // Check for key scenario types
        assert!(scenario_names.iter().any(|&name| name.contains("valid")),
                "Should have valid case scenarios");
        assert!(scenario_names.iter().any(|&name| name.contains("constraint_violation")),
                "Should have constraint violation scenarios");
        assert!(scenario_names.iter().any(|&name| name.contains("inference")),
                "Should have inference testing scenarios");
    }

    #[test]
    fn test_node_property_completeness() {
        let graph = load_sample_graph().unwrap();
        let nodes = graph["nodes"].as_array().unwrap();

        // Verify person nodes have required properties
        for node in nodes {
            if node["type"] == "person" {
                let properties = node["properties"].as_object().unwrap();
                assert!(properties.contains_key("test:hasName"),
                        "Person node {} should have test:hasName", node["id"]);

                // Verify age is reasonable if present
                if let Some(age) = properties.get("test:hasAge") {
                    let age_val = age.as_i64().unwrap();
                    assert!(age_val >= 0 && age_val <= 120,
                            "Age should be between 0-120, got: {}", age_val);
                }
            }
        }

        // Verify company nodes have required properties
        for node in nodes {
            if node["type"] == "company" {
                let properties = node["properties"].as_object().unwrap();
                assert!(properties.contains_key("test:hasCompanyName"),
                        "Company node {} should have test:hasCompanyName", node["id"]);
            }
        }
    }

    #[test]
    fn test_department_company_consistency() {
        let graph = load_sample_graph().unwrap();
        let edges = graph["edges"].as_array().unwrap();

        // Map departments to their companies
        let mut dept_to_company = HashMap::new();
        for edge in edges {
            if edge["type"] == "belongs_to" {
                let dept = edge["from"].as_str().unwrap();
                let company = edge["to"].as_str().unwrap();
                dept_to_company.insert(dept, company);
            }
        }

        // Check person-department-company consistency
        let mut person_departments = HashMap::new();
        let mut person_companies = HashMap::new();

        for edge in edges {
            match edge["type"].as_str().unwrap() {
                "works_in_department" => {
                    let person = edge["from"].as_str().unwrap();
                    let dept = edge["to"].as_str().unwrap();
                    person_departments.insert(person, dept);
                },
                "works_for" => {
                    let person = edge["from"].as_str().unwrap();
                    let company = edge["to"].as_str().unwrap();
                    person_companies.insert(person, company);
                },
                _ => {}
            }
        }

        // Verify consistency: if person works in department,
        // person should work for department's company
        for (person, dept) in &person_departments {
            if let Some(dept_company) = dept_to_company.get(dept) {
                if let Some(person_company) = person_companies.get(person) {
                    assert_eq!(dept_company, person_company,
                        "Person {} works in department {} (belongs to {}) but works for {}",
                        person, dept, dept_company, person_company);
                }
            }
        }
    }

    #[test]
    fn test_validation_rules_coverage() {
        let mapping_content = load_mapping_config().unwrap();

        // Verify comprehensive validation rules exist
        assert!(mapping_content.contains("age_range_0_120"), "Should have age range validation");
        assert!(mapping_content.contains("email_format_validation"), "Should have email format validation");
        assert!(mapping_content.contains("employee_id_unique"), "Should have employee ID uniqueness validation");
        assert!(mapping_content.contains("no_self_employment"), "Should have self-employment prevention");
        assert!(mapping_content.contains("colleague_symmetry_check"), "Should have colleague symmetry validation");
        assert!(mapping_content.contains("department_employment_consistency"), "Should have department consistency validation");
    }

    /// Integration test demonstrating complete fixture usage workflow
    #[test]
    fn test_complete_fixture_integration_workflow() {
        // 1. Load all fixtures
        let graph = load_sample_graph().expect("Should load graph data");
        let mapping_content = load_mapping_config().expect("Should load mapping config");
        let ontology_content = load_ontology().expect("Should load ontology");

        // 2. Verify cross-references
        let nodes = graph["nodes"].as_array().unwrap();
        for node in nodes {
            let ontology_class = node["ontology_class"].as_str().unwrap();
            assert!(ontology_content.contains(ontology_class),
                   "Cross-reference check: ontology should contain {}", ontology_class);
        }

        // 3. Verify mapping configuration covers all node types
        let node_types: std::collections::HashSet<_> = nodes.iter()
            .map(|n| n["type"].as_str().unwrap())
            .collect();

        for node_type in node_types {
            let mapping_key = format!("[node_mappings.{}]", node_type);
            assert!(mapping_content.contains(&mapping_key),
                   "Mapping should cover node type: {}", node_type);
        }

        // 4. Verify test scenarios are actionable
        let scenarios = graph["test_scenarios"].as_array().unwrap();
        assert!(!scenarios.is_empty(), "Should have actionable test scenarios");

        println!("✅ Complete fixture integration test passed!");
        println!("📊 Loaded {} nodes, {} edges, {} test scenarios",
                nodes.len(),
                graph["edges"].as_array().unwrap().len(),
                scenarios.len());
    }
}

/// Example usage demonstrating how to use the test fixtures in practice
#[cfg(test)]
mod fixture_usage_examples {
    use super::fixture_integration_tests::*;

    /// Example: How to extract specific test data for unit tests
    #[test]
    fn example_extract_person_data() {
        let graph = load_sample_graph().unwrap();
        let nodes = graph["nodes"].as_array().unwrap();

        // Extract all person nodes for testing
        let persons: Vec<_> = nodes.iter()
            .filter(|n| n["type"] == "person")
            .collect();

        assert_eq!(persons.len(), 5, "Should find 5 person nodes");

        // Extract specific person for property testing
        let john_smith = persons.iter()
            .find(|p| p["id"] == "john_smith")
            .expect("Should find john_smith");

        let properties = john_smith["properties"].as_object().unwrap();
        assert_eq!(properties["test:hasName"], "John Smith");
        assert_eq!(properties["test:hasAge"], 35);

        println!("✅ Successfully extracted person data for testing");
    }

    /// Example: How to extract relationship data for validation testing
    #[test]
    fn example_extract_employment_relationships() {
        let graph = load_sample_graph().unwrap();
        let edges = graph["edges"].as_array().unwrap();

        // Extract employment relationships
        let employment_edges: Vec<_> = edges.iter()
            .filter(|e| e["type"] == "works_for" || e["type"] == "employs")
            .collect();

        assert!(!employment_edges.is_empty(), "Should find employment relationships");

        // Build employment map for testing
        let mut employments = std::collections::HashMap::new();
        for edge in &employment_edges {
            if edge["type"] == "works_for" {
                let person = edge["from"].as_str().unwrap();
                let company = edge["to"].as_str().unwrap();
                employments.insert(person, company);
            }
        }

        assert!(employments.contains_key("john_smith"), "John Smith should be employed");
        assert_eq!(employments["john_smith"], "acme_corp", "John Smith should work for ACME Corp");

        println!("✅ Successfully extracted employment relationships");
    }
}