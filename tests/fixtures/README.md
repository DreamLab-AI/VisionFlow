# Test Fixtures for Ontology Validation

This directory contains comprehensive test fixtures for validating graph data against ontological constraints and relationships.

## Directory Structure

```
tests/fixtures/
├── ontology/
│   ├── sample.ttl              # Sample OWL ontology with Person/Company classes
│   ├── test_mapping.toml       # Mapping configuration for graph validation
│   └── sample_graph.json       # Sample graph data with test scenarios
└── README.md                   # This documentation file
```

## Overview

The test fixtures provide a complete testing environment for ontology-based graph validation, including:

- **Ontology Definition**: OWL/RDF ontology with classes, properties, and constraints
- **Mapping Configuration**: Rules for mapping graph elements to ontology concepts
- **Sample Data**: Comprehensive graph data with various test scenarios
- **Validation Cases**: Examples of both valid and invalid data patterns

## Files Description

### 1. sample.ttl - Sample Ontology

A comprehensive OWL ontology defining:

**Classes:**
- `test:Person` - Human individuals with employment relationships
- `test:Company` - Business organisations that employ people
- `test:Department` - Subdivisions within companies

**Key Relationships:**
- `test:employs` / `test:worksFor` - Bidirectional employment (inverse properties)
- `test:hasDepartment` / `test:belongsTo` - Company-department relationships
- `test:worksInDepartment` - Person-department assignments
- `test:colleagueOf` - Symmetric colleague relationships

**Constraints:**
- Disjoint classes (Person ≠ Company ≠ Department)
- Domain/range restrictions on properties
- Cardinality constraints (e.g., person must have exactly one name)
- Value restrictions (e.g., age between 0-120)

**Sample Instances:**
- 3 persons with complete property sets
- 2 companies with industry details
- 3 departments with budgets
- Employment and colleague relationships

### 2. test_mapping.toml - Mapping Configuration

Comprehensive mapping rules including:

**Node Mappings:**
- Graph node types → OWL classes
- Required vs optional properties
- Validation rules per node type

**Edge Mappings:**
- Graph edges → OWL object properties
- Domain/range constraints
- Inverse relationship handling
- Symmetric property support

**Validation Rules:**
- 80+ comprehensive validation rules
- Range checks, format validation, uniqueness constraints
- Relationship consistency checks
- Cardinality and constraint validation

**Test Cases:**
- Valid relationship examples
- Constraint violation scenarios
- Inference testing cases
- Edge case handling

### 3. sample_graph.json - Test Graph Data

Rich graph dataset containing:

**Nodes (12 total):**
- 5 Person nodes with complete properties
- 3 Company nodes with business details
- 4 Department nodes with budgets

**Edges (27 total):**
- 10 Employment relationships (bidirectional)
- 8 Department ownership/membership relations
- 6 Colleague relationships (symmetric)
- 3 Cross-department collaborations

**Test Scenarios:**
- Valid employment graph
- Symmetric relationship testing
- Constraint violation examples
- Inference validation cases
- Performance testing dataset specs

**Validation Cases:**
- Self-employment violations
- Disjoint class violations
- Property range violations
- Missing required properties
- Employment consistency checks
- Email format validation
- Employee ID uniqueness

## Usage Examples

### Basic Ontology Loading
```rust
use std::fs;

// Load the ontology
let ontology_content = fs::read_to_string("tests/fixtures/ontology/sample.ttl")?;

// Load mapping configuration
let mapping_content = fs::read_to_string("tests/fixtures/ontology/test_mapping.toml")?;
let mapping: MappingConfig = toml::from_str(&mapping_content)?;

// Load sample graph data
let graph_content = fs::read_to_string("tests/fixtures/ontology/sample_graph.json")?;
let graph_data: GraphData = serde_json::from_str(&graph_content)?;
```

### Validation Testing
```rust
// Test valid employment relationships
#[test]
fn test_valid_employment() {
    let scenario = graph_data.test_scenarios.iter()
        .find(|s| s.name == "valid_employment_graph")
        .unwrap();

    let validation_result = validate_graph_scenario(scenario);
    assert_eq!(validation_result.status, ValidationStatus::Valid);
}

// Test constraint violations
#[test]
fn test_self_employment_violation() {
    let test_case = find_constraint_violation_case("self_employment");
    let validation_result = validate_modified_graph(test_case);

    assert_eq!(validation_result.status, ValidationStatus::Invalid);
    assert!(validation_result.errors.contains(&"no_self_employment"));
}
```

### Property Validation
```rust
// Test age range constraints
#[test]
fn test_age_range_validation() {
    let person_node = create_person_node("test_person", "Test Person", -5); // Invalid age
    let validation_result = validate_node(&person_node, &mapping);

    assert!(validation_result.has_error("age_range_0_120"));
}

// Test email format validation
#[test]
fn test_email_format_validation() {
    let person_node = create_person_with_email("test_person", "invalid-email");
    let validation_result = validate_node(&person_node, &mapping);

    assert!(validation_result.has_error("email_format_validation"));
}
```

### Relationship Testing
```rust
// Test symmetric relationships
#[test]
fn test_colleague_symmetry() {
    let colleagues = extract_colleague_relationships(&graph_data);

    for (person_a, person_b) in colleagues {
        // If A is colleague of B, then B should be colleague of A
        assert!(has_symmetric_relationship(person_a, person_b, "colleague_of"));
    }
}

// Test inverse relationships
#[test]
fn test_employment_inverse() {
    let employments = extract_employment_relationships(&graph_data);

    for (person, company) in employments {
        // If person works_for company, then company employs person
        assert!(has_inverse_relationship(person, company, "works_for", "employs"));
    }
}
```

### Advanced Validation
```rust
// Test department employment consistency
#[test]
fn test_department_employment_consistency() {
    let dept_assignments = extract_department_assignments(&graph_data);

    for (person, department) in dept_assignments {
        let dept_company = get_department_company(department);
        let person_company = get_person_employer(person);

        assert_eq!(dept_company, person_company,
            "Person must work for company that owns their department");
    }
}
```

## Test Scenarios

### 1. Valid Employment Graph
Complete employment network with all constraints satisfied:
- 5 employees across 2 companies
- 4 departments properly owned by companies
- Consistent department assignments
- Proper inverse relationships

### 2. Constraint Violations
Examples of various validation failures:
- Self-employment attempts
- Disjoint class violations
- Invalid property values
- Missing required properties
- Inconsistent relationships

### 3. Inference Testing
Tests for ontological reasoning:
- Inverse property inference
- Symmetric relationship completion
- Transitive relationship validation
- Constraint propagation

### 4. Performance Testing
Specifications for large-scale validation:
- 100 person nodes
- 10 company nodes
- 30 department nodes
- Complex relationship networks
- Performance benchmarks

## Configuration Options

The test fixtures support various validation modes:

```toml
[validation_configuration]
strict_mode = true                    # Enable strict constraint checking
inference_enabled = true              # Enable ontological inference
constraint_checking = "comprehensive" # Level of constraint validation
error_reporting = "detailed"          # Error message detail level
performance_monitoring = true         # Track validation performance
```

## Integration with Test Suite

### Unit Tests
Use individual validation rules and small graph fragments:
```rust
#[cfg(test)]
mod ontology_validation_tests {
    use super::*;
    use crate::tests::fixtures::ontology::*;

    #[test]
    fn test_person_class_validation() { /* ... */ }

    #[test]
    fn test_employment_relationship() { /* ... */ }
}
```

### Integration Tests
Use complete graph scenarios:
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_complete_validation_workflow() {
        // Load all fixtures
        // Run complete validation
        // Verify all constraints
    }
}
```

### Performance Tests
Use large dataset specifications:
```rust
#[cfg(test)]
mod performance_tests {
    #[test]
    fn test_large_graph_validation() {
        // Generate large dataset per specs
        // Validate performance requirements
        // Check memory usage
    }
}
```

## Best Practices

1. **Incremental Testing**: Start with simple valid cases, then add complexity
2. **Error Coverage**: Test each validation rule with specific violation cases
3. **Relationship Testing**: Verify both directions of bidirectional relationships
4. **Constraint Combinations**: Test interactions between multiple constraints
5. **Performance Monitoring**: Track validation time for large datasets
6. **Real-world Scenarios**: Include realistic data patterns and edge cases

## Extensions

The test fixtures can be extended with:

- Additional ontology classes (Project, Role, Location)
- More complex relationship patterns
- Temporal constraints (employment start/end dates)
- Hierarchical relationships (manager/employee)
- Multi-company scenarios
- Industry-specific constraints

This comprehensive test fixture suite provides a robust foundation for validating ontology-based graph systems with thorough coverage of constraints, relationships, and edge cases.