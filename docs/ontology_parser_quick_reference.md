# OntologyParser - Quick Reference

## Import

```rust
use webxr::services::parsers::OntologyParser;
```

## Basic Usage

```rust
let parser = OntologyParser::new();
let content = std::fs::read_to_string("ontology.md")?;
let data = parser.parse(&content, "ontology.md")?;
```

## Syntax Cheat Sheet

### Ontology Block Marker
```markdown
- ### OntologyBlock
```

### OWL Class
```markdown
- owl_class:: ClassName
  - label:: Human Readable Name
  - description:: Description text
  - subClassOf:: ParentClass
```

### Object Property
```markdown
- objectProperty:: propertyName
  - label:: human label
  - domain:: SourceClass
  - range:: TargetClass
```

### Data Property
```markdown
- dataProperty:: propertyName
  - label:: human label
  - domain:: SourceClass
  - range:: xsd:datatype
```

### Multiple Parents
```markdown
- owl_class:: ChildClass
  - subClassOf:: Parent1
  - subClassOf:: Parent2
```

### Multiple Domains/Ranges
```markdown
- objectProperty:: prop
  - domain:: Class1, Class2
  - range:: Class3, Class4
```

## Result Structure

```rust
pub struct OntologyData {
    pub classes: Vec<OwlClass>,          // All classes
    pub properties: Vec<OwlProperty>,     // All properties
    pub axioms: Vec<OwlAxiom>,           // All axioms
    pub class_hierarchy: Vec<(String, String)>, // (child, parent)
}
```

## Common Patterns

### Iterate Classes
```rust
for class in &data.classes {
    println!("{}: {}",
        class.iri,
        class.label.as_deref().unwrap_or("unnamed")
    );
}
```

### Find Class by IRI
```rust
let person = data.classes.iter()
    .find(|c| c.iri == "Person")
    .unwrap();
```

### Get Class Parents
```rust
for class in &data.classes {
    println!("{} parents:", class.iri);
    for parent in &class.parent_classes {
        println!("  - {}", parent);
    }
}
```

### Filter by Property Type
```rust
let object_props = data.properties.iter()
    .filter(|p| p.property_type == PropertyType::ObjectProperty)
    .collect::<Vec<_>>();
```

## IRI Formats Supported

```markdown
- owl_class:: SimpleName
- owl_class:: prefix:Name
- owl_class:: http://example.org/ontology#Name
```

## Error Handling

```rust
match parser.parse(&content, filename) {
    Ok(data) => { /* success */ },
    Err(e) => {
        eprintln!("Error: {}", e);
        // Common: "No OntologyBlock found in file"
    }
}
```

## Integration with Repository

```rust
use webxr::ports::ontology_repository::OntologyRepository;

async fn save_ontology(
    repo: &impl OntologyRepository,
    data: OntologyData
) -> Result<(), String> {
    repo.save_ontology(
        &data.classes,
        &data.properties,
        &data.axioms
    ).await.map_err(|e| e.to_string())
}
```

## Run Tests

```bash
cargo test --lib parsers::ontology_parser::tests
```

## Common XSD Datatypes

```
xsd:string
xsd:integer
xsd:float
xsd:double
xsd:boolean
xsd:date
xsd:dateTime
```

## Files

- **Implementation**: `/home/devuser/workspace/project/src/services/parsers/ontology_parser.rs`
- **Full Docs**: `/home/devuser/workspace/project/docs/ontology_parser_usage.md`
- **Summary**: `/home/devuser/workspace/project/docs/ontology_parser_summary.md`
