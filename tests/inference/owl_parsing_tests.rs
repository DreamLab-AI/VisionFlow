// tests/inference/owl_parsing_tests.rs
//! OWL Parsing Integration Tests

#[cfg(test)]
mod tests {
    use webxr::inference::owl_parser::{OWLParser, OWLFormat};

    const SAMPLE_OWL_XML: &str = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.com/animal-ontology"/>

    <owl:Class rdf:about="http://example.com/Animal">
        <rdfs:label>Animal</rdfs:label>
    </owl:Class>

    <owl:Class rdf:about="http://example.com/Mammal">
        <rdfs:label>Mammal</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.com/Animal"/>
    </owl:Class>

    <owl:Class rdf:about="http://example.com/Dog">
        <rdfs:label>Dog</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.com/Mammal"/>
    </owl:Class>
</rdf:RDF>"#;

    #[test]
    fn test_format_detection_owl_xml() {
        let format = OWLParser::detect_format(SAMPLE_OWL_XML);
        assert_eq!(format, OWLFormat::OwlXml);
    }

    #[test]
    fn test_format_detection_turtle() {
        let turtle = "@prefix owl: <http://www.w3.org/2002/07/owl#> .";
        let format = OWLParser::detect_format(turtle);
        assert_eq!(format, OWLFormat::Turtle);
    }

    #[test]
    fn test_format_detection_manchester() {
        let manchester = "Ontology: <http://example.com/ont>\nClass: Animal";
        let format = OWLParser::detect_format(manchester);
        assert_eq!(format, OWLFormat::Manchester);
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn test_parse_owl_xml_basic() {
        let result = OWLParser::parse(SAMPLE_OWL_XML);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert!(parsed.classes.len() >= 2);
        assert!(parsed.axioms.len() >= 1);
        assert!(parsed.stats.parse_time_ms > 0);
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn test_parse_owl_xml_hierarchy() {
        let result = OWLParser::parse(SAMPLE_OWL_XML);
        assert!(result.is_ok());

        let parsed = result.unwrap();

        // Check for SubClassOf axioms
        let has_mammal_animal = parsed.axioms.iter().any(|ax| {
            ax.subject.contains("Mammal") && ax.object.contains("Animal")
        });

        assert!(has_mammal_animal, "Should have Mammal subClassOf Animal");
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn test_parse_with_explicit_format() {
        let result = OWLParser::parse_with_format(SAMPLE_OWL_XML, OWLFormat::OwlXml);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_invalid_owl() {
        let invalid = "This is not valid OWL";
        let result = OWLParser::parse(invalid);

        // Should either fail or return empty results
        if let Ok(parsed) = result {
            assert!(parsed.classes.is_empty() || parsed.axioms.is_empty());
        }
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn test_parse_statistics() {
        let result = OWLParser::parse(SAMPLE_OWL_XML);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert_eq!(parsed.stats.classes_count, parsed.classes.len());
        assert_eq!(parsed.stats.axioms_count, parsed.axioms.len());
        assert!(parsed.stats.parse_time_ms > 0);
    }
}
