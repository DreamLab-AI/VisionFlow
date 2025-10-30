// Regex Test Suite for OWL Block Extraction
// Tests the correctness of regex patterns for extracting OWL Functional Syntax from markdown

use regex::Regex;

/// Primary regex pattern for OWL extraction
const OWL_BLOCK_PATTERN: &str = r"```(?:clojure|owl-functional)\n([\s\S]*?)\n```";

/// Robust pattern with nested backtick protection
const ROBUST_PATTERN: &str = r"```(?:clojure|owl-functional|owl)\s*\n((?:[^`]|`(?!``))*?)\n```";

#[cfg(test)]
mod regex_extraction_tests {
    use super::*;

    #[test]
    fn test_single_clojure_block_extraction() {
        let markdown = r#"
# Virtual Reality Concept

## Definition
A simulated environment created using computer technology.

## OWL Representation
```clojure
Declaration(Class(<http://www.metaverse-ontology.com/ontology#VirtualReality>))
SubClassOf(<http://www.metaverse-ontology.com/ontology#VirtualReality>
           <http://www.metaverse-ontology.com/ontology#ImmersiveTechnology>)
AnnotationAssertion(rdfs:label
                    <http://www.metaverse-ontology.com/ontology#VirtualReality>
                    "Virtual Reality"^^xsd:string)
```

## Additional Context
More text here.
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 1, "Should extract exactly one OWL block");

        let extracted = captures[0].get(1).unwrap().as_str();
        assert!(extracted.contains("Declaration(Class("));
        assert!(extracted.contains("SubClassOf("));
        assert!(extracted.contains("AnnotationAssertion("));
    }

    #[test]
    fn test_owl_functional_syntax_block() {
        let markdown = r#"
```owl-functional
Declaration(Class(<http://example.com/TestClass>))
```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 1);
        let extracted = captures[0].get(1).unwrap().as_str();
        assert_eq!(
            extracted.trim(),
            "Declaration(Class(<http://example.com/TestClass>))"
        );
    }

    #[test]
    fn test_multiple_owl_blocks() {
        let markdown = r#"
# Augmented Reality System

## Core Definition
```owl-functional
Declaration(Class(<http://www.metaverse-ontology.com/ontology#AugmentedReality>))
```

## Properties
```clojure
Declaration(ObjectProperty(<http://www.metaverse-ontology.com/ontology#overlaysOn>))
ObjectPropertyDomain(<http://www.metaverse-ontology.com/ontology#overlaysOn>
                     <http://www.metaverse-ontology.com/ontology#AugmentedReality>)
```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 2, "Should extract both OWL blocks");

        let first = captures[0].get(1).unwrap().as_str();
        let second = captures[1].get(1).unwrap().as_str();

        assert!(first.contains("AugmentedReality"));
        assert!(second.contains("ObjectProperty"));
        assert!(second.contains("ObjectPropertyDomain"));
    }

    #[test]
    fn test_no_owl_blocks_returns_empty() {
        let markdown = r#"
# User Documentation

This is a simple description without any OWL syntax.
Just plain text for human readers.

```python
print("This is Python code, not OWL")
```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 0, "Should not extract non-OWL code blocks");
    }

    #[test]
    fn test_nested_backticks_basic_pattern() {
        let markdown = r#"
```clojure
Declaration(Class(<http://example.com/Test>))
AnnotationAssertion(rdfs:comment <http://example.com/Test>
    "This comment contains a `backtick` character"^^xsd:string)
```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        // Basic pattern [\s\S]*? should handle single backticks correctly
        assert_eq!(captures.len(), 1, "Should extract block with nested single backtick");

        let extracted = captures[0].get(1).unwrap().as_str();
        assert!(extracted.contains("`backtick`"));
    }

    #[test]
    fn test_robust_pattern_with_nested_backticks() {
        let markdown = r#"
```owl-functional
Declaration(Class(<http://example.com/Test>))
AnnotationAssertion(rdfs:comment <http://example.com/Test>
    "Code example: `use regex` for matching"^^xsd:string)
```
"#;

        let re = Regex::new(ROBUST_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 1);
        let extracted = captures[0].get(1).unwrap().as_str();
        assert!(extracted.contains("`use regex`"));
    }

    #[test]
    fn test_multiline_axioms() {
        let markdown = r#"
```clojure
SubClassOf(
    <http://www.metaverse-ontology.com/ontology#VirtualReality>
    ObjectIntersectionOf(
        <http://www.metaverse-ontology.com/ontology#ImmersiveTechnology>
        ObjectSomeValuesFrom(
            <http://www.metaverse-ontology.com/ontology#hasFeature>
            <http://www.metaverse-ontology.com/ontology#3DRendering>
        )
    )
)
```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 1);
        let extracted = captures[0].get(1).unwrap().as_str();

        // Verify multiline structure preserved
        assert!(extracted.contains("SubClassOf("));
        assert!(extracted.contains("ObjectIntersectionOf("));
        assert!(extracted.contains("ObjectSomeValuesFrom("));
        assert!(extracted.contains("3DRendering"));

        // Count parentheses to ensure completeness
        let open_parens = extracted.matches('(').count();
        let close_parens = extracted.matches(')').count();
        assert_eq!(open_parens, close_parens, "Parentheses should be balanced");
    }

    #[test]
    fn test_whitespace_preservation() {
        let markdown = r#"
```clojure
Declaration(Class(<http://example.com/Test>))

SubClassOf(<http://example.com/Test> <http://example.com/Parent>)

AnnotationAssertion(rdfs:label <http://example.com/Test> "Test"^^xsd:string)
```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 1);
        let extracted = captures[0].get(1).unwrap().as_str();

        // Verify blank lines are preserved
        let lines: Vec<_> = extracted.lines().collect();
        assert!(lines.iter().any(|l| l.trim().is_empty()),
                "Should preserve blank lines for readability");
    }

    #[test]
    fn test_unicode_in_labels() {
        let markdown = r#"
```clojure
Declaration(Class(<http://example.com/测试>))
AnnotationAssertion(rdfs:label <http://example.com/测试>
    "中文标签"^^xsd:string)
AnnotationAssertion(rdfs:comment <http://example.com/测试>
    "Αρχαία ελληνικά"^^xsd:string)
```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        assert_eq!(captures.len(), 1);
        let extracted = captures[0].get(1).unwrap().as_str();

        assert!(extracted.contains("测试"));
        assert!(extracted.contains("中文标签"));
        assert!(extracted.contains("Αρχαία ελληνικά"));
    }

    #[test]
    fn test_performance_large_markdown() {
        use std::time::Instant;

        // Generate large markdown with 100 OWL blocks
        let mut markdown = String::from("# Large Ontology\n\n");
        for i in 0..100 {
            markdown.push_str(&format!(r#"
## Class {}

```clojure
Declaration(Class(<http://example.com/Class{}>))
SubClassOf(<http://example.com/Class{}> <http://example.com/BaseClass>)
AnnotationAssertion(rdfs:label <http://example.com/Class{}> "Class {}"^^xsd:string)
```

"#, i, i, i, i, i));
        }

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();

        let start = Instant::now();
        let captures: Vec<_> = re.captures_iter(&markdown).collect();
        let elapsed = start.elapsed();

        assert_eq!(captures.len(), 100);
        assert!(
            elapsed.as_millis() < 50,
            "Regex extraction should complete in <50ms for 100 blocks, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_edge_case_empty_block() {
        let markdown = r#"
```clojure
```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        // Empty block should not match (requires content)
        assert_eq!(captures.len(), 0, "Empty blocks should not be extracted");
    }

    #[test]
    fn test_edge_case_only_whitespace() {
        let markdown = r#"
```clojure



```
"#;

        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        let captures: Vec<_> = re.captures_iter(markdown).collect();

        if captures.len() == 1 {
            let extracted = captures[0].get(1).unwrap().as_str();
            assert!(
                extracted.trim().is_empty(),
                "Should extract but content should be whitespace only"
            );
        }
    }
}

#[cfg(test)]
mod extraction_function_tests {
    use super::*;

    /// Simulates the actual extraction function
    fn extract_owl_blocks(markdown: &str) -> Vec<String> {
        let re = Regex::new(OWL_BLOCK_PATTERN).unwrap();
        re.captures_iter(markdown)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str().to_string())
            .collect()
    }

    #[test]
    fn test_extraction_function_integration() {
        let markdown = r#"
# Test Ontology

```clojure
Declaration(Class(<http://test.com/A>))
```

Some text.

```owl-functional
Declaration(Class(<http://test.com/B>))
```
"#;

        let blocks = extract_owl_blocks(markdown);

        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].contains("test.com/A"));
        assert!(blocks[1].contains("test.com/B"));
    }

    #[test]
    fn test_extraction_preserves_original_format() {
        let original_owl = r#"Declaration(Class(<http://example.com/Test>))
SubClassOf(<http://example.com/Test> <http://example.com/Parent>)
AnnotationAssertion(rdfs:label <http://example.com/Test> "Test"^^xsd:string)"#;

        let markdown = format!("```clojure\n{}\n```", original_owl);
        let blocks = extract_owl_blocks(&markdown);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].trim(), original_owl.trim());
    }
}

fn main() {
    println!("Run with: cargo test --test regex-test-suite");
}
