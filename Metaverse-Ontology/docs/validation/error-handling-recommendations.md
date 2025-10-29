# Error Handling Recommendations for OWL Extraction Pipeline

## Overview
This document provides concrete recommendations for implementing robust error handling in the Database → OWL Extraction → Parsing pipeline.

---

## 1. Error Type Hierarchy

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OwlExtractionError {
    /// Database-level errors
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    /// OWL parsing errors with context
    #[error("OWL syntax error in entry {entry_id} at line {line}: {message}")]
    OWLSyntaxError {
        entry_id: i64,
        message: String,
        line: usize,
        raw_block: String,
    },

    /// Invalid IRI format
    #[error("Invalid IRI format in entry {entry_id}: {iri}")]
    InvalidIRI { entry_id: i64, iri: String },

    /// No OWL blocks found (warning level)
    #[error("No OWL blocks found in markdown for entry {entry_id}")]
    NoOwlBlocks { entry_id: i64 },

    /// UTF-8 encoding issues
    #[error("Invalid UTF-8 in markdown for entry {entry_id}")]
    InvalidUtf8 { entry_id: i64 },

    /// Regex compilation errors
    #[error("Regex compilation failed: {0}")]
    RegexError(#[from] regex::Error),

    /// IO errors
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Ontology merging errors
    #[error("Failed to merge ontologies: {0}")]
    MergeError(String),

    /// Multiple errors accumulated during batch processing
    #[error("Multiple extraction failures: {count} entries failed")]
    BatchError {
        count: usize,
        failures: Vec<(i64, Box<OwlExtractionError>)>,
    },
}

/// Result type alias for extraction operations
pub type ExtractionResult<T> = Result<T, OwlExtractionError>;
```

---

## 2. Fallible Iteration Pattern

### Problem
Current implementation may halt entire pipeline on single failure.

### Solution: Partial Success Pattern

```rust
use std::collections::HashMap;

pub struct ExtractionResults {
    pub successes: Vec<ExtractedOwl>,
    pub failures: HashMap<i64, OwlExtractionError>,
    pub warnings: Vec<String>,
}

impl ExtractionResults {
    pub fn success_rate(&self) -> f64 {
        let total = self.successes.len() + self.failures.len();
        if total == 0 {
            return 0.0;
        }
        self.successes.len() as f64 / total as f64
    }

    pub fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }
}

pub fn extract_all_ontologies(
    repo: &SqliteOntologyRepository,
) -> ExtractionResult<ExtractionResults> {
    let entries = repo.get_all_entries()
        .map_err(OwlExtractionError::DatabaseError)?;

    let mut successes = Vec::new();
    let mut failures = HashMap::new();
    let mut warnings = Vec::new();

    for entry in entries {
        match extract_owl_from_entry(&entry) {
            Ok(extracted) => {
                if extracted.axiom_count == 0 {
                    warnings.push(format!(
                        "Entry {} has no axioms (empty ontology)",
                        entry.id
                    ));
                }
                successes.push(extracted);
            }
            Err(e) => {
                log::error!(
                    "Failed to extract OWL from entry {}: {}",
                    entry.id,
                    e
                );
                failures.insert(entry.id, e);
            }
        }
    }

    Ok(ExtractionResults {
        successes,
        failures,
        warnings,
    })
}
```

---

## 3. UTF-8 Validation Layer

### Problem
Invalid UTF-8 in database can cause panics.

### Solution: Lossy Conversion

```rust
pub fn safe_extract_markdown(entry: &OntologyEntry) -> String {
    // If markdown is stored as bytes in database
    String::from_utf8_lossy(&entry.markdown_bytes).into_owned()
}

pub fn extract_owl_from_entry(
    entry: &OntologyEntry,
) -> ExtractionResult<ExtractedOwl> {
    // Safely convert markdown to UTF-8 string
    let markdown = safe_extract_markdown(entry);

    // Continue with extraction...
    extract_owl_from_markdown(&markdown, entry.id, &entry.iri)
}
```

---

## 4. Regex Error Handling

### Problem
Regex compilation can fail at runtime.

### Solution: Lazy Static Compilation

```rust
use once_cell::sync::Lazy;
use regex::Regex;

static OWL_BLOCK_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"```(?:clojure|owl-functional)\n([\s\S]*?)\n```")
        .expect("OWL_BLOCK_REGEX: regex compilation failed at startup")
});

pub fn extract_owl_blocks(markdown: &str) -> Vec<String> {
    OWL_BLOCK_REGEX
        .captures_iter(markdown)
        .filter_map(|cap| cap.get(1))
        .map(|m| m.as_str().to_string())
        .collect()
}
```

**Benefit**: Regex compiled once at startup; any compilation error fails fast.

---

## 5. OWL Parsing Error Context

### Problem
horned-functional errors lack context about source entry.

### Solution: Enriched Error Context

```rust
use horned_functional::reader::read;
use std::io::Cursor;

pub fn parse_owl_block(
    owl_text: &str,
    entry_id: i64,
    block_index: usize,
) -> ExtractionResult<Ontology> {
    let cursor = Cursor::new(owl_text.as_bytes());

    read(cursor).map_err(|e| {
        let error_msg = e.to_string();
        let line_number = extract_line_number(&error_msg);

        OwlExtractionError::OWLSyntaxError {
            entry_id,
            message: error_msg,
            line: line_number,
            raw_block: truncate_for_error(owl_text, 500),
        }
    })
}

fn extract_line_number(error_msg: &str) -> usize {
    // Parse error message like "Syntax error at line 42"
    error_msg
        .split_whitespace()
        .find_map(|word| word.parse::<usize>().ok())
        .unwrap_or(0)
}

fn truncate_for_error(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}... (truncated)", &text[..max_len])
    }
}
```

---

## 6. Ontology Merge Error Handling

### Problem
Merging multiple ontologies can fail silently.

### Solution: Validated Merge

```rust
pub fn build_complete_ontology(
    owl_blocks: Vec<String>,
    entry_id: i64,
) -> ExtractionResult<AnnotatedOntology> {
    if owl_blocks.is_empty() {
        return Err(OwlExtractionError::NoOwlBlocks { entry_id });
    }

    let mut complete_ontology = Ontology::new();
    let mut total_axioms = 0;

    for (idx, block) in owl_blocks.iter().enumerate() {
        let parsed = parse_owl_block(block, entry_id, idx)?;

        let axiom_count_before = complete_ontology.axiom_count();

        // Merge axioms
        for axiom in parsed.axiom_iter() {
            complete_ontology.insert(axiom.clone());
        }

        let axiom_count_after = complete_ontology.axiom_count();
        let added = axiom_count_after - axiom_count_before;

        log::debug!(
            "Block {}: added {} axioms (total: {})",
            idx,
            added,
            axiom_count_after
        );

        total_axioms += added;
    }

    if total_axioms == 0 {
        log::warn!("Entry {} produced ontology with 0 axioms", entry_id);
    }

    log::info!(
        "Built complete ontology for entry {} with {} axioms from {} blocks",
        entry_id,
        total_axioms,
        owl_blocks.len()
    );

    Ok(complete_ontology)
}
```

---

## 7. Structured Logging for Debugging

```rust
use tracing::{error, warn, info, debug, instrument};

#[instrument(skip(repo), fields(total_entries = repo.count()))]
pub fn extract_all_ontologies(
    repo: &SqliteOntologyRepository,
) -> ExtractionResult<ExtractionResults> {
    info!("Starting batch OWL extraction");

    let entries = repo.get_all_entries()?;
    info!("Retrieved {} entries from database", entries.len());

    let mut successes = Vec::new();
    let mut failures = HashMap::new();

    for (idx, entry) in entries.iter().enumerate() {
        debug!(
            "Processing entry {}/{}: id={}, iri={}",
            idx + 1,
            entries.len(),
            entry.id,
            entry.iri
        );

        match extract_owl_from_entry(entry) {
            Ok(extracted) => {
                info!(
                    "✓ Entry {}: {} axioms, {} classes",
                    entry.id,
                    extracted.axiom_count,
                    extracted.class_count
                );
                successes.push(extracted);
            }
            Err(e) => {
                error!("✗ Entry {} failed: {}", entry.id, e);
                failures.insert(entry.id, e);
            }
        }
    }

    let success_rate = successes.len() as f64 / entries.len() as f64 * 100.0;
    info!(
        "Extraction complete: {}/{} succeeded ({:.1}%)",
        successes.len(),
        entries.len(),
        success_rate
    );

    Ok(ExtractionResults {
        successes,
        failures,
        warnings: Vec::new(),
    })
}
```

---

## 8. Retry Logic for Database Operations

```rust
use std::time::Duration;
use std::thread::sleep;

const MAX_RETRIES: u32 = 3;
const RETRY_DELAY: Duration = Duration::from_millis(100);

pub fn get_all_entries_with_retry(
    repo: &SqliteOntologyRepository,
) -> ExtractionResult<Vec<OntologyEntry>> {
    let mut attempts = 0;

    loop {
        match repo.get_all_entries() {
            Ok(entries) => return Ok(entries),
            Err(e) if attempts < MAX_RETRIES => {
                attempts += 1;
                log::warn!(
                    "Database query failed (attempt {}/{}): {}. Retrying...",
                    attempts,
                    MAX_RETRIES,
                    e
                );
                sleep(RETRY_DELAY * attempts);
            }
            Err(e) => {
                return Err(OwlExtractionError::DatabaseError(e));
            }
        }
    }
}
```

---

## 9. Integration Test for Error Scenarios

```rust
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_handles_empty_markdown_gracefully() {
        let entry = OntologyEntry {
            id: 1,
            name: "Empty".to_string(),
            iri: "http://example.com/Empty".to_string(),
            markdown: "".to_string(),
            parent_iri: None,
        };

        let result = extract_owl_from_entry(&entry);

        // Should either return empty ontology or NoOwlBlocks error
        match result {
            Ok(extracted) => assert_eq!(extracted.axiom_count, 0),
            Err(OwlExtractionError::NoOwlBlocks { .. }) => {
                // Also acceptable
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }

    #[test]
    fn test_malformed_owl_does_not_crash() {
        let markdown = r#"
```clojure
Declaration(Class(<http://example.com/Broken>
SubClassOf(<http://example.com/Broken> INVALID_SYNTAX!!!
```
"#;

        let entry = OntologyEntry {
            id: 2,
            name: "Malformed".to_string(),
            iri: "http://example.com/Malformed".to_string(),
            markdown: markdown.to_string(),
            parent_iri: None,
        };

        let result = extract_owl_from_entry(&entry);

        // Must return error, not panic
        assert!(result.is_err());
        if let Err(OwlExtractionError::OWLSyntaxError { entry_id, .. }) = result {
            assert_eq!(entry_id, 2);
        } else {
            panic!("Expected OWLSyntaxError");
        }
    }

    #[test]
    fn test_invalid_utf8_handled() {
        let invalid_bytes = vec![0xFF, 0xFE, 0xFD]; // Invalid UTF-8

        let entry = OntologyEntry {
            id: 3,
            name: "Invalid UTF-8".to_string(),
            iri: "http://example.com/InvalidUtf8".to_string(),
            markdown: String::from_utf8_lossy(&invalid_bytes).into_owned(),
            parent_iri: None,
        };

        // Should not panic, either succeeds with replacement chars or errors
        let result = extract_owl_from_entry(&entry);
        // Just ensure no panic
        let _ = result;
    }

    #[test]
    fn test_batch_continues_after_failure() {
        // Create mix of valid and invalid entries
        let entries = vec![
            create_valid_entry(1),
            create_malformed_entry(2),
            create_valid_entry(3),
        ];

        let repo = MockRepository::new(entries);
        let results = extract_all_ontologies(&repo).unwrap();

        assert_eq!(results.successes.len(), 2);
        assert_eq!(results.failures.len(), 1);
        assert!(results.failures.contains_key(&2));
    }
}
```

---

## 10. Recommended Error Handling Policy

### Critical Errors (Fail Fast)
- Database connection failure → Propagate immediately
- Regex compilation failure → Panic at startup
- System resource exhaustion → Propagate immediately

### Recoverable Errors (Continue Processing)
- Single entry OWL parsing failure → Log, skip, continue
- Empty markdown content → Log warning, skip
- No OWL blocks found → Log info, skip
- Malformed OWL syntax → Log error with context, skip

### Warning-Level Issues (Log Only)
- Empty ontology (0 axioms) → Log warning
- Duplicate axioms → Log debug
- Missing annotations → Log info

---

## 11. Error Reporting Dashboard

```rust
pub fn print_extraction_report(results: &ExtractionResults) {
    println!("\n=== OWL Extraction Report ===");
    println!("Total entries: {}", results.successes.len() + results.failures.len());
    println!("✓ Successes: {} ({:.1}%)",
             results.successes.len(),
             results.success_rate() * 100.0);
    println!("✗ Failures: {}", results.failures.len());
    println!("⚠ Warnings: {}", results.warnings.len());

    if !results.failures.is_empty() {
        println!("\n=== Failed Entries ===");
        for (entry_id, error) in &results.failures {
            println!("  Entry {}: {}", entry_id, error);
        }
    }

    if !results.warnings.is_empty() {
        println!("\n=== Warnings ===");
        for warning in &results.warnings {
            println!("  {}", warning);
        }
    }

    println!("\nTotal axioms extracted: {}",
             results.successes.iter().map(|e| e.axiom_count).sum::<usize>());
}
```

---

## Implementation Priority

1. **HIGH PRIORITY** (Implement first):
   - Fallible iteration pattern
   - UTF-8 validation layer
   - Enriched OWL parsing error context

2. **MEDIUM PRIORITY**:
   - Retry logic for database
   - Structured logging
   - Error reporting dashboard

3. **LOW PRIORITY** (Nice to have):
   - Advanced error recovery strategies
   - Error analytics and trending

---

## Success Metrics

After implementing these recommendations:
- ✅ Pipeline should handle 988 classes without crashing
- ✅ Single malformed entry does not halt batch processing
- ✅ All errors provide actionable context (entry ID, line number)
- ✅ Success rate >95% for well-formed ontology database
- ✅ Zero panics on invalid UTF-8 or malformed OWL

---

**Document End**
