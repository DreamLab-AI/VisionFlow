//! Data preprocessing utilities

use crate::errors::{AutoSchemaError, Result};
use crate::text_processing;
use serde::{Deserialize, Serialize};

/// Represents a document to be processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: std::collections::HashMap<String, String>,
    pub content_hash: String,
}

impl Document {
    /// Create a new document with generated hash
    pub fn new(id: String, content: String, metadata: std::collections::HashMap<String, String>) -> Self {
        let content_hash = crate::hashing::sha256_hash(&content);
        Self {
            id,
            content,
            metadata,
            content_hash,
        }
    }

    /// Preprocess the document content
    pub fn preprocess(&mut self, steps: &[String]) -> Result<()> {
        for step in steps {
            match step.as_str() {
                "normalize_whitespace" => {
                    self.content = text_processing::normalize_whitespace(&self.content)?;
                }
                "remove_html" => {
                    self.content = text_processing::remove_html_tags(&self.content)?;
                }
                "normalize_unicode" => {
                    self.content = text_processing::unicode_normalize(&self.content);
                }
                "full_normalize" => {
                    self.content = text_processing::normalize_text(&self.content)?;
                }
                _ => {
                    log::warn!("Unknown preprocessing step: {}", step);
                }
            }
        }

        // Update hash after preprocessing
        self.content_hash = crate::hashing::sha256_hash(&self.content);
        Ok(())
    }

    /// Check if document is valid (non-empty content)
    pub fn is_valid(&self) -> bool {
        !self.content.trim().is_empty() && self.content.len() > 10
    }
}

/// Batch preprocessing for multiple documents
pub fn preprocess_documents(
    documents: &mut [Document],
    steps: &[String],
) -> Result<Vec<String>> {
    let mut errors = Vec::new();

    for doc in documents {
        if let Err(e) = doc.preprocess(steps) {
            log::error!("Failed to preprocess document {}: {}", doc.id, e);
            errors.push(format!("Document {}: {}", doc.id, e));
        }
    }

    Ok(errors)
}

/// Filter documents by minimum length and validity
pub fn filter_documents(documents: Vec<Document>, min_length: usize) -> Vec<Document> {
    documents
        .into_iter()
        .filter(|doc| doc.is_valid() && doc.content.len() >= min_length)
        .collect()
}

/// Deduplicate documents based on content hash
pub fn deduplicate_documents(documents: Vec<Document>) -> Vec<Document> {
    let mut seen_hashes = std::collections::HashSet::new();
    let mut unique_docs = Vec::new();

    for doc in documents {
        if seen_hashes.insert(doc.content_hash.clone()) {
            unique_docs.push(doc);
        } else {
            log::debug!("Skipping duplicate document: {}", doc.id);
        }
    }

    log::info!(
        "Deduplicated {} documents to {} unique documents",
        seen_hashes.len() + unique_docs.len() - seen_hashes.len(),
        unique_docs.len()
    );

    unique_docs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new(
            "test1".to_string(),
            "Hello world".to_string(),
            std::collections::HashMap::new(),
        );
        assert_eq!(doc.id, "test1");
        assert!(!doc.content_hash.is_empty());
        assert!(doc.is_valid());
    }

    #[test]
    fn test_document_preprocessing() {
        let mut doc = Document::new(
            "test1".to_string(),
            "<p>Hello   world!</p>".to_string(),
            std::collections::HashMap::new(),
        );

        let steps = vec!["remove_html".to_string(), "normalize_whitespace".to_string()];
        doc.preprocess(&steps).unwrap();

        assert_eq!(doc.content.trim(), "Hello world!");
    }

    #[test]
    fn test_deduplication() {
        let doc1 = Document::new("1".to_string(), "content".to_string(), std::collections::HashMap::new());
        let doc2 = Document::new("2".to_string(), "content".to_string(), std::collections::HashMap::new());
        let doc3 = Document::new("3".to_string(), "different".to_string(), std::collections::HashMap::new());

        let docs = vec![doc1, doc2, doc3];
        let unique = deduplicate_documents(docs);

        assert_eq!(unique.len(), 2);
    }
}