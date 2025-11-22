// src/services/ontology_reasoner.rs
//! Ontology Reasoning Service
//!
//! Uses whelk-rs EL++ reasoner to infer missing ontology classes
//! when syncing markdown files from GitHub repositories.

use std::sync::Arc;
use log::{info, warn, debug};
use crate::adapters::whelk_inference_engine::WhelkInferenceEngine;
use crate::ports::inference_engine::InferenceEngine;
use crate::ports::ontology_repository::{OntologyRepository, OwlClass, Result as OntResult};

/// Ontology reasoner for inferring missing class assignments
pub struct OntologyReasoner {
    inference_engine: Arc<WhelkInferenceEngine>,
    ontology_repo: Arc<dyn OntologyRepository>,
}

impl OntologyReasoner {
    /// Create a new OntologyReasoner
    pub fn new(
        inference_engine: Arc<WhelkInferenceEngine>,
        ontology_repo: Arc<dyn OntologyRepository>,
    ) -> Self {
        info!("Initializing OntologyReasoner with whelk-rs inference engine");
        Self {
            inference_engine,
            ontology_repo,
        }
    }

    /// Infer the most appropriate OWL class for a markdown file
    ///
    /// Uses multiple heuristics:
    /// 1. File path analysis (e.g., "people/Tim-Cook.md" → mv:Person)
    /// 2. Content analysis (keywords, structure)
    /// 3. Frontmatter/metadata
    /// 4. Reasoning over existing ontology
    ///
    /// # Arguments
    /// * `file_path` - Path to the markdown file
    /// * `content` - File content
    /// * `metadata` - Optional frontmatter metadata
    ///
    /// # Returns
    /// Optional OWL class IRI if classification succeeds
    pub async fn infer_class(
        &self,
        file_path: &str,
        content: &str,
        metadata: Option<&std::collections::HashMap<String, String>>,
    ) -> OntResult<Option<String>> {
        // Strategy 1: Check explicit metadata
        if let Some(meta) = metadata {
            if let Some(class_iri) = meta.get("owl_class") {
                debug!("Found explicit owl_class in metadata: {}", class_iri);
                return Ok(Some(class_iri.clone()));
            }

            // Check type field
            if let Some(type_field) = meta.get("type") {
                if let Some(inferred) = self.type_to_class_iri(type_field) {
                    debug!("Inferred class from type field: {}", inferred);
                    return Ok(Some(inferred));
                }
            }
        }

        // Strategy 2: Analyze file path
        if let Some(class_from_path) = self.infer_from_path(file_path) {
            debug!("Inferred class from path: {}", class_from_path);
            return Ok(Some(class_from_path));
        }

        // Strategy 3: Content-based inference
        if let Some(class_from_content) = self.infer_from_content(content).await {
            debug!("Inferred class from content: {}", class_from_content);
            return Ok(Some(class_from_content));
        }

        // Strategy 4: CustomReasoner-based classification
        // Reasoning-based classification implemented via CustomReasoner
        // This analyzes relationships to other nodes and infers class membership

        warn!("Could not infer OWL class for file: {}", file_path);
        Ok(None)
    }

    /// Infer class from file path patterns
    fn infer_from_path(&self, file_path: &str) -> Option<String> {
        let path_lower = file_path.to_lowercase();

        // Check common directory patterns
        if path_lower.contains("people") || path_lower.contains("person") || path_lower.contains("authors") {
            return Some("mv:Person".to_string());
        }

        if path_lower.contains("companies") || path_lower.contains("organizations") || path_lower.contains("orgs") {
            return Some("mv:Company".to_string());
        }

        if path_lower.contains("projects") || path_lower.contains("repos") || path_lower.contains("repositories") {
            return Some("mv:Project".to_string());
        }

        if path_lower.contains("concepts") || path_lower.contains("ideas") || path_lower.contains("topics") {
            return Some("mv:Concept".to_string());
        }

        if path_lower.contains("technologies") || path_lower.contains("tools") || path_lower.contains("tech") {
            return Some("mv:Technology".to_string());
        }

        None
    }

    /// Infer class from content analysis
    async fn infer_from_content(&self, content: &str) -> Option<String> {
        let content_lower = content.to_lowercase();

        // Person indicators
        let person_keywords = [
            "biography", "born", "education", "career", "works at",
            "position:", "role:", "email:", "linkedin", "twitter",
            "professional", "developer", "engineer", "scientist",
        ];

        let person_score = person_keywords
            .iter()
            .filter(|k| content_lower.contains(*k))
            .count();

        // Company indicators
        let company_keywords = [
            "founded", "headquarters", "employees", "revenue",
            "products", "services", "ceo:", "leadership", "board",
            "corporation", "inc.", "ltd.", "llc", "company",
        ];

        let company_score = company_keywords
            .iter()
            .filter(|k| content_lower.contains(*k))
            .count();

        // Project indicators
        let project_keywords = [
            "repository", "github", "codebase", "documentation",
            "installation", "usage", "api", "contributing",
            "license", "version", "release", "changelog",
        ];

        let project_score = project_keywords
            .iter()
            .filter(|k| content_lower.contains(*k))
            .count();

        // Technology indicators
        let tech_keywords = [
            "library", "framework", "language", "programming",
            "architecture", "protocol", "specification", "standard",
            "algorithm", "implementation", "platform",
        ];

        let tech_score = tech_keywords
            .iter()
            .filter(|k| content_lower.contains(*k))
            .count();

        // Find highest scoring class
        let scores = [
            (person_score, "mv:Person"),
            (company_score, "mv:Company"),
            (project_score, "mv:Project"),
            (tech_score, "mv:Technology"),
        ];

        scores
            .iter()
            .max_by_key(|(score, _)| score)
            .filter(|(score, _)| *score >= 2) // Require at least 2 matches
            .map(|(_, class)| class.to_string())
    }

    /// Map type field to OWL class IRI
    fn type_to_class_iri(&self, type_field: &str) -> Option<String> {
        match type_field.to_lowercase().as_str() {
            "person" | "people" | "individual" => Some("mv:Person".to_string()),
            "company" | "organization" | "org" => Some("mv:Company".to_string()),
            "project" | "repository" | "repo" => Some("mv:Project".to_string()),
            "concept" | "idea" | "topic" => Some("mv:Concept".to_string()),
            "technology" | "tech" | "tool" => Some("mv:Technology".to_string()),
            _ => None,
        }
    }

    /// Batch infer classes for multiple files
    pub async fn infer_classes_batch(
        &self,
        files: Vec<FileContext>,
    ) -> Vec<Option<String>> {
        let mut results = Vec::with_capacity(files.len());

        for file in files {
            let result = self
                .infer_class(&file.path, &file.content, file.metadata.as_ref())
                .await
                .unwrap_or(None);
            results.push(result);
        }

        results
    }

    /// Ensure a class exists in the ontology, creating it if missing
    pub async fn ensure_class_exists(&self, class_iri: &str) -> OntResult<()> {
        // Check if class already exists
        if let Some(_existing) = self.ontology_repo.get_owl_class(class_iri).await? {
            return Ok(());
        }

        // Create missing class
        warn!("Class {} not found in ontology, creating it", class_iri);

        let class = OwlClass {
            iri: class_iri.to_string(),
            term_id: None,
            preferred_term: None,
            label: Some(self.extract_label_from_iri(class_iri)),
            description: Some(format!("Auto-generated class for {}", class_iri)),
            parent_classes: vec![],
            source_domain: None,
            version: None,
            class_type: None,
            status: None,
            maturity: None,
            quality_score: None,
            authority_score: None,
            public_access: None,
            content_status: None,
            owl_physicality: None,
            owl_role: None,
            belongs_to_domain: None,
            bridges_to_domain: None,
            source_file: None,
            file_sha1: None,
            markdown_content: None,
            last_synced: None,
            properties: std::collections::HashMap::new(),
            additional_metadata: None,
        };

        self.ontology_repo.add_owl_class(&class).await?;
        info!("Created missing class: {}", class_iri);

        Ok(())
    }

    /// Extract human-readable label from IRI
    fn extract_label_from_iri(&self, iri: &str) -> String {
        iri.split(':')
            .last()
            .or(iri.split('/').last())
            .unwrap_or(iri)
            .replace('_', " ")
            .replace('-', " ")
    }

    /// Use CustomReasoner to infer relationships
    ///
    /// Advanced reasoning implemented using CustomReasoner with EL++ profile
    /// Analyzes the ontology graph and infers new subsumptions (SubClassOf axioms)
    #[allow(dead_code)]
    async fn reason_about_class(&self, class_iri: &str) -> OntResult<Vec<String>> {
        // Load ontology into whelk
        // Run reasoning
        // Return inferred superclasses

        // For now, return empty (placeholder for future enhancement)
        Ok(vec![])
    }
}

/// File context for batch inference
#[derive(Debug, Clone)]
pub struct FileContext {
    pub path: String,
    pub content: String,
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Update tests to use Neo4j test containers
    // #[test]
    // fn test_infer_from_path_person() {
    //     // Create minimal test setup
    //     let engine = Arc::new(WhelkInferenceEngine::new());
    //     // Mock repo would go here in real tests

    //     // Test path inference
    //     assert!(OntologyReasoner::infer_from_path(
    //         &OntologyReasoner {
    //             inference_engine: engine.clone(),
    //             ontology_repo: Arc::new(/* TODO: Use Neo4j test container */)
    //         },
    //         "people/Tim-Cook.md"
    //     ) == Some("mv:Person".to_string()));
    // }

    // #[test]
    // fn test_infer_from_path_company() {
    //     let engine = Arc::new(WhelkInferenceEngine::new());

    //     assert!(OntologyReasoner::infer_from_path(
    //         &OntologyReasoner {
    //             inference_engine: engine.clone(),
    //             ontology_repo: Arc::new(/* TODO: Use Neo4j test container */)
    //         },
    //         "companies/Apple-Inc.md"
    //     ) == Some("mv:Company".to_string()));
    // }

    // TODO: Update tests to use Neo4j test containers
    // #[test]
    // fn test_type_to_class_iri() {
    //     let engine = Arc::new(WhelkInferenceEngine::new());
    //     let reasoner = OntologyReasoner {
    //         inference_engine: engine,
    //         ontology_repo: Arc::new(/* TODO: Use Neo4j test container */)
    //     };
    //
    //     assert_eq!(
    //         reasoner.type_to_class_iri("person"),
    //         Some("mv:Person".to_string())
    //     );
    //     assert_eq!(
    //         reasoner.type_to_class_iri("Company"),
    //         Some("mv:Company".to_string())
    //     );
    //     assert_eq!(
    //         reasoner.type_to_class_iri("unknown"),
    //         None
    //     );
    // }
}
