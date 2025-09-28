//! Triple extraction from natural language text

use crate::triple::{Triple, Entity, Relation, EntityType, RelationType};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utils::{AutoSchemaError, Result};

/// Result of triple extraction from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub triples: Vec<Triple>,
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub confidence: f32,
    pub processing_time_ms: u64,
    pub metadata: HashMap<String, String>,
}

/// Configuration for triple extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    pub max_triples_per_chunk: usize,
    pub confidence_threshold: f32,
    pub enable_entity_linking: bool,
    pub enable_coreference_resolution: bool,
    pub language: String,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            max_triples_per_chunk: 20,
            confidence_threshold: 0.7,
            enable_entity_linking: true,
            enable_coreference_resolution: true,
            language: "en".to_string(),
        }
    }
}

/// Trait for triple extraction implementations
#[async_trait]
pub trait TripleExtractor: Send + Sync {
    /// Extract triples from a text chunk
    async fn extract_triples(&self, text: &str, config: &ExtractionConfig) -> Result<ExtractionResult>;

    /// Extract entities from text
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>>;

    /// Extract relations from text
    async fn extract_relations(&self, text: &str) -> Result<Vec<Relation>>;

    /// Get the name of this extractor
    fn name(&self) -> &str;
}

/// LLM-based triple extractor
pub struct LlmTripleExtractor {
    pub name: String,
    pub llm_client: Box<dyn llm_generator::LlmClient>,
}

impl LlmTripleExtractor {
    /// Create a new LLM-based extractor
    pub fn new(name: String, llm_client: Box<dyn llm_generator::LlmClient>) -> Self {
        Self { name, llm_client }
    }

    /// Generate the extraction prompt
    fn create_extraction_prompt(&self, text: &str, config: &ExtractionConfig) -> String {
        format!(
            r#"Extract semantic triples from the following text. Each triple should be in the format (subject, predicate, object).
Focus on factual relationships and avoid speculation.

Text: "{}"

Instructions:
1. Extract up to {} triples
2. Only include triples with confidence >= {}
3. Use clear, specific entity names
4. Use standard predicate types when possible
5. Output as JSON array of triples

Format each triple as:
{{
  "subject": {{ "label": "...", "type": "..." }},
  "predicate": {{ "label": "...", "type": "..." }},
  "object": {{ "label": "...", "type": "..." }},
  "confidence": 0.0-1.0
}}

Entity types: Person, Organization, Location, Event, Concept, Product, Unknown
Relation types: isA, partOf, locatedIn, worksFor, createdBy, relatedTo, causes, before, after, similar, unknown
"#,
            text, config.max_triples_per_chunk, config.confidence_threshold
        )
    }

    /// Parse the LLM response into triples
    fn parse_llm_response(&self, response: &str) -> Result<Vec<Triple>> {
        // Try to parse as JSON array
        let parsed: serde_json::Value = serde_json::from_str(response)
            .map_err(|e| AutoSchemaError::kg_construction(format!("Failed to parse LLM response: {}", e)))?;

        let triples_array = parsed
            .as_array()
            .ok_or_else(|| AutoSchemaError::kg_construction("LLM response is not an array"))?;

        let mut triples = Vec::new();

        for triple_value in triples_array {
            let triple = self.parse_triple_json(triple_value)?;
            triples.push(triple);
        }

        Ok(triples)
    }

    /// Parse a single triple from JSON
    fn parse_triple_json(&self, value: &serde_json::Value) -> Result<Triple> {
        let obj = value
            .as_object()
            .ok_or_else(|| AutoSchemaError::kg_construction("Triple is not a JSON object"))?;

        // Parse subject
        let subject_obj = obj
            .get("subject")
            .and_then(|s| s.as_object())
            .ok_or_else(|| AutoSchemaError::kg_construction("Missing subject"))?;

        let subject = Entity::new(
            subject_obj
                .get("label")
                .and_then(|l| l.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            self.parse_entity_type(
                subject_obj
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("Unknown"),
            ),
        );

        // Parse predicate
        let predicate_obj = obj
            .get("predicate")
            .and_then(|p| p.as_object())
            .ok_or_else(|| AutoSchemaError::kg_construction("Missing predicate"))?;

        let predicate = Relation::new(
            predicate_obj
                .get("label")
                .and_then(|l| l.as_str())
                .unwrap_or("unknown")
                .to_string(),
            self.parse_relation_type(
                predicate_obj
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("unknown"),
            ),
        );

        // Parse object
        let object_obj = obj
            .get("object")
            .and_then(|o| o.as_object())
            .ok_or_else(|| AutoSchemaError::kg_construction("Missing object"))?;

        let object = Entity::new(
            object_obj
                .get("label")
                .and_then(|l| l.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            self.parse_entity_type(
                object_obj
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("Unknown"),
            ),
        );

        // Parse confidence
        let confidence = obj
            .get("confidence")
            .and_then(|c| c.as_f64())
            .unwrap_or(0.5) as f32;

        let triple = Triple::new(subject, predicate, object, confidence, "llm".to_string());
        Ok(triple)
    }

    /// Parse entity type from string
    fn parse_entity_type(&self, type_str: &str) -> EntityType {
        match type_str.to_lowercase().as_str() {
            "person" => EntityType::Person,
            "organization" => EntityType::Organization,
            "location" => EntityType::Location,
            "event" => EntityType::Event,
            "concept" => EntityType::Concept,
            "product" => EntityType::Product,
            _ => EntityType::Unknown,
        }
    }

    /// Parse relation type from string
    fn parse_relation_type(&self, type_str: &str) -> RelationType {
        match type_str.to_lowercase().as_str() {
            "isa" | "is_a" => RelationType::IsA,
            "partof" | "part_of" => RelationType::PartOf,
            "locatedin" | "located_in" => RelationType::LocatedIn,
            "worksfor" | "works_for" => RelationType::WorksFor,
            "createdby" | "created_by" => RelationType::CreatedBy,
            "relatedto" | "related_to" => RelationType::RelatedTo,
            "causes" => RelationType::Causes,
            "before" => RelationType::Before,
            "after" => RelationType::After,
            "similar" => RelationType::Similar,
            _ => RelationType::Unknown,
        }
    }
}

#[async_trait]
impl TripleExtractor for LlmTripleExtractor {
    async fn extract_triples(&self, text: &str, config: &ExtractionConfig) -> Result<ExtractionResult> {
        let start_time = std::time::Instant::now();

        // Create extraction prompt
        let prompt = self.create_extraction_prompt(text, config);

        // Get LLM response
        let response = self
            .llm_client
            .generate(&prompt)
            .await
            .map_err(|e| AutoSchemaError::kg_construction(format!("LLM generation failed: {}", e)))?;

        // Parse triples from response
        let triples = self.parse_llm_response(&response)?;

        // Filter by confidence threshold
        let filtered_triples: Vec<Triple> = triples
            .into_iter()
            .filter(|t| t.confidence >= config.confidence_threshold)
            .collect();

        // Extract unique entities and relations
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let mut entity_ids = std::collections::HashSet::new();
        let mut relation_ids = std::collections::HashSet::new();

        for triple in &filtered_triples {
            if entity_ids.insert(triple.subject.id.clone()) {
                entities.push(triple.subject.clone());
            }
            if entity_ids.insert(triple.object.id.clone()) {
                entities.push(triple.object.clone());
            }
            if relation_ids.insert(triple.predicate.id.clone()) {
                relations.push(triple.predicate.clone());
            }
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        let confidence = if filtered_triples.is_empty() {
            0.0
        } else {
            filtered_triples.iter().map(|t| t.confidence).sum::<f32>() / filtered_triples.len() as f32
        };

        let mut metadata = HashMap::new();
        metadata.insert("extractor".to_string(), self.name.clone());
        metadata.insert("original_triple_count".to_string(), filtered_triples.len().to_string());

        Ok(ExtractionResult {
            triples: filtered_triples,
            entities,
            relations,
            confidence,
            processing_time_ms,
            metadata,
        })
    }

    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>> {
        // This is a simplified implementation
        // In practice, you'd use NER models or LLM prompts specifically for entities
        let config = ExtractionConfig::default();
        let result = self.extract_triples(text, &config).await?;
        Ok(result.entities)
    }

    async fn extract_relations(&self, text: &str) -> Result<Vec<Relation>> {
        let config = ExtractionConfig::default();
        let result = self.extract_triples(text, &config).await?;
        Ok(result.relations)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_config_default() {
        let config = ExtractionConfig::default();
        assert_eq!(config.max_triples_per_chunk, 20);
        assert_eq!(config.confidence_threshold, 0.7);
        assert!(config.enable_entity_linking);
    }

    #[test]
    fn test_entity_type_parsing() {
        let extractor = LlmTripleExtractor::new(
            "test".to_string(),
            Box::new(MockLlmClient::new()),
        );

        assert_eq!(extractor.parse_entity_type("Person"), EntityType::Person);
        assert_eq!(extractor.parse_entity_type("organization"), EntityType::Organization);
        assert_eq!(extractor.parse_entity_type("unknown"), EntityType::Unknown);
    }

    // Mock LLM client for testing
    struct MockLlmClient;

    impl MockLlmClient {
        fn new() -> Self {
            Self
        }
    }

    #[async_trait]
    impl llm_generator::LlmClient for MockLlmClient {
        async fn generate(&self, _prompt: &str) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
            Ok(r#"[
                {
                    "subject": {"label": "Alice", "type": "Person"},
                    "predicate": {"label": "worksFor", "type": "worksFor"},
                    "object": {"label": "Company", "type": "Organization"},
                    "confidence": 0.9
                }
            ]"#.to_string())
        }

        fn name(&self) -> &str {
            "mock"
        }
    }
}