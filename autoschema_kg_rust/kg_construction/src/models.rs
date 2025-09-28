//! Data models for knowledge graph construction

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for different extraction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub model_name: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub confidence_threshold: f32,
}

/// Types of models supported for extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Large Language Model (OpenAI, Anthropic, etc.)
    LLM,
    /// Named Entity Recognition model
    NER,
    /// Relation Extraction model
    RelationExtraction,
    /// Custom model
    Custom(String),
}

/// Named Entity Recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerResult {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

/// Relation extraction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationResult {
    pub subject: NerResult,
    pub predicate: String,
    pub object: NerResult,
    pub confidence: f32,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(model_type: ModelType, model_name: String) -> Self {
        Self {
            model_type,
            model_name,
            parameters: HashMap::new(),
            confidence_threshold: 0.5,
        }
    }

    /// Add a parameter to the model configuration
    pub fn with_parameter<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Set the confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::new(ModelType::LLM, "gpt-3.5-turbo".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig::new(ModelType::LLM, "gpt-4".to_string())
            .with_parameter("temperature", 0.1)
            .with_confidence_threshold(0.8);

        assert_eq!(config.model_name, "gpt-4");
        assert_eq!(config.confidence_threshold, 0.8);
        assert_eq!(config.parameters.len(), 1);
    }
}