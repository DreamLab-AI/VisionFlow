//! Configuration management for AutoSchemaKG
//!
//! This module provides centralized configuration management with support for
//! environment variables, configuration files, and runtime overrides.

use crate::errors::{AutoSchemaError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Main configuration structure for AutoSchemaKG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Database configuration
    pub database: DatabaseConfig,

    /// LLM provider configurations
    pub llm: LlmConfig,

    /// Vector store configuration
    pub vectorstore: VectorStoreConfig,

    /// Text processing configuration
    pub text_processing: TextProcessingConfig,

    /// Knowledge graph construction settings
    pub kg_construction: KgConstructionConfig,

    /// Retrieval settings
    pub retrieval: RetrievalConfig,

    /// Logging configuration
    pub logging: LoggingConfig,

    /// Performance settings
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub connection_timeout_ms: u64,
    pub query_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub default_provider: String,
    pub providers: HashMap<String, LlmProviderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_ms: u64,
    pub rate_limit_requests_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    pub backend: String, // "faiss", "hnswlib", "flat"
    pub embedding_model: String,
    pub dimension: usize,
    pub index_parameters: HashMap<String, serde_json::Value>,
    pub batch_size: usize,
    pub storage_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingConfig {
    pub max_text_length: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub language: String,
    pub enable_preprocessing: bool,
    pub preprocessing_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KgConstructionConfig {
    pub max_triples_per_chunk: usize,
    pub confidence_threshold: f32,
    pub entity_linking_enabled: bool,
    pub concept_generation_enabled: bool,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub top_k: usize,
    pub similarity_threshold: f32,
    pub rerank_enabled: bool,
    pub context_window_size: usize,
    pub cache_size: usize,
    pub cache_ttl_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub output: String, // "stdout", "file", "json"
    pub file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub parallel_processing: bool,
    pub max_concurrent_tasks: usize,
    pub memory_limit_mb: usize,
    pub enable_metrics: bool,
    pub metrics_port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            database: DatabaseConfig {
                url: "neo4j://localhost:7687".to_string(),
                max_connections: 10,
                connection_timeout_ms: 5000,
                query_timeout_ms: 30000,
            },
            llm: LlmConfig {
                default_provider: "openai".to_string(),
                providers: {
                    let mut providers = HashMap::new();
                    providers.insert(
                        "openai".to_string(),
                        LlmProviderConfig {
                            api_key: None,
                            base_url: "https://api.openai.com/v1".to_string(),
                            model: "gpt-3.5-turbo".to_string(),
                            max_tokens: 2048,
                            temperature: 0.1,
                            timeout_ms: 30000,
                            rate_limit_requests_per_minute: 60,
                        },
                    );
                    providers
                },
            },
            vectorstore: VectorStoreConfig {
                backend: "flat".to_string(),
                embedding_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                dimension: 384,
                index_parameters: HashMap::new(),
                batch_size: 32,
                storage_path: "./data/vectorstore".to_string(),
            },
            text_processing: TextProcessingConfig {
                max_text_length: 1_000_000,
                chunk_size: 512,
                chunk_overlap: 50,
                language: "en".to_string(),
                enable_preprocessing: true,
                preprocessing_steps: vec![
                    "normalize_whitespace".to_string(),
                    "remove_html".to_string(),
                    "normalize_unicode".to_string(),
                ],
            },
            kg_construction: KgConstructionConfig {
                max_triples_per_chunk: 20,
                confidence_threshold: 0.7,
                entity_linking_enabled: true,
                concept_generation_enabled: true,
                batch_size: 16,
            },
            retrieval: RetrievalConfig {
                top_k: 10,
                similarity_threshold: 0.5,
                rerank_enabled: false,
                context_window_size: 2048,
                cache_size: 1000,
                cache_ttl_seconds: 3600,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "json".to_string(),
                output: "stdout".to_string(),
                file_path: None,
            },
            performance: PerformanceConfig {
                parallel_processing: true,
                max_concurrent_tasks: 4,
                memory_limit_mb: 2048,
                enable_metrics: false,
                metrics_port: 9090,
            },
        }
    }
}

impl Config {
    /// Load configuration from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| AutoSchemaError::configuration(format!("Failed to read config file: {}", e)))?;

        let config: Config = serde_json::from_str(&content)
            .map_err(|e| AutoSchemaError::configuration(format!("Failed to parse config: {}", e)))?;

        Ok(config)
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Config::default();

        // Database configuration
        if let Ok(url) = std::env::var("DATABASE_URL") {
            config.database.url = url;
        }

        // LLM API keys
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if let Some(provider) = config.llm.providers.get_mut("openai") {
                provider.api_key = Some(api_key);
            }
        }

        // Logging level
        if let Ok(level) = std::env::var("LOG_LEVEL") {
            config.logging.level = level;
        }

        Ok(config)
    }

    /// Save configuration to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| AutoSchemaError::configuration(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path.as_ref(), content)
            .map_err(|e| AutoSchemaError::configuration(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate database URL
        if self.database.url.is_empty() {
            return Err(AutoSchemaError::validation("database.url", "Database URL cannot be empty"));
        }

        // Validate vector store dimension
        if self.vectorstore.dimension == 0 {
            return Err(AutoSchemaError::validation("vectorstore.dimension", "Dimension must be > 0"));
        }

        // Validate text processing chunk size
        if self.text_processing.chunk_size == 0 {
            return Err(AutoSchemaError::validation("text_processing.chunk_size", "Chunk size must be > 0"));
        }

        // Validate retrieval top_k
        if self.retrieval.top_k == 0 {
            return Err(AutoSchemaError::validation("retrieval.top_k", "top_k must be > 0"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(config.database.url, deserialized.database.url);
    }

    #[test]
    fn test_config_file_operations() {
        let config = Config::default();
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("config.json");

        // Save config
        config.save_to_file(&config_path).unwrap();

        // Load config
        let loaded_config = Config::from_file(&config_path).unwrap();
        assert_eq!(config.database.url, loaded_config.database.url);
    }
}