//! Text processors for knowledge graph construction

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use utils::{Result, UtilsError};

/// Configuration for text processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub enable_preprocessing: bool,
    pub max_concurrent_chunks: usize,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 50,
            enable_preprocessing: true,
            max_concurrent_chunks: 4,
        }
    }
}

/// Result of text processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorResult {
    pub chunks: Vec<TextChunk>,
    pub metadata: std::collections::HashMap<String, String>,
    pub processing_time_ms: u64,
}

/// A chunk of processed text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub id: String,
    pub content: String,
    pub start_position: usize,
    pub end_position: usize,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Trait for text processors
#[async_trait]
pub trait TextProcessor: Send + Sync {
    /// Process raw text into structured chunks
    async fn process(&self, text: &str, config: &ProcessorConfig) -> Result<ProcessorResult>;

    /// Get the name of this processor
    fn name(&self) -> &str;
}

/// Basic text processor that splits text into chunks
pub struct BasicTextProcessor {
    name: String,
}

impl BasicTextProcessor {
    /// Create a new basic text processor
    pub fn new() -> Self {
        Self {
            name: "basic".to_string(),
        }
    }

    /// Split text into sentences
    fn split_sentences(&self, text: &str) -> Vec<&str> {
        text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && s.len() > 3)
            .collect()
    }

    /// Create chunks from sentences
    fn create_chunks(&self, sentences: Vec<&str>, config: &ProcessorConfig) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start = 0;
        let mut chunk_id = 0;

        for sentence in sentences {
            if current_chunk.len() + sentence.len() > config.chunk_size && !current_chunk.is_empty() {
                // Create chunk
                let chunk = TextChunk {
                    id: format!("chunk_{}", chunk_id),
                    content: current_chunk.trim().to_string(),
                    start_position: current_start,
                    end_position: current_start + current_chunk.len(),
                    metadata: std::collections::HashMap::new(),
                };

                chunks.push(chunk);
                chunk_id += 1;

                // Handle overlap
                if config.chunk_overlap > 0 && current_chunk.len() > config.chunk_overlap {
                    let overlap_start = current_chunk.len() - config.chunk_overlap;
                    current_chunk = current_chunk[overlap_start..].to_string();
                    current_start += overlap_start;
                } else {
                    current_chunk.clear();
                    current_start += current_chunk.len();
                }
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence);
        }

        // Add the last chunk if it's not empty
        if !current_chunk.trim().is_empty() {
            let chunk = TextChunk {
                id: format!("chunk_{}", chunk_id),
                content: current_chunk.trim().to_string(),
                start_position: current_start,
                end_position: current_start + current_chunk.len(),
                metadata: std::collections::HashMap::new(),
            };
            chunks.push(chunk);
        }

        chunks
    }
}

impl Default for BasicTextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TextProcessor for BasicTextProcessor {
    async fn process(&self, text: &str, config: &ProcessorConfig) -> Result<ProcessorResult> {
        let start_time = std::time::Instant::now();

        // Preprocess text if enabled
        let processed_text = if config.enable_preprocessing {
            // Simple preprocessing: normalize whitespace
            text.split_whitespace().collect::<Vec<&str>>().join(" ")
        } else {
            text.to_string()
        };

        // Split into sentences
        let sentences = self.split_sentences(&processed_text);

        // Create chunks
        let chunks = self.create_chunks(sentences, config);

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("processor".to_string(), self.name.clone());
        metadata.insert("chunk_count".to_string(), chunks.len().to_string());
        metadata.insert("original_length".to_string(), text.len().to_string());

        Ok(ProcessorResult {
            chunks,
            metadata,
            processing_time_ms,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_processor() {
        let processor = BasicTextProcessor::new();
        let config = ProcessorConfig::default();
        let text = "First sentence. Second sentence! Third sentence?";

        let result = processor.process(text, &config).await.unwrap();
        assert!(!result.chunks.is_empty());
        assert!(result.processing_time_ms > 0);
    }

    #[test]
    fn test_sentence_splitting() {
        let processor = BasicTextProcessor::new();
        let text = "First sentence. Second sentence! Third sentence?";
        let sentences = processor.split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_chunk_creation() {
        let processor = BasicTextProcessor::new();
        let config = ProcessorConfig {
            chunk_size: 20,
            chunk_overlap: 5,
            ..Default::default()
        };

        let sentences = vec!["Short sentence", "Another short sentence", "Third sentence"];
        let chunks = processor.create_chunks(sentences, &config);
        assert!(!chunks.is_empty());
    }
}