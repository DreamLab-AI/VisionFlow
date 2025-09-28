//! Dataset processing functionality for knowledge graph extraction

use crate::chunker::TextChunker;
use crate::config::ProcessingConfig;
use crate::error::{KgConstructionError, Result};
use crate::prompts::TripleInstructions;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use futures::stream::{self, StreamExt};

/// Represents a sample from the dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, Value>,
}

/// Represents a processed chunk from a dataset sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedChunk {
    pub id: String,
    pub text: String,
    pub chunk_id: usize,
    pub metadata: HashMap<String, Value>,
}

/// Dataset processing configuration
#[derive(Debug, Clone)]
pub struct DatasetProcessorConfig {
    pub supported_languages: Vec<String>,
    pub remove_doc_spaces: bool,
    pub debug_mode: bool,
    pub max_debug_chunks: usize,
}

impl Default for DatasetProcessorConfig {
    fn default() -> Self {
        Self {
            supported_languages: vec!["en".to_string(), "zh-CN".to_string(), "zh-HK".to_string()],
            remove_doc_spaces: false,
            debug_mode: false,
            max_debug_chunks: 20,
        }
    }
}

/// Processes and prepares dataset for knowledge graph extraction
pub struct DatasetProcessor {
    config: ProcessingConfig,
    processor_config: DatasetProcessorConfig,
    chunker: TextChunker,
    whitespace_regex: Regex,
    triple_instructions: TripleInstructions,
}

impl DatasetProcessor {
    /// Create a new dataset processor
    pub fn new(config: ProcessingConfig) -> Self {
        let processor_config = DatasetProcessorConfig {
            remove_doc_spaces: config.remove_doc_spaces,
            debug_mode: config.debug_mode,
            ..Default::default()
        };

        Self {
            config,
            processor_config,
            chunker: TextChunker::default(),
            whitespace_regex: Regex::new(r"\s+").expect("Invalid regex"),
            triple_instructions: TripleInstructions::new(),
        }
    }

    /// Create with custom processor configuration
    pub fn with_processor_config(mut self, processor_config: DatasetProcessorConfig) -> Self {
        self.processor_config = processor_config;
        self
    }

    /// Create with custom chunker
    pub fn with_chunker(mut self, chunker: TextChunker) -> Self {
        self.chunker = chunker;
        self
    }

    /// Check if content is in a supported language
    pub fn filter_language_content(&self, sample: &DataSample) -> bool {
        let language = sample.metadata
            .get("lang")
            .and_then(|v| v.as_str())
            .unwrap_or("en");

        self.triple_instructions.is_language_supported(language) ||
        self.processor_config.supported_languages.contains(&language.to_string())
    }

    /// Create chunks from a single sample
    pub async fn create_sample_chunks(&self, sample: &DataSample) -> Result<Vec<ProcessedChunk>> {
        let mut original_text = sample.text.clone();

        // Remove extra whitespace if configured
        if self.processor_config.remove_doc_spaces {
            original_text = self.whitespace_regex.replace_all(&original_text, " ").trim().to_string();
        }

        // Split text into chunks
        let text_chunks = self.chunker.split_text(&original_text)?;
        let mut chunks = Vec::with_capacity(text_chunks.len());

        for (chunk_idx, chunk_text) in text_chunks.into_iter().enumerate() {
            let chunk = ProcessedChunk {
                id: sample.id.clone(),
                text: chunk_text,
                chunk_id: chunk_idx,
                metadata: sample.metadata.clone(),
            };
            chunks.push(chunk);
        }

        Ok(chunks)
    }

    /// Process raw dataset into chunks suitable for processing
    pub async fn prepare_dataset(&self, raw_dataset: Vec<DataSample>) -> Result<Vec<ProcessedChunk>> {
        let total_texts = raw_dataset.len();

        if total_texts == 0 {
            log::warn!(
                "No texts found for shard {}/{}",
                self.config.current_shard_triple + 1,
                self.config.total_shards_triple
            );
            return Ok(vec![]);
        }

        // Calculate shard boundaries using the same logic as Python
        let (start_idx, end_idx) = self.calculate_shard_boundaries(total_texts);

        log::info!(
            "Processing shard {}/{} (texts {}-{} of {}, {} documents)",
            self.config.current_shard_triple + 1,
            self.config.total_shards_triple,
            start_idx,
            end_idx.saturating_sub(1),
            total_texts,
            end_idx - start_idx
        );

        // Process documents in assigned shard
        let mut processed_samples = Vec::new();
        let shard_data = &raw_dataset[start_idx..end_idx];

        // Use parallel processing for better performance
        let chunk_streams = shard_data.iter().map(|sample| async move {
            // Filter by language
            if !self.filter_language_content(sample) {
                log::debug!("Unsupported language in sample {}, skipping.", sample.id);
                return Ok(vec![]);
            }

            // Create chunks
            self.create_sample_chunks(sample).await
        });

        // Process samples concurrently
        let chunk_results: Vec<Result<Vec<ProcessedChunk>>> =
            stream::iter(chunk_streams)
                .buffer_unordered(self.config.max_workers)
                .collect()
                .await;

        // Collect results
        for chunk_result in chunk_results {
            let chunks = chunk_result?;
            processed_samples.extend(chunks);

            // Debug mode early termination
            if self.processor_config.debug_mode &&
               processed_samples.len() >= self.processor_config.max_debug_chunks {
                log::info!("Debug mode: Stopping at {} chunks", self.processor_config.max_debug_chunks);
                break;
            }
        }

        log::info!(
            "Generated {} chunks for shard {}/{}",
            processed_samples.len(),
            self.config.current_shard_triple + 1,
            self.config.total_shards_triple
        );

        Ok(processed_samples)
    }

    /// Calculate shard boundaries using the same algorithm as Python
    fn calculate_shard_boundaries(&self, total_texts: usize) -> (usize, usize) {
        let base_texts_per_shard = total_texts / self.config.total_shards_triple;
        let remainder = total_texts % self.config.total_shards_triple;

        // Calculate start index
        let start_idx = if self.config.current_shard_triple < remainder {
            self.config.current_shard_triple * (base_texts_per_shard + 1)
        } else {
            remainder * (base_texts_per_shard + 1) +
            (self.config.current_shard_triple - remainder) * base_texts_per_shard
        };

        // Calculate end index
        let end_idx = if self.config.current_shard_triple < remainder {
            start_idx + (base_texts_per_shard + 1)
        } else {
            start_idx + base_texts_per_shard
        };

        // Ensure indices are within bounds
        let start_idx = start_idx.min(total_texts);
        let end_idx = end_idx.min(total_texts);

        (start_idx, end_idx)
    }

    /// Load dataset from directory
    pub async fn load_dataset(&self) -> Result<Vec<DataSample>> {
        let data_path = &self.config.data_directory;

        if !data_path.exists() {
            return Err(KgConstructionError::DatasetError(
                format!("Data directory does not exist: {:?}", data_path)
            ));
        }

        // Find valid data files
        let valid_files = self.find_valid_files(data_path).await?;

        if valid_files.is_empty() {
            return Err(KgConstructionError::DatasetError(
                format!("No valid data files found in {:?}", data_path)
            ));
        }

        log::info!("Found data files: {:?}", valid_files);

        // Load data from all valid files
        let mut all_samples = Vec::new();
        for file_path in valid_files {
            let samples = self.load_file(&file_path).await?;
            all_samples.extend(samples);
        }

        log::info!("Loaded {} samples from dataset", all_samples.len());
        Ok(all_samples)
    }

    /// Find valid data files in the directory
    async fn find_valid_files(&self, data_path: &Path) -> Result<Vec<std::path::PathBuf>> {
        let mut entries = fs::read_dir(data_path).await.map_err(|e| {
            KgConstructionError::DatasetError(format!("Failed to read directory: {}", e))
        })?;

        let mut valid_files = Vec::new();

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            KgConstructionError::DatasetError(format!("Failed to read directory entry: {}", e))
        })? {
            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                if filename.starts_with(&self.config.filename_pattern) &&
                   (filename.ends_with(".json.gz") ||
                    filename.ends_with(".json") ||
                    filename.ends_with(".jsonl") ||
                    filename.ends_with(".jsonl.gz")) {
                    valid_files.push(path);
                }
            }
        }

        Ok(valid_files)
    }

    /// Load data from a single file
    async fn load_file(&self, file_path: &Path) -> Result<Vec<DataSample>> {
        let content = fs::read(file_path).await.map_err(|e| {
            KgConstructionError::DatasetError(format!("Failed to read file {:?}: {}", file_path, e))
        })?;

        // Handle compression
        let text_content = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            self.decompress_gzip(&content)?
        } else {
            String::from_utf8(content).map_err(|e| {
                KgConstructionError::DatasetError(format!("Invalid UTF-8 in file {:?}: {}", file_path, e))
            })?
        };

        // Parse based on file type
        if file_path.to_string_lossy().contains(".jsonl") {
            self.parse_jsonl(&text_content)
        } else {
            self.parse_json(&text_content)
        }
    }

    /// Decompress gzip content
    fn decompress_gzip(&self, content: &[u8]) -> Result<String> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(content);
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed).map_err(|e| {
            KgConstructionError::DatasetError(format!("Failed to decompress gzip: {}", e))
        })?;

        Ok(decompressed)
    }

    /// Parse JSON content
    fn parse_json(&self, content: &str) -> Result<Vec<DataSample>> {
        let value: Value = serde_json::from_str(content).map_err(|e| {
            KgConstructionError::DatasetError(format!("Failed to parse JSON: {}", e))
        })?;

        match value {
            Value::Array(arr) => {
                let mut samples = Vec::with_capacity(arr.len());
                for (idx, item) in arr.into_iter().enumerate() {
                    samples.push(self.parse_sample_json(item, idx)?);
                }
                Ok(samples)
            }
            Value::Object(_) => {
                // Single object
                Ok(vec![self.parse_sample_json(value, 0)?])
            }
            _ => Err(KgConstructionError::DatasetError(
                "JSON must be an object or array of objects".to_string()
            ))
        }
    }

    /// Parse JSONL content
    fn parse_jsonl(&self, content: &str) -> Result<Vec<DataSample>> {
        let mut samples = Vec::new();

        for (idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let value: Value = serde_json::from_str(line).map_err(|e| {
                KgConstructionError::DatasetError(format!("Failed to parse JSONL line {}: {}", idx + 1, e))
            })?;

            samples.push(self.parse_sample_json(value, idx)?);
        }

        Ok(samples)
    }

    /// Parse a single JSON sample
    fn parse_sample_json(&self, value: Value, default_idx: usize) -> Result<DataSample> {
        let obj = value.as_object().ok_or_else(|| {
            KgConstructionError::DatasetError("Sample must be a JSON object".to_string())
        })?;

        // Extract ID
        let id = obj.get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("sample_{}", default_idx));

        // Extract text
        let text = obj.get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Extract metadata
        let mut metadata = HashMap::new();
        for (key, value) in obj {
            if key != "text" {
                metadata.insert(key.clone(), value.clone());
            }
        }

        Ok(DataSample { id, text, metadata })
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> ProcessorStats {
        ProcessorStats {
            total_shards: self.config.total_shards_triple,
            current_shard: self.config.current_shard_triple,
            supported_languages: self.processor_config.supported_languages.clone(),
            chunker_config: self.chunker.config(),
            debug_mode: self.processor_config.debug_mode,
        }
    }
}

/// Statistics about the dataset processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorStats {
    pub total_shards: usize,
    pub current_shard: usize,
    pub supported_languages: Vec<String>,
    pub chunker_config: crate::chunker::ChunkerConfig,
    pub debug_mode: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProcessingConfig;
    use std::path::PathBuf;

    fn create_test_config() -> ProcessingConfig {
        ProcessingConfig {
            data_directory: PathBuf::from("./test_data"),
            filename_pattern: "test".to_string(),
            total_shards_triple: 2,
            current_shard_triple: 0,
            debug_mode: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_processor_creation() {
        let config = create_test_config();
        let processor = DatasetProcessor::new(config);
        assert_eq!(processor.config.total_shards_triple, 2);
        assert!(processor.processor_config.debug_mode);
    }

    #[test]
    fn test_language_filtering() {
        let config = create_test_config();
        let processor = DatasetProcessor::new(config);

        let mut metadata = HashMap::new();
        metadata.insert("lang".to_string(), Value::String("en".to_string()));

        let sample = DataSample {
            id: "test".to_string(),
            text: "test text".to_string(),
            metadata,
        };

        assert!(processor.filter_language_content(&sample));

        let mut unsupported_metadata = HashMap::new();
        unsupported_metadata.insert("lang".to_string(), Value::String("unsupported".to_string()));

        let unsupported_sample = DataSample {
            id: "test".to_string(),
            text: "test text".to_string(),
            metadata: unsupported_metadata,
        };

        assert!(!processor.filter_language_content(&unsupported_sample));
    }

    #[tokio::test]
    async fn test_sample_chunking() {
        let config = create_test_config();
        let processor = DatasetProcessor::new(config);

        let mut metadata = HashMap::new();
        metadata.insert("lang".to_string(), Value::String("en".to_string()));

        let sample = DataSample {
            id: "test_sample".to_string(),
            text: "This is a test text that should be chunked properly.".to_string(),
            metadata,
        };

        let chunks = processor.create_sample_chunks(&sample).await.unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].id, "test_sample");
        assert_eq!(chunks[0].chunk_id, 0);
    }

    #[test]
    fn test_shard_boundaries() {
        let config = ProcessingConfig {
            total_shards_triple: 3,
            current_shard_triple: 1,
            ..Default::default()
        };
        let processor = DatasetProcessor::new(config);

        let (start, end) = processor.calculate_shard_boundaries(10);
        // With 10 items and 3 shards: [0-4), [4-7), [7-10)
        // Shard 1 should get items 4-6 (3 items)
        assert_eq!(start, 4);
        assert_eq!(end, 7);
    }

    #[test]
    fn test_parse_sample_json() {
        let config = create_test_config();
        let processor = DatasetProcessor::new(config);

        let json_str = r#"{"id": "test_id", "text": "test text", "lang": "en"}"#;
        let value: Value = serde_json::from_str(json_str).unwrap();

        let sample = processor.parse_sample_json(value, 0).unwrap();
        assert_eq!(sample.id, "test_id");
        assert_eq!(sample.text, "test text");
        assert_eq!(sample.metadata.get("lang").unwrap().as_str().unwrap(), "en");
    }
}