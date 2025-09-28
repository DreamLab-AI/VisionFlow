//! Configuration structures for the triple extraction pipeline

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for text processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Model path for knowledge extraction
    pub model_path: String,
    /// Directory containing input data
    pub data_directory: PathBuf,
    /// Filename pattern to match
    pub filename_pattern: String,
    /// Batch size for triple processing
    pub batch_size_triple: usize,
    /// Batch size for concept processing
    pub batch_size_concept: usize,
    /// Output directory for results
    pub output_directory: PathBuf,
    /// Total number of data shards for triple processing
    pub total_shards_triple: usize,
    /// Current shard index for triple processing
    pub current_shard_triple: usize,
    /// Total number of data shards for concept processing
    pub total_shards_concept: usize,
    /// Current shard index for concept processing
    pub current_shard_concept: usize,
    /// Use 8-bit quantization
    pub use_8bit: bool,
    /// Enable debug mode
    pub debug_mode: bool,
    /// Resume from specific batch
    pub resume_from: usize,
    /// Record usage statistics
    pub record: bool,
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,
    /// Maximum number of worker threads
    pub max_workers: usize,
    /// Remove document spaces during preprocessing
    pub remove_doc_spaces: bool,
    /// ML inference backend to use
    pub inference_backend: InferenceBackend,
    /// Device to use for inference
    pub device: Device,
}

/// ML inference backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceBackend {
    /// Use Candle framework
    Candle,
    /// Use ONNX Runtime
    Ort,
    /// Use remote API endpoint
    RemoteApi { endpoint: String, api_key: Option<String> },
}

/// Device options for ML inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Device {
    /// Use CPU
    Cpu,
    /// Use CUDA GPU
    Cuda(usize),
    /// Use Metal (macOS)
    Metal,
    /// Automatic device selection
    Auto,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            model_path: "meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
            data_directory: PathBuf::from("./data"),
            filename_pattern: "en_simple_wiki_v0".to_string(),
            batch_size_triple: 16,
            batch_size_concept: 64,
            output_directory: PathBuf::from("./generation_result"),
            total_shards_triple: 1,
            current_shard_triple: 0,
            total_shards_concept: 1,
            current_shard_concept: 0,
            use_8bit: false,
            debug_mode: false,
            resume_from: 0,
            record: false,
            max_new_tokens: 8192,
            max_workers: 8,
            remove_doc_spaces: false,
            inference_backend: InferenceBackend::Candle,
            device: Device::Auto,
        }
    }
}

impl ProcessingConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model path
    pub fn with_model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = path.into();
        self
    }

    /// Set the data directory
    pub fn with_data_directory(mut self, dir: impl Into<PathBuf>) -> Self {
        self.data_directory = dir.into();
        self
    }

    /// Set the output directory
    pub fn with_output_directory(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_directory = dir.into();
        self
    }

    /// Set batch size for triple processing
    pub fn with_batch_size_triple(mut self, size: usize) -> Self {
        self.batch_size_triple = size;
        self
    }

    /// Enable debug mode
    pub fn with_debug_mode(mut self, debug: bool) -> Self {
        self.debug_mode = debug;
        self
    }

    /// Set inference backend
    pub fn with_inference_backend(mut self, backend: InferenceBackend) -> Self {
        self.inference_backend = backend;
        self
    }

    /// Set device for inference
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.batch_size_triple == 0 {
            return Err(crate::error::KgConstructionError::ConfigError(
                "Batch size for triple processing must be greater than 0".to_string()
            ));
        }

        if self.batch_size_concept == 0 {
            return Err(crate::error::KgConstructionError::ConfigError(
                "Batch size for concept processing must be greater than 0".to_string()
            ));
        }

        if self.max_new_tokens == 0 {
            return Err(crate::error::KgConstructionError::ConfigError(
                "Maximum new tokens must be greater than 0".to_string()
            ));
        }

        if !self.data_directory.exists() {
            return Err(crate::error::KgConstructionError::ConfigError(
                format!("Data directory does not exist: {:?}", self.data_directory)
            ));
        }

        Ok(())
    }
}