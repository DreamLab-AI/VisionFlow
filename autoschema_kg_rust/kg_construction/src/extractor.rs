//! Main knowledge graph extraction pipeline

use crate::config::ProcessingConfig;
use crate::dataset::{DataSample, DatasetProcessor};
use crate::error::{KgConstructionError, Result};
use crate::loader::{CustomDataLoader, DataBatch, LoaderConfig};
use crate::ml_inference::{InferenceConfig, ModelInference, UsageStats};
use crate::parser::{OutputParser, ParserConfig, Triple, EventEntity, EventRelation};
use crate::prompts::{ProcessingStage, TripleInstructions};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};

/// Results from processing a single stage
#[derive(Debug, Clone)]
pub struct StageResults {
    pub outputs: Vec<String>,
    pub usage_stats: Option<Vec<UsageStats>>,
}

/// Combined results from all three processing stages
#[derive(Debug, Clone)]
pub struct TripleExtractionResults {
    pub stage_1: StageResults, // Entity-Relation
    pub stage_2: StageResults, // Event-Entity
    pub stage_3: StageResults, // Event-Relation
}

/// A complete result entry for a processed chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub id: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub original_text: String,
    pub entity_relation_dict: Vec<Triple>,
    pub event_entity_relation_dict: Vec<EventEntity>,
    pub event_relation_dict: Vec<EventRelation>,
    pub output_stage_one: String,
    pub output_stage_two: String,
    pub output_stage_three: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_stage_one: Option<UsageStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_stage_two: Option<UsageStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_stage_three: Option<UsageStats>,
}

/// Statistics about the extraction process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionStats {
    pub total_chunks_processed: usize,
    pub total_batches_processed: usize,
    pub total_entities_extracted: usize,
    pub total_events_extracted: usize,
    pub total_relations_extracted: usize,
    pub processing_time_seconds: f64,
    pub average_chunk_time_ms: f64,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
}

/// Main class for knowledge graph extraction pipeline
pub struct KnowledgeGraphExtractor {
    config: ProcessingConfig,
    model_inference: ModelInference,
    dataset_processor: DatasetProcessor,
    output_parser: OutputParser,
    triple_instructions: TripleInstructions,
    stats: ExtractionStats,
}

impl KnowledgeGraphExtractor {
    /// Create a new knowledge graph extractor
    pub async fn new(config: ProcessingConfig) -> Result<Self> {
        // Initialize ML inference
        let inference_config = InferenceConfig {
            model_path: config.model_path.clone(),
            device: config.device.clone(),
            max_tokens: config.max_new_tokens,
            batch_size: config.batch_size_triple,
            use_quantization: config.use_8bit,
            ..Default::default()
        };

        let model_inference = ModelInference::new(config.inference_backend.clone(), inference_config).await?;

        // Initialize other components
        let dataset_processor = DatasetProcessor::new(config.clone());
        let parser_config = ParserConfig {
            repair_json: true,
            strict_validation: !config.debug_mode,
            fallback_empty: true,
            ..Default::default()
        };
        let output_parser = OutputParser::with_config(parser_config);

        let stats = ExtractionStats {
            total_chunks_processed: 0,
            total_batches_processed: 0,
            total_entities_extracted: 0,
            total_events_extracted: 0,
            total_relations_extracted: 0,
            processing_time_seconds: 0.0,
            average_chunk_time_ms: 0.0,
            start_time: Utc::now(),
            end_time: None,
        };

        Ok(Self {
            config,
            model_inference,
            dataset_processor,
            output_parser,
            triple_instructions: TripleInstructions::new(),
            stats,
        })
    }

    /// Load dataset from configured directory
    pub async fn load_dataset(&self) -> Result<Vec<DataSample>> {
        log::info!("Loading dataset from: {:?}", self.config.data_directory);
        self.dataset_processor.load_dataset().await
    }

    /// Process a single stage of extraction
    async fn process_stage(
        &self,
        stage_messages: &[Vec<crate::loader::MessagePair>],
        stage: ProcessingStage,
    ) -> Result<StageResults> {
        log::debug!("Processing stage: {:?} with {} messages", stage, stage_messages.len());

        // Convert messages to the format expected by the inference engine
        let prompts: Vec<String> = stage_messages
            .iter()
            .map(|msg_array| {
                let mut prompt = String::new();
                for msg in msg_array {
                    prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
                }
                prompt
            })
            .collect();

        // Run inference
        let outputs = self.model_inference
            .generate_batch_optimized(&prompts, self.config.max_new_tokens)
            .await?;

        // Record usage if enabled
        let usage_stats = if self.config.record {
            Some(vec![UsageStats::default(); outputs.len()])
        } else {
            None
        };

        Ok(StageResults {
            outputs,
            usage_stats,
        })
    }

    /// Process all three stages for a batch
    async fn process_all_stages(&self, batch: &DataBatch) -> Result<TripleExtractionResults> {
        // Process all stages in parallel for better performance
        let (stage_1_result, stage_2_result, stage_3_result) = tokio::try_join!(
            self.process_stage(&batch.instructions.stage_1, ProcessingStage::EntityRelation),
            self.process_stage(&batch.instructions.stage_2, ProcessingStage::EventEntity),
            self.process_stage(&batch.instructions.stage_3, ProcessingStage::EventRelation),
        )?;

        Ok(TripleExtractionResults {
            stage_1: stage_1_result,
            stage_2: stage_2_result,
            stage_3: stage_3_result,
        })
    }

    /// Parse stage results into structured data
    fn parse_stage_results(
        &mut self,
        stage_results: &TripleExtractionResults,
    ) -> Result<(Vec<Vec<Triple>>, Vec<Vec<EventEntity>>, Vec<Vec<EventRelation>>)> {
        // Parse entity-relation results
        let entity_relations = self.output_parser
            .extract_entity_relations(&stage_results.stage_1.outputs)?;

        // Parse event-entity results
        let event_entities = self.output_parser
            .extract_event_entities(&stage_results.stage_2.outputs)?;

        // Parse event-relation results
        let event_relations = self.output_parser
            .extract_event_relations(&stage_results.stage_3.outputs)?;

        Ok((entity_relations, event_entities, event_relations))
    }

    /// Prepare individual processing results
    fn prepare_processing_results(
        &mut self,
        batch: &DataBatch,
        stage_results: &TripleExtractionResults,
        parsed_results: (Vec<Vec<Triple>>, Vec<Vec<EventEntity>>, Vec<Vec<EventRelation>>),
    ) -> Result<Vec<ProcessingResult>> {
        let (entity_relations, event_entities, event_relations) = parsed_results;
        let mut results = Vec::with_capacity(batch.batch_ids.len());

        for i in 0..batch.batch_ids.len() {
            // Handle date serialization in metadata
            let mut metadata = batch.batch_metadata[i].clone();
            if let Some(date_value) = metadata.get_mut("date_download") {
                if let Some(date_str) = date_value.as_str() {
                    metadata.insert("date_download".to_string(), serde_json::Value::String(date_str.to_string()));
                }
            }

            let result = ProcessingResult {
                id: batch.batch_ids[i].clone(),
                metadata,
                original_text: batch.batch_texts[i].clone(),
                entity_relation_dict: entity_relations[i].clone(),
                event_entity_relation_dict: event_entities[i].clone(),
                event_relation_dict: event_relations[i].clone(),
                output_stage_one: stage_results.stage_1.outputs[i].clone(),
                output_stage_two: stage_results.stage_2.outputs[i].clone(),
                output_stage_three: stage_results.stage_3.outputs[i].clone(),
                usage_stage_one: stage_results.stage_1.usage_stats.as_ref().map(|u| u[i].clone()),
                usage_stage_two: stage_results.stage_2.usage_stats.as_ref().map(|u| u[i].clone()),
                usage_stage_three: stage_results.stage_3.usage_stats.as_ref().map(|u| u[i].clone()),
            };

            // Update statistics
            self.stats.total_entities_extracted += result.entity_relation_dict.len();
            self.stats.total_events_extracted += result.event_entity_relation_dict.len();
            self.stats.total_relations_extracted += result.event_relation_dict.len();

            results.push(result);
        }

        Ok(results)
    }

    /// Create output filename with timestamp and shard information
    fn create_output_filename(&self) -> PathBuf {
        let timestamp = Utc::now().format("%Y%m%d%H%M%S");
        let model_name_safe = self.config.model_path.replace('/', "_");

        let filename = format!(
            "{}_{}_{}_{}_{}_in_{}.jsonl",
            model_name_safe,
            self.config.filename_pattern,
            "output",
            timestamp,
            self.config.current_shard_triple + 1,
            self.config.total_shards_triple
        );

        let extraction_dir = self.config.output_directory.join("kg_extraction");
        extraction_dir.join(filename)
    }

    /// Debug print result for debugging mode
    fn debug_print_result(&self, result: &ProcessingResult) {
        if self.config.debug_mode {
            log::debug!("Processing Result:");
            log::debug!("ID: {}", result.id);
            log::debug!("Text: {}", result.original_text);
            log::debug!("Entity Relations: {} extracted", result.entity_relation_dict.len());
            log::debug!("Event Entities: {} extracted", result.event_entity_relation_dict.len());
            log::debug!("Event Relations: {} extracted", result.event_relation_dict.len());
            log::debug!("{}", "-".repeat(80));
        }
    }

    /// Run the complete knowledge graph extraction pipeline
    pub async fn run_extraction(&mut self) -> Result<String> {
        self.stats.start_time = Utc::now();
        log::info!("Starting knowledge graph extraction pipeline");

        // Create output directory
        let extraction_dir = self.config.output_directory.join("kg_extraction");
        tokio::fs::create_dir_all(&extraction_dir).await.map_err(|e| {
            KgConstructionError::IoError(format!("Failed to create output directory: {}", e))
        })?;

        // Load dataset
        let dataset = self.load_dataset().await?;
        log::info!("Loaded {} samples from dataset", dataset.len());

        if self.config.debug_mode {
            log::info!("Debug mode: Processing limited samples");
        }

        // Create data processor and loader
        let processed_data = self.dataset_processor.prepare_dataset(dataset).await?;
        log::info!("Prepared {} chunks for processing", processed_data.len());

        let loader_config = LoaderConfig {
            batch_size: self.config.batch_size_triple,
            resume_from_batch: self.config.resume_from,
            max_concurrent_batches: self.config.max_workers,
            show_progress: !self.config.debug_mode,
            ..Default::default()
        };

        let data_loader = CustomDataLoader::new(processed_data, loader_config);

        // Create output file
        let output_file_path = self.create_output_filename();
        log::info!("Writing results to: {:?}", output_file_path);

        let output_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&output_file_path)
            .await
            .map_err(|e| KgConstructionError::IoError(format!("Failed to create output file: {}", e)))?;

        let mut writer = BufWriter::new(output_file);

        // Process batches
        let mut batch_counter = 0;
        let mut stream = data_loader.into_stream();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            batch_counter += 1;

            log::debug!("Processing batch {}", batch_counter);

            // Process all three stages
            let stage_results = self.process_all_stages(&batch).await?;

            // Parse results
            let parsed_results = self.parse_stage_results(&stage_results)?;

            // Prepare final results
            let processing_results = self.prepare_processing_results(&batch, &stage_results, parsed_results)?;

            // Write results to file
            for result in &processing_results {
                self.debug_print_result(result);

                let json_line = serde_json::to_string(result).map_err(|e| {
                    KgConstructionError::SerializationError(format!("Failed to serialize result: {}", e))
                })?;

                writer.write_all(json_line.as_bytes()).await.map_err(|e| {
                    KgConstructionError::IoError(format!("Failed to write to output file: {}", e))
                })?;
                writer.write_all(b"\n").await.map_err(|e| {
                    KgConstructionError::IoError(format!("Failed to write newline: {}", e))
                })?;
            }

            writer.flush().await.map_err(|e| {
                KgConstructionError::IoError(format!("Failed to flush output file: {}", e))
            })?;

            self.stats.total_batches_processed += 1;
            self.stats.total_chunks_processed += batch.batch_ids.len();

            log::info!(
                "Processed {} batches ({} chunks total)",
                batch_counter,
                self.stats.total_chunks_processed
            );
        }

        // Finalize statistics
        self.stats.end_time = Some(Utc::now());
        self.stats.processing_time_seconds = self.stats.end_time.unwrap()
            .signed_duration_since(self.stats.start_time)
            .num_milliseconds() as f64 / 1000.0;

        if self.stats.total_chunks_processed > 0 {
            self.stats.average_chunk_time_ms =
                (self.stats.processing_time_seconds * 1000.0) / self.stats.total_chunks_processed as f64;
        }

        log::info!("Extraction completed successfully");
        log::info!("Total processing time: {:.2} seconds", self.stats.processing_time_seconds);
        log::info!("Average time per chunk: {:.2} ms", self.stats.average_chunk_time_ms);
        log::info!("Extracted {} entities, {} events, {} relations",
            self.stats.total_entities_extracted,
            self.stats.total_events_extracted,
            self.stats.total_relations_extracted
        );

        Ok(output_file_path.to_string_lossy().to_string())
    }

    /// Get extraction statistics
    pub fn get_stats(&self) -> &ExtractionStats {
        &self.stats
    }

    /// Get parser statistics
    pub fn get_parser_stats(&self) -> &crate::parser::ParsingStats {
        self.output_parser.stats()
    }

    /// Get model backend information
    pub fn get_backend_info(&self) -> crate::ml_inference::BackendInfo {
        self.model_inference.backend_info()
    }

    /// Reset all statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExtractionStats {
            total_chunks_processed: 0,
            total_batches_processed: 0,
            total_entities_extracted: 0,
            total_events_extracted: 0,
            total_relations_extracted: 0,
            processing_time_seconds: 0.0,
            average_chunk_time_ms: 0.0,
            start_time: Utc::now(),
            end_time: None,
        };
        self.output_parser.reset_stats();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{InferenceBackend, Device};
    use std::collections::HashMap;

    async fn create_test_extractor() -> KnowledgeGraphExtractor {
        let config = ProcessingConfig {
            model_path: "test-model".to_string(),
            data_directory: std::env::temp_dir().join("test_data"),
            filename_pattern: "test".to_string(),
            output_directory: std::env::temp_dir().join("test_output"),
            inference_backend: InferenceBackend::Candle,
            device: Device::Cpu,
            debug_mode: true,
            max_new_tokens: 100,
            batch_size_triple: 2,
            ..Default::default()
        };

        KnowledgeGraphExtractor::new(config).await.unwrap()
    }

    #[tokio::test]
    async fn test_extractor_creation() {
        let extractor = create_test_extractor().await;
        assert_eq!(extractor.config.batch_size_triple, 2);
        assert!(extractor.config.debug_mode);
    }

    #[tokio::test]
    async fn test_output_filename_generation() {
        let extractor = create_test_extractor().await;
        let filename = extractor.create_output_filename();

        let filename_str = filename.to_string_lossy();
        assert!(filename_str.contains("test-model"));
        assert!(filename_str.contains("test"));
        assert!(filename_str.contains("1_in_1")); // shard info
        assert!(filename_str.ends_with(".jsonl"));
    }

    #[test]
    fn test_statistics_initialization() {
        let stats = ExtractionStats {
            total_chunks_processed: 0,
            total_batches_processed: 0,
            total_entities_extracted: 0,
            total_events_extracted: 0,
            total_relations_extracted: 0,
            processing_time_seconds: 0.0,
            average_chunk_time_ms: 0.0,
            start_time: Utc::now(),
            end_time: None,
        };

        assert_eq!(stats.total_chunks_processed, 0);
        assert_eq!(stats.processing_time_seconds, 0.0);
        assert!(stats.end_time.is_none());
    }
}