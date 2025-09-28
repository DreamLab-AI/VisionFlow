//! Custom data loader for knowledge graph extraction with async batch processing

use crate::dataset::{DatasetProcessor, ProcessedChunk};
use crate::error::{KgConstructionError, Result};
use crate::prompts::{ProcessingStage, TripleInstructions};
use futures::stream::{self, Stream, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use tokio::sync::mpsc;

/// Batch of data for processing
#[derive(Debug, Clone)]
pub struct DataBatch {
    pub instructions: StageInstructions,
    pub batch_ids: Vec<String>,
    pub batch_texts: Vec<String>,
    pub batch_metadata: Vec<HashMap<String, serde_json::Value>>,
}

/// Instructions for all three processing stages
#[derive(Debug, Clone)]
pub struct StageInstructions {
    pub stage_1: Vec<Vec<MessagePair>>,
    pub stage_2: Vec<Vec<MessagePair>>,
    pub stage_3: Vec<Vec<MessagePair>>,
}

/// A message pair for chat-based models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePair {
    pub role: String,
    pub content: String,
}

/// Configuration for the data loader
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    pub batch_size: usize,
    pub resume_from_batch: usize,
    pub max_concurrent_batches: usize,
    pub show_progress: bool,
    pub buffer_size: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            resume_from_batch: 0,
            max_concurrent_batches: 4,
            show_progress: true,
            buffer_size: 100,
        }
    }
}

/// Custom data loader for knowledge graph extraction
pub struct CustomDataLoader {
    processed_data: Vec<ProcessedChunk>,
    triple_instructions: TripleInstructions,
    config: LoaderConfig,
    progress_bar: Option<ProgressBar>,
}

impl CustomDataLoader {
    /// Create a new data loader
    pub fn new(
        processed_data: Vec<ProcessedChunk>,
        config: LoaderConfig,
    ) -> Self {
        let progress_bar = if config.show_progress {
            let pb = ProgressBar::new(processed_data.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} batches ({eta})")
                    .expect("Invalid progress bar template")
                    .progress_chars("#>-")
            );
            Some(pb)
        } else {
            None
        };

        Self {
            processed_data,
            triple_instructions: TripleInstructions::new(),
            config,
            progress_bar,
        }
    }

    /// Create loader from dataset processor
    pub async fn from_processor(
        processor: &DatasetProcessor,
        raw_dataset: Vec<crate::dataset::DataSample>,
        config: LoaderConfig,
    ) -> Result<Self> {
        let processed_data = processor.prepare_dataset(raw_dataset).await?;
        Ok(Self::new(processed_data, config))
    }

    /// Get the total number of items
    pub fn len(&self) -> usize {
        self.processed_data.len()
    }

    /// Check if the loader is empty
    pub fn is_empty(&self) -> bool {
        self.processed_data.is_empty()
    }

    /// Get the number of batches
    pub fn num_batches(&self) -> usize {
        (self.processed_data.len() + self.config.batch_size - 1) / self.config.batch_size
    }

    /// Create batch instructions for a set of processed chunks
    fn create_batch_instructions(&self, batch_data: &[ProcessedChunk]) -> StageInstructions {
        let mut stage_1 = Vec::with_capacity(batch_data.len());
        let mut stage_2 = Vec::with_capacity(batch_data.len());
        let mut stage_3 = Vec::with_capacity(batch_data.len());

        for item in batch_data {
            // Get language from metadata
            let language = item.metadata
                .get("lang")
                .and_then(|v| v.as_str())
                .unwrap_or("en");

            let instructions = self.triple_instructions.get_instructions_or_default(language);

            // Create messages for each stage
            let system_message = MessagePair {
                role: "system".to_string(),
                content: instructions.system.clone(),
            };

            // Stage 1: Entity-Relation
            let stage_1_user = MessagePair {
                role: "user".to_string(),
                content: format!(
                    "{}{}\n{}",
                    instructions.entity_relation,
                    instructions.passage_start,
                    item.text
                ),
            };
            stage_1.push(vec![system_message.clone(), stage_1_user]);

            // Stage 2: Event-Entity
            let stage_2_user = MessagePair {
                role: "user".to_string(),
                content: format!(
                    "{}{}\n{}",
                    instructions.event_entity,
                    instructions.passage_start,
                    item.text
                ),
            };
            stage_2.push(vec![system_message.clone(), stage_2_user]);

            // Stage 3: Event-Relation
            let stage_3_user = MessagePair {
                role: "user".to_string(),
                content: format!(
                    "{}{}\n{}",
                    instructions.event_relation,
                    instructions.passage_start,
                    item.text
                ),
            };
            stage_3.push(vec![system_message, stage_3_user]);
        }

        StageInstructions {
            stage_1,
            stage_2,
            stage_3,
        }
    }

    /// Create an async stream of batches
    pub fn into_stream(self) -> Pin<Box<dyn Stream<Item = Result<DataBatch>> + Send>> {
        let batch_size = self.config.batch_size;
        let start_idx = self.config.resume_from_batch * batch_size;
        let data = self.processed_data;
        let instructions = self.triple_instructions;
        let progress_bar = self.progress_bar;

        let stream = stream::iter((start_idx..data.len()).step_by(batch_size))
            .map(move |i| {
                let batch_data: Vec<_> = data
                    .iter()
                    .skip(i)
                    .take(batch_size)
                    .cloned()
                    .collect();

                if batch_data.is_empty() {
                    return Err(KgConstructionError::LoaderError(
                        "Empty batch encountered".to_string()
                    ));
                }

                // Create instructions
                let batch_instructions = Self::create_batch_instructions_static(&instructions, &batch_data);

                // Extract batch information
                let batch_ids: Vec<String> = batch_data.iter().map(|item| item.id.clone()).collect();
                let batch_texts: Vec<String> = batch_data.iter().map(|item| item.text.clone()).collect();
                let batch_metadata: Vec<_> = batch_data.iter().map(|item| item.metadata.clone()).collect();

                // Update progress bar
                if let Some(ref pb) = progress_bar {
                    pb.inc(batch_data.len() as u64);
                }

                Ok(DataBatch {
                    instructions: batch_instructions,
                    batch_ids,
                    batch_texts,
                    batch_metadata,
                })
            });

        Box::pin(stream)
    }

    /// Static version of create_batch_instructions for use in closures
    fn create_batch_instructions_static(
        instructions: &TripleInstructions,
        batch_data: &[ProcessedChunk],
    ) -> StageInstructions {
        let mut stage_1 = Vec::with_capacity(batch_data.len());
        let mut stage_2 = Vec::with_capacity(batch_data.len());
        let mut stage_3 = Vec::with_capacity(batch_data.len());

        for item in batch_data {
            let language = item.metadata
                .get("lang")
                .and_then(|v| v.as_str())
                .unwrap_or("en");

            let lang_instructions = instructions.get_instructions_or_default(language);

            let system_message = MessagePair {
                role: "system".to_string(),
                content: lang_instructions.system.clone(),
            };

            // Stage 1: Entity-Relation
            let stage_1_user = MessagePair {
                role: "user".to_string(),
                content: format!(
                    "{}{}\n{}",
                    lang_instructions.entity_relation,
                    lang_instructions.passage_start,
                    item.text
                ),
            };
            stage_1.push(vec![system_message.clone(), stage_1_user]);

            // Stage 2: Event-Entity
            let stage_2_user = MessagePair {
                role: "user".to_string(),
                content: format!(
                    "{}{}\n{}",
                    lang_instructions.event_entity,
                    lang_instructions.passage_start,
                    item.text
                ),
            };
            stage_2.push(vec![system_message.clone(), stage_2_user]);

            // Stage 3: Event-Relation
            let stage_3_user = MessagePair {
                role: "user".to_string(),
                content: format!(
                    "{}{}\n{}",
                    lang_instructions.event_relation,
                    lang_instructions.passage_start,
                    item.text
                ),
            };
            stage_3.push(vec![system_message, stage_3_user]);
        }

        StageInstructions {
            stage_1,
            stage_2,
            stage_3,
        }
    }

    /// Create a buffered loader for high-throughput processing
    pub fn into_buffered_stream(
        self,
        buffer_size: usize,
    ) -> Pin<Box<dyn Stream<Item = Result<Vec<DataBatch>>> + Send>> {
        let stream = self.into_stream();

        let buffered = stream
            .chunks(buffer_size)
            .map(|chunk| {
                let mut batches = Vec::with_capacity(chunk.len());
                for result in chunk {
                    batches.push(result?);
                }
                Ok(batches)
            });

        Box::pin(buffered)
    }

    /// Process all batches with a custom handler
    pub async fn process_all<F, Fut, T>(
        self,
        mut handler: F,
    ) -> Result<Vec<T>>
    where
        F: FnMut(DataBatch) -> Fut + Send,
        Fut: std::future::Future<Output = Result<T>> + Send,
        T: Send,
    {
        let mut results = Vec::new();
        let mut stream = self.into_stream();

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let result = handler(batch).await?;
            results.push(result);
        }

        Ok(results)
    }

    /// Process batches concurrently with controlled parallelism
    pub async fn process_concurrent<F, Fut, T>(
        self,
        handler: F,
        concurrency: usize,
    ) -> Result<Vec<T>>
    where
        F: Fn(DataBatch) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        let stream = self.into_stream();
        let results: Vec<Result<T>> = stream
            .map(|batch_result| {
                let handler = handler.clone();
                async move {
                    let batch = batch_result?;
                    handler(batch).await
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        // Collect all results, returning error if any failed
        let mut final_results = Vec::with_capacity(results.len());
        for result in results {
            final_results.push(result?);
        }

        Ok(final_results)
    }

    /// Create a channel-based producer-consumer pattern
    pub async fn spawn_producer(
        self,
        buffer_size: usize,
    ) -> (
        mpsc::Receiver<Result<DataBatch>>,
        tokio::task::JoinHandle<Result<()>>,
    ) {
        let (tx, rx) = mpsc::channel(buffer_size);

        let handle = tokio::spawn(async move {
            let mut stream = self.into_stream();

            while let Some(batch_result) = stream.next().await {
                if tx.send(batch_result).await.is_err() {
                    // Receiver dropped
                    break;
                }
            }

            Ok(())
        });

        (rx, handle)
    }

    /// Get loader statistics
    pub fn stats(&self) -> LoaderStats {
        LoaderStats {
            total_items: self.processed_data.len(),
            batch_size: self.config.batch_size,
            total_batches: self.num_batches(),
            resume_from_batch: self.config.resume_from_batch,
            estimated_remaining_items: self.processed_data.len()
                .saturating_sub(self.config.resume_from_batch * self.config.batch_size),
        }
    }

    /// Finish and clean up progress bar
    pub fn finish(&self) {
        if let Some(ref pb) = self.progress_bar {
            pb.finish_with_message("Processing complete");
        }
    }
}

/// Statistics about the data loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderStats {
    pub total_items: usize,
    pub batch_size: usize,
    pub total_batches: usize,
    pub resume_from_batch: usize,
    pub estimated_remaining_items: usize,
}

/// Iterator implementation for backwards compatibility
pub struct DataLoaderIterator {
    loader: CustomDataLoader,
    current_batch: usize,
}

impl DataLoaderIterator {
    pub fn new(loader: CustomDataLoader) -> Self {
        Self {
            loader,
            current_batch: 0,
        }
    }
}

impl Iterator for DataLoaderIterator {
    type Item = Result<DataBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch_size = self.loader.config.batch_size;
        let start_idx = (self.loader.config.resume_from_batch + self.current_batch) * batch_size;

        if start_idx >= self.loader.processed_data.len() {
            return None;
        }

        let batch_data: Vec<_> = self.loader.processed_data
            .iter()
            .skip(start_idx)
            .take(batch_size)
            .cloned()
            .collect();

        if batch_data.is_empty() {
            return None;
        }

        self.current_batch += 1;

        // Create instructions
        let instructions = self.loader.create_batch_instructions(&batch_data);

        // Extract batch information
        let batch_ids = batch_data.iter().map(|item| item.id.clone()).collect();
        let batch_texts = batch_data.iter().map(|item| item.text.clone()).collect();
        let batch_metadata = batch_data.iter().map(|item| item.metadata.clone()).collect();

        Some(Ok(DataBatch {
            instructions,
            batch_ids,
            batch_texts,
            batch_metadata,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DataSample, ProcessedChunk};
    use std::collections::HashMap;

    fn create_test_chunks() -> Vec<ProcessedChunk> {
        let mut metadata = HashMap::new();
        metadata.insert("lang".to_string(), serde_json::Value::String("en".to_string()));

        vec![
            ProcessedChunk {
                id: "sample1".to_string(),
                text: "This is the first test chunk.".to_string(),
                chunk_id: 0,
                metadata: metadata.clone(),
            },
            ProcessedChunk {
                id: "sample2".to_string(),
                text: "This is the second test chunk.".to_string(),
                chunk_id: 0,
                metadata: metadata.clone(),
            },
            ProcessedChunk {
                id: "sample3".to_string(),
                text: "This is the third test chunk.".to_string(),
                chunk_id: 0,
                metadata,
            },
        ]
    }

    #[test]
    fn test_loader_creation() {
        let chunks = create_test_chunks();
        let config = LoaderConfig {
            batch_size: 2,
            show_progress: false,
            ..Default::default()
        };

        let loader = CustomDataLoader::new(chunks, config);
        assert_eq!(loader.len(), 3);
        assert_eq!(loader.num_batches(), 2); // 3 items with batch size 2 = 2 batches
    }

    #[test]
    fn test_batch_instructions() {
        let chunks = create_test_chunks();
        let config = LoaderConfig {
            show_progress: false,
            ..Default::default()
        };

        let loader = CustomDataLoader::new(chunks, config);
        let batch_data = &loader.processed_data[0..2];
        let instructions = loader.create_batch_instructions(batch_data);

        assert_eq!(instructions.stage_1.len(), 2);
        assert_eq!(instructions.stage_2.len(), 2);
        assert_eq!(instructions.stage_3.len(), 2);

        // Check that each stage has system and user messages
        assert_eq!(instructions.stage_1[0].len(), 2);
        assert_eq!(instructions.stage_1[0][0].role, "system");
        assert_eq!(instructions.stage_1[0][1].role, "user");
    }

    #[tokio::test]
    async fn test_stream_processing() {
        let chunks = create_test_chunks();
        let config = LoaderConfig {
            batch_size: 2,
            show_progress: false,
            ..Default::default()
        };

        let loader = CustomDataLoader::new(chunks, config);
        let mut stream = loader.into_stream();

        let mut batch_count = 0;
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.unwrap();
            batch_count += 1;

            assert!(!batch.batch_ids.is_empty());
            assert_eq!(batch.batch_ids.len(), batch.batch_texts.len());
            assert_eq!(batch.batch_ids.len(), batch.batch_metadata.len());
        }

        assert_eq!(batch_count, 2); // Should have 2 batches
    }

    #[tokio::test]
    async fn test_concurrent_processing() {
        let chunks = create_test_chunks();
        let config = LoaderConfig {
            batch_size: 1,
            show_progress: false,
            ..Default::default()
        };

        let loader = CustomDataLoader::new(chunks, config);

        let results = loader.process_concurrent(
            |batch| async move {
                // Simulate some processing
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                Ok(batch.batch_ids.len())
            },
            2, // Process 2 batches concurrently
        ).await.unwrap();

        assert_eq!(results.len(), 3); // Should process all 3 chunks
        assert!(results.iter().all(|&count| count == 1)); // Each batch has 1 item
    }

    #[test]
    fn test_iterator_interface() {
        let chunks = create_test_chunks();
        let config = LoaderConfig {
            batch_size: 2,
            show_progress: false,
            ..Default::default()
        };

        let loader = CustomDataLoader::new(chunks, config);
        let mut iterator = DataLoaderIterator::new(loader);

        let first_batch = iterator.next().unwrap().unwrap();
        assert_eq!(first_batch.batch_ids.len(), 2);

        let second_batch = iterator.next().unwrap().unwrap();
        assert_eq!(second_batch.batch_ids.len(), 1); // Last batch has 1 item

        assert!(iterator.next().is_none()); // No more batches
    }
}