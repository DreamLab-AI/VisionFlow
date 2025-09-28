//! Main concept generation module - Rust implementation of Python concept_generation.py

use crate::{
    batch_processing::{build_comprehensive_batched_data, create_all_batches, validate_batch_data},
    csv_utils::{write_concept_nodes_csv, clean_text_for_csv, CsvConfig},
    data_loading::{load_node_data_with_types, StreamingDataLoader},
    error::{KgConstructionError, Result},
    graph_traversal::{ContextExtractor, Graph, compute_hash_id},
    llm_integration::{batched_inference, LlmGenerator, clean_text, remove_nul},
    statistics::{ProcessingLogger, LogLevel, print_statistics_table, print_performance_summary},
    types::{
        BatchResult, ConceptNode, NodeType, ProcessingConfig, ShardConfig, Statistics,
        TokenUsage, GraphTraversal,
    },
};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Main concept generator implementation
pub struct ConceptGenerator {
    config: ProcessingConfig,
    llm: Arc<dyn LlmGenerator>,
    logger: Option<ProcessingLogger>,
    graph: Option<Graph>,
    context_extractor: Option<ContextExtractor>,
}

impl ConceptGenerator {
    pub fn new(
        config: ProcessingConfig,
        llm: Arc<dyn LlmGenerator>,
        logger: Option<ProcessingLogger>,
    ) -> Self {
        Self {
            config,
            llm,
            logger,
            graph: None,
            context_extractor: None,
        }
    }

    /// Load graph for context extraction (equivalent to loading pickle file)
    pub fn load_graph(&mut self, graph: Graph) -> Result<()> {
        let traversal_config = GraphTraversal::default();
        let context_extractor = ContextExtractor::new(graph.clone(), traversal_config);

        self.graph = Some(graph);
        self.context_extractor = Some(context_extractor);
        Ok(())
    }

    /// Generate concepts for nodes (main function equivalent to Python generate_concept)
    pub async fn generate_concept(
        &mut self,
        input_file: impl AsRef<Path>,
        output_folder: impl AsRef<Path>,
        output_filename: &str,
        shard_config: Option<ShardConfig>,
        record_usage: bool,
    ) -> Result<Statistics> {
        let start_time = Instant::now();

        // Ensure output directory exists
        if !output_folder.as_ref().exists() {
            std::fs::create_dir_all(&output_folder)?;
        }

        // Load data with sharding
        let all_missing_nodes = load_node_data_with_types(&input_file, shard_config)?;

        if let Some(ref mut logger) = self.logger {
            logger.log(
                LogLevel::Info,
                &format!("Loaded {} nodes for processing", all_missing_nodes.len()),
            )?;
        }

        // Validate data
        validate_batch_data(&all_missing_nodes)?;

        // Build batched data
        let batched_data = build_comprehensive_batched_data(&all_missing_nodes, self.config.batch_size)?;
        let all_batches = create_all_batches(&batched_data);

        if let Some(ref mut logger) = self.logger {
            logger.log(
                LogLevel::Info,
                &format!("Created {} batches for processing", all_batches.len()),
            )?;
        }

        // Prepare output file
        let shard_suffix = shard_config
            .map(|sc| format!("_shard_{}", sc.shard_idx))
            .unwrap_or_default();
        let output_file = output_folder.as_ref().join(format!(
            "{}{}.csv",
            output_filename.trim_end_matches(".csv").trim_end_matches(".json"),
            shard_suffix
        ));

        // Process batches and write to CSV
        let mut all_concept_nodes = Vec::new();
        let mut total_stats = Statistics::default();

        for (batch_type, batch) in all_batches {
            let batch_result = self.process_batch(batch_type, batch, record_usage).await?;

            all_concept_nodes.extend(batch_result.generated_concepts);
            total_stats.total_batches_processed += 1;
            total_stats.total_nodes_processed += batch_result.processed_nodes;

            match batch_type {
                NodeType::Entity => total_stats.entities_processed += batch_result.processed_nodes,
                NodeType::Event => total_stats.events_processed += batch_result.processed_nodes,
                NodeType::Relation => total_stats.relations_processed += batch_result.processed_nodes,
            }

            if !batch_result.errors.is_empty() {
                total_stats.errors_encountered += batch_result.errors.len();
                if let Some(ref mut logger) = self.logger {
                    for error in &batch_result.errors {
                        logger.log(LogLevel::Error, error)?;
                    }
                }
            }
        }

        // Write results to CSV
        write_concept_nodes_csv(&output_file, &all_concept_nodes, Some(CsvConfig::default()))?;

        // Calculate statistics from generated concepts
        let (unique_concepts, concept_stats) = self.calculate_concept_statistics(&all_concept_nodes);
        total_stats.unique_concepts_generated = unique_concepts.len();
        total_stats.concepts_by_type = concept_stats;
        total_stats.processing_time_ms = start_time.elapsed().as_millis();

        // Log final summary
        if let Some(ref mut logger) = self.logger {
            logger.log_final_summary()?;
        }

        // Print statistics
        self.print_final_statistics(&total_stats, &unique_concepts);

        Ok(total_stats)
    }

    /// Process a single batch of nodes
    async fn process_batch(
        &mut self,
        batch_type: NodeType,
        batch: Vec<String>,
        record_usage: bool,
    ) -> Result<BatchResult> {
        let start_time = Instant::now();
        let mut errors = Vec::new();

        if let Some(ref mut logger) = self.logger {
            logger.log_batch_start(batch_type, batch.len())?;
        }

        // Get concept template based on batch type
        let (template, replace_token, replace_context_token) = self.get_concept_template(batch_type);

        // Prepare inputs for LLM
        let mut inputs = Vec::new();
        for node in &batch {
            match self.prepare_node_input(node, &template, &replace_token, replace_context_token.as_deref(), batch_type) {
                Ok(input) => inputs.push(input),
                Err(e) => {
                    errors.push(format!("Failed to prepare input for node '{}': {}", node, e));
                    continue;
                }
            }
        }

        if inputs.is_empty() {
            return Ok(BatchResult {
                batch_type,
                processed_nodes: 0,
                generated_concepts: Vec::new(),
                processing_time_ms: start_time.elapsed().as_millis(),
                errors,
            });
        }

        // Perform batched inference
        let (answers, usages) = match batched_inference(
            self.llm.as_ref(),
            inputs,
            record_usage,
            Some(self.config.max_workers),
        ).await {
            Ok((answers, usages)) => (answers, usages),
            Err(e) => {
                errors.push(format!("LLM inference failed: {}", e));
                return Ok(BatchResult {
                    batch_type,
                    processed_nodes: 0,
                    generated_concepts: Vec::new(),
                    processing_time_ms: start_time.elapsed().as_millis(),
                    errors,
                });
            }
        };

        // Process results
        let mut generated_concepts = Vec::new();
        for (i, (node, answer)) in batch.iter().zip(answers.iter()).enumerate() {
            // Log token usage if available
            if let Some(ref usage_vec) = usages {
                if let Some(usage) = usage_vec.get(i) {
                    if let Some(ref mut logger) = self.logger {
                        logger.log_token_usage(node, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)?;
                    }
                }
            }

            // Filter and clean concepts
            let cleaned_concepts: Vec<String> = answer
                .iter()
                .map(|c| clean_text_for_csv(c))
                .filter(|c| !c.trim().is_empty())
                .collect();

            if !cleaned_concepts.is_empty() {
                generated_concepts.push(ConceptNode {
                    node: node.clone(),
                    conceptualized_node: cleaned_concepts,
                    node_type: batch_type,
                });
            }
        }

        let processing_time = start_time.elapsed().as_millis();

        if let Some(ref mut logger) = self.logger {
            logger.log_batch_complete(
                batch_type,
                batch.len(),
                generated_concepts.len(),
                processing_time,
            )?;
        }

        Ok(BatchResult {
            batch_type,
            processed_nodes: batch.len(),
            generated_concepts,
            processing_time_ms: processing_time,
            errors,
        })
    }

    /// Get concept template for a specific node type
    fn get_concept_template(&self, node_type: NodeType) -> (String, String, Option<String>) {
        // These templates should match the CONCEPT_INSTRUCTIONS from the Python code
        match node_type {
            NodeType::Event => (
                "Generate 3-5 high-level concept categories that the event '[EVENT]' belongs to. Provide only the concept names separated by commas.".to_string(),
                "[EVENT]".to_string(),
                None,
            ),
            NodeType::Entity => (
                "Generate 3-5 high-level concept categories that the entity '[ENTITY]' belongs to, considering its context: [CONTEXT]. Provide only the concept names separated by commas.".to_string(),
                "[ENTITY]".to_string(),
                Some("[CONTEXT]".to_string()),
            ),
            NodeType::Relation => (
                "Generate 3-5 high-level concept categories that the relation '[RELATION]' belongs to. Provide only the concept names separated by commas.".to_string(),
                "[RELATION]".to_string(),
                None,
            ),
        }
    }

    /// Prepare input for a single node
    fn prepare_node_input(
        &self,
        node: &str,
        template: &str,
        replace_token: &str,
        replace_context_token: Option<&str>,
        node_type: NodeType,
    ) -> Result<Vec<HashMap<String, String>>> {
        let cleaned_node = remove_nul(&clean_text(node));
        let mut prompt = template.replace(replace_token, &cleaned_node);

        // Add context if needed and available
        if let Some(context_token) = replace_context_token {
            let context = if let Some(ref extractor) = self.context_extractor {
                extractor.extract_typed_context(&cleaned_node, &node_type.to_string())
                    .unwrap_or_default()
            } else {
                String::new()
            };
            prompt = prompt.replace(context_token, &context);
        }

        let input = vec![
            [("role".to_string(), "system".to_string()),
             ("content".to_string(), "You are a helpful AI assistant.".to_string())]
            .iter().cloned().collect(),
            [("role".to_string(), "user".to_string()),
             ("content".to_string(), prompt)]
            .iter().cloned().collect(),
        ];

        Ok(input)
    }

    /// Calculate concept statistics from generated concepts
    fn calculate_concept_statistics(&self, concept_nodes: &[ConceptNode]) -> (HashSet<String>, HashMap<String, usize>) {
        let mut all_concepts = HashSet::new();
        let mut concept_stats = HashMap::new();

        for concept_node in concept_nodes {
            let node_type_str = concept_node.node_type.to_string();

            for concept in &concept_node.conceptualized_node {
                let cleaned_concept = concept.trim().to_lowercase();
                if !cleaned_concept.is_empty() {
                    all_concepts.insert(cleaned_concept.clone());
                    *concept_stats.entry(node_type_str.clone()).or_insert(0) += 1;
                }
            }
        }

        (all_concepts, concept_stats)
    }

    /// Print final statistics (equivalent to Python print statements)
    fn print_final_statistics(&self, stats: &Statistics, unique_concepts: &HashSet<String>) {
        println!("Number of unique conceptualized nodes: {}", unique_concepts.len());

        let unique_events: HashSet<String> = unique_concepts
            .iter()
            .filter(|c| stats.concepts_by_type.get("event").unwrap_or(&0) > &0)
            .cloned()
            .collect();
        println!("Number of unique conceptualized events: {}", unique_events.len());

        let unique_entities: HashSet<String> = unique_concepts
            .iter()
            .filter(|c| stats.concepts_by_type.get("entity").unwrap_or(&0) > &0)
            .cloned()
            .collect();
        println!("Number of unique conceptualized entities: {}", unique_entities.len());

        let unique_relations: HashSet<String> = unique_concepts
            .iter()
            .filter(|c| stats.concepts_by_type.get("relation").unwrap_or(&0) > &0)
            .cloned()
            .collect();
        println!("Number of unique conceptualized relations: {}", unique_relations.len());

        print_statistics_table(stats);
        print_performance_summary(stats);
    }
}

/// Streaming concept generator for large datasets
pub struct StreamingConceptGenerator {
    generator: ConceptGenerator,
    stream_loader: StreamingDataLoader,
}

impl StreamingConceptGenerator {
    pub fn new(
        config: ProcessingConfig,
        llm: Arc<dyn LlmGenerator>,
        input_file: impl AsRef<Path>,
        shard_config: Option<ShardConfig>,
    ) -> Result<Self> {
        let logger = ProcessingLogger::new("concept_generation.log").ok();
        let generator = ConceptGenerator::new(config, llm, logger);
        let stream_loader = StreamingDataLoader::new(input_file, shard_config)?;

        Ok(Self {
            generator,
            stream_loader,
        })
    }

    /// Process data in streaming fashion
    pub async fn process_streaming(
        &mut self,
        output_folder: impl AsRef<Path>,
        output_filename: &str,
        record_usage: bool,
    ) -> Result<Statistics> {
        let start_time = Instant::now();
        let mut total_stats = Statistics::default();
        let mut all_concept_nodes = Vec::new();

        // Ensure output directory exists
        if !output_folder.as_ref().exists() {
            std::fs::create_dir_all(&output_folder)?;
        }

        while self.stream_loader.has_more() {
            let batch_data = self.stream_loader.next_batch(self.generator.config.batch_size)?;
            if batch_data.is_empty() {
                break;
            }

            // Convert to node list format
            let node_list: Vec<(String, String)> = batch_data
                .into_iter()
                .filter(|row| row.len() >= 2)
                .map(|row| (row[0].clone(), row[1].clone()))
                .collect();

            if node_list.is_empty() {
                continue;
            }

            // Validate and process batch
            validate_batch_data(&node_list)?;
            let batched_data = build_comprehensive_batched_data(&node_list, self.generator.config.batch_size)?;
            let all_batches = create_all_batches(&batched_data);

            for (batch_type, batch) in all_batches {
                let batch_result = self.generator.process_batch(batch_type, batch, record_usage).await?;

                all_concept_nodes.extend(batch_result.generated_concepts);
                total_stats.total_batches_processed += 1;
                total_stats.total_nodes_processed += batch_result.processed_nodes;

                match batch_type {
                    NodeType::Entity => total_stats.entities_processed += batch_result.processed_nodes,
                    NodeType::Event => total_stats.events_processed += batch_result.processed_nodes,
                    NodeType::Relation => total_stats.relations_processed += batch_result.processed_nodes,
                }

                total_stats.errors_encountered += batch_result.errors.len();
            }
        }

        // Write results
        let output_file = output_folder.as_ref().join(format!("{}.csv", output_filename));
        write_concept_nodes_csv(&output_file, &all_concept_nodes, Some(CsvConfig::default()))?;

        // Calculate final statistics
        let (unique_concepts, concept_stats) = self.generator.calculate_concept_statistics(&all_concept_nodes);
        total_stats.unique_concepts_generated = unique_concepts.len();
        total_stats.concepts_by_type = concept_stats;
        total_stats.processing_time_ms = start_time.elapsed().as_millis();

        self.generator.print_final_statistics(&total_stats, &unique_concepts);

        Ok(total_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_integration::MockLlmGenerator;
    use tempfile::TempDir;
    use std::io::Write;

    #[tokio::test]
    async fn test_concept_generator() {
        let temp_dir = TempDir::new().unwrap();

        // Create test input file
        let input_file = temp_dir.path().join("input.csv");
        let mut file = std::fs::File::create(&input_file).unwrap();
        writeln!(file, "node,type").unwrap();
        writeln!(file, "test_entity,entity").unwrap();
        writeln!(file, "test_event,event").unwrap();
        writeln!(file, "test_relation,relation").unwrap();

        let config = ProcessingConfig {
            batch_size: 2,
            max_workers: 1,
            ..Default::default()
        };

        let llm = Arc::new(MockLlmGenerator::new(10));
        let mut generator = ConceptGenerator::new(config, llm, None);

        let stats = generator.generate_concept(
            &input_file,
            temp_dir.path(),
            "output.csv",
            None,
            false,
        ).await.unwrap();

        assert!(stats.total_nodes_processed > 0);
        assert!(stats.total_batches_processed > 0);

        // Check output file exists
        let output_file = temp_dir.path().join("output.csv");
        assert!(output_file.exists());
    }

    #[test]
    fn test_get_concept_template() {
        let config = ProcessingConfig::default();
        let llm = Arc::new(MockLlmGenerator::new(0));
        let generator = ConceptGenerator::new(config, llm, None);

        let (template, token, context) = generator.get_concept_template(NodeType::Entity);
        assert!(template.contains("[ENTITY]"));
        assert_eq!(token, "[ENTITY]");
        assert!(context.is_some());

        let (template, token, context) = generator.get_concept_template(NodeType::Event);
        assert!(template.contains("[EVENT]"));
        assert_eq!(token, "[EVENT]");
        assert!(context.is_none());
    }
}