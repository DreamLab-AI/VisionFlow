//! Knowledge graph construction pipeline

use crate::extractors::{TripleExtractor, ExtractionConfig};
use crate::graph::{KnowledgeGraph, GraphBuilder};
use crate::triple::Triple;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use utils::{Result, UtilsError};

/// Configuration for the knowledge graph construction pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub max_triples_per_chunk: usize,
    pub confidence_threshold: f32,
    pub enable_entity_linking: bool,
    pub enable_concept_generation: bool,
    pub batch_size: usize,
    pub extraction_config: ExtractionConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_triples_per_chunk: 20,
            confidence_threshold: 0.7,
            enable_entity_linking: true,
            enable_concept_generation: true,
            batch_size: 16,
            extraction_config: ExtractionConfig::default(),
        }
    }
}

/// Main pipeline for knowledge graph construction
pub struct ConstructionPipeline {
    config: PipelineConfig,
    extractors: Vec<Arc<dyn TripleExtractor>>,
    graph_builder: GraphBuilder,
}

impl ConstructionPipeline {
    /// Create a new construction pipeline
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            extractors: Vec::new(),
            graph_builder: GraphBuilder::new(),
        }
    }

    /// Add an extractor to the pipeline
    pub fn add_extractor(&mut self, extractor: Arc<dyn TripleExtractor>) {
        self.extractors.push(extractor);
    }

    /// Process a single document and add to knowledge graph
    pub async fn process_document(
        &mut self,
        document_id: &str,
        content: &str,
    ) -> Result<KnowledgeGraph> {
        // Split content into chunks
        let chunks = self.split_into_chunks(content)?;

        let mut all_triples = Vec::new();

        // Process each chunk
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            log::debug!("Processing chunk {} for document {}", chunk_idx, document_id);

            // Extract triples using all extractors
            for extractor in &self.extractors {
                match extractor.extract_triples(chunk, &self.config.extraction_config).await {
                    Ok(result) => {
                        log::debug!(
                            "Extracted {} triples from chunk {} using {}",
                            result.triples.len(),
                            chunk_idx,
                            extractor.name()
                        );
                        all_triples.extend(result.triples);
                    }
                    Err(e) => {
                        log::error!(
                            "Failed to extract triples from chunk {} using {}: {}",
                            chunk_idx,
                            extractor.name(),
                            e
                        );
                    }
                }
            }
        }

        // Filter and deduplicate triples
        let filtered_triples = self.filter_and_deduplicate_triples(all_triples)?;

        // Build knowledge graph
        self.graph_builder.add_triples(filtered_triples)?;

        if self.config.enable_entity_linking {
            self.graph_builder.link_entities()?;
        }

        let graph = self.graph_builder.build()?;

        log::info!(
            "Processed document {} - {} entities, {} relations, {} triples",
            document_id,
            graph.entity_count(),
            graph.relation_count(),
            graph.triple_count()
        );

        Ok(graph)
    }

    /// Process multiple documents in batches
    pub async fn process_documents(
        &mut self,
        documents: Vec<(String, String)>, // (id, content) pairs
    ) -> Result<KnowledgeGraph> {
        let mut final_graph = KnowledgeGraph::new();

        // Process documents in batches
        for batch in documents.chunks(self.config.batch_size) {
            let mut batch_graphs = Vec::new();

            for (doc_id, content) in batch {
                match self.process_document(doc_id, content).await {
                    Ok(graph) => batch_graphs.push(graph),
                    Err(e) => {
                        log::error!("Failed to process document {}: {}", doc_id, e);
                    }
                }
            }

            // Merge batch graphs into final graph
            for graph in batch_graphs {
                final_graph.merge(graph)?;
            }
        }

        Ok(final_graph)
    }

    /// Split content into manageable chunks
    fn split_into_chunks(&self, content: &str) -> Result<Vec<String>> {
        // Simple sentence-based chunking
        // In practice, you might want more sophisticated chunking
        let sentences: Vec<&str> = content
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let max_chunk_size = 2000; // characters

        for sentence in sentences {
            let sentence = sentence.trim();
            if current_chunk.len() + sentence.len() > max_chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.trim().to_string());
                current_chunk.clear();
            }
            current_chunk.push_str(sentence);
            current_chunk.push('.');
        }

        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        Ok(chunks)
    }

    /// Filter and deduplicate extracted triples
    fn filter_and_deduplicate_triples(&self, triples: Vec<Triple>) -> Result<Vec<Triple>> {
        // Filter by confidence threshold
        let mut filtered: Vec<Triple> = triples
            .into_iter()
            .filter(|t| t.confidence >= self.config.confidence_threshold)
            .collect();

        // Deduplicate based on canonical string representation
        let mut seen = std::collections::HashSet::new();
        filtered.retain(|triple| {
            let canonical = triple.to_canonical_string();
            seen.insert(canonical)
        });

        // Sort by confidence (highest first)
        filtered.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Limit to max triples per chunk if configured
        if self.config.max_triples_per_chunk > 0 {
            filtered.truncate(self.config.max_triples_per_chunk);
        }

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractors::LlmTripleExtractor;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = ConstructionPipeline::new(config);
        assert_eq!(pipeline.extractors.len(), 0);
    }

    #[test]
    fn test_chunk_splitting() {
        let pipeline = ConstructionPipeline::new(PipelineConfig::default());
        let content = "First sentence. Second sentence! Third sentence?";
        let chunks = pipeline.split_into_chunks(content).unwrap();
        assert_eq!(chunks.len(), 1); // Should be in one chunk as it's short
    }

    #[test]
    fn test_triple_filtering() {
        let mut pipeline = ConstructionPipeline::new(PipelineConfig {
            confidence_threshold: 0.8,
            ..Default::default()
        });

        let triples = vec![
            create_test_triple(0.9),
            create_test_triple(0.7), // Below threshold
            create_test_triple(0.85),
        ];

        let filtered = pipeline.filter_and_deduplicate_triples(triples).unwrap();
        assert_eq!(filtered.len(), 2); // Only two above threshold
    }

    fn create_test_triple(confidence: f32) -> Triple {
        use crate::triple::{Entity, Relation, EntityType, RelationType};

        let subject = Entity::new("Test Subject".to_string(), EntityType::Concept);
        let predicate = Relation::new("relatedTo".to_string(), RelationType::RelatedTo);
        let object = Entity::new("Test Object".to_string(), EntityType::Concept);

        Triple::new(subject, predicate, object, confidence, "test".to_string())
    }
}