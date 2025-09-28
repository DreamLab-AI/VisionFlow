//! Context window management for RAG retrieval
//!
//! Advanced context assembly, overlap handling, and compression for optimal retrieval

use crate::error::{Result, RetrieverError};
use crate::config::{ContextConfig, ContextAssemblyStrategy, OverlapHandling, ContextCompressionConfig, CompressionStrategy};
use crate::search::SearchResult;
use crate::ranking::ScoredDocument;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use unicode_segmentation::UnicodeSegmentation;

/// Context window containing assembled text and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindow {
    /// Final assembled context text
    pub content: String,

    /// Context segments with metadata
    pub segments: Vec<ContextSegment>,

    /// Total token count estimate
    pub token_count: usize,

    /// Assembly strategy used
    pub assembly_strategy: String,

    /// Compression applied
    pub compression_info: Option<CompressionInfo>,

    /// Context quality score
    pub quality_score: f32,

    /// Assembly metadata
    pub metadata: ContextMetadata,
}

/// Individual context segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSegment {
    /// Segment ID
    pub id: String,

    /// Segment text content
    pub content: String,

    /// Source document ID
    pub source_id: String,

    /// Relevance score
    pub relevance: f32,

    /// Position in original document
    pub original_position: usize,

    /// Position in final context
    pub context_position: usize,

    /// Segment type
    pub segment_type: SegmentType,

    /// Overlap information
    pub overlap_info: Option<OverlapInfo>,
}

/// Type of context segment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SegmentType {
    Document,
    Passage,
    Sentence,
    Summary,
    KeyPhrase,
    Entity,
    Bridge, // Connecting text between segments
}

/// Information about text overlaps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapInfo {
    /// Overlapping text
    pub overlap_text: String,

    /// Overlap percentage with previous segment
    pub overlap_percentage: f32,

    /// Resolution strategy applied
    pub resolution: String,
}

/// Compression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Original length before compression
    pub original_length: usize,

    /// Compressed length
    pub compressed_length: usize,

    /// Compression ratio
    pub compression_ratio: f32,

    /// Compression strategy used
    pub strategy: String,

    /// Quality preservation score
    pub quality_preservation: f32,
}

/// Context assembly metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMetadata {
    /// Assembly timestamp
    pub timestamp: i64,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Number of source documents
    pub source_count: usize,

    /// Assembly strategy details
    pub strategy_details: HashMap<String, String>,

    /// Warnings during assembly
    pub warnings: Vec<String>,
}

/// Context assembler trait for different assembly strategies
#[async_trait]
pub trait ContextAssembler: Send + Sync {
    /// Assemble context from search results
    async fn assemble(&self, results: &[ScoredDocument], max_tokens: usize) -> Result<ContextWindow>;

    /// Get assembler name
    fn name(&self) -> &str;
}

/// Simple concatenation assembler
pub struct ConcatenationAssembler {
    config: ContextConfig,
}

impl ConcatenationAssembler {
    pub fn new(config: ContextConfig) -> Self {
        Self { config }
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        // Simple token estimation: ~4 characters per token
        (text.len() as f32 / 4.0).ceil() as usize
    }

    fn split_into_segments(&self, result: &ScoredDocument) -> Vec<ContextSegment> {
        let sentences: Vec<&str> = result.result.content
            .unicode_sentences()
            .collect();

        sentences.into_iter()
            .enumerate()
            .map(|(i, sentence)| ContextSegment {
                id: format!("{}_{}", result.result.id, i),
                content: sentence.trim().to_string(),
                source_id: result.result.id.clone(),
                relevance: result.ranking_score,
                original_position: i,
                context_position: 0, // Will be set during assembly
                segment_type: SegmentType::Sentence,
                overlap_info: None,
            })
            .filter(|seg| !seg.content.is_empty())
            .collect()
    }
}

#[async_trait]
impl ContextAssembler for ConcatenationAssembler {
    async fn assemble(&self, results: &[ScoredDocument], max_tokens: usize) -> Result<ContextWindow> {
        let start_time = std::time::Instant::now();
        let mut segments = Vec::new();
        let mut current_tokens = 0;
        let mut warnings = Vec::new();

        // Convert results to segments
        for result in results {
            let result_segments = self.split_into_segments(result);

            for segment in result_segments {
                let segment_tokens = self.estimate_tokens(&segment.content);

                if current_tokens + segment_tokens > max_tokens {
                    warnings.push(format!("Truncated at {} tokens", current_tokens));
                    break;
                }

                current_tokens += segment_tokens;
                segments.push(segment);
            }

            if current_tokens >= max_tokens {
                break;
            }
        }

        // Handle overlaps
        let processed_segments = self.handle_overlaps(segments).await?;

        // Update context positions
        let mut final_segments = Vec::new();
        for (i, mut segment) in processed_segments.into_iter().enumerate() {
            segment.context_position = i;
            final_segments.push(segment);
        }

        // Assemble final content
        let content = final_segments.iter()
            .map(|seg| seg.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ContextWindow {
            content,
            token_count: current_tokens,
            segments: final_segments,
            assembly_strategy: "concatenation".to_string(),
            compression_info: None,
            quality_score: self.calculate_quality_score(&final_segments),
            metadata: ContextMetadata {
                timestamp: chrono::Utc::now().timestamp(),
                processing_time_ms: processing_time,
                source_count: results.len(),
                strategy_details: HashMap::from([
                    ("max_tokens".to_string(), max_tokens.to_string()),
                    ("overlap_handling".to_string(), format!("{:?}", self.config.overlap_handling)),
                ]),
                warnings,
            },
        })
    }

    fn name(&self) -> &str {
        "concatenation"
    }
}

impl ConcatenationAssembler {
    async fn handle_overlaps(&self, segments: Vec<ContextSegment>) -> Result<Vec<ContextSegment>> {
        if segments.is_empty() {
            return Ok(segments);
        }

        let mut processed = Vec::new();
        processed.push(segments[0].clone());

        for current_segment in segments.into_iter().skip(1) {
            let previous_segment = processed.last().unwrap();
            let overlap = self.detect_overlap(&previous_segment.content, &current_segment.content);

            match self.config.overlap_handling {
                OverlapHandling::Remove => {
                    if overlap.overlap_percentage < 0.5 {
                        processed.push(current_segment);
                    }
                }
                OverlapHandling::Merge => {
                    if overlap.overlap_percentage > 0.3 {
                        let merged_content = self.merge_segments(&previous_segment.content, &current_segment.content);
                        let mut merged_segment = current_segment.clone();
                        merged_segment.content = merged_content;
                        merged_segment.overlap_info = Some(overlap);
                        processed.pop(); // Remove previous
                        processed.push(merged_segment);
                    } else {
                        processed.push(current_segment);
                    }
                }
                OverlapHandling::Preserve => {
                    let mut segment_with_overlap = current_segment;
                    segment_with_overlap.overlap_info = Some(overlap);
                    processed.push(segment_with_overlap);
                }
                OverlapHandling::Deduplicate => {
                    let deduplicated_content = self.deduplicate_content(&previous_segment.content, &current_segment.content);
                    let mut deduplicated_segment = current_segment;
                    deduplicated_segment.content = deduplicated_content;
                    deduplicated_segment.overlap_info = Some(overlap);
                    processed.push(deduplicated_segment);
                }
            }
        }

        Ok(processed)
    }

    fn detect_overlap(&self, text1: &str, text2: &str) -> OverlapInfo {
        let words1: Vec<&str> = text1.split_whitespace().collect();
        let words2: Vec<&str> = text2.split_whitespace().collect();

        let set1: HashSet<&str> = words1.iter().cloned().collect();
        let set2: HashSet<&str> = words2.iter().cloned().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        let overlap_percentage = if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        };

        // Find longest common substring for overlap text
        let overlap_text = self.longest_common_substring(text1, text2);

        OverlapInfo {
            overlap_text,
            overlap_percentage,
            resolution: "detected".to_string(),
        }
    }

    fn longest_common_substring(&self, s1: &str, s2: &str) -> String {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let mut max_length = 0;
        let mut ending_pos = 0;

        let mut dp = vec![vec![0; chars2.len() + 1]; chars1.len() + 1];

        for i in 1..=chars1.len() {
            for j in 1..=chars2.len() {
                if chars1[i - 1] == chars2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    if dp[i][j] > max_length {
                        max_length = dp[i][j];
                        ending_pos = i;
                    }
                }
            }
        }

        if max_length == 0 {
            String::new()
        } else {
            chars1[ending_pos - max_length..ending_pos].iter().collect()
        }
    }

    fn merge_segments(&self, text1: &str, text2: &str) -> String {
        let overlap = self.longest_common_substring(text1, text2);
        if overlap.is_empty() {
            format!("{} {}", text1, text2)
        } else {
            let text1_without_overlap = text1.replace(&overlap, "");
            format!("{} {} {}", text1_without_overlap.trim(), overlap, text2)
        }
    }

    fn deduplicate_content(&self, text1: &str, text2: &str) -> String {
        let words1: HashSet<&str> = text1.split_whitespace().collect();
        let words2: Vec<&str> = text2.split_whitespace().collect();

        let deduplicated_words: Vec<&str> = words2.into_iter()
            .filter(|word| !words1.contains(word))
            .collect();

        deduplicated_words.join(" ")
    }

    fn calculate_quality_score(&self, segments: &[ContextSegment]) -> f32 {
        if segments.is_empty() {
            return 0.0;
        }

        let avg_relevance = segments.iter().map(|s| s.relevance).sum::<f32>() / segments.len() as f32;
        let coherence_score = self.calculate_coherence(segments);
        let coverage_score = self.calculate_coverage(segments);

        (avg_relevance + coherence_score + coverage_score) / 3.0
    }

    fn calculate_coherence(&self, segments: &[ContextSegment]) -> f32 {
        if segments.len() < 2 {
            return 1.0;
        }

        let mut coherence_sum = 0.0;
        for window in segments.windows(2) {
            let similarity = self.calculate_text_similarity(&window[0].content, &window[1].content);
            coherence_sum += similarity;
        }

        coherence_sum / (segments.len() - 1) as f32
    }

    fn calculate_coverage(&self, segments: &[ContextSegment]) -> f32 {
        let total_words: HashSet<&str> = segments.iter()
            .flat_map(|s| s.content.split_whitespace())
            .collect();

        let unique_sources: HashSet<&str> = segments.iter()
            .map(|s| s.source_id.as_str())
            .collect();

        // Coverage is higher when we have diverse content from multiple sources
        let word_diversity = total_words.len() as f32 / segments.len().max(1) as f32;
        let source_diversity = unique_sources.len() as f32 / segments.len().max(1) as f32;

        ((word_diversity / 10.0).min(1.0) + source_diversity) / 2.0
    }

    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        let words1: HashSet<&str> = text1.split_whitespace().collect();
        let words2: HashSet<&str> = text2.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

/// Hierarchical context assembler
pub struct HierarchicalAssembler {
    config: ContextConfig,
}

impl HierarchicalAssembler {
    pub fn new(config: ContextConfig) -> Self {
        Self { config }
    }

    fn organize_by_hierarchy(&self, results: &[ScoredDocument]) -> Vec<HierarchyLevel> {
        let mut levels = Vec::new();

        // Level 1: Summaries and key phrases
        let summaries: Vec<_> = results.iter()
            .filter(|r| r.result.result_type == crate::search::ResultType::Summary)
            .cloned()
            .collect();

        if !summaries.is_empty() {
            levels.push(HierarchyLevel {
                level: 1,
                title: "Summaries".to_string(),
                documents: summaries,
            });
        }

        // Level 2: Main documents
        let documents: Vec<_> = results.iter()
            .filter(|r| r.result.result_type == crate::search::ResultType::Document)
            .cloned()
            .collect();

        if !documents.is_empty() {
            levels.push(HierarchyLevel {
                level: 2,
                title: "Documents".to_string(),
                documents,
            });
        }

        // Level 3: Passages and entities
        let details: Vec<_> = results.iter()
            .filter(|r| matches!(r.result.result_type,
                crate::search::ResultType::Passage |
                crate::search::ResultType::Entity))
            .cloned()
            .collect();

        if !details.is_empty() {
            levels.push(HierarchyLevel {
                level: 3,
                title: "Details".to_string(),
                documents: details,
            });
        }

        levels
    }
}

#[derive(Debug, Clone)]
struct HierarchyLevel {
    level: usize,
    title: String,
    documents: Vec<ScoredDocument>,
}

#[async_trait]
impl ContextAssembler for HierarchicalAssembler {
    async fn assemble(&self, results: &[ScoredDocument], max_tokens: usize) -> Result<ContextWindow> {
        let start_time = std::time::Instant::now();
        let levels = self.organize_by_hierarchy(results);

        let mut segments = Vec::new();
        let mut current_tokens = 0;
        let mut warnings = Vec::new();

        // Allocate tokens proportionally to hierarchy levels
        let token_allocation = self.allocate_tokens(&levels, max_tokens);

        for (level, allocation) in levels.iter().zip(token_allocation.iter()) {
            // Add level header
            let header_segment = ContextSegment {
                id: format!("header_{}", level.level),
                content: format!("=== {} ===", level.title),
                source_id: "hierarchy".to_string(),
                relevance: 1.0,
                original_position: 0,
                context_position: segments.len(),
                segment_type: SegmentType::Summary,
                overlap_info: None,
            };

            let header_tokens = header_segment.content.len() / 4;
            if current_tokens + header_tokens <= max_tokens {
                current_tokens += header_tokens;
                segments.push(header_segment);
            }

            // Add documents from this level
            let concat_assembler = ConcatenationAssembler::new(self.config.clone());
            let level_context = concat_assembler.assemble(&level.documents, *allocation).await?;

            for mut segment in level_context.segments {
                if current_tokens + segment.content.len() / 4 > max_tokens {
                    warnings.push(format!("Truncated at level {} due to token limit", level.level));
                    break;
                }

                segment.context_position = segments.len();
                current_tokens += segment.content.len() / 4;
                segments.push(segment);
            }

            if current_tokens >= max_tokens {
                break;
            }
        }

        // Assemble final content
        let content = segments.iter()
            .map(|seg| seg.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(ContextWindow {
            content,
            token_count: current_tokens,
            segments,
            assembly_strategy: "hierarchical".to_string(),
            compression_info: None,
            quality_score: 0.8, // Hierarchical tends to be well-organized
            metadata: ContextMetadata {
                timestamp: chrono::Utc::now().timestamp(),
                processing_time_ms: processing_time,
                source_count: results.len(),
                strategy_details: HashMap::from([
                    ("levels".to_string(), levels.len().to_string()),
                    ("max_tokens".to_string(), max_tokens.to_string()),
                ]),
                warnings,
            },
        })
    }

    fn name(&self) -> &str {
        "hierarchical"
    }
}

impl HierarchicalAssembler {
    fn allocate_tokens(&self, levels: &[HierarchyLevel], max_tokens: usize) -> Vec<usize> {
        let total_docs: usize = levels.iter().map(|l| l.documents.len()).sum();
        if total_docs == 0 {
            return vec![];
        }

        levels.iter()
            .map(|level| {
                let proportion = level.documents.len() as f32 / total_docs as f32;
                (proportion * max_tokens as f32) as usize
            })
            .collect()
    }
}

/// Context compressor for reducing context size while preserving quality
pub struct ContextCompressor {
    config: ContextCompressionConfig,
}

impl ContextCompressor {
    pub fn new(config: ContextCompressionConfig) -> Self {
        Self { config }
    }

    /// Compress context window
    pub async fn compress(&self, context: ContextWindow) -> Result<ContextWindow> {
        if !self.config.enabled {
            return Ok(context);
        }

        let target_length = (context.content.len() as f32 * self.config.ratio) as usize;

        let compressed_content = match self.config.strategy {
            CompressionStrategy::Summarization => self.summarize(&context.content, target_length).await?,
            CompressionStrategy::KeySentences => self.extract_key_sentences(&context, target_length).await?,
            CompressionStrategy::Clustering => self.cluster_and_represent(&context, target_length).await?,
            CompressionStrategy::Abstractive => self.abstractive_compression(&context.content, target_length).await?,
            CompressionStrategy::Extractive => self.extractive_compression(&context, target_length).await?,
        };

        let compression_info = CompressionInfo {
            original_length: context.content.len(),
            compressed_length: compressed_content.len(),
            compression_ratio: compressed_content.len() as f32 / context.content.len() as f32,
            strategy: format!("{:?}", self.config.strategy),
            quality_preservation: self.calculate_quality_preservation(&context.content, &compressed_content),
        };

        Ok(ContextWindow {
            content: compressed_content,
            token_count: (context.token_count as f32 * compression_info.compression_ratio) as usize,
            segments: context.segments, // Keep original segments for reference
            assembly_strategy: context.assembly_strategy,
            compression_info: Some(compression_info),
            quality_score: context.quality_score * compression_info.quality_preservation,
            metadata: context.metadata,
        })
    }

    async fn summarize(&self, content: &str, target_length: usize) -> Result<String> {
        // Simple extractive summarization
        let sentences: Vec<&str> = content.unicode_sentences().collect();
        let target_sentences = (sentences.len() as f32 * self.config.ratio).ceil() as usize;

        // Score sentences by position and length
        let mut scored_sentences: Vec<(usize, f32)> = sentences.iter()
            .enumerate()
            .map(|(i, sentence)| {
                let position_score = 1.0 - (i as f32 / sentences.len() as f32); // Earlier is better
                let length_score = (sentence.len() as f32 / 100.0).min(1.0); // Moderate length is better
                (i, position_score + length_score)
            })
            .collect();

        scored_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected_indices: Vec<usize> = scored_sentences.iter()
            .take(target_sentences)
            .map(|(i, _)| *i)
            .collect();

        let mut selected_indices = selected_indices;
        selected_indices.sort();

        Ok(selected_indices.iter()
            .map(|&i| sentences[i])
            .collect::<Vec<_>>()
            .join(" "))
    }

    async fn extract_key_sentences(&self, context: &ContextWindow, target_length: usize) -> Result<String> {
        // Extract sentences with highest relevance scores
        let mut sentence_scores: Vec<(String, f32)> = Vec::new();

        for segment in &context.segments {
            let sentences: Vec<&str> = segment.content.unicode_sentences().collect();
            for sentence in sentences {
                sentence_scores.push((sentence.to_string(), segment.relevance));
            }
        }

        sentence_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected_content = String::new();
        for (sentence, _) in sentence_scores {
            if selected_content.len() + sentence.len() > target_length {
                break;
            }
            if !selected_content.is_empty() {
                selected_content.push(' ');
            }
            selected_content.push_str(&sentence);
        }

        Ok(selected_content)
    }

    async fn cluster_and_represent(&self, context: &ContextWindow, target_length: usize) -> Result<String> {
        // Simple clustering by source and select representative sentences
        let mut source_groups: HashMap<String, Vec<&ContextSegment>> = HashMap::new();

        for segment in &context.segments {
            source_groups.entry(segment.source_id.clone()).or_default().push(segment);
        }

        let mut representatives = Vec::new();
        for (_source, segments) in source_groups {
            // Select best segment from each source
            if let Some(best_segment) = segments.iter().max_by(|a, b| a.relevance.partial_cmp(&b.relevance).unwrap_or(std::cmp::Ordering::Equal)) {
                representatives.push(best_segment.content.clone());
            }
        }

        let mut result = representatives.join(" ");
        if result.len() > target_length {
            result.truncate(target_length);
            if let Some(last_space) = result.rfind(' ') {
                result.truncate(last_space);
            }
        }

        Ok(result)
    }

    async fn abstractive_compression(&self, content: &str, target_length: usize) -> Result<String> {
        // Placeholder for abstractive compression using language models
        // In practice, you'd use a trained summarization model
        self.summarize(content, target_length).await
    }

    async fn extractive_compression(&self, context: &ContextWindow, target_length: usize) -> Result<String> {
        self.extract_key_sentences(context, target_length).await
    }

    fn calculate_quality_preservation(&self, original: &str, compressed: &str) -> f32 {
        let original_words: HashSet<&str> = original.split_whitespace().collect();
        let compressed_words: HashSet<&str> = compressed.split_whitespace().collect();

        let preserved_words = original_words.intersection(&compressed_words).count();
        if original_words.is_empty() {
            1.0
        } else {
            preserved_words as f32 / original_words.len() as f32
        }
    }
}

/// Main context manager orchestrating assembly and compression
pub struct ContextManager {
    config: ContextConfig,
    assembler: Box<dyn ContextAssembler>,
    compressor: Option<ContextCompressor>,
}

impl ContextManager {
    /// Create new context manager
    pub fn new(config: ContextConfig) -> Result<Self> {
        let assembler: Box<dyn ContextAssembler> = match config.assembly_strategy {
            ContextAssemblyStrategy::Concatenation => Box::new(ConcatenationAssembler::new(config.clone())),
            ContextAssemblyStrategy::Hierarchical => Box::new(HierarchicalAssembler::new(config.clone())),
            ContextAssemblyStrategy::GraphBased => {
                // TODO: Implement graph-based assembler
                Box::new(ConcatenationAssembler::new(config.clone()))
            }
            ContextAssemblyStrategy::Summarization => {
                // TODO: Implement summarization-based assembler
                Box::new(ConcatenationAssembler::new(config.clone()))
            }
            ContextAssemblyStrategy::Adaptive => {
                // TODO: Implement adaptive assembler
                Box::new(HierarchicalAssembler::new(config.clone()))
            }
        };

        let compressor = if config.compression.enabled {
            Some(ContextCompressor::new(config.compression.clone()))
        } else {
            None
        };

        Ok(Self {
            config,
            assembler,
            compressor,
        })
    }

    /// Create context window from search results
    pub async fn create_context(&self, results: &[ScoredDocument]) -> Result<ContextWindow> {
        if results.is_empty() {
            return Ok(ContextWindow {
                content: String::new(),
                segments: Vec::new(),
                token_count: 0,
                assembly_strategy: self.assembler.name().to_string(),
                compression_info: None,
                quality_score: 0.0,
                metadata: ContextMetadata {
                    timestamp: chrono::Utc::now().timestamp(),
                    processing_time_ms: 0,
                    source_count: 0,
                    strategy_details: HashMap::new(),
                    warnings: vec!["No results provided".to_string()],
                },
            });
        }

        // Assemble initial context
        let mut context = self.assembler.assemble(results, self.config.max_window_size).await?;

        // Apply compression if configured
        if let Some(ref compressor) = self.compressor {
            context = compressor.compress(context).await?;
        }

        Ok(context)
    }

    /// Update context with new results
    pub async fn update_context(&self, current_context: ContextWindow, new_results: &[ScoredDocument]) -> Result<ContextWindow> {
        // Combine current and new results
        let mut combined_results = Vec::new();

        // Convert current segments back to scored documents
        for segment in &current_context.segments {
            combined_results.push(ScoredDocument {
                result: crate::search::SearchResult {
                    id: segment.id.clone(),
                    score: segment.relevance,
                    content: segment.content.clone(),
                    result_type: match segment.segment_type {
                        SegmentType::Document => crate::search::ResultType::Document,
                        SegmentType::Passage => crate::search::ResultType::Passage,
                        SegmentType::Summary => crate::search::ResultType::Summary,
                        _ => crate::search::ResultType::Document,
                    },
                    metadata: HashMap::new(),
                    source_strategy: "context".to_string(),
                    strategy_scores: HashMap::new(),
                    explanation: "From existing context".to_string(),
                    snippet: None,
                },
                ranking_score: segment.relevance,
                feature_scores: HashMap::new(),
                ranking_explanation: "Existing segment".to_string(),
                original_position: segment.original_position,
                new_position: segment.context_position,
            });
        }

        // Add new results
        combined_results.extend(new_results.iter().cloned());

        // Reassemble with combined results
        self.create_context(&combined_results).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{SearchResult, ResultType};

    fn create_test_scored_document(id: &str, content: &str, score: f32) -> ScoredDocument {
        ScoredDocument {
            result: SearchResult {
                id: id.to_string(),
                score,
                content: content.to_string(),
                result_type: ResultType::Document,
                metadata: HashMap::new(),
                source_strategy: "test".to_string(),
                strategy_scores: HashMap::new(),
                explanation: "test".to_string(),
                snippet: None,
            },
            ranking_score: score,
            feature_scores: HashMap::new(),
            ranking_explanation: "test".to_string(),
            original_position: 0,
            new_position: 0,
        }
    }

    #[tokio::test]
    async fn test_concatenation_assembler() {
        let config = ContextConfig::default();
        let assembler = ConcatenationAssembler::new(config);

        let results = vec![
            create_test_scored_document("doc1", "This is the first document. It contains important information.", 0.9),
            create_test_scored_document("doc2", "This is the second document. It has additional details.", 0.8),
        ];

        let context = assembler.assemble(&results, 1000).await.unwrap();

        assert!(!context.content.is_empty());
        assert_eq!(context.segments.len(), 4); // 2 sentences per document
        assert_eq!(context.assembly_strategy, "concatenation");
    }

    #[tokio::test]
    async fn test_hierarchical_assembler() {
        let config = ContextConfig::default();
        let assembler = HierarchicalAssembler::new(config);

        let results = vec![
            create_test_scored_document("doc1", "This is a document.", 0.9),
            create_test_scored_document("doc2", "This is another document.", 0.8),
        ];

        let context = assembler.assemble(&results, 1000).await.unwrap();

        assert!(!context.content.is_empty());
        assert_eq!(context.assembly_strategy, "hierarchical");
        assert!(context.content.contains("=== Documents ==="));
    }

    #[tokio::test]
    async fn test_context_compression() {
        let config = ContextCompressionConfig {
            enabled: true,
            strategy: CompressionStrategy::KeySentences,
            ratio: 0.5,
        };

        let compressor = ContextCompressor::new(config);

        let context = ContextWindow {
            content: "This is a long document with multiple sentences. Each sentence contains different information. Some sentences are more important than others. The compression should preserve the most relevant content.".to_string(),
            segments: vec![],
            token_count: 100,
            assembly_strategy: "test".to_string(),
            compression_info: None,
            quality_score: 1.0,
            metadata: ContextMetadata {
                timestamp: 0,
                processing_time_ms: 0,
                source_count: 1,
                strategy_details: HashMap::new(),
                warnings: vec![],
            },
        };

        let compressed = compressor.compress(context).await.unwrap();

        assert!(compressed.content.len() < 200); // Should be compressed
        assert!(compressed.compression_info.is_some());
        assert!(compressed.compression_info.unwrap().compression_ratio < 1.0);
    }

    #[test]
    fn test_overlap_detection() {
        let config = ContextConfig::default();
        let assembler = ConcatenationAssembler::new(config);

        let overlap = assembler.detect_overlap(
            "machine learning algorithms",
            "learning algorithms and neural networks"
        );

        assert!(overlap.overlap_percentage > 0.0);
        assert!(!overlap.overlap_text.is_empty());
    }

    #[tokio::test]
    async fn test_context_manager() {
        let config = ContextConfig::default();
        let manager = ContextManager::new(config).unwrap();

        let results = vec![
            create_test_scored_document("doc1", "First document content.", 0.9),
            create_test_scored_document("doc2", "Second document content.", 0.8),
        ];

        let context = manager.create_context(&results).await.unwrap();

        assert!(!context.content.is_empty());
        assert_eq!(context.metadata.source_count, 2);
        assert!(context.quality_score > 0.0);
    }
}
"